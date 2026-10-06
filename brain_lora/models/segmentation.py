"""Shared early-fusion LoRA segmentation model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from brain_lora.lora import inject_qv_lora

from .backbones import load_encoder

DEFAULT_FEATURE_LAYERS = (2, 5, 8, 11)
NORMALIZATION = {
    "dinov3": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "mermed": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "siglip2": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "biomedclip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}


class LoRASegmenter(nn.Module):
    """Frozen ViT, trainable Q/V LoRA, modality fusion, and segmentation head."""

    def __init__(
        self,
        encoder: nn.Module,
        model_name: str,
        rank: int = 8,
        lora_layers: str = "all",
        feature_layers: tuple[int, ...] = DEFAULT_FEATURE_LAYERS,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 4,
        output_size: int = 240,
    ) -> None:
        super().__init__()
        if model_name not in NORMALIZATION:
            raise ValueError(f"No normalization registered for model '{model_name}'.")
        self.encoder = encoder.requires_grad_(False)
        # Keep the historical module names so trainable-only research checkpoints
        # load without a key translation.
        self.lora_layers = nn.ModuleList(inject_qv_lora(self.encoder, rank, lora_layers))
        self.feature_layers = tuple(feature_layers)
        self.output_size = output_size
        self.modality_fusion = nn.Conv2d(4, 3, 1, bias=False)
        nn.init.constant_(self.modality_fusion.weight, 0.25)
        mean, std = NORMALIZATION[model_name]
        self.register_buffer("input_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(std).view(1, 3, 1, 1))
        self.seg_head = nn.Sequential(
            nn.Conv2d(len(feature_layers) * embed_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def encode(self, image: torch.Tensor, reshape: bool = True) -> tuple[torch.Tensor, ...]:
        fused = self.modality_fusion(image)
        normalized = (fused - self.input_mean) / self.input_std
        features = self.encoder.get_intermediate_layers(
            normalized, n=list(self.feature_layers), reshape=reshape, norm=True
        )
        return tuple(features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encode(image)
        logits = self.seg_head(torch.cat(features, dim=1))
        return F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value for name, value in self.state_dict().items() if name in trainable}

    @property
    def adapters(self) -> nn.ModuleList:
        """LoRA modules, exposed under the release's original convenience name."""
        return self.lora_layers

    @property
    def segmentation_head(self) -> nn.Sequential:
        """Segmentation head alias without changing historical checkpoint keys."""
        return self.seg_head


def build_segmenter(
    name: str,
    checkpoint: str | Path,
    rank: int = 8,
    lora_layers: str = "all",
    feature_layers: tuple[int, ...] = DEFAULT_FEATURE_LAYERS,
    hidden_dim: int = 256,
    output_size: int = 240,
) -> LoRASegmenter:
    encoder = load_encoder(name, checkpoint)
    return LoRASegmenter(
        encoder,
        name,
        rank=rank,
        lora_layers=lora_layers,
        feature_layers=feature_layers,
        hidden_dim=hidden_dim,
        output_size=output_size,
    )
