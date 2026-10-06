"""Portable loaders for the four retained vision encoders."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

SUPPORTED_MODELS = ("dinov3", "siglip2", "biomedclip", "mermed")
DINOV3_REPOSITORY = "facebookresearch/dinov3:346f38fee679c56a6888f91c51670fae61d364e0"


def _checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint}")
    return checkpoint


def _unwrap_model(state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state


def _clean_load(model: nn.Module, state: dict[str, torch.Tensor], name: str) -> nn.Module:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("head.")]
    unexpected = [key for key in unexpected if not key.startswith("head.")]
    if missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint does not match its encoder: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model


def load_dinov3(path: str | Path) -> nn.Module:
    checkpoint = _checkpoint(path)
    model = torch.hub.load(
        DINOV3_REPOSITORY,
        "dinov3_vitb16",
        source="github",
        pretrained=False,
        skip_validation=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return _clean_load(model, _unwrap_model(state), "DINOv3")


def load_siglip2(path: str | Path) -> nn.Module:
    checkpoint = _checkpoint(path)
    model = timm.create_model("vit_base_patch16_siglip_256.webli", pretrained=False, img_size=224)
    state = dict(load_file(checkpoint))
    if "pos_embed" in state and state["pos_embed"].shape[1] != 196:
        position = state["pos_embed"]
        old_grid = int(position.shape[1] ** 0.5)
        if old_grid * old_grid != position.shape[1]:
            raise RuntimeError("SigLIP2 positional embedding is not a square patch grid.")
        position = position.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        position = F.interpolate(position, size=(14, 14), mode="bicubic", align_corners=False)
        state["pos_embed"] = position.permute(0, 2, 3, 1).reshape(1, 196, -1)
    return _clean_load(model, state, "SigLIP2")


def load_biomedclip(path: str | Path) -> nn.Module:
    checkpoint = _checkpoint(path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("BioMedCLIP checkpoint must contain a state dictionary.")
    prefix = "visual.trunk."
    trunk = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not trunk:
        raise RuntimeError("BioMedCLIP checkpoint has no visual.trunk tensors.")
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    return _clean_load(model, trunk, "BioMedCLIP")


def load_mermed(path: str | Path) -> nn.Module:
    checkpoint = _checkpoint(path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("MerMED checkpoint must contain a state dictionary.")
    state = state.get("teacher", state.get("model", state))
    prefix = "module.backbone."
    stripped = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    state = stripped or state
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    return _clean_load(model, state, "MerMED")


def load_encoder(name: str, checkpoint: str | Path) -> nn.Module:
    loaders = {
        "dinov3": load_dinov3,
        "siglip2": load_siglip2,
        "biomedclip": load_biomedclip,
        "mermed": load_mermed,
    }
    if name not in loaders:
        raise ValueError(f"Unknown model '{name}'. Expected one of {SUPPORTED_MODELS}.")
    encoder = loaders[name](checkpoint)
    encoder.requires_grad_(False)
    return encoder
