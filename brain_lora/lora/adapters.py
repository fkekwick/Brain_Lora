"""Q/V-only LoRA adapters for fused transformer QKV projections."""

from __future__ import annotations

import math

import torch
from torch import nn


class FusedQKVLoRA(nn.Module):
    """Add independent rank-limited updates to query and value projections."""

    def __init__(self, qkv: nn.Linear, rank: int) -> None:
        super().__init__()
        if qkv.out_features != 3 * qkv.in_features:
            raise ValueError("LoRA expects a fused qkv Linear with out_features=3*in_features.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.linear_a_q = nn.Linear(self.in_features, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, self.in_features, bias=False)
        self.linear_a_v = nn.Linear(self.in_features, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, self.in_features, bias=False)
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.qkv(x)
        query, key, value = base.chunk(3, dim=-1)
        query = query + self.linear_b_q(self.linear_a_q(x))
        value = value + self.linear_b_v(self.linear_a_v(x))
        return torch.cat((query, key, value), dim=-1)


def _selected_blocks(mode: str, count: int) -> set[int]:
    if mode == "all":
        return set(range(count))
    midpoint = count // 2
    if mode == "early":
        return set(range(midpoint))
    if mode == "late":
        return set(range(midpoint, count))
    raise ValueError("lora.layers must be one of: all, early, late")


def inject_qv_lora(encoder: nn.Module, rank: int, layers: str = "all") -> list[FusedQKVLoRA]:
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise TypeError("Encoder must expose transformer blocks as '.blocks'.")
    selected = _selected_blocks(layers, len(blocks))
    adapters: list[FusedQKVLoRA] = []
    for index, block in enumerate(blocks):
        if index not in selected:
            continue
        qkv = getattr(getattr(block, "attn", None), "qkv", None)
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Block {index} does not expose a fused attn.qkv Linear.")
        adapter = FusedQKVLoRA(qkv, rank)
        block.attn.qkv = adapter
        adapters.append(adapter)
    return adapters


def assert_adapter_keys(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    marker = (".linear_a_", ".linear_b_")
    expected = {name for name, _ in model.named_parameters() if any(x in name for x in marker)}
    actual = {name for name in state if any(x in name for x in marker)}
    if expected != actual:
        raise RuntimeError(
            "LoRA checkpoint does not match the configured adapter layers: "
            f"checkpoint={len(actual)} tensors, model={len(expected)} tensors."
        )
