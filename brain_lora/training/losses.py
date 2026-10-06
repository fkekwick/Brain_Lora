"""Segmentation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int = 4, ignore_background: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.cross_entropy = (
            nn.CrossEntropyLoss(ignore_index=0) if ignore_background else nn.CrossEntropyLoss()
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        start = 1 if self.ignore_background else 0
        probabilities = probabilities[:, start:]
        one_hot = one_hot[:, start:]
        axes = (0, *range(2, logits.ndim))
        intersection = (probabilities * one_hot).sum(axes)
        denominator = probabilities.sum(axes) + one_hot.sum(axes)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
        return self.cross_entropy(logits, target) + dice_loss
