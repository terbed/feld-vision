from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background_dice: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("segmentation requires at least two classes")
        if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight <= 0:
            raise ValueError("loss component weights must be non-negative and not both zero")
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError("class_weights length must equal num_classes")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background_dice = include_background_dice
        self.epsilon = epsilon
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("expected logits [B,C,H,W] and target [B,H,W]")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("logits and target batch/spatial dimensions must match")
        valid = target.ne(self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0

        total = logits.new_zeros(())
        if self.ce_weight:
            total = total + self.ce_weight * F.cross_entropy(
                logits,
                target,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
            )
        if self.dice_weight:
            total = total + self.dice_weight * self._dice_loss(logits, target, valid)
        return total

    def _dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2)
        valid_channels = valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid_channels
        one_hot = one_hot.to(dtype=probabilities.dtype) * valid_channels
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
        dice = (2 * intersection + self.epsilon) / (denominator + self.epsilon)
        if not self.include_background_dice:
            dice = dice[1:]
        return 1 - dice.mean()
