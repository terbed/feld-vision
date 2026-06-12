from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int,
        min_lr: float,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if not 0 <= warmup_epochs < total_epochs:
            raise ValueError("warmup_epochs must be in [0, total_epochs)")
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        if any(min_lr > base_lr for base_lr in self.base_lrs):
            raise ValueError("min_lr cannot exceed an optimizer base learning rate")
        self.min_lr = min_lr
        self.last_epoch = -1

    def step(self, epoch: int) -> None:
        self.last_epoch = epoch
        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = self._lr(epoch, base_lr)

    def _lr(self, epoch: int, base_lr: float) -> float:
        if self.warmup_epochs and epoch < self.warmup_epochs:
            return base_lr * (epoch + 1) / self.warmup_epochs
        decay_epochs = self.total_epochs - self.warmup_epochs
        decay_index = max(0, epoch - self.warmup_epochs)
        progress = min(decay_index / max(decay_epochs - 1, 1), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr - self.min_lr) * cosine

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_epoch = int(state["last_epoch"])


@dataclass
class EarlyStopping:
    mode: str = "max"
    start_epoch: int = 10
    patience: int = 15
    min_delta: float = 0.001
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("early-stopping mode must be 'min' or 'max'")
        if self.start_epoch < 0 or self.patience < 1 or self.min_delta < 0:
            raise ValueError("invalid early-stopping parameters")

    def update(self, value: float, epoch: int) -> tuple[bool, bool]:
        improved = self.best is None or self._improved(value)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True, False
        if epoch >= self.start_epoch:
            self.bad_epochs += 1
        return False, epoch >= self.start_epoch and self.bad_epochs >= self.patience

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta
