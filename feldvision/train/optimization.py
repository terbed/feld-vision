from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainingScheduler(Protocol):
    def step_epoch_start(self, epoch: int) -> None: ...

    def step_validation(self, value: float) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


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

    def step_epoch_start(self, epoch: int) -> None:
        self.step(epoch)

    def step_validation(self, value: float) -> None:
        del value


class WarmupReduceLROnPlateauScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_epochs: int,
        mode: str,
        factor: float,
        patience: int,
        min_lr: float,
        threshold: float,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if mode not in {"min", "max"}:
            raise ValueError("plateau mode must be 'min' or 'max'")
        if not 0 < factor < 1:
            raise ValueError("plateau factor must be in (0, 1)")
        if patience < 0:
            raise ValueError("plateau patience cannot be negative")
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        if any(min_lr > base_lr for base_lr in self.base_lrs):
            raise ValueError("min_lr cannot exceed an optimizer base learning rate")
        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode="abs",
            min_lr=min_lr,
        )
        self.last_epoch = -1

    def step_epoch_start(self, epoch: int) -> None:
        self.last_epoch = epoch
        if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
            return
        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = base_lr * (epoch + 1) / self.warmup_epochs

    def step_validation(self, value: float) -> None:
        if self.last_epoch >= self.warmup_epochs:
            self.plateau.step(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_epoch": self.last_epoch,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_epoch = int(state["last_epoch"])
        self.plateau.load_state_dict(state["plateau"])


@dataclass
class EarlyStopping:
    mode: str = "max"
    start_epoch: int = 3
    patience: int = 8
    min_delta: float = 0.001
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("early-stopping mode must be 'min' or 'max'")
        if self.start_epoch < 0 or self.patience < 1 or self.min_delta < 0:
            raise ValueError("invalid early-stopping parameters")

    def update(self, value: float, validation_check: int) -> tuple[bool, bool]:
        improved = self.best is None or self._improved(value)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True, False
        if validation_check >= self.start_epoch:
            self.bad_epochs += 1
        return (
            False,
            validation_check >= self.start_epoch and self.bad_epochs >= self.patience,
        )

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta
