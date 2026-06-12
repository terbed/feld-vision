from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from feldvision.logging import ExperimentLogger, NullLogger
from feldvision.train.metrics import SegmentationMetrics
from feldvision.train.optimization import EarlyStopping, WarmupCosineScheduler


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: dict[str, float]


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: nn.Module,
    *,
    class_names: tuple[str, ...],
    ignore_index: int,
    device: torch.device,
    optimizer: Optimizer | None = None,
    amp_mode: str = "off",
    gradient_clip_norm: float | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(class_names=class_names, ignore_index=ignore_index)
    total_loss = 0.0
    total_items = 0
    autocast_enabled = (amp_mode == "bf16" and device.type in {"cuda", "cpu"}) or (
        amp_mode == "fp16" and device.type == "cuda"
    )
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16

    for batch in loader:
        detail = batch["detail"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        context = batch.get("context")
        if context is not None:
            context = context.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(detail, context)
                loss = loss_fn(logits, target)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()

        batch_size = len(detail)
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        style_ids = batch["meta"]["style_id"]
        metrics.update(logits.detach(), target, style_ids)

    if total_items == 0:
        raise ValueError("data loader produced no batches")
    return EpochResult(loss=total_loss / total_items, metrics=metrics.compute())


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: WarmupCosineScheduler,
        loss_fn: nn.Module,
        class_names: tuple[str, ...],
        ignore_index: int,
        device: torch.device,
        epochs: int,
        checkpoint_dir: str | Path,
        early_stopping: EarlyStopping,
        monitor: str = "val/miou",
        amp_mode: str = "off",
        gradient_clip_norm: float | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping = early_stopping
        self.monitor = monitor
        self.amp_mode = amp_mode
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger or NullLogger()
        scaler_enabled = amp_mode == "fp16" and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def fit(
        self,
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]],
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(self.epochs):
            self.scheduler.step(epoch)
            train_result = run_epoch(
                self.model,
                train_loader,
                self.loss_fn,
                class_names=self.class_names,
                ignore_index=self.ignore_index,
                device=self.device,
                optimizer=self.optimizer,
                amp_mode=self.amp_mode,
                gradient_clip_norm=self.gradient_clip_norm,
                scaler=self.scaler,
            )
            with torch.no_grad():
                val_result = run_epoch(
                    self.model,
                    val_loader,
                    self.loss_fn,
                    class_names=self.class_names,
                    ignore_index=self.ignore_index,
                    device=self.device,
                    amp_mode=self.amp_mode,
                )
            record = _epoch_record(
                train_result,
                val_result,
                lr=float(self.optimizer.param_groups[0]["lr"]),
            )
            history.append(record)
            self.logger.log_metrics(record, step=epoch)

            monitor_key = self.monitor.removeprefix("val/")
            if monitor_key not in val_result.metrics:
                raise KeyError(f"monitor metric {self.monitor!r} was not produced")
            monitor_value = val_result.metrics[monitor_key]
            if not math.isfinite(monitor_value):
                raise RuntimeError(f"monitor metric {self.monitor!r} is not finite")
            improved, should_stop = self.early_stopping.update(monitor_value, epoch)
            self._save_checkpoint("last.pt", epoch, monitor_value)
            if improved:
                best_path = self._save_checkpoint("best.pt", epoch, monitor_value)
                self.logger.log_artifact("best_checkpoint", best_path)
            if should_stop:
                break
        return history

    def _save_checkpoint(self, name: str, epoch: int, monitor_value: float) -> Path:
        destination = self.checkpoint_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        state: Mapping[str, Any] = {
            "epoch": epoch,
            "monitor": monitor_value,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, temporary)
        temporary.replace(destination)
        return destination


def _epoch_record(
    train_result: EpochResult,
    val_result: EpochResult,
    *,
    lr: float,
) -> dict[str, float]:
    record = {
        "train/loss": train_result.loss,
        "val/loss": val_result.loss,
        "lr": lr,
    }
    record.update({f"train/{key}": value for key, value in train_result.metrics.items()})
    record.update({f"val/{key}": value for key, value in val_result.metrics.items()})
    return record
