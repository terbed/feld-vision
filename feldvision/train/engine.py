from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from PIL import Image

from feldvision.logging import ExperimentLogger, NullLogger
from feldvision.train.metrics import SegmentationMetrics
from feldvision.train.optimization import EarlyStopping, TrainingScheduler

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(3, 1, 1)
IGNORE_COLOR = np.asarray((255, 0, 255), dtype=np.uint8)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: dict[str, float]
    global_step: int = 0


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
    logger: ExperimentLogger | None = None,
    phase: str = "train",
    epoch: int = 0,
    total_epochs: int = 1,
    global_step: int = 0,
    log_interval_batches: int = 100,
    debug_sample_count: int = 0,
    debug_dir: str | Path | None = None,
    class_palette: np.ndarray | None = None,
    on_log_interval: Callable[[int, int], None] | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    logger = logger or NullLogger()
    metrics = SegmentationMetrics(class_names=class_names, ignore_index=ignore_index)
    total_loss = 0.0
    total_items = 0
    interval_loss = 0.0
    interval_items = 0
    autocast_enabled = (amp_mode == "bf16" and device.type in {"cuda", "cpu"}) or (
        amp_mode == "fp16" and device.type == "cuda"
    )
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
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
        loss_value = float(loss.detach())
        total_loss += loss_value * batch_size
        total_items += batch_size
        interval_loss += loss_value * batch_size
        interval_items += batch_size
        style_ids = batch["meta"]["style_id"]
        metrics.update(logits.detach(), target, style_ids)
        if (
            not training
            and batch_index == 1
            and debug_sample_count > 0
            and debug_dir is not None
            and class_palette is not None
        ):
            debug_path = _save_debug_samples(
                detail=detail,
                target=target,
                logits=logits,
                palette=class_palette,
                ignore_index=ignore_index,
                output_dir=debug_dir,
                epoch=epoch,
                max_samples=debug_sample_count,
            )
            logger.log_media("validation_predictions", debug_path, step=epoch)
            print(f"epoch {epoch + 1}/{total_epochs} logged debug samples {debug_path}", flush=True)
        if training:
            global_step += 1
            should_log = batch_index % log_interval_batches == 0 or batch_index == total_batches
            if should_log:
                interval_average = interval_loss / max(interval_items, 1)
                running_average = total_loss / max(total_items, 1)
                logger.log_metrics(
                    {
                        f"{phase}/batch_loss": interval_average,
                        f"{phase}/running_loss": running_average,
                    },
                    step=global_step,
                )
                print(
                    f"epoch {epoch + 1}/{total_epochs} "
                    f"{phase} batch {batch_index}/{total_batches} "
                    f"batch_loss={interval_average:.6f} "
                    f"running_loss={running_average:.6f}",
                    flush=True,
                )
                if on_log_interval is not None:
                    on_log_interval(global_step, batch_index)
                    model.train(training)
                interval_loss = 0.0
                interval_items = 0

    if total_items == 0:
        raise ValueError("data loader produced no batches")
    return EpochResult(
        loss=total_loss / total_items,
        metrics=metrics.compute(),
        global_step=global_step,
    )


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: TrainingScheduler,
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
        log_interval_batches: int = 100,
        debug_sample_count: int = 4,
        class_palette: np.ndarray | None = None,
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
        self.log_interval_batches = log_interval_batches
        self.debug_sample_count = debug_sample_count
        self.class_palette = class_palette
        self.debug_dir = self.checkpoint_dir.parent / "debug_samples"
        self.validation_checks = 0
        scaler_enabled = amp_mode == "fp16" and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def fit(
        self,
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]],
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        global_step = 0
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        for epoch in range(self.epochs):
            self.scheduler.step_epoch_start(epoch)
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.logger.log_metrics({"lr": lr}, step=global_step)
            print(
                f"epoch {epoch + 1}/{self.epochs} starting "
                f"lr={lr:.8g} train_batches={train_batches} val_batches={val_batches}",
                flush=True,
            )
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
                logger=self.logger,
                phase="train",
                epoch=epoch,
                total_epochs=self.epochs,
                global_step=global_step,
                log_interval_batches=self.log_interval_batches,
            )
            global_step = train_result.global_step
            print(f"epoch {epoch + 1}/{self.epochs} validating", flush=True)
            with torch.no_grad():
                val_result = run_epoch(
                    self.model,
                    val_loader,
                    self.loss_fn,
                    class_names=self.class_names,
                    ignore_index=self.ignore_index,
                    device=self.device,
                    amp_mode=self.amp_mode,
                    phase="val",
                    epoch=epoch,
                    total_epochs=self.epochs,
                    global_step=global_step,
                    log_interval_batches=self.log_interval_batches,
                    logger=self.logger,
                    debug_sample_count=self.debug_sample_count,
                    debug_dir=self.debug_dir,
                    class_palette=self.class_palette,
                )
            record = _epoch_record(
                train_result,
                val_result,
                lr=lr,
            )
            history.append(record)
            self.logger.log_metrics(record, step=epoch)

            monitor_key = self.monitor.removeprefix("val/")
            if monitor_key not in val_result.metrics:
                raise KeyError(f"monitor metric {self.monitor!r} was not produced")
            monitor_value = val_result.metrics[monitor_key]
            if not math.isfinite(monitor_value):
                raise RuntimeError(f"monitor metric {self.monitor!r} is not finite")
            self.scheduler.step_validation(monitor_value)
            updated_lr = float(self.optimizer.param_groups[0]["lr"])
            if updated_lr != lr:
                self.logger.log_metrics({"lr": updated_lr}, step=global_step)
                print(f"lr reduced to {updated_lr:.8g}", flush=True)
            self.validation_checks += 1
            improved, should_stop = self.early_stopping.update(
                monitor_value,
                self.validation_checks,
            )
            self._save_checkpoint("last.pt", epoch, monitor_value)
            if improved:
                best_path = self._save_checkpoint("best.pt", epoch, monitor_value)
                self.logger.log_artifact("best_checkpoint", best_path)
            print(_format_epoch_summary(epoch, self.epochs, record, self.monitor), flush=True)
            if should_stop:
                print(f"early stopping after epoch {epoch + 1}", flush=True)
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


def _format_epoch_summary(
    epoch: int,
    total_epochs: int,
    record: Mapping[str, float],
    monitor: str,
) -> str:
    parts = [
        f"epoch {epoch + 1}/{total_epochs} complete",
        f"train/loss={record['train/loss']:.6f}",
        f"val/loss={record['val/loss']:.6f}",
    ]
    if monitor in record:
        parts.append(f"{monitor}={record[monitor]:.6f}")
    if "val/miou" in record and monitor != "val/miou":
        parts.append(f"val/miou={record['val/miou']:.6f}")
    return " ".join(parts)


def _save_debug_samples(
    *,
    detail: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    palette: np.ndarray,
    ignore_index: int,
    output_dir: str | Path,
    epoch: int,
    max_samples: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions = logits.detach().argmax(dim=1).to(device="cpu", dtype=torch.long)
    detail_cpu = detail.detach().to(device="cpu", dtype=torch.float32)
    target_cpu = target.detach().to(device="cpu", dtype=torch.long)
    rows: list[np.ndarray] = []
    count = min(max_samples, len(detail_cpu))
    separator = np.full((detail_cpu.shape[-2], 4, 3), 255, dtype=np.uint8)
    for index in range(count):
        image = _denormalize_image(detail_cpu[index])
        target_image = _colorize_mask(target_cpu[index], palette, ignore_index)
        prediction_image = _colorize_mask(predictions[index], palette, ignore_index)
        rows.append(np.concatenate([image, separator, target_image, separator, prediction_image], axis=1))
    grid = np.concatenate(rows, axis=0)
    destination = output_path / f"val_debug_epoch_{epoch + 1:03d}.png"
    Image.fromarray(grid).save(destination)
    return destination


def _denormalize_image(image: torch.Tensor) -> np.ndarray:
    restored = image * IMAGENET_STD + IMAGENET_MEAN
    restored = restored.clamp(0, 1).permute(1, 2, 0).numpy()
    return (restored * 255).round().astype(np.uint8)


def _colorize_mask(mask: torch.Tensor, palette: np.ndarray, ignore_index: int) -> np.ndarray:
    values = mask.numpy()
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    valid = (values >= 0) & (values < len(palette))
    output[valid] = palette[values[valid]]
    output[values == ignore_index] = IGNORE_COLOR
    return output
