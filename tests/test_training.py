from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from feldvision.train.engine import Trainer, run_epoch
from feldvision.train.losses import SegmentationLoss
from feldvision.train.optimization import (
    EarlyStopping,
    WarmupCosineScheduler,
    WarmupReduceLROnPlateauScheduler,
)


class TinyDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        target = torch.full((4, 4), index % 2, dtype=torch.long)
        return {
            "detail": torch.ones(3, 4, 4) * index,
            "target": target,
            "meta": {"style_id": index % 2, "chip_id": str(index)},
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 2, kernel_size=1)

    def forward(
        self,
        detail: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        return self.layer(detail)


def test_run_epoch_updates_model_and_returns_metrics() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    before = model.layer.weight.detach().clone()

    result = run_epoch(
        model,
        loader,
        SegmentationLoss(num_classes=2),
        class_names=("background", "river"),
        ignore_index=255,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    assert result.loss > 0
    assert "miou" in result.metrics
    assert not torch.equal(before, model.layer.weight)


def test_scheduler_warms_up_and_ends_at_minimum_lr() -> None:
    model = TinyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=6,
        warmup_epochs=2,
        min_lr=1e-5,
    )

    learning_rates = []
    for epoch in range(6):
        scheduler.step(epoch)
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == 5e-4
    assert learning_rates[1] == 1e-3
    assert learning_rates[-1] == 1e-5


def test_reduce_on_plateau_scheduler_reduces_after_validation_plateau() -> None:
    model = TinyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupReduceLROnPlateauScheduler(
        optimizer,
        warmup_epochs=1,
        mode="max",
        factor=0.5,
        patience=0,
        min_lr=1e-5,
        threshold=0.01,
    )

    scheduler.step_epoch_start(0)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    scheduler.step_validation(0.5)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    scheduler.step_epoch_start(1)
    scheduler.step_validation(0.505)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    scheduler.step_epoch_start(2)
    scheduler.step_validation(0.505)

    assert optimizer.param_groups[0]["lr"] == 5e-4


def test_early_stopping_respects_start_epoch_and_patience() -> None:
    stopping = EarlyStopping(mode="max", start_epoch=2, patience=2, min_delta=0.01)

    assert stopping.update(0.5, 0) == (True, False)
    assert stopping.update(0.505, 1) == (False, False)
    assert stopping.update(0.505, 2) == (False, False)
    assert stopping.update(0.505, 3) == (False, True)


def test_trainer_writes_best_and_last_checkpoints(tmp_path: Path) -> None:
    model = TinyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=WarmupCosineScheduler(
            optimizer,
            total_epochs=2,
            warmup_epochs=0,
            min_lr=1e-5,
        ),
        loss_fn=SegmentationLoss(num_classes=2),
        class_names=("background", "river"),
        ignore_index=255,
        device=torch.device("cpu"),
        epochs=2,
        checkpoint_dir=tmp_path,
        early_stopping=EarlyStopping(start_epoch=0, patience=2),
    )
    loader = DataLoader(TinyDataset(), batch_size=2)

    history = trainer.fit(loader, loader)

    assert history
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
