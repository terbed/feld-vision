from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from feldvision.config import ExperimentConfig, load_experiment_config, resolve_project_path
from feldvision.data.dataset import ChipDataset
from feldvision.data.sampler import (
    ClassBalancedSampler,
    compute_class_weights,
    compute_sampling_weights,
)
from feldvision.logging import ClearMLLogger, CompositeLogger, LocalLogger
from feldvision.models import build_model
from feldvision.taxonomy import Taxonomy, load_taxonomy
from feldvision.train.engine import Trainer
from feldvision.train.losses import SegmentationLoss
from feldvision.train.optimization import EarlyStopping, WarmupCosineScheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config_and_taxonomy(
    config_path: str | Path,
) -> tuple[ExperimentConfig, Taxonomy]:
    config = load_experiment_config(config_path)
    taxonomy_path = resolve_project_path(config.taxonomy, config_path)
    taxonomy = load_taxonomy(taxonomy_path)
    if taxonomy.ignore_index != config.loss.ignore_index:
        raise ValueError("taxonomy and loss ignore_index values differ")
    return config, taxonomy


def build_loaders(
    config: ExperimentConfig,
    taxonomy: Taxonomy,
    *,
    config_path: str | Path,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    pd.DataFrame,
]:
    chips_path = resolve_project_path(config.data.chips, config_path)
    chips = pd.read_parquet(chips_path)
    chip_sizes = chips["size"].dropna().unique()
    if len(chip_sizes) != 1 or int(chip_sizes[0]) != config.split.chip_size:
        raise ValueError(
            "chip table size does not match split.chip_size: "
            f"found {chip_sizes.tolist()}, expected {config.split.chip_size}"
        )
    basemap = resolve_project_path(config.data.basemap, config_path)
    mask = resolve_project_path(config.data.mask, config_path)
    include_context = config.model.name != "segformer_b2_single"
    train_dataset = ChipDataset(
        chips,
        split="train",
        basemap_path=basemap,
        mask_path=mask,
        taxonomy=taxonomy,
        augmentation=config.augmentation,
        include_context=include_context,
        context_scale=config.model.context_scale,
        seed=config.runtime.seed,
    )
    val_dataset = ChipDataset(
        chips,
        split="val",
        basemap_path=basemap,
        mask_path=mask,
        taxonomy=taxonomy,
        augmentation=config.augmentation,
        include_context=include_context,
        context_scale=config.model.context_scale,
        seed=config.runtime.seed,
    )
    sample_count = config.loader.samples_per_epoch or len(train_dataset)
    if config.sampler.strategy == "class_balanced_targeted":
        sampler = ClassBalancedSampler(
            train_dataset.chips,
            taxonomy,
            num_samples=sample_count,
            background_weight=config.sampler.w_background,
            seed=config.runtime.seed,
        )
    else:
        sampling = compute_sampling_weights(train_dataset.chips, taxonomy, config.sampler)
        generator = torch.Generator().manual_seed(config.runtime.seed)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sampling.values),
            num_samples=sample_count,
            replacement=True,
            generator=generator,
        )
    common = {
        "batch_size": config.loader.batch_size,
        "num_workers": config.loader.num_workers,
        "pin_memory": config.loader.pin_memory,
        "persistent_workers": (config.loader.persistent_workers and config.loader.num_workers > 0),
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, train_dataset.chips


def run_training(config_path: str | Path) -> list[dict[str, float]]:
    config, taxonomy = load_config_and_taxonomy(config_path)
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    train_loader, val_loader, train_chips = build_loaders(
        config,
        taxonomy,
        config_path=config_path,
    )
    model = build_model(config.model, taxonomy.num_classes)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=config.optim.epochs,
        warmup_epochs=config.scheduler.warmup_epochs,
        min_lr=config.scheduler.min_lr,
    )
    class_weights = compute_class_weights(
        train_chips,
        taxonomy,
        beta=config.loss.class_weight_beta,
        max_ratio=config.loss.max_class_weight_ratio,
    )
    ce_weight = config.loss.ce_weight if config.loss.type != "dice" else 0.0
    dice_weight = config.loss.dice_weight if config.loss.type != "cross_entropy" else 0.0
    loss_fn = SegmentationLoss(
        num_classes=taxonomy.num_classes,
        ignore_index=taxonomy.ignore_index,
        class_weights=torch.from_numpy(class_weights),
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )

    output_root = resolve_project_path(config.runtime.output_dir, config_path)
    run_dir = output_root / config.runtime.experiment_name
    config_mapping = asdict(config)
    local_logger = LocalLogger(run_dir, config_mapping)
    loggers = [local_logger]
    if config.clearml.enabled:
        loggers.append(
            ClearMLLogger(
                project=config.clearml.project,
                task_name=config.clearml.task,
                config=config_mapping,
            )
        )
    logger = CompositeLogger(*loggers)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        class_names=taxonomy.class_names,
        ignore_index=taxonomy.ignore_index,
        device=device,
        epochs=config.optim.epochs,
        checkpoint_dir=run_dir / "checkpoints",
        early_stopping=EarlyStopping(
            mode=config.early_stopping.mode,
            start_epoch=config.early_stopping.start_epoch,
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
        ),
        monitor=config.early_stopping.monitor,
        amp_mode=config.optim.amp,
        gradient_clip_norm=config.optim.gradient_clip_norm,
        logger=logger,
    )
    try:
        return trainer.fit(train_loader, val_loader)
    finally:
        logger.close()
