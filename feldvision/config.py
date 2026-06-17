from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class DataConfig:
    basemap: str
    mask: str
    chips: str
    stylesheets: str
    mask_metadata: str


@dataclass(frozen=True)
class SplitConfig:
    test_style_ids: tuple[int, ...] = ()
    val_fraction: float = 0.20
    val_cell_size_px: int = 1024
    chip_size: int = 256
    stride: int = 128
    seed: int = 0


@dataclass(frozen=True)
class SamplerConfig:
    strategy: Literal[
        "rarest_class",
        "uniform",
        "inverse_freq",
        "class_balanced_targeted",
    ] = "rarest_class"
    alpha: float = 0.5
    w_background: float = 0.01
    max_class_oversample_factor: float = 10.0
    min_sampling_chips: int = 100


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    jitter: bool = True
    max_jitter_px: int = 64
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    brightness_contrast_p: float = 0.3
    hue_saturation_p: float = 0.2


@dataclass(frozen=True)
class LossConfig:
    type: Literal["ce_dice", "cross_entropy", "dice"] = "ce_dice"
    class_weight_beta: float = 0.5
    max_class_weight_ratio: float = 5.0
    ignore_index: int = 255
    ce_weight: float = 1.0
    dice_weight: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    name: str = "segformer_b2_single"
    pretrained: bool = True
    pretrained_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    context_scale: int = 4


@dataclass(frozen=True)
class OptimConfig:
    name: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 40
    amp: Literal["off", "fp16", "bf16"] = "fp16"
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class SchedulerConfig:
    name: Literal["cosine", "reduce_on_plateau"] = "reduce_on_plateau"
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    factor: float = 0.5
    patience: int = 2
    threshold: float = 0.001


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: str = "val/miou"
    mode: Literal["min", "max"] = "max"
    start_epoch: int = 3
    patience: int = 8
    min_delta: float = 0.001


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    samples_per_epoch: int | None = 12800


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs"
    experiment_name: str = "experiment"


@dataclass(frozen=True)
class ClearMLConfig:
    enabled: bool = False
    project: str = "feld-vision"
    task: str = "experiment"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    taxonomy: str
    split: SplitConfig = field(default_factory=SplitConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)

    @property
    def buffer_steps(self) -> int:
        jitter = self.augmentation.max_jitter_px if self.augmentation.jitter else 0
        numerator = self.split.chip_size + jitter
        return (numerator + self.split.stride - 1) // self.split.stride - 1

    def validate(self) -> None:
        if not 0 < self.split.val_fraction < 1:
            raise ConfigError("split.val_fraction must be between 0 and 1")
        if self.split.chip_size <= 0 or self.split.stride <= 0:
            raise ConfigError("split chip_size and stride must be positive")
        if self.split.val_cell_size_px < self.split.chip_size:
            raise ConfigError("split.val_cell_size_px must be at least chip_size")
        if self.split.val_cell_size_px % self.split.stride:
            raise ConfigError("split.val_cell_size_px must be divisible by stride")
        if self.augmentation.max_jitter_px < 0:
            raise ConfigError("augmentation.max_jitter_px cannot be negative")
        if self.augmentation.max_jitter_px > self.split.stride // 2:
            raise ConfigError("augmentation.max_jitter_px cannot exceed half the stride")
        for name in (
            "horizontal_flip_p",
            "vertical_flip_p",
            "rotate90_p",
            "brightness_contrast_p",
            "hue_saturation_p",
        ):
            value = getattr(self.augmentation, name)
            if not 0 <= value <= 1:
                raise ConfigError(f"augmentation.{name} must be between 0 and 1")
        if self.model.context_scale < 1:
            raise ConfigError("model.context_scale must be at least 1")
        if self.loader.num_workers == 0 and self.loader.persistent_workers:
            raise ConfigError("loader.persistent_workers requires num_workers > 0")
        if self.optim.epochs <= 0:
            raise ConfigError("optim.epochs must be positive")
        if self.scheduler.warmup_epochs >= self.optim.epochs:
            raise ConfigError("scheduler.warmup_epochs must be less than optim.epochs")
        if not 0 < self.scheduler.factor < 1:
            raise ConfigError("scheduler.factor must be in (0, 1)")
        if self.scheduler.patience < 0:
            raise ConfigError("scheduler.patience cannot be negative")
        if self.scheduler.threshold < 0:
            raise ConfigError("scheduler.threshold cannot be negative")
        if self.loss.ignore_index != 255:
            raise ConfigError("v1 requires loss.ignore_index=255")


def _is_union(origin: Any) -> bool:
    return origin in (Union, UnionType)


def _convert_value(expected: Any, value: Any, path: str) -> Any:
    origin = get_origin(expected)
    args = get_args(expected)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path} must be one of {args}, got {value!r}")
        return value
    if _is_union(origin):
        if value is None and type(None) in args:
            return None
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _convert_value(non_none[0], value, path)
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a sequence")
        item_type = args[0] if args else Any
        converted = [_convert_value(item_type, item, path) for item in value]
        return tuple(converted) if origin is tuple else converted
    if hasattr(expected, "__dataclass_fields__"):
        return _dataclass_from_mapping(expected, value, path)
    if expected is Any:
        return value
    if expected is float and isinstance(value, (int, float)):
        return float(value)
    if expected is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected is bool and isinstance(value, bool):
        return value
    if expected is str and isinstance(value, str):
        return value
    if isinstance(expected, type) and isinstance(value, expected):
        return value
    raise ConfigError(f"{path} has invalid type: expected {expected}, got {type(value).__name__}")


def _dataclass_from_mapping(cls: type[Any], data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    field_map = {item.name: item for item in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ConfigError(f"{path} contains unknown keys: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {
        name: _convert_value(hints[name], value, f"{path}.{name}") for name, value in data.items()
    }
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{path} is incomplete: {exc}") from exc


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = _dataclass_from_mapping(ExperimentConfig, raw, "config")
    config.validate()
    return config


def resolve_project_path(value: str | Path, config_path: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_file = Path(config_path).resolve()
    for parent in (config_file.parent, *config_file.parents):
        if (parent / "pyproject.toml").exists():
            return (parent / path).resolve()
    return (config_file.parent / path).resolve()
