from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from feldvision.config import ModelConfig


class ModelRegistryError(ValueError):
    """Raised when a requested model is unknown or registered twice."""


ModelBuilder = Callable[..., nn.Module]
_MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _MODEL_BUILDERS:
            raise ModelRegistryError(f"model {name!r} is already registered")
        _MODEL_BUILDERS[name] = builder
        return builder

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_BUILDERS))


def build_model(
    config: ModelConfig,
    num_classes: int,
    **kwargs: Any,
) -> nn.Module:
    try:
        builder = _MODEL_BUILDERS[config.name]
    except KeyError as exc:
        raise ModelRegistryError(
            f"unknown model {config.name!r}; available: {available_models()}"
        ) from exc
    return builder(config=config, num_classes=num_classes, **kwargs)
