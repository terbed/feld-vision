"""Model registry and built-in SegFormer variants."""

from feldvision.models.registry import available_models, build_model, register_model
from feldvision.models.segformer import (
    SegformerDualStream,
    SegformerSingleStream,
    segformer_b2_config,
)

__all__ = [
    "SegformerDualStream",
    "SegformerSingleStream",
    "available_models",
    "build_model",
    "register_model",
    "segformer_b2_config",
]
