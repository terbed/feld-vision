import pytest
import torch
from transformers import SegformerConfig

from feldvision.config import ModelConfig
from feldvision.models import SegformerDualStream, available_models, build_model


def tiny_config(num_classes: int = 3) -> SegformerConfig:
    return SegformerConfig(
        num_channels=3,
        depths=[1, 1, 1, 1],
        sr_ratios=[4, 2, 2, 1],
        hidden_sizes=[8, 16, 32, 64],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 1, 2, 4],
        mlp_ratios=[2, 2, 2, 2],
        decoder_hidden_size=16,
        num_labels=num_classes,
    )


@pytest.mark.parametrize(
    "name",
    [
        "segformer_b2_single",
        "segformer_b2_shared_context",
        "segformer_b2_separate_context",
    ],
)
def test_all_variants_return_detail_resolution_logits(name: str) -> None:
    model = build_model(
        ModelConfig(name=name, pretrained=False),
        num_classes=3,
        hf_config=tiny_config(),
    )
    detail = torch.randn(2, 3, 64, 64)
    context = None if name.endswith("single") else torch.randn(2, 3, 64, 64)

    logits = model(detail, context)

    assert logits.shape == (2, 3, 64, 64)


def test_shared_and_separate_encoder_ownership() -> None:
    shared = build_model(
        ModelConfig(name="segformer_b2_shared_context", pretrained=False),
        num_classes=3,
        hf_config=tiny_config(),
    )
    separate = build_model(
        ModelConfig(name="segformer_b2_separate_context", pretrained=False),
        num_classes=3,
        hf_config=tiny_config(),
    )

    assert isinstance(shared, SegformerDualStream)
    assert isinstance(separate, SegformerDualStream)
    assert shared.context_encoder is None
    assert separate.context_encoder is not None
    assert next(separate.detail_encoder.parameters()) is not next(
        separate.context_encoder.parameters()
    )


def test_dual_stream_requires_context() -> None:
    model = build_model(
        ModelConfig(name="segformer_b2_shared_context", pretrained=False),
        num_classes=3,
        hf_config=tiny_config(),
    )

    with pytest.raises(ValueError, match="context"):
        model(torch.randn(1, 3, 64, 64))


def test_registry_contains_exact_builtin_variants() -> None:
    assert available_models() == (
        "segformer_b2_separate_context",
        "segformer_b2_shared_context",
        "segformer_b2_single",
    )
