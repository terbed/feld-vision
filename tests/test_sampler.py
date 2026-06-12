import numpy as np
import pandas as pd
import pytest

from feldvision.config import SamplerConfig
from feldvision.data.sampler import (
    ClassBalancedSampler,
    compute_class_weights,
    compute_sampling_weights,
)
from feldvision.taxonomy import ClassDefinition, Taxonomy


def taxonomy() -> Taxonomy:
    return Taxonomy(
        classes=(
            ClassDefinition("background", 0, (0,), (0, 0, 0)),
            ClassDefinition("common", 1, (10,), (0, 0, 255)),
            ClassDefinition("rare", 2, (11,), (0, 255, 255)),
        ),
        ignored_raw_ids=frozenset({17}),
    )


def test_rare_class_gets_higher_weight_and_ignore_has_no_effect() -> None:
    chips = pd.DataFrame(
        {
            "px_bg": [100, 100, 100, 100],
            "px_10": [100, 100, 100, 0],
            "px_11": [0, 0, 1, 0],
            "px_17": [0, 999, 0, 999],
        }
    )
    config = SamplerConfig(min_sampling_chips=1, w_background=0.01)

    result = compute_sampling_weights(chips, taxonomy(), config)

    assert result.values[2] > result.values[0]
    assert result.values[0] == pytest.approx(result.values[1])
    assert result.values[3] < result.values[0]
    assert result.class_factors["rare"] <= config.max_class_oversample_factor


def test_under_supported_class_does_not_drive_oversampling() -> None:
    chips = pd.DataFrame(
        {
            "px_bg": [100, 100, 100],
            "px_10": [100, 100, 0],
            "px_11": [0, 0, 1],
        }
    )
    config = SamplerConfig(min_sampling_chips=2)

    with pytest.warns(UserWarning, match="rare"):
        result = compute_sampling_weights(chips, taxonomy(), config)

    assert result.class_factors["rare"] == 1.0
    assert result.guarded_classes == ("rare",)


def test_class_weights_are_normalized_and_capped() -> None:
    chips = pd.DataFrame(
        {
            "px_bg": [1000],
            "px_10": [100],
            "px_11": [1],
        }
    )

    weights = compute_class_weights(chips, taxonomy(), beta=0.5, max_ratio=5.0)

    assert weights.mean() == pytest.approx(1.0)
    assert weights.max() / weights.min() <= 5.0 + 1e-6
    assert np.isfinite(weights).all()


def test_class_balanced_sampler_draws_from_class_pools_deterministically() -> None:
    chips = pd.DataFrame(
        {
            "px_bg": [100, 100, 100, 100],
            "px_10": [100, 100, 100, 0],
            "px_11": [0, 0, 1, 0],
        }
    )
    first = ClassBalancedSampler(
        chips,
        taxonomy(),
        num_samples=40,
        background_weight=0.1,
        seed=9,
    )
    second = ClassBalancedSampler(
        chips,
        taxonomy(),
        num_samples=40,
        background_weight=0.1,
        seed=9,
    )

    first_draw = list(first)
    assert first_draw == list(second)
    assert 2 in first_draw
    assert 3 in first_draw
    assert set(first_draw) <= {0, 1, 2, 3}
