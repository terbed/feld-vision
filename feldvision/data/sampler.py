from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from feldvision.config import SamplerConfig
from feldvision.taxonomy import Taxonomy


class SamplingError(ValueError):
    """Raised when sampling weights cannot be computed."""


@dataclass(frozen=True)
class SamplingWeights:
    values: np.ndarray
    class_factors: dict[str, float]
    guarded_classes: tuple[str, ...]


class ClassBalancedSampler(Sampler[int]):
    def __init__(
        self,
        chips: pd.DataFrame,
        taxonomy: Taxonomy,
        *,
        num_samples: int,
        background_weight: float,
        seed: int,
    ) -> None:
        if num_samples <= 0:
            raise SamplingError("num_samples must be positive")
        if background_weight < 0:
            raise SamplingError("background_weight cannot be negative")
        presence = taxonomy.class_presence(chips)
        pools: list[np.ndarray] = []
        category_weights: list[float] = []
        for name in presence:
            pool = np.flatnonzero(presence[name].to_numpy())
            if len(pool):
                pools.append(pool)
                category_weights.append(1.0)
        background_pool = np.flatnonzero(~presence.any(axis=1).to_numpy())
        if len(background_pool) and background_weight > 0:
            pools.append(background_pool)
            category_weights.append(background_weight)
        if not pools:
            raise SamplingError("no non-empty class or background pools are available")
        self.pools = pools
        self.category_weights = torch.tensor(category_weights, dtype=torch.float64)
        self.category_weights /= self.category_weights.sum()
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        categories = torch.multinomial(
            self.category_weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        for category in categories.tolist():
            pool = self.pools[category]
            offset = int(torch.randint(len(pool), (1,), generator=generator).item())
            yield int(pool[offset])


def compute_sampling_weights(
    train_chips: pd.DataFrame,
    taxonomy: Taxonomy,
    config: SamplerConfig,
) -> SamplingWeights:
    if train_chips.empty:
        raise SamplingError("cannot compute sampling weights for an empty training split")
    if config.strategy == "class_balanced_targeted":
        raise SamplingError("class_balanced_targeted uses ClassBalancedSampler, not static weights")
    if config.strategy == "uniform":
        return SamplingWeights(
            values=np.ones(len(train_chips), dtype=np.float64),
            class_factors={item.name: 1.0 for item in taxonomy.foreground_classes},
            guarded_classes=(),
        )

    counts = taxonomy.foreground_count_frame(train_chips)
    presence = counts.gt(0)
    pixel_totals = counts.sum(axis=0).astype(np.float64)
    total_active = float(pixel_totals.sum())
    if total_active <= 0:
        raise SamplingError("training split contains no taxonomy-active foreground pixels")
    frequencies = pixel_totals / total_active
    positive_frequencies = frequencies[frequencies.gt(0)]
    if positive_frequencies.empty:
        raise SamplingError("no active class has positive frequency")
    reference = float(positive_frequencies.max())

    exponent = 1.0 if config.strategy == "inverse_freq" else config.alpha
    factors: dict[str, float] = {}
    guarded: list[str] = []
    for class_def in taxonomy.foreground_classes:
        name = class_def.name
        support = int(presence[name].sum())
        frequency = float(frequencies[name])
        if support < config.min_sampling_chips:
            factor = 1.0
            guarded.append(name)
        elif frequency <= 0:
            factor = 1.0
        else:
            factor = min(
                (reference / frequency) ** exponent,
                config.max_class_oversample_factor,
            )
        factors[name] = float(factor)

    if guarded:
        warnings.warn(
            "classes below min_sampling_chips do not drive oversampling: " + ", ".join(guarded),
            stacklevel=2,
        )

    factor_array = np.asarray([factors[name] for name in counts.columns], dtype=np.float64)
    active_weights = np.where(presence.to_numpy(), factor_array, 0.0).max(axis=1)
    values = np.where(presence.any(axis=1).to_numpy(), active_weights, config.w_background)
    mean = float(values.mean())
    if mean <= 0 or not np.isfinite(mean):
        raise SamplingError("sampling weights are not finite and positive")
    values /= mean
    return SamplingWeights(
        values=values,
        class_factors=factors,
        guarded_classes=tuple(guarded),
    )


def compute_class_weights(
    train_chips: pd.DataFrame,
    taxonomy: Taxonomy,
    *,
    beta: float,
    max_ratio: float,
) -> np.ndarray:
    if beta < 0:
        raise SamplingError("class-weight beta cannot be negative")
    if max_ratio < 1:
        raise SamplingError("max class-weight ratio must be at least one")

    counts = taxonomy.count_frame(train_chips).sum(axis=0).to_numpy(dtype=np.float64)
    if counts.sum() <= 0:
        raise SamplingError("cannot derive class weights from zero pixels")
    frequencies = counts / counts.sum()
    positive = frequencies[frequencies > 0]
    reference = float(positive.max())
    raw = np.empty_like(frequencies)
    raw[frequencies > 0] = (reference / frequencies[frequencies > 0]) ** beta
    raw[frequencies == 0] = np.inf

    minimum = float(np.min(raw[np.isfinite(raw)]))
    clipped = np.minimum(raw, minimum * max_ratio)
    clipped /= clipped.mean()
    return clipped.astype(np.float32)
