from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from feldvision.config import SplitConfig
from feldvision.taxonomy import Taxonomy


class SplitError(ValueError):
    """Raised when split construction cannot satisfy its basic contract."""


@dataclass(frozen=True)
class SplitResult:
    chips: pd.DataFrame
    summary: dict[str, Any]


def derive_buffer_steps(chip_size: int, stride: int, max_jitter_px: int) -> int:
    if min(chip_size, stride) <= 0 or max_jitter_px < 0:
        raise SplitError("chip_size and stride must be positive; jitter cannot be negative")
    return (chip_size + max_jitter_px + stride - 1) // stride - 1


def build_splits(
    chips: pd.DataFrame,
    taxonomy: Taxonomy,
    config: SplitConfig,
    *,
    max_jitter_px: int,
) -> SplitResult:
    _validate_split_input(chips, config)
    output = chips.copy()
    output["split"] = "excluded"

    valid_style = output["style_id"].notna()
    test_mask = valid_style & output["style_id"].astype("Int64").isin(config.test_style_ids)
    output.loc[test_mask, "split"] = "test"
    eligible_mask = valid_style & ~test_mask
    eligible = output.loc[eligible_mask].copy()
    if eligible.empty:
        raise SplitError("no chips remain after null-style and test exclusions")

    eligible["_cell_row"] = eligible["row_off"] // config.val_cell_size_px
    eligible["_cell_col"] = eligible["col_off"] // config.val_cell_size_px
    foreground_counts = taxonomy.foreground_count_frame(eligible)
    presence = foreground_counts.gt(0)
    eligible["_active_background"] = ~presence.any(axis=1)
    for name in foreground_counts:
        eligible[f"_present_{name}"] = presence[name].astype("int64")
        eligible[f"_pixels_{name}"] = foreground_counts[name].astype("float64")

    group_columns = ["style_id", "_cell_row", "_cell_col"]
    aggregations: dict[str, tuple[str, str]] = {
        "chip_count": ("chip_id", "size"),
        "background_chips": ("_active_background", "sum"),
    }
    for name in foreground_counts:
        aggregations[f"present_{name}"] = (f"_present_{name}", "sum")
        aggregations[f"pixels_{name}"] = (f"_pixels_{name}", "sum")
    groups = (
        eligible.groupby(group_columns, sort=True, observed=True).agg(**aggregations).reset_index()
    )

    feature_columns = ["chip_count", "background_chips"]
    feature_columns.extend(f"present_{name}" for name in foreground_counts)
    pixel_scale = float(config.chip_size**2)
    for name in foreground_counts:
        groups[f"pixel_equiv_{name}"] = groups[f"pixels_{name}"] / pixel_scale
        feature_columns.append(f"pixel_equiv_{name}")

    selected_group_keys = _select_validation_groups(
        groups,
        feature_columns=feature_columns,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    selected_group_keys = _ensure_validation_class_coverage(
        groups,
        selected_group_keys,
        class_names=list(foreground_counts.columns),
        seed=config.seed,
    )
    eligible_keys = pd.MultiIndex.from_frame(eligible[group_columns])
    validation_mask = eligible_keys.isin(selected_group_keys)
    val_indices = eligible.index[validation_mask]
    output.loc[val_indices, "split"] = "val"

    buffer_steps = derive_buffer_steps(config.chip_size, config.stride, max_jitter_px)
    candidate_train = eligible.loc[~validation_mask]
    excluded_by_buffer = _buffer_exclusion_mask(
        candidate_train,
        eligible.loc[validation_mask],
        stride=config.stride,
        buffer_steps=buffer_steps,
    )
    train_indices = candidate_train.index[~excluded_by_buffer]
    output.loc[train_indices, "split"] = "train"

    coverage_failures = _coverage_failures(output, taxonomy, groups)
    summary = _build_summary(
        output,
        taxonomy,
        eligible_indices=eligible.index,
        eligible_count=len(eligible),
        selected_group_count=len(selected_group_keys),
        total_group_count=len(groups),
        buffer_steps=buffer_steps,
        coverage_failures=coverage_failures,
    )
    return SplitResult(chips=output, summary=summary)


def _validate_split_input(chips: pd.DataFrame, config: SplitConfig) -> None:
    required = {"chip_id", "row_off", "col_off", "style_id"}
    missing = sorted(required - set(chips.columns))
    if missing:
        raise SplitError(f"chip index is missing split columns: {missing}")
    if chips["chip_id"].duplicated().any():
        raise SplitError("chip_id must be unique")
    if not 0 < config.val_fraction < 1:
        raise SplitError("val_fraction must be between zero and one")
    if config.val_cell_size_px % config.stride:
        raise SplitError("val_cell_size_px must be divisible by stride")


def _select_validation_groups(
    groups: pd.DataFrame,
    *,
    feature_columns: list[str],
    val_fraction: float,
    seed: int,
) -> pd.MultiIndex:
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for _, style_groups in groups.groupby("style_id", sort=True, observed=True):
        indices = style_groups.index.to_numpy()
        features = style_groups[feature_columns].to_numpy(dtype=np.float64)
        totals = features.sum(axis=0)
        target = totals * val_fraction
        target_chips = max(1.0, target[0])
        selected = np.zeros(len(style_groups), dtype=bool)
        current = np.zeros_like(target)
        tie_break = rng.random(len(style_groups))

        while current[0] < target_chips and (~selected).any():
            remaining = np.flatnonzero(~selected)
            candidates = current + features[remaining]
            scales = np.maximum(target, 1.0)
            deviations = np.abs(candidates - target) / scales
            scores = deviations.mean(axis=1)
            overshoot = np.maximum(candidates[:, 0] - target_chips, 0.0) / target_chips
            scores += overshoot
            best_local = np.lexsort((tie_break[remaining], scores))[0]
            chosen = remaining[best_local]
            selected[chosen] = True
            current += features[chosen]

        selected_indices.extend(indices[selected].tolist())

    selected = groups.loc[selected_indices, ["style_id", "_cell_row", "_cell_col"]]
    return pd.MultiIndex.from_frame(selected)


def _buffer_exclusion_mask(
    candidates: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    stride: int,
    buffer_steps: int,
) -> np.ndarray:
    if candidates.empty or validation.empty:
        return np.zeros(len(candidates), dtype=bool)
    validation_grid = {
        (int(row // stride), int(col // stride))
        for row, col in validation[["row_off", "col_off"]].itertuples(index=False, name=None)
    }
    forbidden: set[tuple[int, int]] = set()
    for grid_row, grid_col in validation_grid:
        for row_delta in range(-buffer_steps, buffer_steps + 1):
            for col_delta in range(-buffer_steps, buffer_steps + 1):
                forbidden.add((grid_row + row_delta, grid_col + col_delta))
    return np.fromiter(
        (
            (int(row // stride), int(col // stride)) in forbidden
            for row, col in candidates[["row_off", "col_off"]].itertuples(index=False, name=None)
        ),
        dtype=bool,
        count=len(candidates),
    )


def _ensure_validation_class_coverage(
    groups: pd.DataFrame,
    selected_keys: pd.MultiIndex,
    *,
    class_names: list[str],
    seed: int,
) -> pd.MultiIndex:
    key_columns = ["style_id", "_cell_row", "_cell_col"]
    selected = set(selected_keys.tolist())
    rng = np.random.default_rng(seed + 1)
    tie_break = rng.random(len(groups))

    for class_name in class_names:
        support = groups.loc[groups[f"present_{class_name}"].gt(0)]
        if len(support) < 2:
            continue
        support_keys = [
            tuple(values) for values in support[key_columns].itertuples(index=False, name=None)
        ]
        selected_support = [key for key in support_keys if key in selected]
        if not selected_support:
            candidates = support.copy()
            candidates["_tie"] = tie_break[candidates.index]
            chosen = candidates.sort_values(
                [f"present_{class_name}", "_tie"],
                ascending=[False, True],
            ).iloc[0]
            selected.add(tuple(chosen[column] for column in key_columns))
        elif len(selected_support) == len(support_keys):
            removable = support.copy()
            removable["_tie"] = tie_break[removable.index]
            chosen = removable.sort_values(
                [f"present_{class_name}", "_tie"],
                ascending=[True, True],
            ).iloc[0]
            selected.discard(tuple(chosen[column] for column in key_columns))

    selected_frame = pd.DataFrame(sorted(selected), columns=key_columns)
    return pd.MultiIndex.from_frame(selected_frame)


def _coverage_failures(
    chips: pd.DataFrame,
    taxonomy: Taxonomy,
    groups: pd.DataFrame,
) -> list[str]:
    failures: list[str] = []
    for class_def in taxonomy.foreground_classes:
        support_groups = int(groups[f"present_{class_def.name}"].gt(0).sum())
        if support_groups < 2:
            continue
        for split in ("train", "val"):
            subset = chips.loc[chips["split"].eq(split)]
            raw_columns = [f"px_{raw_id}" for raw_id in class_def.raw_ids]
            if subset[raw_columns].sum(axis=None) == 0:
                failures.append(f"{class_def.name}:{split}")
    return failures


def _build_summary(
    chips: pd.DataFrame,
    taxonomy: Taxonomy,
    *,
    eligible_indices: pd.Index,
    eligible_count: int,
    selected_group_count: int,
    total_group_count: int,
    buffer_steps: int,
    coverage_failures: list[str],
) -> dict[str, Any]:
    split_counts = (
        chips["split"].value_counts().reindex(["train", "val", "test", "excluded"], fill_value=0)
    )
    per_style_frame = (
        chips.loc[chips["style_id"].notna()]
        .groupby(["style_id", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test", "excluded"], fill_value=0)
    )
    per_style = {
        str(int(style_id)): {split: int(count) for split, count in row.items()}
        for style_id, row in per_style_frame.iterrows()
    }
    per_class: dict[str, dict[str, int]] = {}
    for class_def in taxonomy.foreground_classes:
        raw_columns = [f"px_{raw_id}" for raw_id in class_def.raw_ids]
        per_class[class_def.name] = {
            split: int(chips.loc[chips["split"].eq(split), raw_columns].sum(axis=1).gt(0).sum())
            for split in ("train", "val", "test", "excluded")
        }
    eligible = chips.loc[eligible_indices]
    validation = chips.loc[chips["split"].eq("val")]
    return {
        "counts": {name: int(count) for name, count in split_counts.items()},
        "counts_per_style": per_style,
        "eligible_chip_count": eligible_count,
        "achieved_val_fraction": float(split_counts["val"] / eligible_count),
        "validation_group_count": selected_group_count,
        "eligible_group_count": total_group_count,
        "buffer_steps": buffer_steps,
        "coverage_failures": coverage_failures,
        "class_chip_counts": per_class,
        "distribution_deltas": _distribution_deltas(eligible, validation, taxonomy),
    }


def _distribution_deltas(
    eligible: pd.DataFrame,
    validation: pd.DataFrame,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    eligible_style = eligible["style_id"].value_counts(normalize=True)
    validation_style = validation["style_id"].value_counts(normalize=True)
    style_ids = sorted(set(eligible_style.index) | set(validation_style.index))
    style_delta = {
        str(int(style_id)): float(
            validation_style.get(style_id, 0.0) - eligible_style.get(style_id, 0.0)
        )
        for style_id in style_ids
    }

    eligible_foreground = taxonomy.foreground_count_frame(eligible)
    validation_foreground = taxonomy.foreground_count_frame(validation)
    eligible_background = ~eligible_foreground.gt(0).any(axis=1)
    validation_background = ~validation_foreground.gt(0).any(axis=1)
    eligible_pixel_total = max(float(eligible_foreground.to_numpy().sum()), 1.0)
    validation_pixel_total = max(float(validation_foreground.to_numpy().sum()), 1.0)
    class_chip_delta: dict[str, float] = {}
    class_pixel_delta: dict[str, float] = {}
    for name in eligible_foreground:
        class_chip_delta[name] = float(
            validation_foreground[name].gt(0).mean() - eligible_foreground[name].gt(0).mean()
        )
        class_pixel_delta[name] = float(
            validation_foreground[name].sum() / validation_pixel_total
            - eligible_foreground[name].sum() / eligible_pixel_total
        )
    return {
        "style_fraction": style_delta,
        "active_background_fraction": float(
            validation_background.mean() - eligible_background.mean()
        ),
        "class_chip_fraction": class_chip_delta,
        "class_active_pixel_fraction": class_pixel_delta,
    }
