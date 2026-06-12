from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_FOREGROUND_IDS = tuple(range(10, 18))
COUNT_COLUMNS = ("px_bg", *(f"px_{raw_id}" for raw_id in RAW_FOREGROUND_IDS))
REQUIRED_COLUMNS = (
    "chip_id",
    "row_off",
    "col_off",
    "size",
    "style_id",
    "label_version",
    *COUNT_COLUMNS,
)


class ChipIndexError(ValueError):
    """Raised when a chip index violates the data contract."""


@dataclass(frozen=True)
class ChipIndexSummary:
    row_count: int
    null_style_count: int
    label_bearing_count: int
    rows_per_style: dict[str, int]
    chips_per_raw_class: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "null_style_count": self.null_style_count,
            "label_bearing_count": self.label_bearing_count,
            "rows_per_style": self.rows_per_style,
            "chips_per_raw_class": self.chips_per_raw_class,
        }


def load_chip_index(path: str | Path) -> pd.DataFrame:
    index_path = Path(path)
    if index_path.suffix == ".parquet":
        return pd.read_parquet(index_path)
    if index_path.suffix == ".csv":
        return pd.read_csv(index_path)
    raise ChipIndexError(f"unsupported chip-index format: {index_path.suffix}")


def load_label_version(metadata_path: str | Path) -> str:
    with Path(metadata_path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    try:
        return str(metadata["label_version"])
    except KeyError as exc:
        raise ChipIndexError("mask metadata has no label_version") from exc


def validate_chip_index(
    chips: pd.DataFrame,
    *,
    expected_label_version: str | None = None,
    known_style_ids: Collection[int] | None = None,
    expected_stride: int | None = None,
) -> ChipIndexSummary:
    missing = sorted(set(REQUIRED_COLUMNS) - set(chips.columns))
    if missing:
        raise ChipIndexError(f"chip index is missing columns: {missing}")
    if chips.empty:
        raise ChipIndexError("chip index is empty")
    if chips["chip_id"].isna().any() or chips["chip_id"].duplicated().any():
        raise ChipIndexError("chip_id values must be non-null and unique")

    for column in ("row_off", "col_off", "size", *COUNT_COLUMNS):
        values = chips[column]
        if values.isna().any() or not np.issubdtype(values.dtype, np.number):
            raise ChipIndexError(f"{column} must contain non-null numeric values")
        if (values < 0).any():
            raise ChipIndexError(f"{column} cannot contain negative values")

    if (chips["size"] == 0).any():
        raise ChipIndexError("chip size must be positive")
    counted = chips.loc[:, list(COUNT_COLUMNS)].sum(axis=1)
    expected = chips["size"].astype("int64").pow(2)
    invalid_counts = counted.ne(expected)
    if invalid_counts.any():
        first = chips.loc[invalid_counts, "chip_id"].iloc[0]
        raise ChipIndexError(f"raw pixel counts do not sum to size^2 for chip {first}")

    expected_ids = "r" + chips["row_off"].astype(str) + "_c" + chips["col_off"].astype(str)
    if not chips["chip_id"].astype(str).eq(expected_ids).all():
        raise ChipIndexError("chip_id must match r{row_off}_c{col_off}")

    if expected_stride is not None:
        if expected_stride <= 0:
            raise ChipIndexError("expected_stride must be positive")
        if (chips["row_off"] % expected_stride).any() or (chips["col_off"] % expected_stride).any():
            raise ChipIndexError("chip offsets are not aligned to the configured stride")

    if expected_label_version is not None:
        versions = set(chips["label_version"].dropna().astype(str))
        if versions != {expected_label_version}:
            message = (
                f"label_version mismatch: expected {expected_label_version!r}, "
                f"got {sorted(versions)}"
            )
            raise ChipIndexError(message)

    if known_style_ids is not None:
        actual = set(chips["style_id"].dropna().astype(int))
        unknown = actual - set(known_style_ids)
        if unknown:
            raise ChipIndexError(f"chip index contains unknown style ids: {sorted(unknown)}")

    return summarize_chip_index(chips)


def summarize_chip_index(chips: pd.DataFrame) -> ChipIndexSummary:
    style_keys = chips["style_id"].map(
        lambda value: "null" if pd.isna(value) else str(int(value)),
        na_action=None,
    )
    rows_per_style = style_keys.value_counts().sort_index().astype(int).to_dict()
    chips_per_raw_class = {
        str(raw_id): int(chips[f"px_{raw_id}"].gt(0).sum()) for raw_id in RAW_FOREGROUND_IDS
    }
    return ChipIndexSummary(
        row_count=len(chips),
        null_style_count=int(chips["style_id"].isna().sum()),
        label_bearing_count=int(chips[list(COUNT_COLUMNS[1:])].sum(axis=1).gt(0).sum()),
        rows_per_style=rows_per_style,
        chips_per_raw_class=chips_per_raw_class,
    )


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
