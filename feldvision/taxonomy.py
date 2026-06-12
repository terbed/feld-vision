from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class TaxonomyError(ValueError):
    """Raised when a taxonomy cannot produce an unambiguous training target."""


@dataclass(frozen=True)
class ClassDefinition:
    name: str
    train_id: int
    raw_ids: tuple[int, ...]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Taxonomy:
    classes: tuple[ClassDefinition, ...]
    ignored_raw_ids: frozenset[int]
    ignore_index: int = 255

    def __post_init__(self) -> None:
        train_ids = [item.train_id for item in self.classes]
        if train_ids != list(range(len(self.classes))):
            raise TaxonomyError("class train_id values must be ordered and contiguous from zero")
        if not self.classes or self.classes[0].name != "background":
            raise TaxonomyError("train_id 0 must be named 'background'")
        if self.ignore_index in train_ids:
            raise TaxonomyError("ignore_index cannot be an active train_id")

        active_raw_ids = [raw_id for item in self.classes for raw_id in item.raw_ids]
        duplicates = _duplicates(active_raw_ids)
        if duplicates:
            raise TaxonomyError(f"raw ids occur in multiple active classes: {duplicates}")
        overlap = set(active_raw_ids) & self.ignored_raw_ids
        if overlap:
            raise TaxonomyError(f"raw ids cannot be both active and ignored: {sorted(overlap)}")
        for item in self.classes:
            if not item.raw_ids:
                raise TaxonomyError(f"class {item.name!r} has no raw ids")
            if len(item.color) != 3 or any(not 0 <= value <= 255 for value in item.color):
                raise TaxonomyError(f"class {item.name!r} has an invalid RGB color")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def foreground_classes(self) -> tuple[ClassDefinition, ...]:
        return self.classes[1:]

    @property
    def active_raw_ids(self) -> frozenset[int]:
        return frozenset(raw_id for item in self.classes for raw_id in item.raw_ids)

    @property
    def raw_to_train(self) -> dict[int, int]:
        return {raw_id: item.train_id for item in self.classes for raw_id in item.raw_ids}

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    @property
    def palette(self) -> np.ndarray:
        return np.asarray([item.color for item in self.classes], dtype=np.uint8)

    def remap(self, raw_mask: np.ndarray) -> np.ndarray:
        if not np.issubdtype(raw_mask.dtype, np.integer):
            raise TaxonomyError("raw mask must contain integer class ids")
        output = np.full(raw_mask.shape, self.ignore_index, dtype=np.uint8)
        for raw_id, train_id in self.raw_to_train.items():
            output[raw_mask == raw_id] = train_id
        return output

    def count_frame(self, chips: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=chips.index)
        for item in self.classes:
            columns = [f"px_{raw_id}" if raw_id else "px_bg" for raw_id in item.raw_ids]
            missing = [column for column in columns if column not in chips.columns]
            if missing:
                raise TaxonomyError(f"chip index is missing count columns: {missing}")
            result[item.name] = chips[columns].sum(axis=1).astype("int64")
        return result

    def foreground_count_frame(self, chips: pd.DataFrame) -> pd.DataFrame:
        counts = self.count_frame(chips)
        return counts.loc[:, list(self.class_names[1:])]

    def active_pixel_counts(self, chips: pd.DataFrame) -> pd.Series:
        return self.foreground_count_frame(chips).sum(axis=1)

    def class_presence(self, chips: pd.DataFrame) -> pd.DataFrame:
        return self.foreground_count_frame(chips).gt(0)


def _duplicates(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _require_keys(data: Mapping[str, Any], keys: set[str], path: str) -> None:
    missing = keys - set(data)
    if missing:
        raise TaxonomyError(f"{path} is missing keys: {sorted(missing)}")


def load_taxonomy(path: str | Path) -> Taxonomy:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TaxonomyError("taxonomy root must be a mapping")
    _require_keys(raw, {"ignore_index", "classes"}, "taxonomy")
    if not isinstance(raw["classes"], list):
        raise TaxonomyError("taxonomy.classes must be a list")

    classes: list[ClassDefinition] = []
    for index, item in enumerate(raw["classes"]):
        if not isinstance(item, dict):
            raise TaxonomyError(f"taxonomy.classes[{index}] must be a mapping")
        _require_keys(item, {"name", "train_id", "raw_ids", "color"}, f"classes[{index}]")
        classes.append(
            ClassDefinition(
                name=str(item["name"]),
                train_id=int(item["train_id"]),
                raw_ids=tuple(int(value) for value in item["raw_ids"]),
                color=tuple(int(value) for value in item["color"]),
            )
        )

    ignored: set[int] = set()
    for index, item in enumerate(raw.get("ignored", [])):
        if not isinstance(item, dict) or "raw_ids" not in item:
            raise TaxonomyError(f"taxonomy.ignored[{index}] must define raw_ids")
        ignored.update(int(value) for value in item["raw_ids"])

    return Taxonomy(
        classes=tuple(classes),
        ignored_raw_ids=frozenset(ignored),
        ignore_index=int(raw["ignore_index"]),
    )
