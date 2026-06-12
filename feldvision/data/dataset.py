from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import Dataset

from feldvision.config import AugmentationConfig
from feldvision.data.augmentation import build_transform
from feldvision.taxonomy import Taxonomy


class DatasetError(ValueError):
    """Raised when raster data cannot satisfy the chip dataset contract."""


class ChipDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        chips: pd.DataFrame | str | Path,
        *,
        split: str,
        basemap_path: str | Path,
        mask_path: str | Path,
        taxonomy: Taxonomy,
        augmentation: AugmentationConfig,
        include_context: bool = False,
        context_scale: int = 4,
        seed: int = 0,
    ) -> None:
        frame = pd.read_parquet(chips) if isinstance(chips, (str, Path)) else chips.copy()
        if "split" not in frame:
            raise DatasetError("chip table has no split column")
        self.chips = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        if self.chips.empty:
            raise DatasetError(f"split {split!r} contains no chips")
        if self.chips["style_id"].isna().any():
            raise DatasetError("trainable dataset splits cannot contain null style_id")
        sizes = self.chips["size"].unique()
        if len(sizes) != 1:
            raise DatasetError("a dataset instance requires one fixed chip size")
        if context_scale < 1:
            raise DatasetError("context_scale must be at least one")

        self.split = split
        self.basemap_path = str(basemap_path)
        self.mask_path = str(mask_path)
        self.taxonomy = taxonomy
        self.chip_size = int(sizes[0])
        self.include_context = include_context
        self.context_scale = context_scale
        self.seed = seed
        self.max_jitter_px = (
            augmentation.max_jitter_px if split == "train" and augmentation.jitter else 0
        )
        self.transform = build_transform(
            augmentation,
            train=split == "train",
            include_context=include_context,
        )
        self._basemap: rasterio.io.DatasetReader | None = None
        self._mask: rasterio.io.DatasetReader | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.chips.iloc[index]
        basemap, mask = self._open_rasters()
        row_off, col_off = self._jittered_offsets(
            int(row["row_off"]),
            int(row["col_off"]),
            raster_height=mask.height,
            raster_width=mask.width,
        )
        window = Window(col_off, row_off, self.chip_size, self.chip_size)
        detail = basemap.read(
            indexes=(1, 2, 3),
            window=window,
            out_dtype=np.uint8,
        ).transpose(1, 2, 0)
        raw_target = mask.read(1, window=window, out_dtype=np.uint8)
        target = self.taxonomy.remap(raw_target)

        transformed_input: dict[str, Any] = {"image": detail, "mask": target}
        if self.include_context:
            transformed_input["context"] = self._read_context(
                basemap,
                row_off=row_off,
                col_off=col_off,
            )
        transformed = self.transform(**transformed_input)

        sample: dict[str, Any] = {
            "detail": transformed["image"].to(dtype=torch.float32),
            "target": transformed["mask"].to(dtype=torch.long),
            "meta": {
                "chip_id": str(row["chip_id"]),
                "style_id": int(row["style_id"]),
                "row_off": row_off,
                "col_off": col_off,
            },
        }
        if self.include_context:
            sample["context"] = transformed["context"].to(dtype=torch.float32)
        return sample

    def _open_rasters(
        self,
    ) -> tuple[rasterio.io.DatasetReader, rasterio.io.DatasetReader]:
        if self._basemap is None:
            self._basemap = rasterio.open(self.basemap_path)
        if self._mask is None:
            self._mask = rasterio.open(self.mask_path)
        if self._basemap.count < 3:
            raise DatasetError("basemap must have at least three bands")
        if self._mask.count != 1:
            raise DatasetError("mask must have exactly one band")
        if (
            self._basemap.width != self._mask.width
            or self._basemap.height != self._mask.height
            or self._basemap.transform != self._mask.transform
        ):
            raise DatasetError("basemap and mask are not pixel-aligned")
        return self._basemap, self._mask

    def _jittered_offsets(
        self,
        row_off: int,
        col_off: int,
        *,
        raster_height: int,
        raster_width: int,
    ) -> tuple[int, int]:
        if self.max_jitter_px:
            if self._rng is None:
                worker_seed = torch.initial_seed() % 2**32
                self._rng = np.random.default_rng(worker_seed + self.seed)
            row_off += int(self._rng.integers(-self.max_jitter_px, self.max_jitter_px + 1))
            col_off += int(self._rng.integers(-self.max_jitter_px, self.max_jitter_px + 1))
        row_off = int(np.clip(row_off, 0, raster_height - self.chip_size))
        col_off = int(np.clip(col_off, 0, raster_width - self.chip_size))
        return row_off, col_off

    def _read_context(
        self,
        basemap: rasterio.io.DatasetReader,
        *,
        row_off: int,
        col_off: int,
    ) -> np.ndarray:
        context_size = self.chip_size * self.context_scale
        center_row = row_off + self.chip_size / 2
        center_col = col_off + self.chip_size / 2
        context_window = Window(
            center_col - context_size / 2,
            center_row - context_size / 2,
            context_size,
            context_size,
        )
        return basemap.read(
            indexes=(1, 2, 3),
            window=context_window,
            out_shape=(3, self.chip_size, self.chip_size),
            out_dtype=np.uint8,
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)

    def close(self) -> None:
        if self._basemap is not None:
            self._basemap.close()
            self._basemap = None
        if self._mask is not None:
            self._mask.close()
            self._mask = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_basemap"] = None
        state["_mask"] = None
        state["_rng"] = None
        return state

    def __del__(self) -> None:
        self.close()
