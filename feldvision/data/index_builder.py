from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import Window
from shapely import STRtree
from tqdm import tqdm

from feldvision.data.index import (
    COUNT_COLUMNS,
    RAW_FOREGROUND_IDS,
    ChipIndexError,
    summarize_chip_index,
    write_json,
)


@dataclass(frozen=True)
class ChipIndexBuildConfig:
    chip_size: int = 256
    stride: int = 128
    block_size: int = 4096

    def validate(self) -> None:
        if self.chip_size <= 0 or self.stride <= 0 or self.block_size <= 0:
            raise ChipIndexError("chip_size, stride, and block_size must be positive")
        if self.block_size % self.stride:
            raise ChipIndexError("block_size must be divisible by stride")


def build_chip_index(
    *,
    mask_path: str | Path,
    stylesheets_path: str | Path,
    metadata_path: str | Path,
    config: ChipIndexBuildConfig | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    config = config or ChipIndexBuildConfig()
    config.validate()
    metadata = _load_metadata(metadata_path)
    sheets = gpd.read_file(stylesheets_path, columns=["style_id", "geometry"])
    if sheets.empty or "style_id" not in sheets:
        raise ChipIndexError("stylesheets must contain style_id and geometry")
    if sheets["style_id"].isna().any() or sheets["style_id"].duplicated().any():
        raise ChipIndexError("stylesheet style_id values must be non-null and unique")
    sheets = sheets.loc[~sheets.geometry.is_empty & sheets.geometry.notna()].copy()
    sheets.geometry = sheets.geometry.make_valid()

    with rasterio.open(mask_path) as mask:
        _validate_raster(mask, metadata, config)
        sheets = sheets.to_crs(mask.crs)
        geometries = np.asarray(sheets.geometry.array)
        union = shapely.union_all(geometries)
        row_offsets, col_offsets = _candidate_offsets(
            mask,
            union,
            config=config,
        )
        style_ids = _assign_style_ids(
            mask,
            row_offsets,
            col_offsets,
            geometries=geometries,
            sheet_style_ids=sheets["style_id"].to_numpy(dtype=np.int32),
            chip_size=config.chip_size,
        )
        counts = _count_mask_classes(
            mask,
            row_offsets,
            col_offsets,
            config=config,
            progress=progress,
        )
        centers_x, centers_y = _center_coordinates(
            mask,
            row_offsets,
            col_offsets,
            config.chip_size,
        )
        raster_crs = mask.crs

    transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(centers_x, centers_y)
    frame = _build_frame(
        row_offsets=row_offsets,
        col_offsets=col_offsets,
        chip_size=config.chip_size,
        centers_x=centers_x,
        centers_y=centers_y,
        lon=np.asarray(lon),
        lat=np.asarray(lat),
        style_ids=style_ids,
        counts=counts,
        label_version=str(metadata["label_version"]),
    )
    return frame.sort_values(["row_off", "col_off"], ignore_index=True)


def write_chip_index(
    chips: pd.DataFrame,
    *,
    output_path: str | Path,
    write_csv: bool = False,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    chips.to_parquet(output, index=False)
    if write_csv:
        chips.to_csv(output.with_suffix(".csv"), index=False)
    write_json(summarize_chip_index(chips).to_dict(), output.with_suffix(".summary.json"))


def _load_metadata(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {"label_version", "raster_width_px", "raster_height_px"}
    missing = required - set(metadata)
    if missing:
        raise ChipIndexError(f"mask metadata is missing keys: {sorted(missing)}")
    return metadata


def _validate_raster(
    mask: rasterio.io.DatasetReader,
    metadata: dict[str, object],
    config: ChipIndexBuildConfig,
) -> None:
    if mask.count != 1:
        raise ChipIndexError("mask must have exactly one band")
    if mask.crs is None:
        raise ChipIndexError("mask must define a CRS")
    if mask.width != int(metadata["raster_width_px"]) or mask.height != int(
        metadata["raster_height_px"]
    ):
        raise ChipIndexError("mask dimensions do not match metadata")
    if mask.width < config.chip_size or mask.height < config.chip_size:
        raise ChipIndexError("chip_size exceeds raster dimensions")
    if mask.transform.b != 0 or mask.transform.d != 0:
        raise ChipIndexError("rotated rasters are not supported")


def _candidate_offsets(
    mask: rasterio.io.DatasetReader,
    union: object,
    *,
    config: ChipIndexBuildConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(
        0,
        mask.height - config.chip_size + 1,
        config.stride,
        dtype=np.int32,
    )
    cols = np.arange(
        0,
        mask.width - config.chip_size + 1,
        config.stride,
        dtype=np.int32,
    )
    center_cols = cols.astype(np.float64) + config.chip_size / 2
    center_rows = rows.astype(np.float64) + config.chip_size / 2
    x = mask.transform.c + mask.transform.a * center_cols
    y = mask.transform.f + mask.transform.e * center_rows
    inside = shapely.intersects_xy(union, x[None, :], y[:, None])
    row_indices, col_indices = np.nonzero(inside)
    return rows[row_indices], cols[col_indices]


def _assign_style_ids(
    mask: rasterio.io.DatasetReader,
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    *,
    geometries: np.ndarray,
    sheet_style_ids: np.ndarray,
    chip_size: int,
) -> pd.arrays.IntegerArray:
    left = mask.transform.c + mask.transform.a * col_offsets
    right = mask.transform.c + mask.transform.a * (col_offsets + chip_size)
    top = mask.transform.f + mask.transform.e * row_offsets
    bottom = mask.transform.f + mask.transform.e * (row_offsets + chip_size)
    boxes = shapely.box(
        np.minimum(left, right),
        np.minimum(bottom, top),
        np.maximum(left, right),
        np.maximum(bottom, top),
    )
    pairs = STRtree(geometries).query(boxes, predicate="intersects")
    match_counts = np.bincount(pairs[0], minlength=len(boxes))
    single_pair = match_counts[pairs[0]] == 1
    box_indices = pairs[0, single_pair]
    sheet_indices = pairs[1, single_pair]
    covered = shapely.covers(geometries[sheet_indices], boxes[box_indices])
    values = np.full(len(boxes), -1, dtype=np.int32)
    values[box_indices[covered]] = sheet_style_ids[sheet_indices[covered]]
    return pd.array(np.where(values >= 0, values, None), dtype="Int32")


def _count_mask_classes(
    mask: rasterio.io.DatasetReader,
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    *,
    config: ChipIndexBuildConfig,
    progress: bool,
) -> dict[str, np.ndarray]:
    result = {column: np.zeros(len(row_offsets), dtype=np.int32) for column in COUNT_COLUMNS}
    block_rows = row_offsets // config.block_size
    block_cols = col_offsets // config.block_size
    block_keys = np.column_stack((block_rows, block_cols))
    unique_keys, inverse = np.unique(block_keys, axis=0, return_inverse=True)
    iterator = tqdm(
        enumerate(unique_keys),
        total=len(unique_keys),
        desc="Counting mask classes",
        disable=not progress,
    )
    known_ids = {0, *RAW_FOREGROUND_IDS}

    for group_index, (block_row, block_col) in iterator:
        indices = np.flatnonzero(inverse == group_index)
        source_row = int(block_row * config.block_size)
        source_col = int(block_col * config.block_size)
        max_row = int(row_offsets[indices].max() + config.chip_size)
        max_col = int(col_offsets[indices].max() + config.chip_size)
        data = mask.read(
            1,
            window=Window(
                source_col,
                source_row,
                max_col - source_col,
                max_row - source_row,
            ),
            out_dtype=np.uint8,
        )
        present_ids = set(np.unique(data).tolist())
        unknown = present_ids - known_ids
        if unknown:
            raise ChipIndexError(f"mask contains unknown raw class ids: {sorted(unknown)}")
        local_rows = row_offsets[indices] - source_row
        local_cols = col_offsets[indices] - source_col
        for raw_id in sorted(present_ids):
            column = "px_bg" if raw_id == 0 else f"px_{raw_id}"
            result[column][indices] = _window_sums(
                data == raw_id,
                local_rows,
                local_cols,
                config.chip_size,
            )
    return result


def _window_sums(
    values: np.ndarray,
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    size: int,
) -> np.ndarray:
    integral = values.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    row_end = row_offsets + size
    col_end = col_offsets + size
    return (
        integral[row_end, col_end]
        - integral[row_offsets, col_end]
        - integral[row_end, col_offsets]
        + integral[row_offsets, col_offsets]
    )


def _center_coordinates(
    mask: rasterio.io.DatasetReader,
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    chip_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    center_cols = col_offsets.astype(np.float64) + chip_size / 2
    center_rows = row_offsets.astype(np.float64) + chip_size / 2
    x = mask.transform.c + mask.transform.a * center_cols
    y = mask.transform.f + mask.transform.e * center_rows
    return x, y


def _build_frame(
    *,
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    chip_size: int,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    style_ids: pd.arrays.IntegerArray,
    counts: dict[str, np.ndarray],
    label_version: str,
) -> pd.DataFrame:
    foreground = np.column_stack([counts[f"px_{raw_id}"] for raw_id in RAW_FOREGROUND_IDS])
    raw_ids = np.asarray(RAW_FOREGROUND_IDS, dtype=np.int16)
    dominant_indices = foreground.argmax(axis=1)
    has_label = foreground.sum(axis=1) > 0
    dominant = np.where(has_label, raw_ids[dominant_indices], -1).astype(np.int16)
    frame = pd.DataFrame(
        {
            "chip_id": "r" + row_offsets.astype(str) + "_c" + col_offsets.astype(str),
            "row_off": row_offsets,
            "col_off": col_offsets,
            "size": np.full(len(row_offsets), chip_size, dtype=np.int16),
            "center_x": centers_x,
            "center_y": centers_y,
            "lon": lon.astype(np.float32),
            "lat": lat.astype(np.float32),
            "style_id": style_ids,
            **counts,
            "n_label_px": foreground.sum(axis=1).astype(np.int32),
            "dominant_class": dominant,
            "label_version": label_version,
        }
    )
    total = frame[list(COUNT_COLUMNS)].sum(axis=1)
    if not total.eq(chip_size**2).all():
        raise ChipIndexError("class counts do not sum to chip_size^2")
    return frame
