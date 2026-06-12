from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from torch import nn

from feldvision.data.augmentation import normalize_image
from feldvision.taxonomy import Taxonomy
from feldvision.train.metrics import metrics_from_prediction


@dataclass(frozen=True)
class SheetReconstruction:
    prediction: np.ndarray
    target: np.ndarray
    sheet_mask: np.ndarray
    evaluation_mask: np.ndarray
    metrics: dict[str, float]
    window: Window
    transform: Affine
    crs: object


def reconstruct_sheet(
    model: nn.Module,
    *,
    geometry: BaseGeometry,
    basemap_path: str | Path,
    mask_path: str | Path | None,
    taxonomy: Taxonomy,
    device: torch.device,
    chip_size: int = 256,
    overlap: int = 64,
    batch_size: int = 4,
    context_scale: int | None = None,
) -> SheetReconstruction:
    if not 0 <= overlap < chip_size:
        raise ValueError("overlap must be in [0, chip_size)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stride = chip_size - overlap
    model.eval()

    with ExitStack() as stack:
        basemap = stack.enter_context(rasterio.open(basemap_path))
        mask = stack.enter_context(rasterio.open(mask_path)) if mask_path is not None else None
        if mask is not None and (basemap.width != mask.width or basemap.height != mask.height):
            raise ValueError("basemap and mask dimensions differ")
        sheet_window = (
            geometry_window(
                basemap,
                [mapping(geometry)],
                pad_x=0,
                pad_y=0,
            )
            .round_offsets()
            .round_lengths()
        )
        sheet_window = _clip_window(sheet_window, basemap.width, basemap.height)
        height, width = int(sheet_window.height), int(sheet_window.width)
        row_starts = _window_starts(height, chip_size, stride)
        col_starts = _window_starts(width, chip_size, stride)
        score_sum = np.zeros((taxonomy.num_classes, height, width), dtype=np.float32)
        score_count = np.zeros((height, width), dtype=np.uint16)

        pending: list[tuple[int, int, torch.Tensor, torch.Tensor | None]] = []
        for local_row in row_starts:
            for local_col in col_starts:
                global_row = int(sheet_window.row_off) + local_row
                global_col = int(sheet_window.col_off) + local_col
                detail = _read_rgb(
                    basemap,
                    row_off=global_row,
                    col_off=global_col,
                    size=chip_size,
                    output_size=chip_size,
                )
                context = None
                if context_scale is not None:
                    context_size = chip_size * context_scale
                    context = _read_rgb(
                        basemap,
                        row_off=global_row - (context_size - chip_size) // 2,
                        col_off=global_col - (context_size - chip_size) // 2,
                        size=context_size,
                        output_size=chip_size,
                    )
                pending.append((local_row, local_col, detail, context))
                if len(pending) == batch_size:
                    _infer_pending(
                        model,
                        pending,
                        score_sum,
                        score_count,
                        device=device,
                        height=height,
                        width=width,
                    )
                    pending.clear()
        if pending:
            _infer_pending(
                model,
                pending,
                score_sum,
                score_count,
                device=device,
                height=height,
                width=width,
            )
        if (score_count == 0).any():
            raise RuntimeError("reconstruction left uncovered pixels")

        prediction = score_sum.argmax(axis=0).astype(np.uint8)
        if mask is None:
            target = np.full((height, width), taxonomy.ignore_index, dtype=np.uint8)
        else:
            target = taxonomy.remap(
                mask.read(
                    1,
                    window=sheet_window,
                    out_shape=(height, width),
                    resampling=Resampling.nearest,
                )
            )
        transform = basemap.window_transform(sheet_window)
        inside = geometry_mask(
            [mapping(geometry)],
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )
        evaluation_mask = inside & (target != taxonomy.ignore_index)
        evaluation_target = target.copy()
        evaluation_target[~inside] = taxonomy.ignore_index
        metrics = (
            metrics_from_prediction(
                torch.from_numpy(prediction),
                torch.from_numpy(evaluation_target),
                class_names=taxonomy.class_names,
                ignore_index=taxonomy.ignore_index,
            )
            if evaluation_mask.any()
            else {}
        )
        return SheetReconstruction(
            prediction=prediction,
            target=target,
            sheet_mask=inside,
            evaluation_mask=evaluation_mask,
            metrics=metrics,
            window=sheet_window,
            transform=transform,
            crs=basemap.crs,
        )


def _window_starts(length: int, chip_size: int, stride: int) -> list[int]:
    if length <= chip_size:
        return [0]
    starts = list(range(0, length - chip_size + 1, stride))
    final = length - chip_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _clip_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(width, int(window.col_off + window.width))
    row_end = min(height, int(window.row_off + window.height))
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _read_rgb(
    source: rasterio.io.DatasetReader,
    *,
    row_off: int,
    col_off: int,
    size: int,
    output_size: int,
) -> torch.Tensor:
    image = source.read(
        indexes=(1, 2, 3),
        window=Window(col_off, row_off, size, size),
        out_shape=(3, output_size, output_size),
        out_dtype=np.uint8,
        boundless=True,
        fill_value=0,
        resampling=Resampling.bilinear,
    ).transpose(1, 2, 0)
    return normalize_image(image)


def _infer_pending(
    model: nn.Module,
    pending: list[tuple[int, int, torch.Tensor, torch.Tensor | None]],
    score_sum: np.ndarray,
    score_count: np.ndarray,
    *,
    device: torch.device,
    height: int,
    width: int,
) -> None:
    details = torch.stack([item[2] for item in pending]).to(device)
    contexts = None
    if pending[0][3] is not None:
        contexts = torch.stack([item[3] for item in pending if item[3] is not None]).to(device)
    with torch.inference_mode():
        scores = model(details, contexts).float().cpu().numpy()
    for (row, col, _, _), chip_scores in zip(pending, scores, strict=True):
        valid_height = min(chip_scores.shape[1], height - row)
        valid_width = min(chip_scores.shape[2], width - col)
        score_sum[:, row : row + valid_height, col : col + valid_width] += chip_scores[
            :, :valid_height, :valid_width
        ]
        score_count[row : row + valid_height, col : col + valid_width] += 1


def save_prediction(
    reconstruction: SheetReconstruction,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": reconstruction.prediction.shape[0],
        "width": reconstruction.prediction.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": reconstruction.crs,
        "transform": reconstruction.transform,
        "compress": "deflate",
        "nodata": 255,
        "tiled": True,
    }
    prediction = reconstruction.prediction.copy()
    prediction[~reconstruction.sheet_mask] = 255
    with rasterio.open(output, "w", **profile) as destination:
        destination.write(prediction, 1)
    return output


def save_prediction_image(
    reconstruction: SheetReconstruction,
    *,
    taxonomy: Taxonomy,
    path: str | Path,
) -> Path:
    labels = reconstruction.prediction
    in_range = labels < taxonomy.num_classes
    colors = np.zeros((*labels.shape, 4), dtype=np.uint8)
    colors[..., 3] = 255
    colors[in_range, :3] = taxonomy.palette[labels[in_range]]
    colors[~reconstruction.sheet_mask, 3] = 0

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colors, mode="RGBA").save(output, format="PNG")
    return output


def save_triptych(
    reconstruction: SheetReconstruction,
    *,
    basemap_path: str | Path,
    taxonomy: Taxonomy,
    path: str | Path,
    max_panel_size: int = 2048,
    alpha: float = 0.55,
) -> Path:
    height, width = reconstruction.prediction.shape
    scale = min(1.0, max_panel_size / max(height, width))
    panel_width = max(1, round(width * scale))
    panel_height = max(1, round(height * scale))
    with rasterio.open(basemap_path) as basemap:
        image = basemap.read(
            indexes=(1, 2, 3),
            window=reconstruction.window,
            out_shape=(3, panel_height, panel_width),
            out_dtype=np.uint8,
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)
    base = Image.fromarray(image, mode="RGB")
    target = Image.fromarray(reconstruction.target).resize(
        (panel_width, panel_height),
        resample=Image.Resampling.NEAREST,
    )
    prediction = Image.fromarray(reconstruction.prediction).resize(
        (panel_width, panel_height),
        resample=Image.Resampling.NEAREST,
    )
    sheet_mask = Image.fromarray(reconstruction.sheet_mask).resize(
        (panel_width, panel_height),
        resample=Image.Resampling.NEAREST,
    )
    evaluation_mask = Image.fromarray(reconstruction.evaluation_mask).resize(
        (panel_width, panel_height),
        resample=Image.Resampling.NEAREST,
    )
    ground_truth_overlay = _overlay(base, target, evaluation_mask, taxonomy, alpha)
    prediction_overlay = _overlay(base, prediction, sheet_mask, taxonomy, alpha)
    triptych = Image.new("RGB", (panel_width * 3, panel_height))
    triptych.paste(base, (0, 0))
    triptych.paste(ground_truth_overlay, (panel_width, 0))
    triptych.paste(prediction_overlay, (panel_width * 2, 0))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    triptych.save(output)
    return output


def _overlay(
    base: Image.Image,
    labels: Image.Image,
    valid: Image.Image,
    taxonomy: Taxonomy,
    alpha: float,
) -> Image.Image:
    label_array = np.asarray(labels)
    valid_array = np.asarray(valid, dtype=bool)
    colors = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    in_range = label_array < taxonomy.num_classes
    colors[in_range] = taxonomy.palette[label_array[in_range]]
    overlay = Image.fromarray(colors, mode="RGB")
    mask = valid_array & in_range & (label_array != 0)
    alpha_mask = Image.fromarray((mask * round(alpha * 255)).astype(np.uint8), mode="L")
    return Image.composite(overlay, base, alpha_mask)
