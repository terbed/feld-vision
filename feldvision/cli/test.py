from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import rasterio
import torch
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from feldvision.config import resolve_project_path
from feldvision.logging import ClearMLLogger, CompositeLogger, LocalLogger
from feldvision.models import build_model
from feldvision.pipeline import load_config_and_taxonomy, resolve_device
from feldvision.reconstruct import (
    reconstruct_sheet,
    save_prediction,
    save_prediction_image,
    save_triptych,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct held-out map sheets")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="reconstructions")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--style-id",
        action="append",
        type=int,
        help="Reconstruct only this configured test style; may be repeated",
    )
    parser.add_argument(
        "--max-size-px",
        type=int,
        help="Restrict each sheet to a centered square crop for smoke testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, taxonomy = load_config_and_taxonomy(args.config)
    if not config.split.test_style_ids:
        raise ValueError("split.test_style_ids is empty; there are no test sheets")
    device = resolve_device(config.runtime.device)
    model = build_model(config.model, taxonomy.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    basemap_path = resolve_project_path(config.data.basemap, args.config)
    mask_path = resolve_project_path(config.data.mask, args.config)
    stylesheets_path = resolve_project_path(config.data.stylesheets, args.config)
    sheets = gpd.read_file(stylesheets_path)
    with rasterio.open(basemap_path) as basemap:
        sheets = sheets.to_crs(basemap.crs)
        raster_transform = basemap.transform
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loggers = [LocalLogger(output_dir, asdict(config))]
    if config.clearml.enabled:
        loggers.append(
            ClearMLLogger(
                project=config.clearml.project,
                task_name=f"{config.clearml.task}-test",
                config=asdict(config),
            )
        )
    logger = CompositeLogger(*loggers)
    all_metrics: dict[str, dict[str, float]] = {}
    context_scale = (
        None if config.model.name == "segformer_b2_single" else config.model.context_scale
    )
    requested_style_ids = tuple(args.style_id or config.split.test_style_ids)
    unknown = set(requested_style_ids) - set(config.split.test_style_ids)
    if unknown:
        raise ValueError(f"requested style ids are not configured test sheets: {sorted(unknown)}")
    try:
        for iteration, style_id in enumerate(requested_style_ids):
            matches = sheets.loc[sheets["style_id"].astype(int).eq(style_id)]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one geometry for style_id {style_id}, got {len(matches)}"
                )
            geometry = matches.geometry.iloc[0]
            if args.max_size_px is not None:
                geometry = _bounded_geometry(
                    geometry,
                    pixel_width=abs(raster_transform.a),
                    pixel_height=abs(raster_transform.e),
                    max_size_px=args.max_size_px,
                )
            reconstruction = reconstruct_sheet(
                model,
                geometry=geometry,
                basemap_path=basemap_path,
                mask_path=mask_path,
                taxonomy=taxonomy,
                device=device,
                chip_size=config.split.chip_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                context_scale=context_scale,
            )
            prediction_path = save_prediction(
                reconstruction,
                output_dir / f"style_{style_id}_prediction.tif",
            )
            prediction_image_path = save_prediction_image(
                reconstruction,
                taxonomy=taxonomy,
                path=output_dir / f"style_{style_id}_prediction.png",
            )
            preview_path = save_triptych(
                reconstruction,
                basemap_path=basemap_path,
                taxonomy=taxonomy,
                path=output_dir / f"style_{style_id}_triptych.jpg",
            )
            all_metrics[str(style_id)] = reconstruction.metrics
            logger.log_metrics(
                {
                    f"test/style/{style_id}/{name}": value
                    for name, value in reconstruction.metrics.items()
                },
                step=iteration,
            )
            logger.log_artifact(f"style_{style_id}_prediction", prediction_path)
            logger.log_artifact(
                f"style_{style_id}_prediction_png",
                prediction_image_path,
            )
            logger.log_image(
                f"style_{style_id}_prediction",
                prediction_image_path,
                step=iteration,
            )
            logger.log_image(
                f"style_{style_id}_triptych",
                preview_path,
                step=iteration,
            )
    finally:
        logger.close()
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n"
    )


def _bounded_geometry(
    geometry: BaseGeometry,
    *,
    pixel_width: float,
    pixel_height: float,
    max_size_px: int,
) -> BaseGeometry:
    if max_size_px <= 0:
        raise ValueError("max_size_px must be positive")
    center = geometry.representative_point()
    half_width = pixel_width * max_size_px / 2
    half_height = pixel_height * max_size_px / 2
    bounded = geometry.intersection(
        box(
            center.x - half_width,
            center.y - half_height,
            center.x + half_width,
            center.y + half_height,
        )
    )
    if bounded.is_empty:
        raise ValueError("bounded reconstruction geometry is empty")
    return bounded


if __name__ == "__main__":
    main()
