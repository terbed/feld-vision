from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from feldvision.config import load_experiment_config, resolve_project_path
from feldvision.data.index import (
    load_chip_index,
    load_label_version,
    validate_chip_index,
    write_json,
)
from feldvision.data.splits import build_splits
from feldvision.taxonomy import load_taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic train/val/test splits")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="Raw chips.parquet path")
    parser.add_argument("--output", help="Output chips_split.parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    output_path = (
        Path(args.output) if args.output else resolve_project_path(config.data.chips, args.config)
    )
    if args.input:
        input_path = Path(args.input)
    elif output_path.stem.endswith("_split"):
        input_path = output_path.with_name(output_path.stem.removesuffix("_split") + ".parquet")
    else:
        raise ValueError("--input is required when output is not named *_split.parquet")

    taxonomy = load_taxonomy(resolve_project_path(config.taxonomy, args.config))
    chips = load_chip_index(input_path)
    metadata_path = resolve_project_path(config.data.mask_metadata, args.config)
    stylesheets_path = resolve_project_path(config.data.stylesheets, args.config)
    style_ids = set(gpd.read_file(stylesheets_path, columns=["style_id"])["style_id"].astype(int))
    validate_chip_index(
        chips,
        expected_label_version=load_label_version(metadata_path),
        known_style_ids=style_ids,
        expected_stride=config.split.stride,
    )
    result = build_splits(
        chips,
        taxonomy,
        config.split,
        max_jitter_px=(config.augmentation.max_jitter_px if config.augmentation.jitter else 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.chips.to_parquet(output_path, index=False)
    write_json(result.summary, output_path.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
