from __future__ import annotations

import argparse
from pathlib import Path

from feldvision.config import load_experiment_config, resolve_project_path
from feldvision.data.index_builder import (
    ChipIndexBuildConfig,
    build_chip_index,
    write_chip_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the production chip index")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/chips.parquet")
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    chips = build_chip_index(
        mask_path=resolve_project_path(config.data.mask, args.config),
        stylesheets_path=resolve_project_path(config.data.stylesheets, args.config),
        metadata_path=resolve_project_path(config.data.mask_metadata, args.config),
        config=ChipIndexBuildConfig(
            chip_size=config.split.chip_size,
            stride=config.split.stride,
            block_size=args.block_size,
        ),
    )
    write_chip_index(
        chips,
        output_path=Path(args.output),
        write_csv=args.csv,
    )


if __name__ == "__main__":
    main()
