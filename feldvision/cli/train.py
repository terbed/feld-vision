from __future__ import annotations

import argparse

from feldvision.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a feld-vision segmentation model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
