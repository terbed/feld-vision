from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin

from feldvision.config import AugmentationConfig
from feldvision.data.dataset import ChipDataset
from feldvision.taxonomy import ClassDefinition, Taxonomy


def write_rasters(directory: Path) -> tuple[Path, Path]:
    image_path = directory / "image.tif"
    mask_path = directory / "mask.tif"
    image = np.zeros((3, 32, 32), dtype=np.uint8)
    image[0] = np.arange(32, dtype=np.uint8)[None, :]
    image[1] = np.arange(32, dtype=np.uint8)[:, None]
    image[2] = 100
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 10
    transform = from_origin(0, 32, 1, 1)
    common = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "transform": transform,
        "crs": "EPSG:3857",
        "dtype": "uint8",
    }
    with rasterio.open(image_path, "w", count=3, **common) as destination:
        destination.write(image)
    with rasterio.open(mask_path, "w", count=1, **common) as destination:
        destination.write(mask, 1)
    return image_path, mask_path


def taxonomy() -> Taxonomy:
    return Taxonomy(
        classes=(
            ClassDefinition("background", 0, (0,), (0, 0, 0)),
            ClassDefinition("river", 1, (10,), (0, 0, 255)),
        ),
        ignored_raw_ids=frozenset({15}),
    )


def test_dataset_reads_aligned_detail_target_and_context(tmp_path: Path) -> None:
    image_path, mask_path = write_rasters(tmp_path)
    chips = pd.DataFrame(
        {
            "chip_id": ["r8_c8"],
            "row_off": [8],
            "col_off": [8],
            "size": [8],
            "style_id": [5],
            "split": ["val"],
        }
    )
    dataset = ChipDataset(
        chips,
        split="val",
        basemap_path=image_path,
        mask_path=mask_path,
        taxonomy=taxonomy(),
        augmentation=AugmentationConfig(enabled=False, jitter=False),
        include_context=True,
        context_scale=2,
    )

    sample = dataset[0]

    assert sample["detail"].shape == (3, 8, 8)
    assert sample["context"].shape == (3, 8, 8)
    assert sample["target"].shape == (8, 8)
    assert sample["target"].dtype == torch.long
    assert sample["target"].eq(1).all()
    assert sample["meta"] == {
        "chip_id": "r8_c8",
        "style_id": 5,
        "row_off": 8,
        "col_off": 8,
    }
    assert not torch.equal(sample["detail"], sample["context"])
    dataset.close()


def test_validation_offsets_are_never_jittered(tmp_path: Path) -> None:
    image_path, mask_path = write_rasters(tmp_path)
    chips = pd.DataFrame(
        {
            "chip_id": ["r8_c8"],
            "row_off": [8],
            "col_off": [8],
            "size": [8],
            "style_id": [5],
            "split": ["val"],
        }
    )
    dataset = ChipDataset(
        chips,
        split="val",
        basemap_path=image_path,
        mask_path=mask_path,
        taxonomy=taxonomy(),
        augmentation=AugmentationConfig(max_jitter_px=4),
    )

    assert dataset[0]["meta"]["row_off"] == 8
    assert dataset[0]["meta"]["col_off"] == 8
