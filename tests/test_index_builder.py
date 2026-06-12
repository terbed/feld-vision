import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from feldvision.data.index_builder import ChipIndexBuildConfig, build_chip_index


def test_build_chip_index_emits_candidates_counts_and_null_boundaries(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.tif"
    metadata_path = tmp_path / "mask.json"
    sheets_path = tmp_path / "sheets.geojson"
    mask = np.zeros((16, 24), dtype=np.uint8)
    mask[4:12, 4:12] = 10
    transform = from_origin(0, 16, 1, 1)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=16,
        width=24,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as destination:
        destination.write(mask, 1)
    metadata_path.write_text(
        json.dumps(
            {
                "label_version": "test",
                "raster_width_px": 24,
                "raster_height_px": 16,
            }
        )
    )
    sheets = gpd.GeoDataFrame(
        {"style_id": [1, 2]},
        geometry=[box(0, 0, 12, 16), box(12, 0, 24, 16)],
        crs="EPSG:3857",
    )
    sheets.to_file(sheets_path, driver="GeoJSON")

    chips = build_chip_index(
        mask_path=mask_path,
        stylesheets_path=sheets_path,
        metadata_path=metadata_path,
        config=ChipIndexBuildConfig(chip_size=8, stride=4, block_size=8),
        progress=False,
    )

    assert len(chips) == 15
    assert chips["style_id"].isna().sum() == 9
    assert chips.loc[chips["chip_id"].eq("r4_c4"), "px_10"].item() == 64
    assert chips.loc[chips["chip_id"].eq("r0_c0"), "style_id"].item() == 1
    assert (
        chips[list(["px_bg", *(f"px_{raw_id}" for raw_id in range(10, 18))])]
        .sum(axis=1)
        .eq(64)
        .all()
    )
