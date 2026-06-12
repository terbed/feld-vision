from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box
from torch import nn

from feldvision.reconstruct import (
    reconstruct_sheet,
    save_prediction,
    save_prediction_image,
    save_triptych,
)
from feldvision.taxonomy import ClassDefinition, Taxonomy


class RiverModel(nn.Module):
    def forward(
        self,
        detail: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        batch, _, height, width = detail.shape
        logits = torch.zeros(batch, 2, height, width, device=detail.device)
        logits[:, 1] = 5
        return logits


def write_rasters(directory: Path) -> tuple[Path, Path]:
    basemap_path = directory / "basemap.tif"
    mask_path = directory / "mask.tif"
    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "transform": from_origin(0, 16, 1, 1),
        "crs": "EPSG:3857",
        "dtype": "uint8",
    }
    with rasterio.open(basemap_path, "w", count=3, **profile) as destination:
        destination.write(np.full((3, 16, 16), 100, dtype=np.uint8))
    with rasterio.open(mask_path, "w", count=1, **profile) as destination:
        mask = np.full((16, 16), 10, dtype=np.uint8)
        mask[8, 8] = 15
        destination.write(mask, 1)
    return basemap_path, mask_path


def taxonomy() -> Taxonomy:
    return Taxonomy(
        classes=(
            ClassDefinition("background", 0, (0,), (0, 0, 0)),
            ClassDefinition("river", 1, (10,), (0, 0, 255)),
        ),
        ignored_raw_ids=frozenset({15}),
    )


def test_reconstruction_averages_windows_and_exports_artifacts(tmp_path: Path) -> None:
    basemap, mask = write_rasters(tmp_path)
    reconstruction = reconstruct_sheet(
        RiverModel(),
        geometry=box(2, 2, 14, 14),
        basemap_path=basemap,
        mask_path=mask,
        taxonomy=taxonomy(),
        device=torch.device("cpu"),
        chip_size=8,
        overlap=4,
        batch_size=2,
    )

    assert reconstruction.prediction.shape == (12, 12)
    assert (reconstruction.prediction == 1).all()
    assert reconstruction.metrics["miou"] == 1.0
    prediction_path = save_prediction(reconstruction, tmp_path / "prediction.tif")
    prediction_image_path = save_prediction_image(
        reconstruction,
        taxonomy=taxonomy(),
        path=tmp_path / "prediction.png",
    )
    preview_path = save_triptych(
        reconstruction,
        basemap_path=basemap,
        taxonomy=taxonomy(),
        path=tmp_path / "triptych.jpg",
    )
    assert prediction_path.exists()
    assert prediction_image_path.exists()
    assert preview_path.exists()
    with rasterio.open(prediction_path) as source:
        assert source.crs.to_epsg() == 3857
        assert source.nodata == 255
        exported = source.read(1)
        assert exported[6, 6] == 1
    prediction_image = np.asarray(Image.open(prediction_image_path))
    assert prediction_image.shape == (12, 12, 4)
    assert prediction_image[6, 6].tolist() == [0, 0, 255, 255]


def test_reconstruction_supports_unlabeled_inference(tmp_path: Path) -> None:
    basemap, _ = write_rasters(tmp_path)

    reconstruction = reconstruct_sheet(
        RiverModel(),
        geometry=box(2, 2, 14, 14),
        basemap_path=basemap,
        mask_path=None,
        taxonomy=taxonomy(),
        device=torch.device("cpu"),
        chip_size=8,
        overlap=0,
    )

    assert reconstruction.metrics == {}
    assert reconstruction.sheet_mask.all()
    assert not reconstruction.evaluation_mask.any()
