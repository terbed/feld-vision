from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from torch import nn

from feldvision.pipeline import run_training


class TinyPipelineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(3, 2, kernel_size=1)

    def forward(
        self,
        detail: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        return self.classifier(detail)


def write_pipeline_inputs(directory: Path) -> tuple[Path, Path, Path, Path]:
    basemap_path = directory / "basemap.tif"
    mask_path = directory / "mask.tif"
    chips_path = directory / "chips_split.parquet"
    taxonomy_path = directory / "taxonomy.yaml"
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
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:8, 8:] = 10
    mask[8:, :] = 10
    with rasterio.open(mask_path, "w", count=1, **profile) as destination:
        destination.write(mask, 1)
    pd.DataFrame(
        {
            "chip_id": ["r0_c0", "r0_c8", "r8_c0", "r8_c8"],
            "row_off": [0, 0, 8, 8],
            "col_off": [0, 8, 0, 8],
            "size": [8, 8, 8, 8],
            "style_id": [1, 1, 2, 2],
            "split": ["train", "train", "val", "val"],
            "px_bg": [64, 0, 0, 0],
            "px_10": [0, 64, 64, 64],
        }
    ).to_parquet(chips_path, index=False)
    taxonomy_path.write_text(
        yaml.safe_dump(
            {
                "ignore_index": 255,
                "classes": [
                    {
                        "name": "background",
                        "train_id": 0,
                        "raw_ids": [0],
                        "color": [0, 0, 0],
                    },
                    {
                        "name": "river",
                        "train_id": 1,
                        "raw_ids": [10],
                        "color": [0, 0, 255],
                    },
                ],
                "ignored": [],
            }
        )
    )
    return basemap_path, mask_path, chips_path, taxonomy_path


def test_run_training_executes_one_epoch_and_logs_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    basemap, mask, chips, taxonomy = write_pipeline_inputs(tmp_path)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
    config["data"].update(
        {
            "basemap": str(basemap),
            "mask": str(mask),
            "chips": str(chips),
            "stylesheets": str(tmp_path / "unused.shp"),
            "mask_metadata": str(tmp_path / "unused.json"),
        }
    )
    config["taxonomy"] = str(taxonomy)
    config["split"].update(
        {
            "chip_size": 8,
            "stride": 8,
            "val_cell_size_px": 16,
            "test_style_ids": [],
        }
    )
    config["model"].update({"pretrained": False, "name": "segformer_b2_single"})
    config["augmentation"].update({"enabled": False, "jitter": False, "max_jitter_px": 0})
    config["loader"].update(
        {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )
    config["optim"].update({"epochs": 1, "amp": "off", "lr": 1e-3})
    config["scheduler"].update({"warmup_epochs": 0, "min_lr": 1e-5})
    config["runtime"].update(
        {
            "device": "cpu",
            "output_dir": str(tmp_path / "runs"),
            "experiment_name": "integration",
        }
    )
    config["sampler"]["min_sampling_chips"] = 1
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "feldvision.pipeline.build_model",
        lambda model_config, num_classes: TinyPipelineModel(),
    )

    history = run_training(config_path)

    run_dir = tmp_path / "runs" / "integration"
    assert len(history) == 1
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
