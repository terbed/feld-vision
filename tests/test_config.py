from pathlib import Path

import pytest
import yaml

from feldvision.config import ConfigError, load_experiment_config

ROOT = Path(__file__).parents[1]


def test_default_config_is_valid_and_derives_two_buffer_steps() -> None:
    config = load_experiment_config(ROOT / "configs/default.yaml")

    assert config.model.name == "segformer_b2_single"
    assert config.split.test_style_ids == (53, 54, 22, 18, 121)
    assert config.buffer_steps == 2


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    raw["model"]["unknown"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="unknown keys"):
        load_experiment_config(path)


def test_jitter_cannot_exceed_half_stride(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    raw["augmentation"]["max_jitter_px"] = 65
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="half the stride"):
        load_experiment_config(path)
