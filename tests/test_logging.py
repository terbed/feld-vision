import json
from pathlib import Path

from feldvision.logging import CompositeLogger, LocalLogger, NullLogger


def test_local_and_composite_logger_write_metrics_artifacts_and_images(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    image = tmp_path / "image.png"
    artifact.write_bytes(b"artifact")
    image.write_bytes(b"image")
    logger = CompositeLogger(LocalLogger(tmp_path / "run", {"seed": 3}), NullLogger())

    logger.log_metrics({"val/miou": 0.5}, step=2)
    logger.log_artifact("checkpoint", artifact)
    logger.log_image("preview", image, step=2)
    logger.close()

    run_dir = tmp_path / "run"
    assert json.loads((run_dir / "config.json").read_text()) == {"seed": 3}
    assert json.loads((run_dir / "metrics.jsonl").read_text())["val/miou"] == 0.5
    assert json.loads((run_dir / "artifacts.jsonl").read_text())["name"] == "checkpoint"
    assert json.loads((run_dir / "images.jsonl").read_text())["name"] == "preview"
