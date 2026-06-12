from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class ExperimentLogger(Protocol):
    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None: ...

    def log_artifact(self, name: str, path: str | Path) -> None: ...

    def log_image(self, name: str, path: str | Path, *, step: int) -> None: ...

    def close(self) -> None: ...


class NullLogger:
    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        del metrics, step

    def log_artifact(self, name: str, path: str | Path) -> None:
        del name, path

    def log_image(self, name: str, path: str | Path, *, step: int) -> None:
        del name, path, step

    def close(self) -> None:
        return None


class LocalLogger:
    def __init__(self, output_dir: str | Path, config: Mapping[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        (self.output_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        record = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_artifact(self, name: str, path: str | Path) -> None:
        manifest_path = self.output_dir / "artifacts.jsonl"
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"name": name, "path": str(path)}) + "\n")

    def log_image(self, name: str, path: str | Path, *, step: int) -> None:
        manifest_path = self.output_dir / "images.jsonl"
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"name": name, "path": str(path), "step": step}) + "\n")

    def close(self) -> None:
        return None


class ClearMLLogger:
    def __init__(
        self,
        *,
        project: str,
        task_name: str,
        config: Mapping[str, Any],
    ) -> None:
        from clearml import Task

        self.task = Task.init(project_name=project, task_name=task_name)
        self.task.connect(dict(config))
        self.logger = self.task.get_logger()

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        for key, value in metrics.items():
            series, _, title = key.rpartition("/")
            self.logger.report_scalar(
                title=title or "metrics",
                series=series or key,
                value=value,
                iteration=step,
            )

    def log_artifact(self, name: str, path: str | Path) -> None:
        self.task.upload_artifact(name=name, artifact_object=str(path))

    def log_image(self, name: str, path: str | Path, *, step: int) -> None:
        self.logger.report_image(
            title="reconstruction",
            series=name,
            local_path=str(path),
            iteration=step,
        )

    def close(self) -> None:
        self.task.close()


class CompositeLogger:
    def __init__(self, *loggers: ExperimentLogger) -> None:
        self.loggers = loggers

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)

    def log_artifact(self, name: str, path: str | Path) -> None:
        for logger in self.loggers:
            logger.log_artifact(name, path)

    def log_image(self, name: str, path: str | Path, *, step: int) -> None:
        for logger in self.loggers:
            logger.log_image(name, path, step=step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
