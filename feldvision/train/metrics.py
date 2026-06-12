from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConfusionMatrix:
    num_classes: int
    ignore_index: int = 255
    matrix: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.ndim == 4:
            prediction = prediction.argmax(dim=1)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes do not match")
        prediction = prediction.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        target = target.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        valid = (
            target.ne(self.ignore_index)
            & target.ge(0)
            & target.lt(self.num_classes)
            & prediction.ge(0)
            & prediction.lt(self.num_classes)
        )
        encoded = target[valid] * self.num_classes + prediction[valid]
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def merge(self, other: ConfusionMatrix) -> None:
        if self.num_classes != other.num_classes:
            raise ValueError("cannot merge confusion matrices with different class counts")
        self.matrix += other.matrix

    def iou(self) -> torch.Tensor:
        matrix = self.matrix.to(torch.float64)
        intersection = matrix.diag()
        union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
        return torch.where(
            union > 0,
            intersection / union,
            torch.full_like(union, torch.nan),
        )


class SegmentationMetrics:
    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        ignore_index: int = 255,
    ) -> None:
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.global_matrix = ConfusionMatrix(len(class_names), ignore_index)
        self.style_matrices: dict[int, ConfusionMatrix] = {}

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        style_ids: torch.Tensor | list[int] | tuple[int, ...],
    ) -> None:
        predictions = logits.argmax(dim=1)
        self.global_matrix.update(predictions, target)
        style_tensor = torch.as_tensor(style_ids, dtype=torch.int64).reshape(-1)
        if len(style_tensor) != len(target):
            raise ValueError("one style_id is required per batch item")
        for style_id in style_tensor.unique().tolist():
            batch_mask = style_tensor.eq(style_id)
            matrix = self.style_matrices.setdefault(
                int(style_id),
                ConfusionMatrix(len(self.class_names), self.ignore_index),
            )
            device_mask = batch_mask.to(predictions.device)
            matrix.update(predictions[device_mask], target[device_mask])

    def compute(self) -> dict[str, float]:
        result = _matrix_metrics(self.global_matrix, self.class_names)
        for style_id, matrix in sorted(self.style_matrices.items()):
            style_metrics = _matrix_metrics(matrix, self.class_names)
            for name, value in style_metrics.items():
                result[f"style/{style_id}/{name}"] = value
        return result


def metrics_from_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    class_names: tuple[str, ...],
    ignore_index: int = 255,
) -> dict[str, float]:
    matrix = ConfusionMatrix(len(class_names), ignore_index)
    matrix.update(prediction, target)
    return _matrix_metrics(matrix, class_names)


def _matrix_metrics(
    matrix: ConfusionMatrix,
    class_names: tuple[str, ...],
) -> dict[str, float]:
    iou = matrix.iou()
    foreground = iou[1:]
    finite_foreground = foreground[~foreground.isnan()]
    mean_iou = float(finite_foreground.mean()) if len(finite_foreground) else float("nan")
    result = {
        "miou": mean_iou,
        "background_iou": float(iou[0]),
    }
    result.update(
        {f"iou/{name}": float(value) for name, value in zip(class_names[1:], iou[1:], strict=True)}
    )
    return result
