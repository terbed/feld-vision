import pytest
import torch

from feldvision.train.losses import SegmentationLoss
from feldvision.train.metrics import ConfusionMatrix, SegmentationMetrics


def test_loss_ignores_255_pixels_and_is_differentiable() -> None:
    logits = torch.tensor(
        [
            [
                [[5.0, -5.0], [1.0, 100.0]],
                [[-5.0, 5.0], [2.0, -100.0]],
            ]
        ],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1], [1, 255]]])
    loss_fn = SegmentationLoss(num_classes=2, ignore_index=255)

    loss = loss_fn(logits, target)
    loss.backward()

    assert loss.item() < 0.2
    assert logits.grad is not None
    assert logits.grad[..., 1, 1].eq(0).all()


def test_all_ignored_target_returns_zero_loss() -> None:
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    target = torch.full((1, 2, 2), 255)
    loss = SegmentationLoss(num_classes=3)(logits, target)

    loss.backward()

    assert loss.item() == 0
    assert logits.grad is not None


def test_confusion_matrix_iou_matches_known_values() -> None:
    matrix = ConfusionMatrix(num_classes=2)
    prediction = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([[0, 1], [0, 255]])

    matrix.update(prediction, target)

    assert matrix.iou()[0].item() == pytest.approx(0.5)
    assert matrix.iou()[1].item() == pytest.approx(0.5)


def test_metrics_report_global_style_summary_and_per_style_miou() -> None:
    logits = torch.tensor(
        [
            [[[4.0, -4.0]], [[-4.0, 4.0]]],
            [[[-4.0, 4.0]], [[4.0, -4.0]]],
        ]
    )
    target = torch.tensor([[[0, 1]], [[1, 0]]])
    metrics = SegmentationMetrics(class_names=("background", "river"))

    metrics.update(logits, target, [7, 8])
    result = metrics.compute()

    assert result["miou"] == 1.0
    assert result["style_mean/miou"] == 1.0
    assert result["style_mean/iou/river"] == 1.0
    assert result["style_min/miou"] == 1.0
    assert result["style_max/miou"] == 1.0
    assert result["style_id/7/miou"] == 1.0
    assert result["style_id/8/miou"] == 1.0
    assert "style/8/iou/river" not in result
