from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from feldvision.config import AugmentationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    config: AugmentationConfig,
    *,
    train: bool,
    include_context: bool,
) -> A.Compose:
    transforms: list[Any] = []
    if train and config.enabled:
        transforms.extend(
            [
                A.HorizontalFlip(p=config.horizontal_flip_p),
                A.VerticalFlip(p=config.vertical_flip_p),
                A.RandomRotate90(p=config.rotate90_p),
                A.RandomBrightnessContrast(p=config.brightness_contrast_p),
                A.HueSaturationValue(p=config.hue_saturation_p),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    additional_targets = {"context": "image"} if include_context else {}
    return A.Compose(transforms, additional_targets=additional_targets)


def normalize_image(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB image in HWC layout")
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    tensor /= 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).reshape(3, 1, 1)
    return (tensor - mean) / std
