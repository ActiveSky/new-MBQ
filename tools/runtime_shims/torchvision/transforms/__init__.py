"""Minimal PIL-to-tensor transforms used by the InternVL2 COCO collector."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .functional import InterpolationMode


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class Lambda:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, value):
        return self.fn(value)


class Resize:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = tuple(size) if isinstance(size, (tuple, list)) else (size, size)
        self.interpolation = interpolation

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("runtime Resize supports PIL images only")
        mode = self.interpolation.value if isinstance(self.interpolation, InterpolationMode) else self.interpolation
        return image.resize((self.size[1], self.size[0]), resample=mode)


class ToTensor:
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("runtime ToTensor supports PIL images only")
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean.to(dtype=tensor.dtype)) / self.std.to(dtype=tensor.dtype)


__all__ = ["Compose", "InterpolationMode", "Lambda", "Normalize", "Resize", "ToTensor"]
