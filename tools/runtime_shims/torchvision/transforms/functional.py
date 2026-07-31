"""Minimal interpolation enum used by the local image transform shim."""

from enum import Enum

from PIL import Image


class InterpolationMode(Enum):
    NEAREST = Image.Resampling.NEAREST
    BILINEAR = Image.Resampling.BILINEAR
    BICUBIC = Image.Resampling.BICUBIC
    LANCZOS = Image.Resampling.LANCZOS
