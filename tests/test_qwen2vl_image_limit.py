#!/usr/bin/env python3
"""Tests for Qwen2-VL image pixel limiting in lmms-eval wrapper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models import qwen2_vl


class Qwen2VLImageLimitTest(unittest.TestCase):
    def test_image_message_content_sets_default_max_pixels(self) -> None:
        image = Image.new("RGB", (32, 32), color="white")

        content = qwen2_vl._image_message_content(
            image,
            max_pixels=qwen2_vl.DEFAULT_IMAGE_MAX_PIXELS,
        )

        self.assertEqual(content["type"], "image")
        self.assertEqual(content["max_pixels"], qwen2_vl.DEFAULT_IMAGE_MAX_PIXELS)
        self.assertTrue(content["image"].startswith("data:image/jpeg;base64,"))

    def test_image_message_content_can_disable_max_pixels(self) -> None:
        image = Image.new("RGB", (32, 32), color="white")

        content = qwen2_vl._image_message_content(image, max_pixels=None)

        self.assertNotIn("max_pixels", content)


if __name__ == "__main__":
    unittest.main()
