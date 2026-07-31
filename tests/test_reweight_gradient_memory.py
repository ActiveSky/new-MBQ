#!/usr/bin/env python3
"""Regression test for reweight-gradient memory cleanup."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qmllm.methods.mbq.quantize.pre_quant import _clear_model_gradients


class _ProcessModel:
    def __init__(self) -> None:
        self.model = nn.Linear(4, 4)


class ReweightGradientMemoryTest(unittest.TestCase):
    def test_clears_underlying_model_gradients(self) -> None:
        process_model = _ProcessModel()
        process_model.model(torch.ones(1, 4)).sum().backward()
        self.assertTrue(any(p.grad is not None for p in process_model.model.parameters()))

        _clear_model_gradients(process_model)

        self.assertTrue(all(p.grad is None for p in process_model.model.parameters()))


if __name__ == "__main__":
    unittest.main()
