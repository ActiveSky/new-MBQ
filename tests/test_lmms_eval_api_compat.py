#!/usr/bin/env python3
"""Regression tests for lmms-eval CLI API compatibility."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main import _ensure_lmms_eval_args, parse_eval_args


class LmmsEvalApiCompatTest(unittest.TestCase):
    def test_adds_process_with_media_default_for_programmatic_args(self) -> None:
        args = argparse.Namespace()

        _ensure_lmms_eval_args(args)

        self.assertFalse(args.process_with_media)

    def test_preserves_explicit_process_with_media_value(self) -> None:
        args = argparse.Namespace(process_with_media=True)

        _ensure_lmms_eval_args(args)

        self.assertTrue(args.process_with_media)

    def test_parser_exposes_process_with_media_flag(self) -> None:
        with patch.object(sys, "argv", ["main.py"]):
            self.assertFalse(parse_eval_args().process_with_media)
        with patch.object(sys, "argv", ["main.py", "--process_with_media"]):
            self.assertTrue(parse_eval_args().process_with_media)


if __name__ == "__main__":
    unittest.main()
