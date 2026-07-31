#!/usr/bin/env python3
"""Tests for module-family low-rank truncation cache utility."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "truncate_low_rank_by_module.py"


def low_rank_item(name: str, out_features: int, in_features: int, rank: int) -> dict:
    return {
        "name": name,
        "rank": rank,
        "up": torch.arange(out_features * rank, dtype=torch.float32).reshape(
            out_features, rank
        ),
        "down": torch.arange(rank * in_features, dtype=torch.float32).reshape(
            rank, in_features
        ),
    }


class TruncateLowRankByModuleTest(unittest.TestCase):
    def test_truncates_only_requested_module_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "base.pt"
            reference_path = tmp_path / "reference.pt"
            output_path = tmp_path / "out.pt"

            names = [
                "language_model.model.layers.0.feed_forward.w1",
                "language_model.model.layers.1.feed_forward.w1",
                "language_model.model.layers.0.feed_forward.w3",
                "language_model.model.layers.0.attention.wqkv",
            ]
            reference_cache = {
                "low_rank": [
                    low_rank_item(names[0], 8, 4, 128),
                    low_rank_item(names[1], 8, 4, 128),
                    low_rank_item(names[2], 8, 4, 128),
                    low_rank_item(names[3], 6, 4, 128),
                ]
            }
            base_cache = {
                "linear_bit_map": {name: 2 for name in names},
                "low_rank": [
                    low_rank_item(names[0], 8, 4, 128),
                    low_rank_item(names[1], 8, 4, 128),
                    low_rank_item(names[2], 8, 4, 128),
                    low_rank_item(names[3], 6, 4, 32),
                ],
            }
            torch.save(reference_cache, reference_path)
            torch.save(base_cache, base_path)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-cache",
                    str(base_path),
                    "--reference-cache",
                    str(reference_path),
                    "--module",
                    "w1",
                    "--rank",
                    "64",
                    "--name",
                    "unit_w1_r64",
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            out_cache = torch.load(output_path, map_location="cpu")
            ranks = {item["name"]: item["rank"] for item in out_cache["low_rank"]}
            self.assertEqual(ranks[names[0]], 64)
            self.assertEqual(ranks[names[1]], 64)
            self.assertEqual(ranks[names[2]], 128)
            self.assertEqual(ranks[names[3]], 32)

            for item in out_cache["low_rank"]:
                if item["name"] in names[:2]:
                    self.assertEqual(tuple(item["up"].shape), (8, 64))
                    self.assertEqual(tuple(item["down"].shape), (64, 4))

            summary = json.loads(output_path.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["target_module"], "w1")
            self.assertEqual(summary["target_rank"], 64)
            self.assertEqual(summary["target_module_low_rank_count"], 2)
            self.assertEqual(summary["low_rank_ranks_by_module"]["w1"], {"64": 2})
            self.assertEqual(summary["low_rank_ranks_by_module"]["w3"], {"128": 1})
            self.assertEqual(summary["low_rank_ranks_by_module"]["wqkv"], {"32": 1})
            self.assertLess(summary["actual_bit"], summary["base_actual_bit"])

            reloaded_base = torch.load(base_path, map_location="cpu")
            self.assertEqual(reloaded_base["low_rank"][0]["rank"], 128)

    def test_truncates_multiple_requested_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "base.pt"
            reference_path = tmp_path / "reference.pt"
            output_path = tmp_path / "out.pt"

            names = [
                "language_model.model.layers.0.feed_forward.w1",
                "language_model.model.layers.0.feed_forward.w3",
                "language_model.model.layers.0.attention.wo",
                "language_model.model.layers.0.attention.wqkv",
            ]
            reference_cache = {
                "low_rank": [
                    low_rank_item(names[0], 8, 4, 128),
                    low_rank_item(names[1], 8, 4, 128),
                    low_rank_item(names[2], 4, 4, 128),
                    low_rank_item(names[3], 6, 4, 128),
                ]
            }
            base_cache = {
                "linear_bit_map": {name: 2 for name in names},
                "low_rank": [
                    low_rank_item(names[0], 8, 4, 128),
                    low_rank_item(names[1], 8, 4, 128),
                    low_rank_item(names[2], 4, 4, 128),
                    low_rank_item(names[3], 6, 4, 32),
                ],
            }
            torch.save(reference_cache, reference_path)
            torch.save(base_cache, base_path)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-cache",
                    str(base_path),
                    "--reference-cache",
                    str(reference_path),
                    "--set",
                    "w1:64",
                    "--set",
                    "w3:32",
                    "--name",
                    "unit_w1_r64_w3_r32",
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            out_cache = torch.load(output_path, map_location="cpu")
            ranks = {item["name"]: item["rank"] for item in out_cache["low_rank"]}
            self.assertEqual(ranks[names[0]], 64)
            self.assertEqual(ranks[names[1]], 32)
            self.assertEqual(ranks[names[2]], 128)
            self.assertEqual(ranks[names[3]], 32)

            summary = json.loads(output_path.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["target_module"], "multi")
            self.assertIsNone(summary["target_rank"])
            self.assertEqual(summary["target_rank_set"], {"w1": 64, "w3": 32})
            self.assertEqual(
                summary["target_module_low_rank_counts"], {"w1": 1, "w3": 1}
            )
            self.assertEqual(summary["low_rank_ranks_by_module"]["w1"], {"64": 1})
            self.assertEqual(summary["low_rank_ranks_by_module"]["w3"], {"32": 1})
            self.assertEqual(summary["low_rank_ranks_by_module"]["wo"], {"128": 1})
            self.assertEqual(summary["low_rank_ranks_by_module"]["wqkv"], {"32": 1})
            self.assertEqual(
                summary["truncate_summary"]["changed_counts_by_module"],
                {"w1": 1, "w3": 1},
            )


if __name__ == "__main__":
    unittest.main()
