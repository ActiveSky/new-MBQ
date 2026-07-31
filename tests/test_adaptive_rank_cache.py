#!/usr/bin/env python3
"""Unit tests for tools/build_adaptive_rank_cache.py."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_adaptive_rank_cache",
    REPO_ROOT / "tools" / "build_adaptive_rank_cache.py",
)
adaptive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adaptive)

ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "analyze_ck", REPO_ROOT / "tools" / "analyze_ck.py"
)
analyze_ck = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze_ck)


def low_rank_item(name: str, rank: int = 4) -> dict:
    return {
        "name": name,
        "rank": rank,
        "up": torch.arange(8 * rank, dtype=torch.float32).reshape(8, rank),
        "down": torch.arange(rank * 8, dtype=torch.float32).reshape(rank, 8),
        "svd_quant": True,
        "svd_quant_config": {
            "up": {
                "weight_quant": "per_channel",
                "quant_axis": "out_channel",
                "quant_bit": 4,
                "zero_point": False,
                "q_group_size": -1,
            },
            "down": {
                "weight_quant": "per_channel",
                "quant_axis": "in_channel",
                "quant_bit": 4,
                "zero_point": False,
                "q_group_size": -1,
            },
        },
    }


def normalized_analysis(names) -> dict:
    curves = {
        names[0]: [0.60, 0.80, 0.90, 1.00],
        names[1]: [0.20, 0.40, 0.80, 1.00],
    }
    return {
        "metric": {
            "version": 2,
            "normalized": True,
            "basis_order": "energy_ordered_svd_prefix",
            "coverage_denominator": "projection_score_E",
            "mask_semantics": ["answer_text"],
            "factor_basis": [adaptive.FACTOR_BASIS],
            "token_subsampling": False,
        },
        "per_module": [
            {
                "name": name,
                "family": "attn_in" if index == 0 else "mlp_in",
                "coverage_normalized": True,
                "mask_semantics": "answer_text",
                "factor_prefix_basis": adaptive.FACTOR_BASIS,
                "factor_prefix_file": f"factors/{index}.pt",
                "score_E": 1.0,
                "cum_c": curves[name],
            }
            for index, name in enumerate(names)
        ],
    }


class AdaptiveRankMathTest(unittest.TestCase):
    def test_ck_uses_both_modality_normalizers(self):
        singular_values = torch.tensor([2.0, 3.0])
        right_vectors = torch.eye(2)
        inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        text_mask = torch.tensor([True, False])
        vis_mask = torch.tensor([False, True])
        ck = analyze_ck.compute_ck(
            singular_values,
            right_vectors,
            inputs,
            text_mask,
            vis_mask,
            rho=0.5,
            denominator_text=10.0,
            denominator_vis=20.0,
        )
        torch.testing.assert_close(ck, torch.tensor([0.4, 0.9]))
        self.assertAlmostEqual(float(ck.sum()), 1.3, places=6)

    def test_threshold_selection_and_fallback(self):
        cum = [0.2, 0.4, 0.8, 1.0]
        self.assertEqual(
            adaptive.select_rank_for_tau(cum, [1, 2, 4], 0.4),
            (2, 0.4, True),
        )
        self.assertEqual(
            adaptive.select_rank_for_tau([0.1, 0.2, 0.3], [1, 2, 3], 0.9),
            (3, 0.3, False),
        )

    def test_threshold_breakpoint_is_exact(self):
        tau = math.nextafter(0.4, 1.0)
        self.assertEqual(
            adaptive.select_rank_for_tau([0.2, 0.4, 0.8], [1, 2, 3], tau),
            (3, 0.8, True),
        )

    def test_non_monotonic_coverage_rejected(self):
        with self.assertRaises(ValueError):
            adaptive.validate_cumulative_coverage([0.2, 0.19, 0.4], "bad")

    def test_legacy_analysis_rejected(self):
        with self.assertRaises(ValueError):
            adaptive.analysis_rows_by_name(
                {"metric": {"version": 1, "normalized": False}, "per_module": []}
            )

    def test_non_paper_text_mask_rejected(self):
        names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
        analysis = normalized_analysis(names)
        analysis["metric"]["mask_semantics"] = ["all_text"]
        with self.assertRaises(ValueError):
            adaptive.analysis_rows_by_name(analysis)

    def test_token_subsampling_rejected(self):
        names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
        analysis = normalized_analysis(names)
        analysis["metric"]["token_subsampling"] = True
        with self.assertRaises(ValueError):
            adaptive.analysis_rows_by_name(analysis)

    def test_independent_factor_basis_rejected(self):
        names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
        analysis = normalized_analysis(names)
        analysis["metric"]["factor_basis"] = ["independent_svd"]
        with self.assertRaises(ValueError):
            adaptive.analysis_rows_by_name(analysis)

    def test_compact_statistics_match_direct_ck(self):
        singular_values = torch.tensor([2.0, 3.0])
        text_energy = torch.tensor([1.0, 0.0])
        vis_energy = torch.tensor([0.0, 4.0])
        compact = analyze_ck.compute_ck_from_component_energies(
            singular_values,
            text_energy,
            vis_energy,
            rho=0.5,
            denominator_text=10.0,
            denominator_vis=20.0,
        )
        torch.testing.assert_close(compact, torch.tensor([0.4, 0.9]))

    def test_unobserved_svd_tail_remains_in_reconstruction_error(self):
        error, text_error, vis_error = (
            analyze_ck.reconstruct_resid_error_from_component_energies(
                S=torch.tensor([2.0]),
                text_energy=torch.tensor([1.0]),
                vis_energy=torch.tensor([0.0]),
                rho=0.5,
                denominator_text=10.0,
                denominator_vis=1.0,
                order=torch.tensor([0]),
                r=1,
                numerator_text=10.0,
                numerator_vis=0.0,
            )
        )
        self.assertAlmostEqual(text_error, 6.0)
        self.assertAlmostEqual(vis_error, 0.0)
        self.assertAlmostEqual(error, 0.6)

    def test_compact_analysis_pipeline_requires_no_raw_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "0_L0.q_proj"
            module_dir.mkdir()
            name = "model.layers.0.self_attn.q_proj"
            (root / "_global_meta.json").write_text(
                json.dumps(
                    {
                        "module_index": [
                            {
                                "idx": 0,
                                "name": name,
                                "short": "L0.q_proj",
                                "family": "attn_in",
                                "bit": 2,
                                "rho": 1.0,
                            }
                        ],
                        "scale_path": str(root / "base.pt"),
                        "max_tokens": None,
                        "factor_prefix_rank": 2,
                    }
                )
            )
            (module_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "metric_version": 2,
                        "mask_semantics": "answer_text",
                        "denominator_text": 10.0,
                        "denominator_vis": 10.0,
                        "numerator_text": 1.0,
                        "numerator_vis": 1.0,
                        "residual_frobenius_sq": 2.0,
                        "weight_shape": [2, 2],
                        "factor_prefix_rank": 2,
                        "factor_prefix_file": "svd_factor_prefix.pt",
                        "factor_prefix_basis": adaptive.FACTOR_BASIS,
                    }
                )
            )
            torch.save(torch.tensor([1.0, 1.0]), module_dir / "sigma.pt")
            torch.save(
                {
                    "input_energy_text": torch.tensor([1.0, 0.0]),
                    "input_energy_vis": torch.tensor([0.0, 1.0]),
                },
                module_dir / "component_stats.pt",
            )
            torch.save(
                {
                    "rank": 2,
                    "up": torch.eye(2),
                    "down": torch.eye(2),
                    "quantized": False,
                },
                module_dir / "svd_factor_prefix.pt",
            )

            analyze_ck.main(
                ["--input_dir", str(root), "--ranks", "1", "2", "--no_plot"]
            )

            output = json.loads((root / "_ck_analysis.json").read_text())
            self.assertTrue(output["metric"]["normalized"])
            self.assertEqual(
                output["per_module"][0]["statistics_source"], "component_stats"
            )
            self.assertEqual(
                output["metric"]["factor_basis"], [adaptive.FACTOR_BASIS]
            )
            self.assertAlmostEqual(output["per_module"][0]["cum_c"][-1], 1.0)


class AdaptiveRankCacheTest(unittest.TestCase):
    def setUp(self):
        self.names = [
            "language_model.model.layers.0.attention.wqkv",
            "model.layers.0.mlp.up_proj",
        ]
        self.cache = {
            "linear_bit_map": {name: 2 for name in self.names},
            "linear_score_map": {
                self.names[0]: {"module_family": "attn_in"},
                self.names[1]: {"module_family": "mlp_in"},
            },
            "low_rank": [low_rank_item(name) for name in self.names],
        }
        self.analysis = normalized_analysis(self.names)
        self.rows = adaptive.analysis_rows_by_name(self.analysis)
        self.weight_numel = adaptive.infer_weight_numel_map(self.cache)

    def test_builds_projection_specific_rank_map(self):
        rank_map, rows = adaptive.build_rank_map(
            self.cache, self.rows, [1, 2, 4], tau=0.5
        )
        self.assertEqual(rank_map[self.names[0]], 1)
        self.assertEqual(rank_map[self.names[1]], 4)
        self.assertEqual(len(rows), 2)

    def test_truncation_does_not_mutate_base(self):
        rank_map = {self.names[0]: 1, self.names[1]: 2}
        output = adaptive.truncate_cache(self.cache, rank_map)
        output_ranks = {item["name"]: item["rank"] for item in output["low_rank"]}
        self.assertEqual(output_ranks, rank_map)
        self.assertEqual(tuple(output["low_rank"][0]["up"].shape), (8, 1))
        self.assertEqual(tuple(output["low_rank"][1]["down"].shape), (2, 8))
        self.assertEqual(self.cache["low_rank"][0]["rank"], 4)
        self.assertEqual(tuple(self.cache["low_rank"][0]["up"].shape), (8, 4))

    def test_actual_bit_uses_real_shapes(self):
        actual_bit, detail = adaptive.actual_bit_cost(
            self.cache,
            {self.names[0]: 1, self.names[1]: 1},
            self.weight_numel,
            factor_bit=4,
        )
        self.assertAlmostEqual(actual_bit, 3.0)
        self.assertEqual(detail["target_weight_numel"], 128)
        self.assertEqual(detail["low_rank_factor_numel"], 32)

    def test_factor_bit_mismatch_is_rejected(self):
        self.cache["low_rank"][0]["svd_quant_config"] = {
            "up": {"quant_bit": 8},
            "down": {"quant_bit": 8},
        }
        with self.assertRaises(ValueError):
            adaptive.actual_bit_cost(
                self.cache,
                {self.names[0]: 1, self.names[1]: 1},
                self.weight_numel,
                factor_bit=4,
            )

    def test_budget_mode_selects_highest_feasible_threshold_configuration(self):
        tau, rank_map, _, actual_bit, _ = adaptive.choose_tau_for_budget(
            self.cache,
            self.rows,
            [1, 2, 4],
            target_actual_bit=4.5,
            weight_numel_map=self.weight_numel,
            factor_bit=4,
        )
        self.assertGreater(tau, 0.4)
        self.assertEqual(rank_map, {self.names[0]: 1, self.names[1]: 4})
        self.assertAlmostEqual(actual_bit, 4.5)

    def test_budget_mode_never_exceeds_target_by_tolerance(self):
        with self.assertRaises(ValueError):
            adaptive.choose_tau_for_budget(
                self.cache,
                self.rows,
                [1, 2, 4],
                target_actual_bit=3.0 - 5e-8,
                weight_numel_map=self.weight_numel,
                factor_bit=4,
            )

    def test_missing_projection_analysis_is_rejected(self):
        incomplete = {self.names[0]: self.rows[self.names[0]]}
        with self.assertRaises(ValueError):
            adaptive.build_rank_map(self.cache, incomplete, [1, 2, 4], tau=0.5)

    def test_cli_writes_fixed_projection_specific_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "base.pt"
            analysis_path = tmp_path / "analysis.json"
            output_path = tmp_path / "adaptive.pt"
            torch.save(self.cache, base_path)
            self.analysis["collection"] = {"source_cache": str(base_path.resolve())}
            factors_dir = tmp_path / "factors"
            factors_dir.mkdir()
            for index, item in enumerate(self.cache["low_rank"]):
                torch.save(
                    {
                        "rank": item["rank"],
                        "up": item["up"].half(),
                        "down": item["down"].half(),
                        "quantized": False,
                    },
                    factors_dir / f"{index}.pt",
                )
            analysis_path.write_text(json.dumps(self.analysis))

            result = adaptive.main(
                [
                    "--base-cache",
                    str(base_path),
                    "--analysis",
                    str(analysis_path),
                    "--output",
                    str(output_path),
                    "--ranks",
                    "1",
                    "2",
                    "4",
                    "--tau",
                    "0.5",
                ]
            )

            self.assertEqual(result, 0)
            output = torch.load(output_path, map_location="cpu", weights_only=True)
            ranks = {item["name"]: item["rank"] for item in output["low_rank"]}
            self.assertEqual(ranks, {self.names[0]: 1, self.names[1]: 4})
            self.assertTrue(output["adaptive_rank_config"]["fixed_after_calibration"])
            summary = json.loads(
                output_path.with_suffix(".adaptive-rank.json").read_text()
            )
            self.assertTrue(summary["fixed_after_calibration"])
            self.assertEqual(summary["factor_basis"], adaptive.FACTOR_BASIS)

    def test_cli_rejects_analysis_from_different_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "base.pt"
            analysis_path = tmp_path / "analysis.json"
            torch.save(self.cache, base_path)
            self.analysis["collection"] = {
                "source_cache": str(tmp_path / "another-cache.pt")
            }
            analysis_path.write_text(json.dumps(self.analysis))
            with self.assertRaises(ValueError):
                adaptive.main(
                    [
                        "--base-cache",
                        str(base_path),
                        "--analysis",
                        str(analysis_path),
                        "--output",
                        str(tmp_path / "adaptive.pt"),
                        "--ranks",
                        "1",
                        "2",
                        "4",
                        "--tau",
                        "0.5",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
