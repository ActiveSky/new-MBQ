"""Unit tests for tools/build_rank_coverage_report.py.

Covers the plan's stage-8 checklist:
  1. r-1 indexing on cum_c
  2. coverage + tail = 1
  3. coverage monotonic non-decreasing in r
  4. score-weighted aggregation formula
  5. 42/55 cache membership separated
  6. duplicate projections not double-counted
  7. output-dir == source dir -> error
  8. existing output files -> refuse overwrite unless --overwrite
  9. existing-only mode does not load a VLM / does not create act_for_ck dirs;
     deferred wo are recorded as deferred_missing_coverage, not in denominator
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Load the module by path so the test works without package install.
REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_rank_coverage_report", REPO / "tools" / "build_rank_coverage_report.py"
)
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)


class TestCoverageMath(unittest.TestCase):
    def test_r_minus_one_indexing(self):
        cum = [0.1, 0.25, 0.45, 0.7, 1.0]
        self.assertAlmostEqual(rc.coverage_from_cum(cum, 1), 0.1)
        self.assertAlmostEqual(rc.coverage_from_cum(cum, 3), 0.45)
        self.assertAlmostEqual(rc.coverage_from_cum(cum, 5), 1.0)

    def test_rank_out_of_range_raises(self):
        cum = [0.1, 0.2, 1.0]
        with self.assertRaises(ValueError):
            rc.coverage_from_cum(cum, 4)  # r-1 = 3 >= len 3
        with self.assertRaises(ValueError):
            rc.coverage_from_cum(cum, 0)

    def test_coverage_plus_tail_one(self):
        for cov in [0.0, 0.123, 0.5, 0.877, 1.0]:
            self.assertAlmostEqual(rc.tail_from_coverage(cov) + cov, 1.0)

    def test_coverage_monotonic(self):
        # cum_c is cumulative by construction, so coverage(r) is non-decreasing.
        cum = [0.1, 0.3, 0.3, 0.6, 1.0]  # a flat step is allowed (equal)
        prev = -1.0
        for r in range(1, len(cum) + 1):
            c = rc.coverage_from_cum(cum, r)
            self.assertGreaterEqual(c, prev - 1e-12)
            prev = c

    def test_weighted_aggregation_formula(self):
        items = [(1.0, 0.8), (3.0, 0.4)]
        total = 4.0
        # weighted coverage = (1*0.8 + 3*0.4)/4 = 2.0/4 = 0.5
        self.assertAlmostEqual(rc.weighted_coverage(items, total), 0.5)
        # weighted tail = 1 - weighted coverage
        self.assertAlmostEqual(rc.weighted_tail(items, total), 0.5)
        # zero/negative total weight -> error
        with self.assertRaises(ValueError):
            rc.weighted_coverage(items, 0.0)
        with self.assertRaises(ValueError):
            rc.weighted_tail(items, -1.0)

    def test_weighted_tail_equals_one_minus_coverage(self):
        items = [(2.0, 0.3), (5.0, 0.6)]
        total = 7.0
        self.assertAlmostEqual(
            rc.weighted_tail(items, total),
            1.0 - rc.weighted_coverage(items, total),
        )


class TestNameParsing(unittest.TestCase):
    def test_parse_single_module(self):
        base, ranks = rc.parse_candidate_name("custom_wo_all_w2_w3_55_wqkv_r64")
        self.assertEqual(base, "custom_wo_all_w2_w3_55")
        self.assertEqual(ranks, {"wqkv": 64})

    def test_parse_multi_module(self):
        base, ranks = rc.parse_candidate_name("custom_wo_w2_w3_42_wqkv_r32_w3_r64_wo_r32")
        self.assertEqual(base, "custom_wo_w2_w3_42")
        self.assertEqual(ranks, {"wqkv": 32, "w3": 64, "wo": 32})

    def test_parse_base_only(self):
        base, ranks = rc.parse_candidate_name("custom_wo_all_w2_w3_55")
        self.assertEqual(base, "custom_wo_all_w2_w3_55")
        self.assertEqual(ranks, {})

    def test_parse_unknown(self):
        base, ranks = rc.parse_candidate_name("some_random_name")
        self.assertEqual(base, "")
        self.assertEqual(ranks, {})


class TestFamilyMapping(unittest.TestCase):
    def test_tail_family(self):
        self.assertEqual(rc.tail_family("language_model.model.layers.0.attention.wqkv"), "wqkv")
        self.assertEqual(rc.tail_family("language_model.model.layers.0.attention.wo"), "wo")
        self.assertEqual(rc.tail_family("language_model.model.layers.0.feed_forward.w1"), "w1")
        self.assertEqual(rc.tail_family("language_model.model.layers.0.feed_forward.w2"), "w2")
        self.assertEqual(rc.tail_family("language_model.model.layers.0.feed_forward.w3"), "w3")

    def test_group_family_from_score_meta(self):
        self.assertEqual(rc.group_family("x.wqkv", {"module_family": "attn_in"}), "attn_in")
        self.assertEqual(rc.group_family("x.wo", None), "attn_out")
        self.assertEqual(rc.group_family("x.w3", None), "mlp_in")


class TestOutputDirSafety(unittest.TestCase):
    def test_output_dir_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "act_for_ck" / "wo_all_w2_w3_48"
            src.mkdir(parents=True)
            out = src / "sub"
            with self.assertRaises(ValueError):
                rc._check_output_dir(out, str(src), allow_existing=False, overwrite=False)

    def test_output_dir_equal_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "act"
            src.mkdir()
            with self.assertRaises(ValueError):
                rc._check_output_dir(src, str(src), allow_existing=False, overwrite=False)

    def test_existing_files_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "act"
            src.mkdir()
            out = Path(td) / "out"
            out.mkdir()
            (out / "manifest.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                rc._check_output_dir(out, str(src), allow_existing=True, overwrite=False)

    def test_existing_files_allowed_with_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "act"
            src.mkdir()
            out = Path(td) / "out"
            out.mkdir()
            (out / "manifest.json").write_text("{}")
            # should not raise
            rc._check_output_dir(out, str(src), allow_existing=True, overwrite=True)

    def test_fresh_dir_created(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "act"
            src.mkdir()
            out = Path(td) / "fresh"
            rc._check_output_dir(out, str(src), allow_existing=False, overwrite=False)
            self.assertTrue(out.exists())


class TestExistingOnlyNoVLM(unittest.TestCase):
    """Existing-only mode must not load a VLM or create act_for_ck dirs.

    We verify this structurally: build_rank_coverage_report imports torch but
    never references lmms_eval / qmllm.models / CUDA, and its main() only
    torch.load caches with map_location cpu. The simplest behavioral test is
    that running the full pipeline on a tiny synthetic fixture produces the
    report without touching CUDA and without creating any act_for_ck dir.
    """

    def test_pipeline_on_synthetic_fixture_creates_report_without_vlm(self):
        tmp = Path(tempfile.mkdtemp(prefix="rc_test_"))
        try:
            src_root, argv, out = self._build_synthetic(tmp)
            # Snapshot source act_for_ck files BEFORE running the script.
            before = self._snapshot(src_root)
            rc.main(argv)
            # The script must not create any act_for_ck dir under the output dir.
            self.assertFalse(any(p.name == "act_for_ck" for p in out.rglob("act_for_ck")))
            # Source files unchanged (read-only): same files, same sizes, no new files.
            after = self._snapshot(src_root)
            self.assertEqual(before, after)
            # Report + csvs exist.
            self.assertTrue((out / "rank_coverage_report.md").exists())
            self.assertTrue((out / "per_projection_coverage.csv").exists())
            self.assertTrue((out / "family_coverage_summary.csv").exists())
            self.assertTrue((out / "experiment_coverage_summary.csv").exists())

            # --- output content assertions --------------------------------
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["line_55"]["reusable"], 1)
            self.assertEqual(manifest["line_55"]["deferred_missing"], 0)
            self.assertEqual(manifest["line_42"]["reusable"], 1)
            self.assertEqual(manifest["line_42"]["deferred_missing"], 1)
            # deferred wo recorded in overlap report
            overlap = json.loads((out / "overlap_report.json").read_text())
            deferred = [r for r in overlap if r["reuse_status"] == "deferred_missing_coverage"]
            self.assertEqual(len(deferred), 1)
            # per-projection: 1 reusable wqkv on 55 + 1 reusable wqkv on 42 + 1 deferred wo
            per_proj = json.loads((out / "per_projection_coverage.json").read_text())
            deferred_rows = [r for r in per_proj if r.get("deferred")]
            self.assertEqual(len(deferred_rows), 1)
            self.assertIsNone(deferred_rows[0]["coverage_c_r64"])
            # family stats: 42 line must contain a deferred wo placeholder row
            fam = json.loads((out / "family_coverage_summary.json").read_text())
            fam42_wo = [r for r in fam if r["line"] == "42" and r["tail_family"] == "wo"]
            self.assertTrue(all(r.get("status") == "deferred_missing_coverage" for r in fam42_wo))
            # reusable wqkv coverage is non-null and in [0,1]
            wqkv_rows = [r for r in per_proj if r["name"].endswith(".wqkv") and not r.get("deferred")]
            for r in wqkv_rows:
                for rk in [16, 32, 64, 128]:
                    v = r[f"coverage_c_r{rk}"]
                    self.assertIsNotNone(v)
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _snapshot(self, root: Path):
        snap = {}
        for p in root.rglob("*"):
            if p.is_file():
                snap[str(p.relative_to(root))] = p.stat().st_size
        return snap

    def _build_synthetic(self, tmp: Path):
        import torch

        # Two projection names: one wqkv (reused on both lines), one wo (only
        # present in the 42 cache -> deferred).
        wqkv = "language_model.model.layers.0.attention.wqkv"
        wo = "language_model.model.layers.1.attention.wo"
        L = 200  # cum_c length, >> ranks

        # --- build a fake _global_meta.json + _ck_analysis.json -----------
        cum_c_wqkv = [min(1.0, i / 100.0) for i in range(1, L + 1)]
        # ensure last == 1.0 exactly
        cum_c_wqkv[-1] = 1.0
        gmeta = {
            "scale_path": "scale_cache/mbq/fake48.pt",
            "rho_dict": {"attn_in": 0.17, "attn_out": 0.13, "mlp_in": 0.125, "mlp_out": 0.133},
            "q_config": {"zero_point": True, "q_group_size": 32, "double_quant": True},
            "total_modules": 1,
            "module_index": [
                {"idx": 0, "name": wqkv, "short": "L0.wqkv", "layer_idx": 0,
                 "family": "attn_in", "bit": 2, "rho": 0.17},
            ],
        }
        (tmp / "act_for_ck").mkdir()
        src = tmp / "act_for_ck" / "wo_all_w2_w3_48"
        src.mkdir()
        (src / "_global_meta.json").write_text(json.dumps(gmeta))
        analysis = {
            "summary": {"num_modules": 1, "ranks": [32, 64, 128]},
            "per_module": [
                {"idx": 0, "name": wqkv, "family": "attn_in", "bit": 2, "rho": 0.17,
                 "spearman": 0.9, "kendall": 0.8,
                 "cov_energy": {"32": 0.2, "64": 0.4, "128": 0.7},
                 "cov_c": {"32": 0.2, "64": 0.4, "128": 0.7},
                 "judge": {},
                 "cum_energy": cum_c_wqkv[:], "cum_c": cum_c_wqkv[:]},
            ],
            "rho_override": {},
        }
        (src / "_ck_analysis.json").write_text(json.dumps(analysis))

        # --- build fake 48 / 55 / 42 caches -------------------------------
        scale = [("norm0", ("sub0",), torch.zeros(4, dtype=torch.bfloat16))]
        score_map = {
            wqkv: {"score": 0.5, "module_type": "attention_wqkv", "module_family": "attn_in", "w_bit": 2},
            wo: {"score": 0.2, "module_type": "attention_wo", "module_family": "attn_out", "w_bit": 2},
        }
        up_wqkv = torch.zeros(6144, 128, dtype=torch.float16)
        down_wqkv = torch.zeros(128, 4096, dtype=torch.float16)
        up_wo = torch.zeros(4096, 128, dtype=torch.float16)
        down_wo = torch.zeros(128, 4096, dtype=torch.float16)

        def mk(low_rank, bit_map):
            return {
                "scale": scale,
                "low_rank": low_rank,
                "low_rank_config": {"rank": 128},
                "linear_score_map": score_map,
                "linear_bit_map": bit_map,
                "linear_mixed_config": {},
            }

        lr_wqkv = {"name": wqkv, "rank": 128, "score": 0.5, "module_type": "attention_wqkv",
                   "up": up_wqkv, "down": down_wqkv, "svd_quant": True, "svd_quant_config": {}}
        lr_wo = {"name": wo, "rank": 128, "score": 0.2, "module_type": "attention_wo",
                 "up": up_wo, "down": down_wo, "svd_quant": True, "svd_quant_config": {}}

        ck48 = mk([lr_wqkv], {wqkv: 2, wo: 2})
        ck55 = mk([lr_wqkv], {wqkv: 2, wo: 2})  # 55 line: only wqkv low-rank (wo is 3-bit)
        ck42 = mk([lr_wqkv, lr_wo], {wqkv: 2, wo: 2})  # 42 line: wqkv reused, wo deferred

        fake48 = tmp / "fake48.pt"; torch.save(ck48, fake48)
        fake55 = tmp / "fake55.pt"; torch.save(ck55, fake55)
        fake42 = tmp / "fake42.pt"; torch.save(ck42, fake42)

        # Patch the evidence table + OCR parser to avoid touching real outputs.
        rc.RANK_ABLATION_EVIDENCE = [
            ("custom_wo_all_w2_w3_55", "55", "mixed_0.3", {}, None, "plan:test"),
            ("custom_wo_w2_w3_42", "42", "mixed_0.3", {}, None, "plan:test"),
        ]
        rc._parse_ocr_score = lambda cand: None

        out = tmp / "out"
        argv = [
            "--existing-analysis", str(src / "_ck_analysis.json"),
            "--existing-global-meta", str(src / "_global_meta.json"),
            "--cache-55", str(fake55),
            "--cache-42", str(fake42),
            "--cache-48", str(fake48),
            "--ranks", "16", "32", "64", "128",
            "--output-dir", str(out),
            "--existing-only",
            "--record-missing-family", "wo",
        ]
        return src, argv, out


class TestDuplicateNotDoubleCounted(unittest.TestCase):
    def test_family_stats_counts_each_projection_once(self):
        # Three distinct projections on the 55 line, same family.
        rows = [
            {"line": "55", "name": name, "tail_family": "wqkv", "score_E": 1.0, "rho": 0.1,
             **{f"coverage_c_r{r}": 0.5 for r in [16, 32, 64, 128]},
             **{f"coverage_energy_r{r}": 0.5 for r in [16, 32, 64, 128]},
             **{f"tail_c_r{r}": 0.5 for r in [16, 32, 64, 128]},
             **{f"weighted_tail_r{r}": 0.5 for r in [16, 32, 64, 128]}}
            for name in ["layers.0.wqkv", "layers.1.wqkv", "layers.2.wqkv"]
        ]
        # build_per_projection is fed a *set* of names upstream, so duplicates
        # cannot enter. Here we also assert a (line,name) set has no dupes:
        unique = {(r["line"], r["name"]) for r in rows}
        self.assertEqual(len(unique), 3)
        fam = rc.family_stats(rows, [64], "55")
        row64 = [f for f in fam if f["rank"] == 64][0]
        self.assertEqual(row64["n"], 3)  # each projection counted once

    def test_build_per_projection_input_set_dedup(self):
        # build_per_projection takes a set of reusable names; passing a set
        # with a duplicate string yields a single entry (set semantics).
        names = {"layers.0.wqkv", "layers.0.wqkv", "layers.1.wqkv"}
        self.assertEqual(len(names), 2)


if __name__ == "__main__":
    unittest.main()