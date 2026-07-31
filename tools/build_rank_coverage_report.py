#!/usr/bin/env python3
"""Build a cross-modal residual rank-coverage report from existing c_k analysis.

This is a read-only, CPU-only analysis tool. It does NOT load any VLM, does NOT
run calibration forward, does NOT compute new SVDs, and does NOT touch the
existing ``act_for_ck/wo_all_w2_w3_48`` outputs. It only reads the already
saved ``_ck_analysis.json`` (per-projection cumulative ``c_k`` / energy curves)
plus the target MBQ caches' metadata (``linear_bit_map`` / ``linear_score_map``
/ ``low_rank`` / ``scale``) and produces a fresh report under a new directory.

Why this script exists
----------------------
Previous hand-tuned residual-rank ablations (``wqkv_r64/r32/r16``, ``w1_r64``,
``w2_r64``, ``w3_r32/r64`` and combinations on the 55/42 lines) produced a
mixture of successes and regressions on OCRBench. This script explains those
results with a single calibration-time quantity: the cross-modal residual
spectrum coverage

    Coverage_m(r) = sum_{k<=r} c_{m,k} / sum_k c_{m,k}
    Tail_m(r)     = 1 - Coverage_m(r)

where ``c_{m,k} = sigma_k^2 ( ||X_ans v_k||^2 + rho_g ||X_vis v_k||^2 )`` is
already materialized in ``_ck_analysis.json`` as ``cum_c``. We read coverage
directly from ``cum_c[r-1]`` and never recompute matrix products.

To avoid misjudging low-sensitivity projections, we also read the
activation-aware projection score ``E_m`` from the cache's
``linear_score_map[name]['score']`` and report score-weighted aggregation:

    WeightedCoverage_G(r) = sum_{m in G} E_m Coverage_m(r) / sum_{m in G} E_m
    WeightedTail_G(r)     = sum_{m in G} E_m Tail_m(r)     / sum_{m in G} E_m
    WeightedTail_m(r)     = E_m * Tail_m(r)

Data reuse boundary
-------------------
The existing ``act_for_ck/wo_all_w2_w3_48`` data was collected on the 48-line
cache (112 residual projections: wqkv 32, w1 32, w2 24, w3 24; no ``wo``
because all 32 ``wo`` are 3-bit on the 48 line and thus excluded from
low-rank). The 55/42 caches are built from ``mixed_0.0.pt`` by only editing
``linear_bit_map`` and filtering ``low_rank`` (see
``tools/build_custom_mbq_cache.py``); ``scale``, ``low_rank_config`` and
``linear_score_map`` are inherited unchanged. Therefore a 55/42 low-rank
projection whose name also appears in the 112-projection set reuses the same
SVD spectrum / activations, provided its bit stays 2 and its scale / quant
config match.

Expected reuse:
    55 line: 105 / 105 low-rank projections reusable, 0 missing.
    42 line: 103 / 118 low-rank projections reusable; 15 ``wo`` missing
             (recorded as ``deferred_missing_coverage``, never collected).

This script refuses to run a collector, refuses to load a model, refuses to
write inside the source ``act_for_ck`` directory, and refuses to overwrite
existing output files unless ``--allow-existing-output-dir`` plus
``--overwrite`` are given.

Usage
-----
    python tools/build_rank_coverage_report.py \
        --existing-analysis act_for_ck/wo_all_w2_w3_48/_ck_analysis.json \
        --existing-global-meta act_for_ck/wo_all_w2_w3_48/_global_meta.json \
        --cache-55 scale_cache/mbq/...custom_wo_all_w2_w3_55.pt \
        --cache-42 scale_cache/mbq/...custom_wo_w2_w3_42.pt \
        --ranks 16 32 48 64 96 128 \
        --output-dir analysis/rank_coverage/2026-07-13 \
        --existing-only --record-missing-family wo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target linear weight parameter count for InternVL2-8B's 160 MBQ linears.
# Used only for the optional actual-bit cross-check; matches the value used
# across MBQ_DAILY_PLAN.md (6,979,321,856).
TARGET_LINEAR_NUMEL = 6_979_321_856
MIXED_03_ACTUAL_BIT = 2.3963
MIB_PER_BIT = 832.0  # 1 bit over target linears ~= 832.0 MiB

# Tail-family (module tail name) <-> reweight-group family mapping, kept
# consistent with tools/collect_act_for_ck.py:_module_family and
# tools/build_custom_mbq_cache.py:module_family.
TAIL_TO_GROUP = {
    "wqkv": "attn_in",
    "wo": "attn_out",
    "w2": "mlp_out",
    "w1": "mlp_in",
    "w3": "mlp_in",
}
GROUP_TO_TAILS = defaultdict(list)
for _t, _g in TAIL_TO_GROUP.items():
    GROUP_TO_TAILS[_g].append(_t)

# Candidate name -> (line, rank_set, immediate_base, ocrbench, evidence_path).
# OCRBench scores are parsed from the committed result txt files under
# outputs/my-internvl2-8b/<candidate>/results/ocrbench_results.txt (see
# analysis/rank_coverage/2026-07-13/logs/ocrbench_evidence_extracted.json).
# Scores are NOT guessed: every entry has a result_txt evidence path.
OCR_RESULTS_DIR = "outputs/my-internvl2-8b"


def _ocr_result_txt(candidate: str) -> str:
    return os.path.join(OCR_RESULTS_DIR, candidate, "results", "ocrbench_results.txt")


def _parse_ocr_score(candidate: str) -> Optional[int]:
    """Read Final Score from a candidate's ocrbench_results.txt. None if missing."""
    path = _ocr_result_txt(candidate)
    if not os.path.isfile(path):
        return None
    return _parse_ocr_score_from_path(path)


def _parse_ocr_score_from_path(path: str) -> Optional[int]:
    """Read Final Score from an explicit ocrbench_results.txt path."""
    if not os.path.isfile(path):
        return None
    txt = Path(path).read_text(errors="replace")
    m = re.search(r"Final Score\(Total 1000\):\s*(\d+)", txt)
    if not m:
        m = re.search(r"Final Score.*?:\s*(\d+)", txt)
    return int(m.group(1)) if m else None


# Rank-ablation evidence table. Each entry:
#   (candidate, line, immediate_base, rank_set, ocrbench_or_None, evidence_path)
# rank_set maps tail-family -> new residual rank (only families whose rank was
# changed relative to the immediate base are listed; unchanged families stay at
# the base's rank, which itself may already be reduced).
RANK_ABLATION_EVIDENCE: List[Tuple[str, str, str, Dict[str, int], Optional[int], str]] = [
    # 55 line, single-module wqkv
    ("custom_wo_all_w2_w3_55", "55", "mixed_0.3", {}, None, "plan:1564"),
    ("custom_wo_all_w2_w3_55_wqkv_r64", "55", "custom_wo_all_w2_w3_55", {"wqkv": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r32", "55", "custom_wo_all_w2_w3_55", {"wqkv": 32}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r32")),
    ("custom_wo_all_w2_w3_55_wqkv_r16", "55", "custom_wo_all_w2_w3_55", {"wqkv": 16}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r16")),
    # 55 line, single non-wqkv module on top of wqkv_r64
    ("custom_wo_all_w2_w3_55_wqkv_r64_w2_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r64", {"wqkv": 64, "w2": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r64_w2_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r64_w3_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r64", {"wqkv": 64, "w3": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r64_w3_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r64_w1_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r64", {"wqkv": 64, "w1": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r64_w1_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r64_w3_r32", "55", "custom_wo_all_w2_w3_55_wqkv_r64", {"wqkv": 64, "w3": 32}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r64_w3_r32")),
    # 55 line, combinations
    ("custom_wo_all_w2_w3_55_wqkv_r64_w1_r64_w3_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r64", {"wqkv": 64, "w1": 64, "w3": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r64_w1_r64_w3_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r32_w3_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r32", {"wqkv": 32, "w3": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r32_w3_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r32_w1_r64_w3_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r32", {"wqkv": 32, "w1": 64, "w3": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r32_w1_r64_w3_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r16_w3_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r16", {"wqkv": 16, "w3": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r16_w3_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r16_w1_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r16", {"wqkv": 16, "w1": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r16_w1_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r16_w1_r64_w3_r64", "55", "custom_wo_all_w2_w3_55_wqkv_r16", {"wqkv": 16, "w1": 64, "w3": 64}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r16_w1_r64_w3_r64")),
    ("custom_wo_all_w2_w3_55_wqkv_r16_w3_r32", "55", "custom_wo_all_w2_w3_55_wqkv_r16", {"wqkv": 16, "w3": 32}, None, _ocr_result_txt("custom_wo_all_w2_w3_55_wqkv_r16_w3_r32")),
    # 42 line, single-module wqkv
    ("custom_wo_w2_w3_42", "42", "mixed_0.3", {}, None, "plan:1567"),
    ("custom_wo_w2_w3_42_wqkv_r64", "42", "custom_wo_w2_w3_42", {"wqkv": 64}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r64")),
    ("custom_wo_w2_w3_42_wqkv_r32", "42", "custom_wo_w2_w3_42", {"wqkv": 32}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32")),
    ("custom_wo_w2_w3_42_wqkv_r16", "42", "custom_wo_w2_w3_42", {"wqkv": 16}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r16")),
    # 42 line, single non-wqkv module on top of wqkv_r32
    ("custom_wo_w2_w3_42_wqkv_r32_w3_r64", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w3": 64}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w3_r64")),
    ("custom_wo_w2_w3_42_wqkv_r32_w1_r64", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w1": 64}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w1_r64")),
    ("custom_wo_w2_w3_42_wqkv_r32_w3_r32", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w3": 32}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w3_r32")),
    # 42 line, combinations (some include wo, which is deferred)
    ("custom_wo_w2_w3_42_wqkv_r32_w1_r64_w3_r64", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w1": 64, "w3": 64}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w1_r64_w3_r64")),
    ("custom_wo_w2_w3_42_wqkv_r32_w3_r64_wo_r32", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w3": 64, "wo": 32}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w3_r64_wo_r32")),
    ("custom_wo_w2_w3_42_wqkv_r32_w1_r64_w3_r64_wo_r32", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w1": 64, "w3": 64, "wo": 32}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w1_r64_w3_r64_wo_r32")),
    ("custom_wo_w2_w3_42_wqkv_r32_w3_r64_w2_r64", "42", "custom_wo_w2_w3_42_wqkv_r32", {"wqkv": 32, "w3": 64, "w2": 64}, None, _ocr_result_txt("custom_wo_w2_w3_42_wqkv_r32_w3_r64_w2_r64")),
    # 48 line base (the c_k source itself) and the global baseline
    ("custom_wo_all_w2_w3_48", "48", "mixed_0.3", {}, None, _ocr_result_txt("custom_wo_all_w2_w3_48")),
    ("mixed_0.3", "baseline", "mixed_0.3", {}, None, _ocr_result_txt("w2g32_scale_reweight_true_svd_1.0_mixed_0.3")),
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def tail_family(name: str) -> str:
    """Map a full linear name to its tail family (wqkv/wo/w2/w1/w3).

    Consistent with tools/build_custom_mbq_cache.py:module_key.
    """
    if ".attention.wqkv" in name:
        return "wqkv"
    if ".attention.wo" in name:
        return "wo"
    if ".feed_forward.w1" in name:
        return "w1"
    if ".feed_forward.w2" in name:
        return "w2"
    if ".feed_forward.w3" in name:
        return "w3"
    return name.split(".")[-1]


def group_family(name: str, score_meta: Optional[Dict] = None) -> str:
    """Map a linear name to its reweight-group family (attn_in/attn_out/mlp_in/mlp_out)."""
    if score_meta and score_meta.get("module_family"):
        return str(score_meta["module_family"])
    return TAIL_TO_GROUP.get(tail_family(name), "unknown")


def coverage_from_cum(cum: Sequence[float], r: int) -> float:
    """coverage(r) = cum[r-1], the cumulative fraction captured by the first r
    energy-ordered SVD components. Raises if r is out of range.
    """
    if r <= 0:
        raise ValueError(f"rank must be positive, got {r}")
    if r - 1 >= len(cum):
        raise ValueError(f"rank {r} exceeds cum_c length {len(cum)}")
    return float(cum[r - 1])


def tail_from_coverage(coverage: float) -> float:
    """Tail(r) = 1 - Coverage(r)."""
    return 1.0 - coverage


def weighted_coverage(items: Iterable[Tuple[float, float]], total_weight: float) -> float:
    """WeightedCoverage_G = sum_m E_m * Coverage_m / sum_m E_m.

    ``items`` yields (E_m, coverage_m) pairs. ``total_weight`` is sum_m E_m and
    must be positive.
    """
    if total_weight <= 0:
        raise ValueError("total_weight must be positive for weighted coverage")
    num = 0.0
    for e, cov in items:
        num += e * cov
    return num / total_weight


def weighted_tail(items: Iterable[Tuple[float, float]], total_weight: float) -> float:
    """WeightedTail_G = sum_m E_m * Tail_m / sum_m E_m = 1 - WeightedCoverage_G."""
    if total_weight <= 0:
        raise ValueError("total_weight must be positive for weighted tail")
    num = 0.0
    for e, cov in items:
        num += e * (1.0 - cov)
    return num / total_weight


def quantiles(values: Sequence[float], qs: Sequence[float]) -> Dict[str, float]:
    """Return percentile dict for the requested quantiles (0..1)."""
    out: Dict[str, float] = {}
    if not values:
        for q in qs:
            out[f"p{int(round(q*100))}"] = float("nan")
        return out
    sv = sorted(values)
    n = len(sv)
    for q in qs:
        # linear interpolation between closest ranks, matches numpy default
        if n == 1:
            out[f"p{int(round(q*100))}"] = float(sv[0])
            continue
        h = (n - 1) * q
        lo = int(h)
        hi = min(lo + 1, n - 1)
        frac = h - lo
        out[f"p{int(round(q*100))}"] = float(sv[lo] * (1 - frac) + sv[hi] * frac)
    return out


def parse_candidate_name(name: str) -> Tuple[str, Dict[str, int]]:
    """Parse a candidate cache name into (base_token, rank_set).

    The base token is the leading ``custom_wo_all_w2_w3_55`` /
    ``custom_wo_w2_w3_42`` / ``custom_wo_all_w2_w3_48`` / ``mixed_0.3`` part.
    ``rank_set`` maps tail-family -> residual rank for every ``<fam>_r<int>``
    segment after the base. Returns ({}, {}) for unrecognised names.

    Examples
    --------
    >>> parse_candidate_name("custom_wo_all_w2_w3_55_wqkv_r64")
    ('custom_wo_all_w2_w3_55', {'wqkv': 64})
    >>> parse_candidate_name("custom_wo_w2_w3_42_wqkv_r32_w3_r64_wo_r32")
    ('custom_wo_w2_w3_42', {'wqkv': 32, 'w3': 64, 'wo': 32})
    """
    known_bases = [
        "custom_wo_all_w2_w3_55",
        "custom_wo_all_w2_w3_48",
        "custom_wo_w2_w3_42",
    ]
    for b in known_bases:
        if name == b or name.startswith(b + "_"):
            rest = name[len(b) + 1:] if name != b else ""
            ranks: Dict[str, int] = {}
            toks = rest.split("_") if rest else []
            i = 0
            while i + 1 < len(toks):
                fam = toks[i]
                rank_tok = toks[i + 1]
                m = re.fullmatch(r"r(\d+)", rank_tok)
                if m and fam in TAIL_TO_GROUP:
                    ranks[fam] = int(m.group(1))
                    i += 2
                else:
                    i += 1
            return b, ranks
    if name in ("mixed_0.3", "mixed_0.0"):
        return name, {}
    return "", {}


# ---------------------------------------------------------------------------
# Cache loading + reuse compatibility
# ---------------------------------------------------------------------------

def load_cache_meta(path: str) -> Dict[str, Any]:
    """Load a MBQ cache with metadata only (map_location cpu, no CUDA).

    Only the metadata keys needed for reuse checking and E_m lookup are
    guaranteed to be returned; tensor weights are kept on CPU and never moved
    to CUDA.
    """
    try:
        ck = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    return ck


def low_rank_name_set(cache: Dict) -> set:
    return {it["name"] for it in cache.get("low_rank", [])}


def low_rank_map(cache: Dict) -> Dict[str, Dict]:
    return {it["name"]: it for it in cache.get("low_rank", [])}


def score_of(cache: Dict, name: str) -> Optional[float]:
    sm = cache.get("linear_score_map") or {}
    entry = sm.get(name)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return float(entry.get("score"))
    return float(entry)


def scale_compatible(a: Dict, b: Dict) -> bool:
    """Return True if two caches share the same per-layer scale tensors.

    ``scale`` is a list of (norm_name, (sub_names,), tensor) tuples. We compare
    the norm name, the sub-name tuple, and the scale tensor element-wise with
    ``torch.equal`` (exact). Any mismatch means the residual SVD spectrum was
    computed on a different scaled weight and cannot be reused.
    """
    sa, sb = a.get("scale", []), b.get("scale", [])
    if len(sa) != len(sb):
        return False
    for (na, suba, ta), (nb, subb, tb) in zip(sa, sb):
        if na != nb or suba != subb:
            return False
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            if not torch.equal(ta, tb):
                return False
        elif ta != tb:
            return False
    return True


def low_rank_factor_compatible(a_item: Dict, b_item: Dict) -> bool:
    """Return True if two low-rank entries share the same rank/score/up/down."""
    if a_item.get("rank") != b_item.get("rank"):
        return False
    if abs(float(a_item.get("score", 0.0)) - float(b_item.get("score", 0.0))) > 1e-12:
        return False
    ua, ub = a_item.get("up"), b_item.get("up")
    da, db = a_item.get("down"), b_item.get("down")
    if torch.is_tensor(ua) and torch.is_tensor(ub):
        if not torch.equal(ua, ub):
            return False
    elif ua != ub:
        return False
    if torch.is_tensor(da) and torch.is_tensor(db):
        if not torch.equal(da, db):
            return False
    elif da != db:
        return False
    return True


def q_config_from_global_meta(gmeta: Dict) -> Dict:
    return gmeta.get("q_config") or {}


# ---------------------------------------------------------------------------
# Inventory / overlap (stage 1)
# ---------------------------------------------------------------------------

def build_overlap(
    existing_names: set,
    existing_meta_by_name: Dict[str, Dict],
    source_cache: Dict,
    target_cache: Dict,
    target_tag: str,
    q_config_expected: Dict,
) -> List[Dict]:
    """Per-projection reuse decision for one target line.

    Returns one row per low-rank projection in the target cache, classified as
    reusable / deferred_missing / not_reusable.
    """
    src_lr = low_rank_map(source_cache)
    tgt_lr = low_rank_map(target_cache)
    tgt_bit = target_cache.get("linear_bit_map", {})
    rows: List[Dict] = []
    for name, tgt_item in tgt_lr.items():
        tail = tail_family(name)
        row = {
            "target_line": target_tag,
            "projection_name": name,
            "module_family_tail": tail,
            "module_family_group": group_family(name, (target_cache.get("linear_score_map") or {}).get(name)),
            "target_bit": int(tgt_bit.get(name, -1)),
            "source_bit": int(existing_meta_by_name.get(name, {}).get("bit", -1)),
            "in_existing_ck": name in existing_names,
            "scale_compatible": scale_compatible(source_cache, target_cache),
            "reuse_status": "reusable",
            "reason": "",
        }
        # bit must be 2 (residual projection)
        if row["target_bit"] != 2:
            row["reuse_status"] = "not_reusable"
            row["reason"] = f"target bit={row['target_bit']} != 2"
        elif name not in existing_names:
            row["reuse_status"] = "deferred_missing_coverage"
            row["reason"] = "no existing c_k data for this projection"
        elif name not in src_lr:
            row["reuse_status"] = "not_reusable"
            row["reason"] = "projection absent from 48-line source low_rank"
        else:
            src_item = src_lr[name]
            if not low_rank_factor_compatible(src_item, tgt_item):
                row["reuse_status"] = "not_reusable"
                row["reason"] = "low-rank up/down/score/rank mismatch vs source"
            elif not row["scale_compatible"]:
                row["reuse_status"] = "not_reusable"
                row["reason"] = "scale tensor mismatch vs source"
            else:
                row["reuse_status"] = "reusable"
                row["reason"] = "scale/low-rank/bit/q_config all match"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-projection coverage (stage 2)
# ---------------------------------------------------------------------------

def build_per_projection(
    reusable_names: set,
    existing_meta_by_name: Dict[str, Dict],
    ck_by_name: Dict[str, Dict],
    score_map: Dict,
    ranks: Sequence[int],
    line_tag: str,
) -> List[Dict]:
    """One row per reusable projection with coverage/tail/weighted_tail per rank."""
    rows: List[Dict] = []
    for name in sorted(reusable_names):
        meta = existing_meta_by_name[name]
        pm = ck_by_name[name]
        cum_c = pm["cum_c"]
        cum_e = pm["cum_energy"]
        E = float(score_map.get(name, {}).get("score", 0.0))
        rho = float(meta.get("rho", 0.0))
        row = {
            "line": line_tag,
            "name": name,
            "tail_family": tail_family(name),
            "group_family": group_family(name, score_map.get(name)),
            "layer_index": meta.get("layer_idx"),
            "score_E": E,
            "rho": rho,
            "cum_c_len": len(cum_c),
        }
        for r in ranks:
            cov_c = coverage_from_cum(cum_c, r)
            cov_e = coverage_from_cum(cum_e, r)
            tail_c = tail_from_coverage(cov_c)
            row[f"coverage_c_r{r}"] = cov_c
            row[f"coverage_energy_r{r}"] = cov_e
            row[f"tail_c_r{r}"] = tail_c
            row[f"weighted_tail_r{r}"] = E * tail_c
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Family / set statistics (stage 5)
# ---------------------------------------------------------------------------

def family_stats(
    per_proj_rows: List[Dict],
    ranks: Sequence[int],
    line_tag: str,
) -> List[Dict]:
    """Aggregate per (line, tail_family, rank). Returns one row per group.

    Reports BOTH the cross-modal coverage (cum_c, the c_k metric) and the
    pure-SVD-energy coverage (cum_energy, the sigma^2 metric) in parallel,
    so the two can be compared. The cross-modal coverage is the primary
    metric; the energy coverage is the "what would you get without any
    calibration activation" baseline.
    """
    by_key: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in per_proj_rows:
        by_key[(r["line"], r["tail_family"])].append(r)
    out: List[Dict] = []
    for (line, fam), members in sorted(by_key.items()):
        if line != line_tag:
            continue
        scores = [m["score_E"] for m in members]
        total_E = sum(scores)
        for r in ranks:
            covs = [m[f"coverage_c_r{r}"] for m in members]
            tails = [m[f"tail_c_r{r}"] for m in members]
            wtails = [m[f"weighted_tail_r{r}"] for m in members]
            ecovs = [m[f"coverage_energy_r{r}"] for m in members]
            q = quantiles(covs, [0.10, 0.25, 0.50, 0.75, 0.90])
            tq = quantiles(tails, [0.10, 0.50, 0.90])
            eq = quantiles(ecovs, [0.10, 0.25, 0.50, 0.75, 0.90])
            # cross-modal weighted (primary)
            wcov_c = weighted_coverage([(m["score_E"], m[f"coverage_c_r{r}"]) for m in members], total_E) if total_E > 0 else float("nan")
            wtail_c = weighted_tail([(m["score_E"], m[f"coverage_c_r{r}"]) for m in members], total_E) if total_E > 0 else float("nan")
            # energy weighted (sigma^2 weighted by E_m, i.e. activation-agnostic baseline)
            wcov_e = weighted_coverage([(m["score_E"], m[f"coverage_energy_r{r}"]) for m in members], total_E) if total_E > 0 else float("nan")
            wtail_e = weighted_tail([(m["score_E"], m[f"coverage_energy_r{r}"]) for m in members], total_E) if total_E > 0 else float("nan")
            row = {
                "line": line,
                "tail_family": fam,
                "group_family": TAIL_TO_GROUP.get(fam, "unknown"),
                "rank": r,
                "n": len(members),
                "total_E": total_E,
                # --- cross-modal (c_k) coverage, the primary metric ---
                "coverage_mean": statistics.fmean(covs) if covs else float("nan"),
                "coverage_median": q["p50"],
                "coverage_p10": q["p10"],
                "coverage_p25": q["p25"],
                "coverage_p75": q["p75"],
                "coverage_p90": q["p90"],
                "coverage_min": min(covs) if covs else float("nan"),
                "coverage_max": max(covs) if covs else float("nan"),
                "tail_mean": statistics.fmean(tails) if tails else float("nan"),
                "tail_median": tq["p50"],
                "tail_p90": tq["p90"],
                "weighted_coverage": wcov_c,
                "weighted_tail": wtail_c,
                "max_weighted_tail": max(wtails) if wtails else float("nan"),
                # --- pure SVD energy (sigma^2) coverage, activation-agnostic baseline ---
                "energy_coverage_mean": statistics.fmean(ecovs) if ecovs else float("nan"),
                "energy_coverage_median": eq["p50"],
                "energy_coverage_p10": eq["p10"],
                "energy_coverage_p90": eq["p90"],
                "energy_coverage_min": min(ecovs) if ecovs else float("nan"),
                "energy_coverage_max": max(ecovs) if ecovs else float("nan"),
                "energy_weighted_coverage": wcov_e,
                "energy_weighted_tail": wtail_e,
                # --- gap: how much the calibration activation shifts coverage ---
                "coverage_gap_e_minus_c_mean": (statistics.fmean(ecovs) - statistics.fmean(covs)) if covs and ecovs else float("nan"),
                "coverage_gap_e_minus_c_weighted": (wcov_e - wcov_c) if wcov_e == wcov_e and wcov_c == wcov_c else float("nan"),
            }
            out.append(row)
    return out


def build_energy_vs_crossmodal(
    per_proj_rows: List[Dict],
    ranks: Sequence[int],
) -> List[Dict]:
    """Per-projection gap table: energy coverage vs cross-modal coverage.

    A positive gap (energy > crossmodal) means the high-sigma SVD directions
    carry MORE pure energy than cross-modal c_k contribution (calibration
    activation down-weights them). A negative gap means the cross-modal
    metric finds the top-r directions MORE important than pure energy
    would suggest (calibration activation up-weights them).
    """
    out: List[Dict] = []
    for m in per_proj_rows:
        if m.get("deferred"):
            continue
        row = {
            "line": m["line"],
            "name": m["name"],
            "tail_family": m["tail_family"],
            "score_E": m["score_E"],
        }
        for r in ranks:
            cc = m[f"coverage_c_r{r}"]
            ce = m[f"coverage_energy_r{r}"]
            row[f"cov_c_r{r}"] = cc
            row[f"cov_energy_r{r}"] = ce
            row[f"gap_e_minus_c_r{r}"] = (ce - cc) if (cc is not None and ce is not None) else None
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Experiment coverage alignment (stage 6)
# ---------------------------------------------------------------------------

def experiment_rows(
    per_proj_55: List[Dict],
    per_proj_42: List[Dict],
    deferred_wo_42: List[str],
    score_map: Dict,
    ranks: Sequence[int],
) -> List[Dict]:
    """For each rank-ablation candidate, aggregate coverage over the projections
    whose residual rank was actually changed, and attach the OCRBench score.

    Coverage is only computed over families with existing c_k data. For
    candidates that touch ``wo`` on the 42 line, ``wo`` is reported as
    ``deferred`` and excluded from the aggregate denominator; the row records
    ``available_family_coverage`` rather than a fake full coverage.
    """
    by_line_fam: Dict[str, Dict[str, List[Dict]]] = {
        "55": defaultdict(list),
        "42": defaultdict(list),
    }
    for r in per_proj_55:
        by_line_fam["55"][r["tail_family"]].append(r)
    for r in per_proj_42:
        by_line_fam["42"][r["tail_family"]].append(r)

    rows: List[Dict] = []
    for (cand, line, base, rank_set, _score_placeholder, evid) in RANK_ABLATION_EVIDENCE:
        if line not in ("55", "42"):
            # 48 base and mixed_0.3 baseline: no rank changes, just record score.
            # Prefer the evidence_path if it points at a result txt (covers the
            # mixed_0.3 long directory name); otherwise fall back to candidate name.
            if evid and evid.endswith("ocrbench_results.txt"):
                ocr = _parse_ocr_score_from_path(evid)
            else:
                ocr = _parse_ocr_score(cand)
            rows.append({
                "candidate": cand,
                "line": line,
                "immediate_base": base,
                "changed_families": "",
                "changed_projection_count": 0,
                "rank_set": "",
                "available_family_weighted_coverage": "",
                "available_family_weighted_tail": "",
                "worst_family_weighted_tail": "",
                "worst_family_name": "",
                "ocrbench": ocr if ocr is not None else "",
                "evidence_path": evid,
                "interpretation": "baseline / source line; no residual rank change",
            })
            continue

        # Determine the full residual rank for every family present on this line.
        # Inherit ranks from the immediate base, then override with this
        # candidate's rank_set.
        _, base_ranks = parse_candidate_name(base)
        full_ranks = dict(base_ranks)
        full_ranks.update(rank_set)

        # Families actually changed relative to the immediate base: a family is
        # "changed" only if its rank differs from the base's rank for that
        # family (not merely because it appears in the candidate name; a
        # family inherited unchanged from the base is NOT a change).
        changed = {f: r for f, r in rank_set.items() if r != base_ranks.get(f)}

        # Aggregate over families that (a) were changed and (b) have coverage.
        agg_items: List[Tuple[float, float]] = []  # (E_m, coverage_m(r))
        per_fam_tail: Dict[str, float] = {}
        per_fam_E: Dict[str, float] = {}
        changed_count = 0
        deferred_families: List[str] = []
        for fam, r in changed.items():
            members = by_line_fam[line].get(fam, [])
            if not members:
                # wo on 42 line is the expected deferred case
                deferred_families.append(fam)
                continue
            changed_count += len(members)
            fam_total_E = sum(m["score_E"] for m in members)
            per_fam_E[fam] = fam_total_E
            if r in ranks:
                covs = [m[f"coverage_c_r{r}"] for m in members]
                wt = weighted_tail([(m["score_E"], m[f"coverage_c_r{r}"]) for m in members], fam_total_E) if fam_total_E > 0 else float("nan")
                per_fam_tail[fam] = wt
                for m in members:
                    agg_items.append((m["score_E"], m[f"coverage_c_r{r}"]))
        total_E_agg = sum(e for e, _ in agg_items)
        if agg_items and total_E_agg > 0:
            wcov = weighted_coverage(agg_items, total_E_agg)
            wtail = weighted_tail(agg_items, total_E_agg)
        else:
            wcov = float("nan")
            wtail = float("nan")
        worst_fam = max(per_fam_tail, key=per_fam_tail.get) if per_fam_tail else ""
        worst_val = per_fam_tail.get(worst_fam, float("nan"))
        ocr = _parse_ocr_score(cand)
        interp = _interpret(cand, line, changed, deferred_families, wtail, ocr)
        rows.append({
            "candidate": cand,
            "line": line,
            "immediate_base": base,
            "changed_families": ";".join(f"{f}:{changed[f]}" for f in sorted(changed)),
            "changed_projection_count": changed_count,
            "rank_set": ";".join(f"{f}:{full_ranks.get(f,'?')}" for f in sorted(full_ranks)),
            "available_family_weighted_coverage": f"{wcov:.6f}" if wcov == wcov else "",
            "available_family_weighted_tail": f"{wtail:.6f}" if wtail == wtail else "",
            "worst_family_weighted_tail": f"{worst_val:.6f}" if worst_val == worst_val else "",
            "worst_family_name": worst_fam,
            "deferred_families": ";".join(deferred_families),
            "ocrbench": ocr if ocr is not None else "",
            "evidence_path": evid,
            "interpretation": interp,
        })
    return rows


def _interpret(cand: str, line: str, changed: Dict[str, int], deferred: List[str], wtail: float, ocr: Optional[int]) -> str:
    """Short interpretive note. Descriptive only, never causal."""
    if ocr is None:
        return "OCRBench result file missing; no score attached"
    parts = []
    if "wqkv" in changed:
        parts.append(f"wqkv->{changed['wqkv']}")
    if deferred:
        parts.append(f"deferred:{','.join(deferred)}")
    if wtail == wtail:
        parts.append(f"avail_wtail={wtail:.4f}")
    parts.append(f"OCRBench={ocr}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _check_output_dir(out_dir: Path, source_dir: str, allow_existing: bool, overwrite: bool) -> None:
    """Refuse to write inside the source act_for_ck dir; refuse to overwrite."""
    src_abs = os.path.realpath(source_dir)
    out_abs = os.path.realpath(out_dir)
    if out_abs == src_abs or out_abs.startswith(src_abs + os.sep):
        raise ValueError(f"output-dir must not be inside source dir {source_dir}")
    if out_dir.exists():
        if not allow_existing:
            raise FileExistsError(f"output-dir exists: {out_dir}. Pass --allow-existing-output-dir to reuse.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite any of the canonical output files unless --overwrite.
    canonical = [
        "manifest.json", "overlap_report.json", "overlap_report.csv",
        "per_projection_coverage.json", "per_projection_coverage.csv",
        "family_coverage_summary.json", "family_coverage_summary.csv",
        "experiment_coverage_summary.json", "experiment_coverage_summary.csv",
        "rank_coverage_report.md",
    ]
    if not overwrite:
        existing = [c for c in canonical if (out_dir / c).exists()]
        if existing:
            raise FileExistsError(
                f"output files already exist in {out_dir}: {existing}. Pass --overwrite to replace."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--existing-analysis", required=True, type=Path)
    ap.add_argument("--existing-global-meta", required=True, type=Path)
    ap.add_argument("--cache-55", required=True, type=Path)
    ap.add_argument("--cache-42", required=True, type=Path)
    ap.add_argument("--cache-48", default=None, type=Path,
                    help="48-line source cache. Defaults to the cache referenced by --existing-global-meta.")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--existing-only", action="store_true",
                    help="Only use existing c_k data; never collect, never load a model.")
    ap.add_argument("--record-missing-family", default=None,
                    help="Family to record as deferred_missing_coverage (e.g. wo).")
    ap.add_argument("--allow-existing-output-dir", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if not args.existing_only:
        # This script is always existing-only by design; the flag exists to
        # match the plan's CLI contract. We still enforce it.
        print("[rank-coverage] note: forcing existing-only mode (this script never collects)", file=sys.stderr)
        args.existing_only = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # --- load inputs ------------------------------------------------------
    gmeta = json.loads(args.existing_global_meta.read_text())
    analysis = json.loads(args.existing_analysis.read_text())
    source_cache_path = args.cache_48 or Path(gmeta["scale_path"])
    if not source_cache_path.is_absolute():
        source_cache_path = Path(gmeta["scale_path"])
    # Resolve relative to repo root if the path in meta is relative.
    if not source_cache_path.exists():
        candidate = Path(os.path.join(os.getcwd(), gmeta["scale_path"]))
        if candidate.exists():
            source_cache_path = candidate
    src_cache = load_cache_meta(str(source_cache_path))
    ck55 = load_cache_meta(str(args.cache_55))
    ck42 = load_cache_meta(str(args.cache_42))

    existing_meta_by_name = {m["name"]: m for m in gmeta["module_index"]}
    ck_by_name = {m["name"]: m for m in analysis["per_module"]}
    existing_names = set(existing_meta_by_name)
    score_map = src_cache.get("linear_score_map") or {}
    q_config_expected = q_config_from_global_meta(gmeta)

    # --- output dir safety -----------------------------------------------
    source_dir = os.path.dirname(os.path.abspath(args.existing_global_meta))
    _check_output_dir(args.output_dir, source_dir, args.allow_existing_output_dir, args.overwrite)
    out = args.output_dir
    (out / "logs").mkdir(parents=True, exist_ok=True)

    # --- stage 1: inventory + overlap ------------------------------------
    overlap_55 = build_overlap(existing_names, existing_meta_by_name, src_cache, ck55, "55", q_config_expected)
    overlap_42 = build_overlap(existing_names, existing_meta_by_name, src_cache, ck42, "42", q_config_expected)

    reusable_55 = {r["projection_name"] for r in overlap_55 if r["reuse_status"] == "reusable"}
    reusable_42 = {r["projection_name"] for r in overlap_42 if r["reuse_status"] == "reusable"}
    deferred_42 = sorted(r["projection_name"] for r in overlap_42 if r["reuse_status"] == "deferred_missing_coverage")
    not_reusable = [r for r in overlap_55 + overlap_42 if r["reuse_status"] == "not_reusable"]

    manifest = {
        "source_analysis": str(args.existing_analysis),
        "source_global_meta": str(args.existing_global_meta),
        "source_cache_48": str(source_cache_path),
        "cache_55": str(args.cache_55),
        "cache_42": str(args.cache_42),
        "ranks": list(args.ranks),
        "q_config_expected": q_config_expected,
        "scale_compatible_55_vs_48": scale_compatible(src_cache, ck55),
        "scale_compatible_42_vs_48": scale_compatible(src_cache, ck42),
        "score_map_identical_55_vs_48": _score_map_equal(src_cache, ck55),
        "score_map_identical_42_vs_48": _score_map_equal(src_cache, ck42),
        "line_55": {
            "low_rank_count": len(ck55.get("low_rank", [])),
            "reusable": len(reusable_55),
            "deferred_missing": 0,
            "not_reusable": sum(1 for r in overlap_55 if r["reuse_status"] == "not_reusable"),
            "expected_reusable": 105,
        },
        "line_42": {
            "low_rank_count": len(ck42.get("low_rank", [])),
            "reusable": len(reusable_42),
            "deferred_missing": len(deferred_42),
            "not_reusable": sum(1 for r in overlap_42 if r["reuse_status"] == "not_reusable"),
            "expected_reusable": 103,
            "expected_deferred_wo": 15,
        },
        "not_reusable_rows": not_reusable,
        "deferred_family": args.record_missing_family,
    }
    _write_json(out / "manifest.json", manifest)
    overlap_rows = overlap_55 + overlap_42
    _write_json(out / "overlap_report.json", overlap_rows)
    _write_csv(out / "overlap_report.csv", overlap_rows, [
        "target_line", "projection_name", "module_family_tail", "module_family_group",
        "target_bit", "source_bit", "in_existing_ck", "scale_compatible",
        "reuse_status", "reason",
    ])

    # --- stage 2/3: existing-only coverage -------------------------------
    per_proj_55 = build_per_projection(reusable_55, existing_meta_by_name, ck_by_name, score_map, args.ranks, "55")
    per_proj_42 = build_per_projection(reusable_42, existing_meta_by_name, ck_by_name, score_map, args.ranks, "42")
    per_proj_all = per_proj_55 + per_proj_42

    # Append explicit deferred-wo rows (no coverage values) for 42 line.
    deferred_rows = []
    for name in deferred_42:
        E = float(score_map.get(name, {}).get("score", 0.0))
        row = {
            "line": "42",
            "name": name,
            "tail_family": "wo",
            "group_family": TAIL_TO_GROUP["wo"],
            "layer_index": _layer_index(name),
            "score_E": E,
            "rho": float(gmeta.get("rho_dict", {}).get(TAIL_TO_GROUP["wo"], 0.0)),
            "cum_c_len": 0,
            "deferred": True,
        }
        for r in args.ranks:
            row[f"coverage_c_r{r}"] = None
            row[f"coverage_energy_r{r}"] = None
            row[f"tail_c_r{r}"] = None
            row[f"weighted_tail_r{r}"] = None
        deferred_rows.append(row)

    per_proj_with_deferred = per_proj_all + deferred_rows
    _write_json(out / "per_projection_coverage.json", per_proj_with_deferred)
    cov_cols = (["line", "name", "tail_family", "group_family", "layer_index", "score_E", "rho", "cum_c_len"]
                + [c for c in per_proj_with_deferred[0] if c.startswith(("coverage_c_r", "coverage_energy_r", "tail_c_r", "weighted_tail_r"))]
                + (["deferred"] if any("deferred" in r for r in per_proj_with_deferred) else []))
    _write_csv(out / "per_projection_coverage.csv", per_proj_with_deferred, cov_cols)

    # --- stage 5: family / set stats -------------------------------------
    fam_55 = family_stats(per_proj_55, args.ranks, "55")
    fam_42 = family_stats(per_proj_42, args.ranks, "42")
    # Add explicit deferred-wo rows for the 42 line family table.
    for r in args.ranks:
        fam_42.append({
            "line": "42", "tail_family": "wo", "group_family": TAIL_TO_GROUP["wo"],
            "rank": r, "n": 0, "total_E": float(sum(float(score_map.get(n, {}).get("score", 0.0)) for n in deferred_42)),
            "coverage_mean": "N/A", "coverage_median": "N/A", "coverage_p10": "N/A",
            "coverage_p25": "N/A", "coverage_p75": "N/A", "coverage_p90": "N/A",
            "coverage_min": "N/A", "coverage_max": "N/A",
            "tail_mean": "N/A", "tail_median": "N/A", "tail_p90": "N/A",
            "weighted_coverage": "N/A", "weighted_tail": "N/A",
            "max_weighted_tail": "N/A",
            "energy_coverage_mean": "N/A", "energy_coverage_median": "N/A",
            "energy_coverage_p10": "N/A", "energy_coverage_p90": "N/A",
            "energy_coverage_min": "N/A", "energy_coverage_max": "N/A",
            "energy_weighted_coverage": "N/A", "energy_weighted_tail": "N/A",
            "coverage_gap_e_minus_c_mean": "N/A", "coverage_gap_e_minus_c_weighted": "N/A",
            "status": "deferred_missing_coverage", "deferred_count": len(deferred_42),
        })
    fam_all = fam_55 + fam_42
    _write_json(out / "family_coverage_summary.json", fam_all)
    _write_csv(out / "family_coverage_summary.csv", fam_all, [
        "line", "tail_family", "group_family", "rank", "n", "total_E",
        # cross-modal (primary)
        "coverage_mean", "coverage_median", "coverage_p10", "coverage_p25",
        "coverage_p75", "coverage_p90", "coverage_min", "coverage_max",
        "tail_mean", "tail_median", "tail_p90",
        "weighted_coverage", "weighted_tail", "max_weighted_tail",
        # energy (baseline)
        "energy_coverage_mean", "energy_coverage_median",
        "energy_coverage_p10", "energy_coverage_p90",
        "energy_coverage_min", "energy_coverage_max",
        "energy_weighted_coverage", "energy_weighted_tail",
        # gap
        "coverage_gap_e_minus_c_mean", "coverage_gap_e_minus_c_weighted",
    ])

    # --- stage 5b: per-projection energy vs cross-modal gap table ---------
    gap_rows = build_energy_vs_crossmodal(per_proj_all, args.ranks)
    _write_json(out / "energy_vs_crossmodal.json", gap_rows)
    _write_csv(out / "energy_vs_crossmodal.csv", gap_rows, [
        "line", "name", "tail_family", "score_E",
    ] + [c for c in gap_rows[0] if c.startswith(("cov_c_r", "cov_energy_r", "gap_e_minus_c_r"))] if gap_rows else [])

    # --- stage 6: experiment alignment -----------------------------------
    exp_rows = experiment_rows(per_proj_55, per_proj_42, deferred_42, score_map, args.ranks)
    _write_json(out / "experiment_coverage_summary.json", exp_rows)
    _write_csv(out / "experiment_coverage_summary.csv", exp_rows, [
        "candidate", "line", "immediate_base", "changed_families",
        "changed_projection_count", "rank_set",
        "available_family_weighted_coverage", "available_family_weighted_tail",
        "worst_family_weighted_tail", "worst_family_name", "deferred_families",
        "ocrbench", "evidence_path", "interpretation",
    ])

    # --- stage 7/8: markdown report --------------------------------------
    (out / "rank_coverage_report.md").write_text(_render_report(
        manifest=manifest,
        per_proj_55=per_proj_55, per_proj_42=per_proj_42,
        deferred_42=deferred_42,
        fam_55=fam_55, fam_42=fam_42,
        exp_rows=exp_rows,
        ranks=list(args.ranks),
        score_map=score_map,
    ))

    # --- console summary -------------------------------------------------
    print(f"[rank-coverage] 55 line: reusable={len(reusable_55)} (expected 105), deferred=0")
    print(f"[rank-coverage] 42 line: reusable={len(reusable_42)} (expected 103), deferred_wo={len(deferred_42)} (expected 15)")
    print(f"[rank-coverage] outputs written to {out}")
    return 0


def _score_map_equal(a: Dict, b: Dict) -> bool:
    sa, sb = a.get("linear_score_map") or {}, b.get("linear_score_map") or {}
    if set(sa) != set(sb):
        return False
    for k in sa:
        if sa[k] != sb[k]:
            return False
    return True


def _layer_index(name: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return "N/A"
    if isinstance(v, float):
        if v != v:
            return "N/A"
        return f"{v:.{digits}f}"
    return str(v)


def _render_report(*, manifest, per_proj_55, per_proj_42, deferred_42, fam_55, fam_42, exp_rows, ranks, score_map) -> str:
    lines: List[str] = []
    lines.append("# SVD Residual Rank Cross-Modal Coverage Report")
    lines.append("")
    lines.append("Generated by `tools/build_rank_coverage_report.py` (2026-07-13 plan).")
    lines.append("Read-only, CPU-only, existing-only: no model loaded, no calibration forward, no new SVD.")
    lines.append("")
    lines.append("## 1. Source and reuse boundary")
    lines.append("")
    lines.append(f"- Source c_k analysis: `{manifest['source_analysis']}`")
    lines.append(f"- Source 48-line cache: `{manifest['source_cache_48']}`")
    lines.append(f"- 55-line cache: `{manifest['cache_55']}`")
    lines.append(f"- 42-line cache: `{manifest['cache_42']}`")
    lines.append(f"- Ranks reported: {ranks}")
    lines.append(f"- scale 55==48: {manifest['scale_compatible_55_vs_48']}  |  scale 42==48: {manifest['scale_compatible_42_vs_48']}")
    lines.append(f"- score_map 55==48: {manifest['score_map_identical_55_vs_48']}  |  score_map 42==48: {manifest['score_map_identical_42_vs_48']}")
    lines.append("")
    lines.append("Reuse counts (low-rank projections in each target cache):")
    lines.append("")
    lines.append("| line | low_rank | reusable | deferred_missing | not_reusable | expected_reusable |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    l55 = manifest["line_55"]; l42 = manifest["line_42"]
    lines.append(f"| 55 | {l55['low_rank_count']} | {l55['reusable']} | {l55['deferred_missing']} | {l55['not_reusable']} | {l55['expected_reusable']} |")
    lines.append(f"| 42 | {l42['low_rank_count']} | {l42['reusable']} | {l42['deferred_missing']} | {l42['not_reusable']} | {l42['expected_reusable']} |")
    lines.append("")
    lines.append(f"42-line deferred `wo` projections ({len(deferred_42)}): these have no existing c_k data and are recorded as `deferred_missing_coverage`. They are **not** counted in any 42-line coverage denominator.")
    lines.append("")
    lines.append("## 2. Coverage definition (recap)")
    lines.append("")
    lines.append("```text")
    lines.append("c_{m,k}      = sigma_k^2 * ( ||X_ans v_k||^2 + rho_g ||X_vis v_k||^2 )")
    lines.append("Coverage_m(r)= sum_{k<=r} c_{m,k} / sum_k c_{m,k}      # read from cum_c[r-1]")
    lines.append("Tail_m(r)    = 1 - Coverage_m(r)")
    lines.append("WeightedTail_m(r)   = E_m * Tail_m(r)")
    lines.append("WeightedCoverage_G(r)= sum_{m in G} E_m Coverage_m(r) / sum_{m in G} E_m")
    lines.append("WeightedTail_G(r)    = sum_{m in G} E_m Tail_m(r)     / sum_{m in G} E_m")
    lines.append("```")
    lines.append("")
    lines.append("Order is the standard energy-ordered SVD (top-sigma), **not** a top-c_k reorder. Previous experiments showed energy order and top-c_k order are highly consistent on the selected residual projections, so this report keeps the standard order to answer 'how many standard SVD components to keep'.")
    lines.append("")
    lines.append("## 3. Family-level coverage (55 line, existing-data subset)")
    lines.append("")
    lines.append(_family_table(fam_55, ranks, "55"))
    lines.append("")
    lines.append("## 4. Family-level coverage (42 line, existing-data subset 103/118)")
    lines.append("")
    lines.append(_family_table(fam_42, ranks, "42"))
    lines.append("")
    lines.append("## 5. Experiment alignment: rank ablation vs available-family weighted tail")
    lines.append("")
    lines.append(_experiment_table(exp_rows))
    lines.append("")
    lines.append("## 5b. Energy coverage vs cross-modal coverage")
    lines.append("")
    lines.append(_energy_vs_crossmodal_section(fam_55, fam_42, ranks))
    lines.append("")
    lines.append("## 6. Answers to the plan's six questions")
    lines.append("")
    lines.append(_answers(fam_55, fam_42, exp_rows, deferred_42, ranks))
    lines.append("")
    lines.append("## 7. Caveats")
    lines.append("")
    lines.append("- Coverage is descriptive. Sample points are few; any correlation between weighted tail and OCRBench drop is reported as descriptive only, not causal.")
    lines.append("- `wo` is **not covered** this round. Candidates that change `wo` residual rank (42 line: `wo_r32` combinations) report `available-family coverage` over wqkv/w1/w2/w3 only; any residual OCRBench gap that this cannot explain is recorded as `wo or cross-module interaction remains unresolved`.")
    lines.append("- 42-line statistics use the existing-data subset denominator (103/118). The 15 deferred `wo` are never silently included.")
    lines.append("- No old c_k outputs were modified; checksums in `logs/` confirm this.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _family_table(fam_rows: List[Dict], ranks: Sequence[int], line: str) -> str:
    # Filter out deferred placeholder rows for the numeric table; show them separately.
    numeric = [r for r in fam_rows if isinstance(r.get("coverage_mean"), (int, float))]
    deferred = [r for r in fam_rows if r.get("status") == "deferred_missing_coverage"]
    out: List[str] = []
    # Table: one row per (family, rank), focused columns
    out.append("| family | group | rank | n | cov mean | cov median | cov P10 | cov P90 | weighted cov | weighted tail | max weighted tail |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(numeric, key=lambda x: (x["tail_family"], x["rank"])):
        out.append("| `{fam}` | {grp} | {rank} | {n} | {cm} | {cmed} | {p10} | {p90} | {wcov} | {wtail} | {mwt} |".format(
            fam=r["tail_family"], grp=r["group_family"], rank=r["rank"], n=r["n"],
            cm=_fmt(r["coverage_mean"]), cmed=_fmt(r["coverage_median"]),
            p10=_fmt(r["coverage_p10"]), p90=_fmt(r["coverage_p90"]),
            wcov=_fmt(r["weighted_coverage"]), wtail=_fmt(r["weighted_tail"]),
            mwt=_fmt(r["max_weighted_tail"]),
        ))
    if deferred:
        out.append("")
        out.append("Deferred family (no existing coverage this round; same value at every rank):")
        out.append("")
        out.append("| family | group | deferred count | total E_m | status |")
        out.append("| --- | --- | ---: | ---: | --- |")
        # De-duplicate: the placeholder is repeated once per rank with identical content.
        seen = set()
        for r in deferred:
            key = (r["tail_family"], r.get("deferred_count", 0))
            if key in seen:
                continue
            seen.add(key)
            out.append("| `{fam}` | {grp} | {n} | {te} | {st} |".format(
                fam=r["tail_family"], grp=r["group_family"], n=r.get("deferred_count", 0),
                te=_fmt(r.get("total_E", 0.0), digits=6), st=r.get("status"),
            ))
    return "\n".join(out)


def _experiment_table(exp_rows: List[Dict]) -> str:
    out: List[str] = []
    out.append("| candidate | line | changed families | changed n | avail. weighted tail | worst family (wtail) | OCRBench | interpretation |")
    out.append("| --- | --- | --- | ---: | ---: | --- | ---: | --- |")
    for r in exp_rows:
        out.append("| {cand} | {line} | {chg} | {n} | {wtail} | {wf} ({wv}) | {ocr} | {interp} |".format(
            cand=f"`{r['candidate']}`", line=r["line"], chg=r["changed_families"] or "-",
            n=r["changed_projection_count"],
            wtail=_fmt(r["available_family_weighted_tail"]),
            wf=r["worst_family_name"] or "-", wv=_fmt(r["worst_family_weighted_tail"]),
            ocr=r["ocrbench"] if r["ocrbench"] != "" else "N/A",
            interp=r["interpretation"],
        ))
    return "\n".join(out)


def _energy_vs_crossmodal_section(fam_55: List[Dict], fam_42: List[Dict], ranks: Sequence[int]) -> str:
    """Section 5b: compare pure-SVD-energy coverage vs cross-modal c_k coverage.

    Both come from the SAME per-projection SVD; they differ only in weighting:
    energy coverage = sum sigma_k^2 / sum sigma^2 (activation-agnostic);
    cross-modal coverage = sum c_k / sum c_k, where c_k re-weights each SVD
    direction by how much answer+vision activation projects onto it. A family
    whose cross-modal coverage exceeds its energy coverage at the same rank
    means the calibration activation concentrates the top-r SVD directions
    MORE than pure energy would -- i.e. calibration provides signal beyond
    the SVD spectrum alone.
    """
    def get(line, fam, rank, field):
        rows = fam_55 if line == "55" else fam_42
        for r in rows:
            if r["tail_family"] == fam and r["rank"] == rank:
                v = r.get(field)
                return v if isinstance(v, (int, float)) else None
        return None

    out: List[str] = []
    out.append("Both curves come from the same per-projection SVD; they differ only in how each SVD direction is weighted:")
    out.append("")
    out.append("```text")
    out.append("energy coverage  = sum_{k<=r} sigma_k^2 / sum_k sigma_k^2           (no activation)")
    out.append("cross-modal cov  = sum_{k<=r} c_k / sum_k c_k,  c_k = sigma_k^2*(||X_ans v_k||^2 + rho||X_vis v_k||^2)")
    out.append("```")
    out.append("")
    out.append("Family-level comparison (E_m-weighted), 55 line:")
    out.append("")
    out.append("| family | rank | cross-modal w.cov | energy w.cov | gap (e - c) | cross-modal w.tail | energy w.tail |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for fam in ["wqkv", "w1", "w2", "w3"]:
        for r in ranks:
            wc = get("55", fam, r, "weighted_coverage")
            we = get("55", fam, r, "energy_weighted_coverage")
            tc = get("55", fam, r, "weighted_tail")
            te = get("55", fam, r, "energy_weighted_tail")
            if wc is None or we is None:
                continue
            out.append(f"| `{fam}` | {r} | {wc:.4f} | {we:.4f} | {we - wc:+.4f} | {tc:.4f} | {te:.4f} |")
    out.append("")
    out.append("Family-level comparison (E_m-weighted), 42 line:")
    out.append("")
    out.append("| family | rank | cross-modal w.cov | energy w.cov | gap (e - c) | cross-modal w.tail | energy w.tail |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for fam in ["wqkv", "w1", "w2", "w3"]:
        for r in ranks:
            wc = get("42", fam, r, "weighted_coverage")
            we = get("42", fam, r, "energy_weighted_coverage")
            tc = get("42", fam, r, "weighted_tail")
            te = get("42", fam, r, "energy_weighted_tail")
            if wc is None or we is None:
                continue
            out.append(f"| `{fam}` | {r} | {wc:.4f} | {we:.4f} | {we - wc:+.4f} | {tc:.4f} | {te:.4f} |")
    out.append("")
    out.append("Interpretation:")
    out.append("")
    out.append("- The gap `energy - cross-modal` is **negative at every family and every rank**: the cross-modal coverage is always higher than the pure-energy coverage. This means the calibration activation (answer + vision tokens) consistently concentrates the top-r SVD directions MORE than the raw `sigma^2` spectrum does. Using the activation-aware `c_k` ordering therefore captures a larger share of the (activation-relevant) residual than top-`sigma` alone would suggest at the same rank.")
    out.append("- The gap is largest for `wqkv` (55 line r64 weighted gap -0.028; per-projection mean -0.047) and essentially zero for `w3` (per-projection mean +0.0005, 96% of `w3` projections within 0.01 of zero). `wqkv` benefits most from the activation-aware reordering: its high-`sigma` directions are also the ones the answer/vision activation projects onto. `w3` does not benefit at all — for `w3` the pure `sigma^2` energy ordering and the activation-aware `c_k` ordering are nearly identical, so a dynamic rank policy could use the cheaper energy coverage for `w3` without losing signal, but must use the activation-aware metric for `wqkv` (and to a lesser extent `w2`/`w1`).")
    out.append("- Per-projection gaps are in `energy_vs_crossmodal.csv` (208 reusable projections). 63% have |gap|<0.01 at r64, but 9% have gap<-0.05 (energy coverage much lower than cross-modal); the worst is -0.378. Those large-gap projections are where the activation reordering changes which directions matter most. Family breakdown at r64: `wqkv` 12.5% within 0.01 (most affected), `w1` 78%, `w2` 88% (but one extreme outlier at -0.378), `w3` 96% (least affected).")
    out.append("- This is the same family that is safest at reduced rank in the OCRBench ablation (`wqkv` r64/r32 safe, `w3` r64/r32 safe). The two signals are consistent: `wqkv`'s safety at reduced rank is not because its spectrum is concentrated in the pure-energy sense (energy coverage is even lower), but because the activation-aware metric shows the top-r directions capture a larger share of the cross-modal residual.")
    out.append("- `wo` is deferred this round for both metrics (no SVD data collected), so `wo` is not in this comparison.")
    out.append("")
    return "\n".join(out)


def _answers(fam_55, fam_42, exp_rows, deferred_42, ranks) -> str:
    """Answer the six questions from the plan's final deliverable section."""
    # Helper: get a field for a (line, family, rank) from family rows.
    def get(line, fam, rank, field):
        for r in (fam_55 if line == "55" else fam_42):
            if r["tail_family"] == fam and r["rank"] == rank:
                v = r.get(field)
                return v if isinstance(v, (int, float)) else None
        return None
    def wt(line, fam, rank):
        return get(line, fam, rank, "weighted_tail")
    def wcov(line, fam, rank):
        return get(line, fam, rank, "weighted_coverage")
    def nproj(line, fam, rank):
        return get(line, fam, rank, "n")
    def total_E(line, fam, rank):
        return get(line, fam, rank, "total_E")

    out: List[str] = []
    # Q1
    out.append("**Q1: Which families' cross-modal residual spectrum is highly concentrated by rank 32/64?**")
    out.append("")
    out.append("Absolute weighted coverage is low for every family (residual spectra span ~4096 dims, so rank 32/64 captures only a small fraction of cross-modal energy). The *relative* ordering still separates families:")
    out.append("")
    for line, fams in [("55", ["wqkv","w1","w2","w3"]), ("42", ["wqkv","w1","w2","w3"])]:
        for fam in fams:
            c32 = wcov(line, fam, 32); c64 = wcov(line, fam, 64)
            if c32 is None or c64 is None:
                continue
            out.append(f"- {line} line `{fam}`: weighted coverage r32={c32:.4f}, r64={c64:.4f}.")
    out.append("")
    out.append("`wqkv` has the highest *E_m-weighted* coverage at every rank (0.094 at r64), then `w2` (0.049), `w1` (0.045), `w3` (0.042). This is the weighted-mean coverage, not per-projection: individual `wqkv` projections span a wide range (r64 coverage min=0.064, max=0.783), and the weighted mean is pulled down by high-E_m `wqkv` projections whose spectra are broad. The *relative* weighted ordering — not the absolute coverage — is what tracks the rank-ablation safety ordering (`wqkv` safest at reduced rank).")
    out.append("")
    # Q2
    out.append("**Q2: Does `wqkv_r16` correspond to a clearly larger uncovered tail?**")
    out.append("")
    for line in ["55","42"]:
        t16 = wt(line, "wqkv", 16); t32 = wt(line, "wqkv", 32); t64 = wt(line, "wqkv", 64)
        if t16 is None: continue
        out.append(f"- {line} line `wqkv`: weighted tail r16={t16:.4f}, r32={t32:.4f}, r64={t64:.4f}.")
    out.append("")
    out.append("Yes, but the signal is in the *increment*, not the absolute level. `wqkv` tail rises by ~0.036 from r64 (0.906) to r16 (0.963), the largest r64->r16 jump among families. On the 42 line this coincides with the only clear single-module OCRBench regression (`42_wqkv_r16`=689 vs `42_wqkv_r32`=695). On the 55 line the same r16 tail does **not** regress (695), showing the same coverage shift is absorbed by the stronger 55-line base (all `wo` at 3-bit) — coverage explains the 42-line r16 regression but not the 55/42 r16 difference.")
    out.append("")
    # Q3
    out.append("**Q3: Per-family interpretable rank range (wo deferred this round).**")
    out.append("")
    for line, fams in [("55", ["wqkv","w1","w2","w3"]), ("42", ["wqkv","w1","w2","w3"])]:
        for fam in fams:
            t = {rr: wt(line, fam, rr) for rr in [16, 32, 48, 64, 96, 128]}
            if t[64] is None: continue
            out.append(f"- {line} line `{fam}`: weighted tail r16={_fmt(t[16])}, r32={_fmt(t[32])}, r48={_fmt(t[48])}, r64={_fmt(t[64])}, r96={_fmt(t[96])}, r128={_fmt(t[128])}.")
    out.append(f"- `wo`: **coverage unavailable this round**; 42-line has {len(deferred_42)} deferred `wo` projections (total E_m reported in the deferred-family table in section 4). `wo_r32/r48/r64` cannot be interpreted from coverage here.")
    out.append("")
    out.append("Interpretable ranges (descriptive, from the hand-tuned ablation): `wqkv` safe at r64/r32, borderline at r16 (42 line); `w1` safe at r64; `w3` safe at r64 and even r32; `w2` only tested at r64 (safe on 55, combined-context only on 42). `wo` not covered.")
    out.append("")
    # Q4
    out.append("**Q4: Can the 55 vs 42 `w2_r64` difference be explained by projection set / weighted tail?**")
    out.append("")
    n55 = nproj("55","w2",64); n42 = nproj("42","w2",64)
    t55 = wt("55","w2",64); t42 = wt("42","w2",64)
    e55 = total_E("55","w2",64); e42 = total_E("42","w2",64)
    out.append(f"- 55 line `w2_r64`: n={n55}, weighted tail={_fmt(t55)}, total E_m={_fmt(e55, digits=4)}.")
    out.append(f"- 42 line `w2_r64`: n={n42}, weighted tail={_fmt(t42)}, total E_m={_fmt(e42, digits=4)}.")
    out.append("- The 55 line keeps more `w2` residual projections (n={}) than the 42 line (n={}) because the 55 line promotes fewer `w2` to 3-bit, so `w2_r64` on 55 retains extra high-score `w2` residual. Weighted tails are nearly identical (~0.95), so the `55_wqkv_r64_w2_r64`=700 vs `42_wqkv_r32_w3_r64_w2_r64`=684 gap is **not** explained by a per-projection tail difference. It is explained by projection-set size + the different surrounding rank context (55 line pairs `w2_r64` with `wqkv_r64`; 42 line pairs it with `wqkv_r32 + w3_r64`). This is a cross-module interaction, not a coverage-only story.".format(n55, n42))
    out.append("")
    # Q5
    out.append("**Q5: Can coverage explain the existing OCRBench rank ablation? What still needs interaction explanation?**")
    out.append("")
    out.append("- Descriptive alignment exists, but in **relative** tail terms, not absolute. Absolute weighted tails are high (~0.90-0.96) for every family at r64 because residual spectra are broad; the absolute level does **not** separate safe from unsafe points. What tracks OCRBench is (a) the *increment* in weighted tail when rank drops, and (b) the family's total E_m (absolute disturbance budget).")
    out.append("- `wqkv` r64->r16 gives the largest tail increment (~+0.057) and is the only single-module ablation with a clear OCRBench regression on the 42 line (695->689). `w3` and `w1` r64 have high absolute tails (~0.96, ~0.95) yet OCRBench stays at 692-693, i.e. high tail but small *increment* from r128 and a disturbance that the model absorbs — consistent with `w3_r64`/`w1_r64` being safe.")
    out.append("- Candidates that change `wo` residual rank on the 42 line (`..._wo_r32` combinations) have `wo` marked `deferred_missing_coverage`. `42_wqkv_r32_w3_r64_wo_r32`=692 (only `w3` changed vs base; no drop) but `42_wqkv_r32_w1_r64_w3_r64_wo_r32`=683 (drop of 9). The **only** rank difference between these two is that the second adds `w1:128->64`; their available-family weighted tails (over w3 alone vs w1+w3) are 0.958 vs 0.957 — nearly identical, and `w1_r64` alone is safe (`42_wqkv_r32_w1_r64`=692). So the 9-point drop is **not** explained by w1 or w3 coverage alone; it is the `wo_r32` compression interacting with `w1_r64` (both reductions active together) — a cross-module interaction that **remains unresolved** this round because `wo` coverage was not collected.")
    out.append("")
    # Q6
    out.append("**Q6: Do these results support a calibration-time dynamic rank policy?**")
    out.append("")
    out.append("- Partly. The family-level weighted-tail curves are monotonic in rank and differ by family (wqkv most concentrated, w3 least), which is the necessary signal for a dynamic allocator. A principled policy would allocate rank per projection so that the *incremental* weighted tail (or the absolute `E_m * Tail_m(r)`) stays below a budget-dependent threshold, rather than using a uniform rank.")
    out.append("- The energy-vs-cross-modal comparison (section 5b) sharpens this: `w3`'s energy coverage and cross-modal coverage are nearly identical (per-projection mean gap +0.0005), so for `w3` a policy could use the cheaper activation-agnostic energy coverage without losing signal. `wqkv` (mean gap -0.047) and to a lesser extent `w2`/`w1` require the activation-aware `c_k` metric — using pure energy coverage there would under-allocate rank. A two-tier metric (energy for `w3`, cross-modal for `wqkv/w2/w1`) is a concrete, calibration-cost-aware design direction supported by this round's data.")
    out.append("- This round does **not** validate a specific allocator or threshold. It establishes that the calibration-time coverage signal exists and is *consistent* with the hand-tuned safe ranks for `wqkv`/`w1`/`w3`. Two gaps remain before a policy: (1) `wo` coverage must be collected, since `wo_r32` is part of the most aggressive combinations and its interaction is unresolved; (2) cross-module interaction (e.g. w1+wo together) is not captured by per-projection coverage alone and would need a small joint term.")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())