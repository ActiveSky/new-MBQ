#!/usr/bin/env python3
"""Build a projection-adaptive low-rank MBQ cache.

The allocator implements the paper rule

    r_m = min {r in K_m : eta_m(r) >= tau},

with a max-rank fallback when the threshold is not reached.  ``eta_m`` must
come from the paper-normalized c_k analysis emitted by ``tools/analyze_ck.py``
(metric version 2).  Allocation is performed once after calibration; inference
only consumes the fixed per-projection ranks stored in the output cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRIC_VERSION = 2
DEFAULT_RANKS = (16, 32, 48, 64, 96, 128)
TOL = 1e-7
FACTOR_BASIS = "collector_energy_ordered_svd_prefix"


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_analysis(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def normalize_ranks(ranks: Iterable[int]) -> Tuple[int, ...]:
    normalized = tuple(sorted(set(int(rank) for rank in ranks)))
    if not normalized or normalized[0] <= 0:
        raise ValueError("deployable ranks must be distinct positive integers")
    return normalized


def validate_threshold(tau: float) -> float:
    tau = float(tau)
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    return tau


def analysis_rows_by_name(analysis: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    metric = analysis.get("metric") or {}
    if int(metric.get("version", 0)) < METRIC_VERSION or not metric.get("normalized"):
        raise ValueError(
            "adaptive rank requires paper-normalized metric version 2; "
            "recollect with tools/collect_act_for_ck.py and rerun tools/analyze_ck.py"
        )
    if metric.get("basis_order") != "energy_ordered_svd_prefix":
        raise ValueError("analysis basis must be energy_ordered_svd_prefix")
    if metric.get("coverage_denominator") != "projection_score_E":
        raise ValueError("analysis coverage denominator must be projection_score_E")
    mask_semantics = metric.get("mask_semantics") or []
    if isinstance(mask_semantics, str):
        mask_semantics = [mask_semantics]
    if set(mask_semantics) != {"answer_text"}:
        raise ValueError(
            "paper-aligned adaptive rank requires supervised answer_text masks"
        )
    factor_basis = metric.get("factor_basis") or []
    if isinstance(factor_basis, str):
        factor_basis = [factor_basis]
    if set(factor_basis) != {FACTOR_BASIS}:
        raise ValueError(
            "adaptive rank requires factors from the same energy-ordered SVD "
            "used to compute c_k; recollect with tools/collect_act_for_ck.py"
        )
    if metric.get("token_subsampling"):
        raise ValueError(
            "paper-aligned adaptive rank rejects --max-tokens subsampling; "
            "recollect using all supervised-text and visual calibration tokens"
        )
    if analysis.get("rho_override"):
        raise ValueError(
            "paper-aligned adaptive rank rejects diagnostic rho overrides"
        )

    rows: Dict[str, Dict[str, Any]] = {}
    for row in analysis.get("per_module") or []:
        name = row.get("name")
        if not name:
            raise ValueError("analysis contains a projection without a name")
        if name in rows:
            raise ValueError(f"duplicate projection in analysis: {name}")
        if not row.get("coverage_normalized"):
            raise ValueError(f"projection has legacy unnormalized coverage: {name}")
        if row.get("mask_semantics") != "answer_text":
            raise ValueError(f"projection has non-paper text-mask semantics: {name}")
        if row.get("factor_prefix_basis") != FACTOR_BASIS:
            raise ValueError(f"projection has a mismatched SVD factor basis: {name}")
        if not row.get("factor_prefix_file"):
            raise ValueError(f"projection lacks a deployable SVD factor prefix: {name}")
        score = float(row.get("score_E", 0.0))
        if not math.isfinite(score) or score <= 0:
            raise ValueError(f"projection has invalid score_E: {name}")
        rows[name] = row
    if not rows:
        raise ValueError("analysis contains no projection rows")
    return rows


def validate_cumulative_coverage(cum: Sequence[float], name: str) -> None:
    if not cum:
        raise ValueError(f"empty cumulative coverage for {name}")
    previous = -TOL
    for index, raw_value in enumerate(cum):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite coverage for {name} at component {index + 1}")
        if value < previous - TOL:
            raise ValueError(f"non-monotonic cumulative coverage for {name}")
        if value < -TOL or value > 1.0 + 1e-4:
            raise ValueError(
                f"coverage outside [0, 1] for {name} at component {index + 1}: {value}"
            )
        previous = value


def coverage_at(cum: Sequence[float], rank: int, name: str = "projection") -> float:
    if rank <= 0 or rank > len(cum):
        raise ValueError(
            f"rank {rank} is unavailable for {name}; coverage length is {len(cum)}"
        )
    return min(1.0, max(0.0, float(cum[rank - 1])))


def select_rank_for_tau(
    cum: Sequence[float], ranks: Sequence[int], tau: float, name: str = "projection"
) -> Tuple[int, float, bool]:
    tau = validate_threshold(tau)
    if not ranks:
        raise ValueError(f"no deployable ranks for {name}")
    for rank in ranks:
        coverage = coverage_at(cum, rank, name)
        if coverage >= tau:
            return int(rank), coverage, True
    rank = int(ranks[-1])
    return rank, coverage_at(cum, rank, name), False


def tensor_pair(item: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    up = item.get("up")
    down = item.get("down")
    if not torch.is_tensor(up) or not torch.is_tensor(down):
        raise TypeError(f"low_rank item lacks tensor up/down: {item.get('name')}")
    if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]:
        raise ValueError(
            f"invalid low-rank factor shapes for {item.get('name')}: "
            f"up={tuple(up.shape)}, down={tuple(down.shape)}"
        )
    return up, down


def available_rank(item: Mapping[str, Any]) -> int:
    up, down = tensor_pair(item)
    declared = int(item.get("rank", up.shape[1]))
    return min(declared, int(up.shape[1]), int(down.shape[0]))


def _shape_numel(value: Any) -> Optional[int]:
    if isinstance(value, Mapping):
        out_features = value.get("out_features", value.get("out"))
        in_features = value.get("in_features", value.get("in"))
        if out_features is not None and in_features is not None:
            numel = int(out_features) * int(in_features)
            return numel if numel > 0 else None
        shape = value.get("shape")
        if shape is not None:
            value = shape
    if isinstance(value, (list, tuple)) and len(value) == 2:
        numel = int(value[0]) * int(value[1])
        return numel if numel > 0 else None
    return None


def infer_weight_numel_map(*caches: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for cache in caches:
        if not cache:
            continue
        for name, shape in (cache.get("linear_shape_map") or {}).items():
            numel = _shape_numel(shape)
            if numel is not None:
                result[name] = numel
        for item in cache.get("low_rank") or []:
            up, down = tensor_pair(item)
            result[item["name"]] = int(up.shape[0] * down.shape[1])
    return result


def validate_factor_bit(cache: Mapping[str, Any], factor_bit: int) -> None:
    """Check factor-quantization metadata when the cache provides it."""
    if factor_bit <= 0:
        raise ValueError("factor_bit must be positive")
    configured_bits = set()
    missing_config = []
    for item in cache.get("low_rank") or []:
        config = item.get("svd_quant_config") or {}
        item_bits = []
        for factor_name in ("up", "down"):
            factor_config = config.get(factor_name) or {}
            if factor_config.get("quant_bit") is not None:
                bit = int(factor_config["quant_bit"])
                configured_bits.add(bit)
                item_bits.append(bit)
        if len(item_bits) != 2:
            missing_config.append(str(item.get("name")))
    if missing_config and factor_bit < 16:
        raise ValueError(
            "cannot account low-rank factors below 16 bits without explicit "
            "up/down svd_quant_config metadata; first missing projections: "
            f"{missing_config[:5]}"
        )
    if configured_bits and configured_bits != {int(factor_bit)}:
        raise ValueError(
            f"--factor-bit={factor_bit} disagrees with cache factor bits "
            f"{sorted(configured_bits)}"
        )


def actual_bit_cost(
    cache: Mapping[str, Any],
    rank_map: Mapping[str, int],
    weight_numel_map: Mapping[str, int],
    factor_bit: int,
) -> Tuple[float, Dict[str, int]]:
    if factor_bit <= 0:
        raise ValueError("factor_bit must be positive")
    validate_factor_bit(cache, factor_bit)
    linear_bit_map = cache.get("linear_bit_map") or {}
    if not linear_bit_map:
        raise ValueError("cache has no linear_bit_map")
    missing = sorted(set(linear_bit_map) - set(weight_numel_map))
    if missing:
        raise ValueError(
            "missing shapes for linear_bit_map entries; provide --reference-cache. "
            f"First missing entries: {missing[:5]}"
        )

    target_weight_numel = sum(weight_numel_map[name] for name in linear_bit_map)
    weight_bits = sum(
        weight_numel_map[name] * int(bit) for name, bit in linear_bit_map.items()
    )
    factor_numel = 0
    low_rank_names = set()
    for item in cache.get("low_rank") or []:
        name = item["name"]
        low_rank_names.add(name)
        up, down = tensor_pair(item)
        rank = int(rank_map.get(name, available_rank(item)))
        if rank <= 0 or rank > available_rank(item):
            raise ValueError(f"invalid selected rank {rank} for {name}")
        factor_numel += rank * (int(up.shape[0]) + int(down.shape[1]))
    unknown = sorted(set(rank_map) - low_rank_names)
    if unknown:
        raise ValueError(f"rank map contains non-low-rank projections: {unknown[:5]}")

    low_rank_bits = factor_numel * int(factor_bit)
    total_bits = weight_bits + low_rank_bits
    return total_bits / target_weight_numel, {
        "target_weight_numel": int(target_weight_numel),
        "weight_bits": int(weight_bits),
        "low_rank_factor_numel": int(factor_numel),
        "low_rank_bits": int(low_rank_bits),
        "total_bits": int(total_bits),
    }


def build_rank_map(
    cache: Mapping[str, Any],
    rows_by_name: Mapping[str, Mapping[str, Any]],
    deployable_ranks: Sequence[int],
    tau: float,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    tau = validate_threshold(tau)
    rank_map: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    missing_analysis: List[str] = []
    seen_names = set()

    score_map = cache.get("linear_score_map") or {}
    for item in cache.get("low_rank") or []:
        name = item.get("name")
        if not name or name in seen_names:
            raise ValueError(f"duplicate or empty low-rank projection name: {name}")
        seen_names.add(name)
        if name not in rows_by_name:
            missing_analysis.append(str(name))
            continue
        analysis_row = rows_by_name[name]
        cum = analysis_row.get("cum_c") or []
        validate_cumulative_coverage(cum, name)
        max_available = available_rank(item)
        ranks = tuple(
            rank
            for rank in normalize_ranks(deployable_ranks)
            if rank <= max_available and rank <= len(cum)
        )
        if not ranks:
            raise ValueError(
                f"no deployable rank fits {name}: factor rank={max_available}, "
                f"coverage length={len(cum)}"
            )
        selected, coverage, threshold_met = select_rank_for_tau(cum, ranks, tau, name)
        rank_map[name] = selected
        score_meta = score_map.get(name) or {}
        rows.append(
            {
                "name": name,
                "module_family": score_meta.get(
                    "module_family", analysis_row.get("family", "unknown")
                ),
                "score_E": float(analysis_row.get("score_E", 0.0)),
                "source_rank": max_available,
                "selected_rank": selected,
                "coverage": coverage,
                "threshold_met": threshold_met,
                "max_available_coverage": coverage_at(cum, ranks[-1], name),
                "mask_semantics": analysis_row.get("mask_semantics"),
            }
        )

    if missing_analysis:
        raise ValueError(
            f"missing normalized c_k analysis for {len(missing_analysis)} low-rank "
            f"projections: {missing_analysis[:5]}"
        )
    if not rank_map:
        raise ValueError("cache contains no low_rank entries")
    return rank_map, rows


def threshold_candidates(
    cache: Mapping[str, Any],
    rows_by_name: Mapping[str, Mapping[str, Any]],
    deployable_ranks: Sequence[int],
) -> List[float]:
    candidates = {1e-12, 1.0}
    for item in cache.get("low_rank") or []:
        name = item["name"]
        row = rows_by_name.get(name)
        if row is None:
            continue
        cum = row.get("cum_c") or []
        max_available = available_rank(item)
        ranks = [
            rank
            for rank in normalize_ranks(deployable_ranks)
            if rank <= max_available and rank <= len(cum)
        ]
        for rank in ranks[:-1]:
            coverage = coverage_at(cum, rank, name)
            if coverage < 1.0:
                candidates.add(math.nextafter(max(coverage, 0.0), 1.0))
    return sorted(validate_threshold(value) for value in candidates)


def choose_tau_for_budget(
    cache: Mapping[str, Any],
    rows_by_name: Mapping[str, Mapping[str, Any]],
    deployable_ranks: Sequence[int],
    target_actual_bit: float,
    weight_numel_map: Mapping[str, int],
    factor_bit: int,
) -> Tuple[float, Dict[str, int], List[Dict[str, Any]], float, Dict[str, int]]:
    if not math.isfinite(target_actual_bit) or target_actual_bit <= 0:
        raise ValueError("target_actual_bit must be positive")
    best = None
    minimum = None
    for tau in threshold_candidates(cache, rows_by_name, deployable_ranks):
        rank_map, rows = build_rank_map(cache, rows_by_name, deployable_ranks, tau)
        actual_bit, detail = actual_bit_cost(
            cache, rank_map, weight_numel_map, factor_bit
        )
        candidate = (actual_bit, tau, rank_map, rows, detail)
        if minimum is None or actual_bit < minimum[0]:
            minimum = candidate
        if actual_bit <= target_actual_bit:
            if best is None or (actual_bit, tau) > (best[0], best[1]):
                best = candidate
    if best is None:
        raise ValueError(
            f"target actual bit {target_actual_bit:.6f} is below the minimum "
            f"deployable configuration {minimum[0]:.6f}"
        )
    actual_bit, tau, rank_map, rows, detail = best
    return tau, rank_map, rows, actual_bit, detail


def validate_source_cache(analysis: Mapping[str, Any], base_cache_path: Path) -> None:
    """Prevent applying calibration statistics to a different scale cache."""
    source = (analysis.get("collection") or {}).get("source_cache")
    if not source:
        raise ValueError(
            "analysis lacks source-cache provenance; recollect and rerun analyze_ck.py"
        )
    if Path(source).expanduser().resolve() != base_cache_path.expanduser().resolve():
        raise ValueError(
            "analysis was collected from a different cache: "
            f"analysis={source}, --base-cache={base_cache_path}"
        )


def _normalized_factor_config(
    item: Mapping[str, Any], factor_name: str, factor_bit: int
) -> Dict[str, Any]:
    config = (item.get("svd_quant_config") or {}).get(factor_name)
    if not isinstance(config, Mapping):
        if factor_bit >= 16:
            return {
                "weight_quant": "per_channel",
                "quant_axis": "out_channel" if factor_name == "up" else "in_channel",
                "quant_bit": factor_bit,
                "zero_point": False,
                "q_group_size": -1,
            }
        raise ValueError(
            f"{item.get('name')} lacks {factor_name} factor-quantization metadata"
        )
    normalized = dict(config)
    normalized["weight_quant"] = str(
        normalized.get("weight_quant", "per_channel")
    ).lower()
    normalized["quant_axis"] = str(
        normalized.get(
            "quant_axis", "out_channel" if factor_name == "up" else "in_channel"
        )
    ).lower()
    normalized["quant_bit"] = int(normalized.get("quant_bit", factor_bit))
    normalized["zero_point"] = bool(normalized.get("zero_point", False))
    normalized["q_group_size"] = int(normalized.get("q_group_size", -1))
    if normalized["quant_bit"] != int(factor_bit):
        raise ValueError(
            f"{item.get('name')} {factor_name} quant_bit="
            f"{normalized['quant_bit']} disagrees with --factor-bit={factor_bit}"
        )
    return normalized


@torch.no_grad()
def _quantize_factor(
    tensor: torch.Tensor,
    factor_name: str,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Apply the same pseudo-quantization convention as MBQ cache creation."""
    quant_bit = int(config["quant_bit"])
    if quant_bit >= 16:
        return tensor

    weight_quant = str(config["weight_quant"])
    if weight_quant not in {"per_channel", "per_group", "per_tensor"}:
        raise ValueError(
            f"invalid {factor_name} weight_quant={weight_quant!r}"
        )

    quant_axis = str(config["quant_axis"])
    if factor_name == "up":
        if quant_axis in {"out_channel", "row"}:
            transpose = False
        elif quant_axis in {"rank_channel", "col"}:
            transpose = True
        else:
            raise ValueError(f"invalid up quant_axis={quant_axis!r}")
    elif factor_name == "down":
        if quant_axis in {"rank_channel", "row"}:
            transpose = False
        elif quant_axis in {"in_channel", "col"}:
            transpose = True
        else:
            raise ValueError(f"invalid down quant_axis={quant_axis!r}")
    else:
        raise ValueError(f"unknown factor name: {factor_name}")

    quant_input = tensor.transpose(0, 1).contiguous() if transpose else tensor
    from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

    if weight_quant == "per_tensor":
        quantized = pseudo_quantize_tensor(
            quant_input,
            n_bits=quant_bit,
            zero_point=bool(config["zero_point"]),
            q_group_size=-1,
            per_tensor=True,
            inplace=False,
        )
    else:
        group_size = int(config["q_group_size"]) if weight_quant == "per_group" else -1
        if group_size > 0 and quant_input.shape[-1] % group_size != 0:
            raise ValueError(
                f"{factor_name} dimension {quant_input.shape[-1]} is not "
                f"divisible by q_group_size={group_size}"
            )
        quantized = pseudo_quantize_tensor(
            quant_input,
            n_bits=quant_bit,
            zero_point=bool(config["zero_point"]),
            q_group_size=group_size,
            per_tensor=False,
            inplace=False,
        )
    return quantized.transpose(0, 1).contiguous() if transpose else quantized


def materialize_adaptive_cache(
    cache: Mapping[str, Any],
    rank_map: Mapping[str, int],
    rows_by_name: Mapping[str, Mapping[str, Any]],
    analysis_path: Path,
    factor_bit: int,
) -> Dict[str, Any]:
    """Build factors from the exact SVD basis used by the coverage analysis."""
    validate_factor_bit(cache, factor_bit)
    new_cache = dict(cache)
    new_low_rank = []
    analysis_dir = analysis_path.expanduser().resolve().parent

    for item in cache.get("low_rank") or []:
        name = item["name"]
        if name not in rank_map or name not in rows_by_name:
            raise ValueError(f"missing adaptive-rank state for {name}")
        target_rank = int(rank_map[name])
        row = rows_by_name[name]
        factor_path = Path(str(row["factor_prefix_file"]))
        if not factor_path.is_absolute():
            factor_path = analysis_dir / factor_path
        factor_path = factor_path.resolve()
        if not factor_path.is_file():
            raise FileNotFoundError(f"missing SVD factor prefix for {name}: {factor_path}")
        prefix = load_cache(factor_path)
        if prefix.get("quantized"):
            raise ValueError(f"expected an unquantized factor prefix for {name}")
        source_up, source_down = tensor_pair({"name": name, **prefix})
        source_rank = min(
            int(prefix.get("rank", source_up.shape[1])),
            int(source_up.shape[1]),
            int(source_down.shape[0]),
        )
        if target_rank > source_rank:
            raise ValueError(
                f"selected rank {target_rank} exceeds stored factor prefix "
                f"rank {source_rank} for {name}"
            )

        base_up, base_down = tensor_pair(item)
        expected_shape = (int(base_up.shape[0]), int(base_down.shape[1]))
        source_shape = (int(source_up.shape[0]), int(source_down.shape[1]))
        if source_shape != expected_shape:
            raise ValueError(
                f"factor-prefix shape mismatch for {name}: "
                f"analysis={source_shape}, cache={expected_shape}"
            )
        up = source_up[:, :target_rank].float()
        down = source_down[:target_rank, :].float()
        up = _quantize_factor(
            up, "up", _normalized_factor_config(item, "up", factor_bit)
        ).to(dtype=base_up.dtype, device="cpu")
        down = _quantize_factor(
            down, "down", _normalized_factor_config(item, "down", factor_bit)
        ).to(dtype=base_down.dtype, device="cpu")

        new_item = dict(item)
        new_item["rank"] = target_rank
        new_item["up"] = up.contiguous()
        new_item["down"] = down.contiguous()
        new_item["adaptive_rank_factor_basis"] = FACTOR_BASIS
        new_low_rank.append(new_item)

    if len(new_low_rank) != len(rank_map):
        raise ValueError(
            "rank-map projection count differs from materialized low-rank factors"
        )
    new_cache["low_rank"] = new_low_rank
    return new_cache


def truncate_cache(
    cache: Mapping[str, Any], rank_map: Mapping[str, int]
) -> Dict[str, Any]:
    new_cache = dict(cache)
    new_low_rank = []
    for item in cache.get("low_rank") or []:
        name = item["name"]
        if name not in rank_map:
            raise ValueError(f"missing selected rank for {name}")
        target_rank = int(rank_map[name])
        up, down = tensor_pair(item)
        if target_rank > available_rank(item):
            raise ValueError(f"cannot increase {name} to unavailable rank {target_rank}")
        new_item = dict(item)
        new_item["rank"] = target_rank
        new_item["up"] = up[:, :target_rank].contiguous()
        new_item["down"] = down[:target_rank, :].contiguous()
        new_low_rank.append(new_item)
    new_cache["low_rank"] = new_low_rank
    return new_cache


def summarize(
    *,
    base_cache: Mapping[str, Any],
    rank_map: Mapping[str, int],
    rows: List[Dict[str, Any]],
    tau: float,
    deployable_ranks: Sequence[int],
    factor_bit: int,
    weight_numel_map: Mapping[str, int],
    target_actual_bit: Optional[float],
    base_cache_path: Path,
    analysis_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    actual_bit, actual_detail = actual_bit_cost(
        base_cache, rank_map, weight_numel_map, factor_bit
    )
    base_rank_map = {
        item["name"]: available_rank(item) for item in base_cache.get("low_rank") or []
    }
    base_actual_bit, base_detail = actual_bit_cost(
        base_cache, base_rank_map, weight_numel_map, factor_bit
    )
    rank_counts = Counter(rank_map.values())
    family_rank_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        family_rank_counts[str(row["module_family"])][int(row["selected_rank"])] += 1

    return {
        "policy": "paper_coverage_threshold",
        "metric_version": METRIC_VERSION,
        "factor_basis": FACTOR_BASIS,
        "base_cache": str(base_cache_path),
        "analysis": str(analysis_path),
        "output_cache": str(output_path),
        "deployable_ranks": list(deployable_ranks),
        "tau": float(tau),
        "target_actual_bit": target_actual_bit,
        "factor_bit": int(factor_bit),
        "fixed_after_calibration": True,
        "selected_projection_count": len(rank_map),
        "threshold_met_count": sum(bool(row["threshold_met"]) for row in rows),
        "fallback_to_max_count": sum(not bool(row["threshold_met"]) for row in rows),
        "rank_counts": {str(k): int(v) for k, v in sorted(rank_counts.items())},
        "family_rank_counts": {
            family: {str(k): int(v) for k, v in sorted(counter.items())}
            for family, counter in sorted(family_rank_counts.items())
        },
        "mean_selected_coverage": sum(row["coverage"] for row in rows) / len(rows),
        "actual_bit": actual_bit,
        "base_actual_bit": base_actual_bit,
        "actual_bit_detail": actual_detail,
        "base_actual_bit_detail": base_detail,
        "selected_ranks": rows,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-aligned projection-adaptive MBQ low-rank cache."
    )
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help=(
            "Cache containing low-rank entries or linear_shape_map for every "
            "linear in linear_bit_map; required when the base cache lacks shapes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    allocation = parser.add_mutually_exclusive_group(required=True)
    allocation.add_argument("--tau", type=float)
    allocation.add_argument("--target-actual-bit", type=float)
    parser.add_argument("--factor-bit", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ranks = normalize_ranks(args.ranks)
    base_cache = load_cache(args.base_cache)
    analysis = load_analysis(args.analysis)
    validate_source_cache(analysis, args.base_cache)
    rows_by_name = analysis_rows_by_name(analysis)
    reference_cache = load_cache(args.reference_cache) if args.reference_cache else None
    weight_numel_map = infer_weight_numel_map(reference_cache, base_cache)

    if args.target_actual_bit is not None:
        tau, rank_map, rows, _, _ = choose_tau_for_budget(
            base_cache,
            rows_by_name,
            ranks,
            args.target_actual_bit,
            weight_numel_map,
            args.factor_bit,
        )
    else:
        tau = validate_threshold(args.tau)
        rank_map, rows = build_rank_map(base_cache, rows_by_name, ranks, tau)

    summary_path = args.summary or args.output.with_suffix(".adaptive-rank.json")
    summary = summarize(
        base_cache=base_cache,
        rank_map=rank_map,
        rows=rows,
        tau=tau,
        deployable_ranks=ranks,
        factor_bit=args.factor_bit,
        weight_numel_map=weight_numel_map,
        target_actual_bit=args.target_actual_bit,
        base_cache_path=args.base_cache,
        analysis_path=args.analysis,
        output_path=args.output,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Dry run: no cache or summary written.")
        return 0

    if not args.overwrite and (args.output.exists() or summary_path.exists()):
        raise FileExistsError(
            f"output exists: {args.output} or {summary_path}; use --overwrite"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    new_cache = materialize_adaptive_cache(
        base_cache,
        rank_map,
        rows_by_name,
        args.analysis,
        args.factor_bit,
    )
    adaptive_config = {
        "policy": "paper_coverage_threshold",
        "metric_version": METRIC_VERSION,
        "factor_basis": FACTOR_BASIS,
        "tau": float(tau),
        "deployable_ranks": list(ranks),
        "factor_bit": int(args.factor_bit),
        "target_actual_bit": args.target_actual_bit,
        "selected_count": len(rank_map),
        "fixed_after_calibration": True,
    }
    new_cache["adaptive_rank_config"] = adaptive_config
    new_cache["low_rank_config"] = {
        **(new_cache.get("low_rank_config") or {}),
        "adaptive_rank": adaptive_config,
    }

    tmp_cache = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    torch.save(new_cache, tmp_cache)
    tmp_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_cache, args.output)
    os.replace(tmp_summary, summary_path)
    print(f"Saved cache: {args.output}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
