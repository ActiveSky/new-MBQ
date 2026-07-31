#!/usr/bin/env python3
"""Create MBQ cache variants with lower-rank residuals by module family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


DEFAULT_PREFIX = "internvl2_8b_w2g32_scale_reweight_true_svd_1.0"
DEFAULT_OUT_DIR = Path("scale_cache/mbq")
DEFAULT_REFERENCE_CACHE = DEFAULT_OUT_DIR / f"{DEFAULT_PREFIX}_mixed_0.0.pt"
BASELINE_ACTUAL_BIT = 2.3963
MIB_PER_TARGET_BIT = 832.0
MODULE_CHOICES = ("w1", "w2", "w3", "wo", "wqkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an MBQ cache and truncate low-rank SVD factors for one "
            "or more module families."
        )
    )
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument(
        "--module",
        choices=MODULE_CHOICES,
        default=None,
        help="Single target module family; kept for backward compatibility.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Single target rank; kept for backward compatibility.",
    )
    parser.add_argument(
        "--set",
        dest="rank_sets",
        action="append",
        default=[],
        metavar="MODULE:RANK",
        help=(
            "Target module/rank pair. May be repeated, e.g. "
            "--set w1:64 --set w3:64 --set wo:32."
        ),
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=DEFAULT_REFERENCE_CACHE,
        help="Full mixed_0.0 cache used to recover target weight numels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_rank_sets(args: argparse.Namespace) -> Dict[str, int]:
    rank_sets: Dict[str, int] = {}

    if args.module is not None or args.rank is not None:
        if args.module is None or args.rank is None:
            raise ValueError("--module and --rank must be provided together")
        rank_sets[args.module] = int(args.rank)

    for raw_set in args.rank_sets:
        if ":" not in raw_set:
            raise ValueError(f"--set must be MODULE:RANK, got {raw_set!r}")
        module, raw_rank = raw_set.split(":", 1)
        if module not in MODULE_CHOICES:
            raise ValueError(
                f"invalid module {module!r}; expected one of {MODULE_CHOICES}"
            )
        try:
            rank = int(raw_rank)
        except ValueError as exc:
            raise ValueError(f"invalid rank in --set {raw_set!r}") from exc
        if module in rank_sets:
            raise ValueError(f"duplicate target module {module!r}")
        rank_sets[module] = rank

    if not rank_sets:
        raise ValueError("provide --module/--rank or at least one --set MODULE:RANK")

    invalid = {module: rank for module, rank in rank_sets.items() if rank <= 0}
    if invalid:
        raise ValueError(f"target ranks must be positive: {invalid}")

    return rank_sets


def module_key(name: str) -> str:
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
    return "other"


def tensor_pair(item: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    up = item.get("up")
    down = item.get("down")
    if not torch.is_tensor(up) or not torch.is_tensor(down):
        raise TypeError(f"low_rank item lacks tensor up/down: {item.get('name')}")
    return up, down


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def weight_numel_from_low_rank(item: Dict[str, Any]) -> int:
    up, down = tensor_pair(item)
    return int(up.shape[0] * down.shape[1])


def build_weight_numel_map(reference_cache: Dict[str, Any]) -> Dict[str, int]:
    low_rank = reference_cache.get("low_rank") or []
    if not low_rank:
        raise ValueError("reference cache has no low_rank entries")
    return {item["name"]: weight_numel_from_low_rank(item) for item in low_rank}


def low_rank_factor_numel(items: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for item in items:
        up, down = tensor_pair(item)
        total += int(up.numel() + down.numel())
    return total


def compute_actual_bit(
    cache: Dict[str, Any], weight_numel_map: Dict[str, int]
) -> Tuple[float, Dict[str, Any]]:
    linear_bit_map = cache.get("linear_bit_map") or {}
    if not linear_bit_map:
        raise ValueError("cache has no linear_bit_map")

    missing = sorted(set(linear_bit_map) - set(weight_numel_map))
    if missing:
        raise ValueError(f"missing weight numel for {len(missing)} linears: {missing[:5]}")

    denominator = sum(weight_numel_map[name] for name in linear_bit_map)
    weight_bits = sum(
        weight_numel_map[name] * int(bit) for name, bit in linear_bit_map.items()
    )
    low_rank_items = cache.get("low_rank") or []
    low_rank_bits = low_rank_factor_numel(low_rank_items) * 4
    actual_bits = weight_bits + low_rank_bits
    actual_bit = actual_bits / denominator
    detail = {
        "target_weight_numel": denominator,
        "weight_bits": int(weight_bits),
        "low_rank_factor_numel": int(low_rank_bits // 4),
        "low_rank_bits": int(low_rank_bits),
        "actual_bits": int(actual_bits),
    }
    return actual_bit, detail


def truncate_modules(
    cache: Dict[str, Any], rank_sets: Dict[str, int]
) -> Dict[str, Any]:
    old_low_rank = cache.get("low_rank") or []
    new_low_rank = []
    changed: List[Dict[str, Any]] = []
    changed_counts = Counter()

    for item in old_low_rank:
        name = item.get("name", "")
        new_item = dict(item)
        module = module_key(name)
        if module in rank_sets:
            target_rank = rank_sets[module]
            up, down = tensor_pair(item)
            old_rank = int(item.get("rank", up.shape[1]))
            if up.shape[1] < target_rank or down.shape[0] < target_rank:
                raise ValueError(
                    f"{name} cannot be truncated to rank {target_rank}: "
                    f"up={tuple(up.shape)}, down={tuple(down.shape)}"
                )
            new_item["rank"] = int(target_rank)
            new_item["up"] = up[:, :target_rank].contiguous()
            new_item["down"] = down[:target_rank, :].contiguous()
            changed_counts[module] += 1
            changed.append(
                {
                    "name": name,
                    "module": module,
                    "old_rank": old_rank,
                    "new_rank": int(target_rank),
                    "old_up_shape": list(up.shape),
                    "old_down_shape": list(down.shape),
                    "new_up_shape": list(new_item["up"].shape),
                    "new_down_shape": list(new_item["down"].shape),
                }
            )
        new_low_rank.append(new_item)

    missing_modules = sorted(set(rank_sets) - set(changed_counts))
    if missing_modules:
        raise ValueError(f"no low_rank entries found for modules: {missing_modules}")

    cache["low_rank"] = new_low_rank
    cache["module_rank_ablation"] = {
        "target_rank_set": {module: int(rank) for module, rank in sorted(rank_sets.items())},
        "changed_count": len(changed),
        "changed_counts_by_module": {
            module: int(count) for module, count in sorted(changed_counts.items())
        },
    }
    return {
        "changed_count": len(changed),
        "changed_counts_by_module": {
            module: int(count) for module, count in sorted(changed_counts.items())
        },
        "changed_entries": changed,
    }


def summarize(
    cache: Dict[str, Any],
    base_cache: Dict[str, Any],
    args: argparse.Namespace,
    rank_sets: Dict[str, int],
) -> Dict[str, Any]:
    reference_cache = load_cache(args.reference_cache)
    weight_numel_map = build_weight_numel_map(reference_cache)

    actual_bit, actual_detail = compute_actual_bit(cache, weight_numel_map)
    base_actual_bit, base_actual_detail = compute_actual_bit(base_cache, weight_numel_map)

    linear_bit_map = cache.get("linear_bit_map") or {}
    bit_counts = Counter(int(bit) for bit in linear_bit_map.values())
    high_bit_modules = Counter(
        module_key(name) for name, bit in linear_bit_map.items() if int(bit) == 3
    )
    low_rank_items = cache.get("low_rank") or []
    low_rank_modules = Counter(module_key(item["name"]) for item in low_rank_items)
    ranks_by_module: Dict[str, Counter[int]] = {}
    for item in low_rank_items:
        key = module_key(item["name"])
        ranks_by_module.setdefault(key, Counter())
        ranks_by_module[key][int(item["rank"])] += 1

    target_items_by_module = {
        module: [item for item in low_rank_items if module_key(item["name"]) == module]
        for module in rank_sets
    }
    target_factor_numels = {
        module: low_rank_factor_numel(items)
        for module, items in target_items_by_module.items()
    }
    single_module = next(iter(rank_sets)) if len(rank_sets) == 1 else None
    single_items = target_items_by_module[single_module] if single_module else []
    single_factor_numel = (
        target_factor_numels[single_module] if single_module else sum(target_factor_numels.values())
    )

    return {
        "name": args.name,
        "base_cache": str(args.base_cache),
        "reference_cache": str(args.reference_cache),
        "output_cache": str(args.output),
        "target_module": single_module if single_module else "multi",
        "target_rank": int(rank_sets[single_module]) if single_module else None,
        "target_rank_set": {
            module: int(rank) for module, rank in sorted(rank_sets.items())
        },
        "baseline_actual_bit": BASELINE_ACTUAL_BIT,
        "actual_bit": actual_bit,
        "delta_mib_vs_mixed0.3": (actual_bit - BASELINE_ACTUAL_BIT)
        * MIB_PER_TARGET_BIT,
        "base_actual_bit": base_actual_bit,
        "delta_mib_vs_base": (actual_bit - base_actual_bit) * MIB_PER_TARGET_BIT,
        "actual_bit_detail": actual_detail,
        "base_actual_bit_detail": base_actual_detail,
        "linear_count": len(linear_bit_map),
        "bit_counts": {str(k): int(v) for k, v in sorted(bit_counts.items())},
        "module_counts_3bit": dict(sorted(high_bit_modules.items())),
        "low_rank_count": len(low_rank_items),
        "low_rank_module_counts": dict(sorted(low_rank_modules.items())),
        "low_rank_ranks_by_module": {
            key: {str(rank): int(count) for rank, count in sorted(counter.items())}
            for key, counter in sorted(ranks_by_module.items())
        },
        "target_module_low_rank_count": len(single_items),
        "target_module_low_rank_counts": {
            module: len(items)
            for module, items in sorted(target_items_by_module.items())
        },
        "target_module_low_rank_factor_numel": int(single_factor_numel),
        "target_module_low_rank_factor_numels": {
            module: int(numel)
            for module, numel in sorted(target_factor_numels.items())
        },
        "target_module_low_rank_actual_mib_q4": single_factor_numel
        * 4
        / 8
        / 1024
        / 1024,
        "target_module_low_rank_actual_mib_q4_by_module": {
            module: numel * 4 / 8 / 1024 / 1024
            for module, numel in sorted(target_factor_numels.items())
        },
    }


def main() -> None:
    args = parse_args()
    rank_sets = parse_rank_sets(args)
    if args.output is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.output = args.out_dir / f"{DEFAULT_PREFIX}_{args.name}.pt"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    summary_path = args.output.with_suffix(".summary.json")
    if not args.overwrite and (args.output.exists() or summary_path.exists()):
        raise FileExistsError(
            f"output exists: {args.output} or {summary_path}; use --overwrite"
        )

    base_cache = load_cache(args.base_cache)
    new_cache = dict(base_cache)
    truncate_summary = truncate_modules(new_cache, rank_sets)
    summary = summarize(new_cache, base_cache, args, rank_sets)
    summary["truncate_summary"] = {
        "changed_count": truncate_summary["changed_count"],
        "changed_counts_by_module": truncate_summary["changed_counts_by_module"],
        "sample_changed_entries": truncate_summary["changed_entries"][:5],
    }

    torch.save(new_cache, args.output)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved cache: {args.output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
