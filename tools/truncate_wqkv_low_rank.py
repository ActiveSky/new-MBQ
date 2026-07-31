#!/usr/bin/env python3
"""Create MBQ cache variants with lower-rank wqkv low-rank residuals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


DEFAULT_PREFIX = "internvl2_8b_w2g32_scale_reweight_true_svd_1.0"
DEFAULT_OUT_DIR = Path("scale_cache/mbq")
DEFAULT_REFERENCE_CACHE = (
    DEFAULT_OUT_DIR / f"{DEFAULT_PREFIX}_mixed_0.0.pt"
)
BASELINE_ACTUAL_BIT = 2.3963
MIB_PER_TARGET_BIT = 832.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an MBQ cache and truncate only attention.wqkv low-rank "
            "SVD factors to a smaller rank."
        )
    )
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
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


def truncate_wqkv(cache: Dict[str, Any], target_rank: int) -> Dict[str, Any]:
    if target_rank <= 0:
        raise ValueError("--rank must be positive")

    old_low_rank = cache.get("low_rank") or []
    new_low_rank = []
    changed = []

    for item in old_low_rank:
        name = item.get("name", "")
        new_item = dict(item)
        if module_key(name) == "wqkv":
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
            changed.append(
                {
                    "name": name,
                    "old_rank": old_rank,
                    "new_rank": int(target_rank),
                    "old_up_shape": list(up.shape),
                    "old_down_shape": list(down.shape),
                    "new_up_shape": list(new_item["up"].shape),
                    "new_down_shape": list(new_item["down"].shape),
                }
            )
        new_low_rank.append(new_item)

    if not changed:
        raise ValueError("no attention.wqkv low_rank entries found")

    cache["low_rank"] = new_low_rank
    cache["wqkv_rank_ablation"] = {
        "target_rank": int(target_rank),
        "changed_count": len(changed),
    }
    return {
        "changed_count": len(changed),
        "changed_entries": changed,
    }


def summarize(cache: Dict[str, Any], base_cache: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    reference_cache = load_cache(args.reference_cache)
    weight_numel_map = build_weight_numel_map(reference_cache)

    actual_bit, actual_detail = compute_actual_bit(cache, weight_numel_map)
    base_actual_bit, base_actual_detail = compute_actual_bit(base_cache, weight_numel_map)

    linear_bit_map = cache.get("linear_bit_map") or {}
    bit_counts = Counter(int(bit) for bit in linear_bit_map.values())
    high_bit_modules = Counter(
        module_key(name) for name, bit in linear_bit_map.items() if int(bit) == 3
    )
    low_rank_modules = Counter(module_key(item["name"]) for item in cache.get("low_rank") or [])
    ranks_by_module = {}
    for item in cache.get("low_rank") or []:
        key = module_key(item["name"])
        ranks_by_module.setdefault(key, Counter())
        ranks_by_module[key][int(item["rank"])] += 1

    wqkv_items = [
        item for item in cache.get("low_rank") or [] if module_key(item["name"]) == "wqkv"
    ]
    wqkv_factor_numel = low_rank_factor_numel(wqkv_items)

    return {
        "name": args.name,
        "base_cache": str(args.base_cache),
        "reference_cache": str(args.reference_cache),
        "output_cache": str(args.output),
        "target_wqkv_rank": int(args.rank),
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
        "low_rank_count": len(cache.get("low_rank") or []),
        "low_rank_module_counts": dict(sorted(low_rank_modules.items())),
        "low_rank_ranks_by_module": {
            key: {str(rank): int(count) for rank, count in sorted(counter.items())}
            for key, counter in sorted(ranks_by_module.items())
        },
        "wqkv_low_rank_count": len(wqkv_items),
        "wqkv_low_rank_factor_numel": int(wqkv_factor_numel),
        "wqkv_low_rank_actual_mib_q4": wqkv_factor_numel * 4 / 8 / 1024 / 1024,
    }


def main() -> None:
    args = parse_args()
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
    truncate_summary = truncate_wqkv(new_cache, args.rank)
    summary = summarize(new_cache, base_cache, args)
    summary["truncate_summary"] = {
        "changed_count": truncate_summary["changed_count"],
        "sample_changed_entries": truncate_summary["changed_entries"][:5],
    }

    torch.save(new_cache, args.output)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved cache: {args.output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
