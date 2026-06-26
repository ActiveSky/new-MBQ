#!/usr/bin/env python3
"""Build custom MBQ mixed-bit caches from the mixed_0.0 base cache.

The generated cache keeps the existing MBQ scale/search state, edits
``linear_bit_map`` according to a named sensitive-layer policy, and filters
``low_rank`` so selected 3-bit layers do not also receive SVD compensation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


DEFAULT_BASE_CACHE = Path(
    "scale_cache/mbq/internvl2_8b_w2g32_scale_reweight_true_svd_1.0_mixed_0.0.pt"
)
DEFAULT_OUT_DIR = Path("scale_cache/mbq")
DEFAULT_PREFIX = "internvl2_8b_w2g32_scale_reweight_true_svd_1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a custom MBQ cache by editing linear_bit_map."
    )
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--policy",
        required=True,
        choices=[
            "global_topk",
            "wo_core",
            "wo_all_w2_w3",
            "wo_w2",
            "wo_w2_w3",
            "costaware",
            "layer_list",
        ],
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Maximum number of 3-bit layers. Required for top-k style policies.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Short policy name used in output filenames and metadata.",
    )
    parser.add_argument(
        "--layer-list",
        type=Path,
        default=None,
        help="Text file with one full layer name per line for policy=layer_list.",
    )
    parser.add_argument(
        "--drop-layer",
        action="append",
        default=[],
        help="Full layer name to remove after policy selection. Can be repeated.",
    )
    parser.add_argument(
        "--add-layer",
        action="append",
        default=[],
        help="Full layer name to add after policy selection. Can be repeated.",
    )
    parser.add_argument("--high-bit", type=int, default=3)
    parser.add_argument("--default-bit", type=int, default=2)
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting output files."
    )
    return parser.parse_args()


def layer_index(name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", name)
    if not match:
        return -1
    return int(match.group(1))


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
    return name.split(".")[-1]


def module_family(name: str, score_meta: Dict) -> str:
    if score_meta.get("module_family"):
        return str(score_meta["module_family"])
    if module_key(name) == "wqkv":
        return "attn_in"
    if module_key(name) == "wo":
        return "attn_out"
    if module_key(name) == "w2":
        return "mlp_out"
    if module_key(name) in {"w1", "w3"}:
        return "mlp_in"
    return "unknown"


def score_of(name: str, score_map: Dict[str, Dict]) -> float:
    return float(score_map[name]["score"])


def sorted_by_score(names: Iterable[str], score_map: Dict[str, Dict]) -> List[str]:
    return sorted(names, key=lambda name: score_of(name, score_map), reverse=True)


def load_cache(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def require_budget(policy: str, budget: int | None) -> int:
    if budget is None:
        raise ValueError(f"--budget is required for policy={policy}")
    if budget <= 0:
        raise ValueError("--budget must be positive")
    return int(budget)


def take_unique(
    selected: List[str], candidates: Sequence[str], budget: int | None = None
) -> List[str]:
    seen = set(selected)
    for name in candidates:
        if name in seen:
            continue
        selected.append(name)
        seen.add(name)
        if budget is not None and len(selected) >= budget:
            break
    return selected


def select_global_topk(score_map: Dict[str, Dict], budget: int) -> List[str]:
    return sorted_by_score(score_map.keys(), score_map)[:budget]


def select_wo_core(score_map: Dict[str, Dict], budget: int | None) -> List[str]:
    core_layers = set(range(7, 18)) | {20, 21, 22, 24, 25, 26}
    candidates = [
        name
        for name in score_map
        if module_key(name) == "wo" and layer_index(name) in core_layers
    ]
    selected = sorted_by_score(candidates, score_map)
    if budget is not None:
        selected = selected[:budget]
    return selected


def select_wo_all(score_map: Dict[str, Dict]) -> List[str]:
    return sorted_by_score(
        [name for name in score_map if module_key(name) == "wo"], score_map
    )


def select_wo_w2(score_map: Dict[str, Dict], budget: int) -> List[str]:
    selected: List[str] = []
    wo_core = select_wo_core(score_map, budget=None)
    w2_primary_layers = {3, *range(5, 14)}
    w2_secondary_layers = {1, 2, *range(16, 25), 26, 27, 28, 29, 0, 4, 14, 15}
    w2_primary = sorted_by_score(
        [
            name
            for name in score_map
            if module_key(name) == "w2" and layer_index(name) in w2_primary_layers
        ],
        score_map,
    )
    w2_secondary = sorted_by_score(
        [
            name
            for name in score_map
            if module_key(name) == "w2" and layer_index(name) in w2_secondary_layers
        ],
        score_map,
    )
    w2_all = sorted_by_score(
        [name for name in score_map if module_key(name) == "w2"], score_map
    )

    take_unique(selected, wo_core, budget)
    take_unique(selected, w2_primary, budget)
    take_unique(selected, w2_secondary, budget)
    take_unique(selected, w2_all, budget)
    if len(selected) < budget:
        take_unique(selected, select_global_topk(score_map, len(score_map)), budget)
    return selected[:budget]


def select_wo_w2_w3(score_map: Dict[str, Dict], budget: int) -> List[str]:
    selected = select_wo_w2(score_map, min(budget, max(0, budget - 8)))

    w3_priority_layers = set(range(2, 10)) | set(range(21, 28))
    w3_priority = sorted_by_score(
        [
            name
            for name in score_map
            if module_key(name) == "w3" and layer_index(name) in w3_priority_layers
        ],
        score_map,
    )
    w3_all = sorted_by_score(
        [name for name in score_map if module_key(name) == "w3"], score_map
    )

    take_unique(selected, w3_priority, budget)
    take_unique(selected, w3_all, budget)
    if len(selected) < budget:
        take_unique(selected, select_global_topk(score_map, len(score_map)), budget)
    return selected[:budget]


def select_wo_all_w2_w3(score_map: Dict[str, Dict], budget: int) -> List[str]:
    selected: List[str] = []
    take_unique(selected, select_wo_all(score_map), budget)

    remaining_for_w2_before_w3 = max(0, budget - len(selected) - 8)
    if remaining_for_w2_before_w3:
        take_unique(
            selected,
            select_wo_w2(score_map, len(selected) + remaining_for_w2_before_w3),
            len(selected) + remaining_for_w2_before_w3,
        )

    w3_priority_layers = set(range(2, 10)) | set(range(21, 28))
    w3_priority = sorted_by_score(
        [
            name
            for name in score_map
            if module_key(name) == "w3" and layer_index(name) in w3_priority_layers
        ],
        score_map,
    )
    w3_all = sorted_by_score(
        [name for name in score_map if module_key(name) == "w3"], score_map
    )

    take_unique(selected, w3_priority, budget)
    take_unique(selected, w3_all, budget)
    take_unique(selected, select_wo_w2(score_map, budget), budget)
    if len(selected) < budget:
        take_unique(selected, select_global_topk(score_map, len(score_map)), budget)
    return selected[:budget]


def low_rank_maps(low_rank: Sequence[Dict]) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    lr_by_name = {}
    lr_bytes = {}
    for item in low_rank:
        name = item["name"]
        lr_by_name[name] = item
        tensors = [item.get("up"), item.get("down")]
        lr_bytes[name] = sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if torch.is_tensor(tensor)
        )
    return lr_by_name, lr_bytes


def weight_numel_from_low_rank(item: Dict) -> int:
    up = item.get("up")
    down = item.get("down")
    if not torch.is_tensor(up) or not torch.is_tensor(down):
        return 0
    return int(up.shape[0] * down.shape[1])


def net_cost_bytes(name: str, lr_by_name: Dict[str, Dict], lr_bytes: Dict[str, int]) -> float:
    item = lr_by_name.get(name)
    if item is None:
        return 0.0
    extra_base_bytes = weight_numel_from_low_rank(item) / 8.0
    return extra_base_bytes - float(lr_bytes.get(name, 0))


def select_costaware(
    score_map: Dict[str, Dict], low_rank: Sequence[Dict], budget: int
) -> List[str]:
    lr_by_name, lr_bytes = low_rank_maps(low_rank)

    def key(name: str) -> Tuple[float, float]:
        net = net_cost_bytes(name, lr_by_name, lr_bytes)
        # A negative net cost means promoting the layer may reduce cache size after
        # removing low-rank. Treat it as especially attractive, but keep score as a
        # tiebreaker so tiny low-score layers do not dominate entirely.
        if net <= 0:
            return (float("inf"), score_of(name, score_map))
        return (score_of(name, score_map) / net, score_of(name, score_map))

    return sorted(score_map.keys(), key=key, reverse=True)[:budget]


def select_from_layer_list(path: Path, score_map: Dict[str, Dict]) -> List[str]:
    if path is None:
        raise ValueError("--layer-list is required for policy=layer_list")
    selected = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in score_map:
            raise ValueError(f"Layer from list not found in score_map: {line}")
        selected.append(line)
    return selected


def build_selection(args: argparse.Namespace, cache: Dict) -> List[str]:
    score_map = cache.get("linear_score_map") or {}
    if not score_map:
        raise ValueError("Base cache does not contain linear_score_map")

    if args.policy == "global_topk":
        return select_global_topk(score_map, require_budget(args.policy, args.budget))
    if args.policy == "wo_core":
        return select_wo_core(score_map, args.budget)
    if args.policy == "wo_all_w2_w3":
        return select_wo_all_w2_w3(
            score_map, require_budget(args.policy, args.budget)
        )
    if args.policy == "wo_w2":
        return select_wo_w2(score_map, require_budget(args.policy, args.budget))
    if args.policy == "wo_w2_w3":
        return select_wo_w2_w3(score_map, require_budget(args.policy, args.budget))
    if args.policy == "costaware":
        return select_costaware(
            score_map, cache.get("low_rank") or [], require_budget(args.policy, args.budget)
        )
    if args.policy == "layer_list":
        selected = select_from_layer_list(args.layer_list, score_map)
        if args.budget is not None:
            selected = selected[: args.budget]
        return selected
    raise ValueError(f"Unsupported policy: {args.policy}")


def apply_manual_edits(
    selected: Sequence[str], args: argparse.Namespace, score_map: Dict[str, Dict]
) -> List[str]:
    selected_list = list(selected)
    selected_set = set(selected_list)

    for name in args.drop_layer:
        if name not in score_map:
            raise ValueError(f"--drop-layer not found in score_map: {name}")
        selected_set.discard(name)

    for name in args.add_layer:
        if name not in score_map:
            raise ValueError(f"--add-layer not found in score_map: {name}")
        selected_set.add(name)

    edited = [name for name in selected_list if name in selected_set]
    for name in args.add_layer:
        if name not in edited:
            edited.append(name)
    return edited


def summarize(
    selected: Sequence[str], cache: Dict, output_path: Path, args: argparse.Namespace
) -> Dict:
    selected_set = set(selected)
    score_map = cache.get("linear_score_map") or {}
    old_low_rank = cache.get("low_rank") or []
    lr_by_name, lr_bytes = low_rank_maps(old_low_rank)

    extra_base_bytes_by_name = {}
    removed_lr_bytes_by_name = {}
    for name in selected:
        item = lr_by_name.get(name)
        extra_base_bytes_by_name[name] = (
            weight_numel_from_low_rank(item) / 8.0 if item is not None else 0.0
        )
        removed_lr_bytes_by_name[name] = float(lr_bytes.get(name, 0))

    modules = Counter(module_key(name) for name in selected)
    families = Counter(module_family(name, score_map[name]) for name in selected)
    bit_counts = Counter(
        args.high_bit if name in selected_set else args.default_bit for name in score_map
    )

    selected_rows = []
    rank_by_name = {
        name: rank
        for rank, name in enumerate(sorted_by_score(score_map.keys(), score_map), start=1)
    }
    for name in sorted(selected, key=lambda item: rank_by_name.get(item, 9999)):
        selected_rows.append(
            {
                "rank": rank_by_name.get(name),
                "name": name,
                "layer": layer_index(name),
                "module": module_key(name),
                "family": module_family(name, score_map[name]),
                "score": score_of(name, score_map),
                "extra_base_bytes": extra_base_bytes_by_name[name],
                "removed_low_rank_bytes": removed_lr_bytes_by_name[name],
                "net_bytes": extra_base_bytes_by_name[name]
                - removed_lr_bytes_by_name[name],
            }
        )

    total_extra = sum(extra_base_bytes_by_name.values())
    total_removed = sum(removed_lr_bytes_by_name.values())
    return {
        "policy": args.policy,
        "name": args.name,
        "base_cache": str(args.base_cache),
        "output_cache": str(output_path),
        "high_bit": args.high_bit,
        "default_bit": args.default_bit,
        "manual_drop_layer": list(args.drop_layer),
        "manual_add_layer": list(args.add_layer),
        "selected_count": len(selected),
        "linear_count": len(score_map),
        "bit_counts": {str(k): int(v) for k, v in sorted(bit_counts.items())},
        "old_low_rank_count": len(old_low_rank),
        "new_low_rank_count": len(old_low_rank) - len(selected_set),
        "module_counts": dict(sorted(modules.items())),
        "family_counts": dict(sorted(families.items())),
        "extra_base_bytes": total_extra,
        "removed_low_rank_bytes": total_removed,
        "net_bytes": total_extra - total_removed,
        "extra_base_mib": total_extra / 1024 / 1024,
        "removed_low_rank_mib": total_removed / 1024 / 1024,
        "net_mib": (total_extra - total_removed) / 1024 / 1024,
        "selected_layers": selected_rows,
    }


def main() -> None:
    args = parse_args()
    if not args.base_cache.exists():
        raise FileNotFoundError(args.base_cache)

    name = args.name
    if name is None:
        if args.budget is None:
            name = f"custom_{args.policy}"
        else:
            name = f"custom_{args.policy}_{args.budget}"
    args.name = name

    if args.output is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.out_dir / f"{DEFAULT_PREFIX}_{name}.pt"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output_path.with_suffix(".summary.json")
    if not args.overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Output exists: {output_path} or {summary_path}. Use --overwrite if intended."
        )

    cache = load_cache(args.base_cache)
    score_map = cache.get("linear_score_map") or {}
    selected = build_selection(args, cache)
    selected = apply_manual_edits(selected, args, score_map)
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("Selection contains duplicate layer names")
    missing = selected_set - set(score_map.keys())
    if missing:
        raise ValueError(f"Selected layers missing from score_map: {sorted(missing)}")

    linear_bit_map = {
        name: int(args.high_bit if name in selected_set else args.default_bit)
        for name in score_map.keys()
    }
    old_low_rank = cache.get("low_rank") or []
    new_low_rank = [item for item in old_low_rank if item.get("name") not in selected_set]

    summary = summarize(selected, cache, output_path, args)

    cache["linear_bit_map"] = linear_bit_map
    cache["low_rank"] = new_low_rank
    cache["linear_mixed_config"] = {
        **(cache.get("linear_mixed_config") or {}),
        "high_bit": int(args.high_bit),
        "keep_ratio": float(len(selected_set) / max(1, len(score_map))),
        "exclusive_with_low_rank": True,
        "custom_policy": args.policy,
        "custom_name": args.name,
        "selected_count": len(selected_set),
        "manual_drop_layer": list(args.drop_layer),
        "manual_add_layer": list(args.add_layer),
    }

    cache["custom_policy_summary"] = {
        key: value
        for key, value in summary.items()
        if key not in {"selected_layers"}
    }

    torch.save(cache, output_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved cache: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
