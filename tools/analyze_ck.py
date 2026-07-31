"""离线分析：从 collect_act_for_ck.py 落盘的中间量计算 $c_k$ 并做判别。

优先读取 {output_dir}/{idx}_{short}/ 下的紧凑 component_stats.pt 和
        sigma.pt；旧采集则回退到 X.pt / masks.pt / V.pt。对每个 module 计算：

1. c_k = sigma_k^2 * (
       ||X^{Omega_t} v_k||^2 / D_m(Omega_t)
       + rho * ||X^{Omega_v} v_k||^2 / D_m(Omega_v)
   )
2. 累积能量曲线 sum_{i<=k} sigma_i^2 / sum sigma^2
3. 累积 c_k 曲线     sum_{i<=k} c_i / sum c_k
4. Spearman / Kendall 相关度 between sigma_k^2 与 c_k（描述性证据，注意
   c_k 内含 sigma_k^2 因子，相关度天然偏高，只作参考）
5. 判别性证据：在真实激活下比较
     - top-sigma: 取 sigma 最大的前 r 个分量重构 R_hat
     - top-c    : 取 c_k 最大的前 r 个分量重构 R_hat
   误差指标 = ||X (R - R_hat)||_F^2 （分 text/vis 两个模态报）
   这是真正回答「换排序有没有收益」的指标。

输出：
    {output_dir}/_ck_analysis.json : 全部数值结果
    {output_dir}/_ck_curves.png    : 累积能量 vs 累积 c_k 曲线（每个 module 一张子图）
    {output_dir}/_ck_summary.csv   : per-module 汇总表

用法
----
python tools/analyze_ck.py --input_dir act_for_ck/wo_all_w2_w3_48 \
    --ranks 32 64 128 [--rho_override attn_in=1.5,mlp_out=0.8]
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
def _load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def component_input_energies(X, text_mask, vis_mask, Vh):
    """Compute per-direction ||X_text v_k||^2 and ||X_vis v_k||^2."""
    text_rows = X[text_mask]
    vis_rows = X[vis_mask]
    projected_text = text_rows @ Vh.t() if text_rows.numel() else None
    projected_vis = vis_rows @ Vh.t() if vis_rows.numel() else None
    text_energy = (
        projected_text.pow(2).sum(dim=0)
        if projected_text is not None
        else torch.zeros(Vh.shape[0], device=Vh.device)
    )
    vis_energy = (
        projected_vis.pow(2).sum(dim=0)
        if projected_vis is not None
        else torch.zeros(Vh.shape[0], device=Vh.device)
    )
    return text_energy, vis_energy


def load_module_dir(d, device="cpu"):
    meta = json.loads(Path(d, "meta.json").read_text())
    S = _load_tensor(os.path.join(d, "sigma.pt")).float().to(device)
    compact_path = os.path.join(d, "component_stats.pt")
    if os.path.exists(compact_path):
        stats = _load_tensor(compact_path)
        text_energy = stats["input_energy_text"].double().to(device)
        vis_energy = stats["input_energy_vis"].double().to(device)
        source = "component_stats"
    else:
        X = _load_tensor(os.path.join(d, "X.pt")).float().to(device)
        masks = {
            key: value.to(device)
            for key, value in _load_tensor(os.path.join(d, "masks.pt")).items()
        }
        Vh = _load_tensor(os.path.join(d, "V.pt")).float().to(device)
        text_mask = masks.get("text", masks.get("ans"))
        if text_mask is None:
            raise KeyError(f"{d}/masks.pt has neither 'text' nor legacy 'ans'")
        text_energy, vis_energy = component_input_energies(
            X, text_mask.bool(), masks["vis"].bool(), Vh
        )
        source = "legacy_raw_tensors"
    if text_energy.numel() != S.numel() or vis_energy.numel() != S.numel():
        raise ValueError(
            f"component-stat length mismatch in {d}: sigma={S.numel()}, "
            f"text={text_energy.numel()}, vis={vis_energy.numel()}"
        )
    return meta, S, text_energy, vis_energy, source


def compute_ck_from_component_energies(
    S,
    text_energy,
    vis_energy,
    rho,
    denominator_text,
    denominator_vis,
):
    """Compute paper-normalized c_k from compact per-direction statistics."""
    if denominator_text <= 0 or denominator_vis <= 0:
        raise ValueError("modality denominators must be positive")
    return S.pow(2) * (
        text_energy / float(denominator_text)
        + float(rho) * vis_energy / float(denominator_vis)
    )


def compute_ck(
    S,
    Vh,
    X,
    text_mask,
    vis_mask,
    rho,
    denominator_text,
    denominator_vis,
):
    """Compute the modality-normalized c_k used by the paper.

    S: [r], Vh: [r, in]（第 k 行是 v_k）, X: [n, in]。
    返回 c: [r]。"""
    text_energy, vis_energy = component_input_energies(
        X, text_mask, vis_mask, Vh
    )
    return compute_ck_from_component_energies(
        S,
        text_energy,
        vis_energy,
        rho,
        denominator_text,
        denominator_vis,
    )


def reconstruct_resid_error(
    S,
    Vh,
    X,
    text_mask,
    vis_mask,
    rho,
    denominator_text,
    denominator_vis,
    order,
    r,
    numerator_text=None,
    numerator_vis=None,
):
    """按给定分量顺序 order 取前 r 个重构 R_hat，计算激活下输出误差。

    R = U S V^T；但 U 没存。激活下输出误差：
        err = sum_k' over (未被选中的分量) sigma_k'^2 * ||X v_k'||^2
    因为 ||X (R - R_hat)||_F^2 = sum_{k not in selected} sigma_k^2 ||X v_k||^2
    （V 正交，cross term 为 0）。所以用 c_k 的分模态版直接算残差误差。
    """
    text_energy, vis_energy = component_input_energies(
        X, text_mask, vis_mask, Vh
    )
    return reconstruct_resid_error_from_component_energies(
        S,
        text_energy,
        vis_energy,
        rho,
        denominator_text,
        denominator_vis,
        order,
        r,
        numerator_text=numerator_text,
        numerator_vis=numerator_vis,
    )


def reconstruct_resid_error_from_component_energies(
    S,
    text_energy,
    vis_energy,
    rho,
    denominator_text,
    denominator_vis,
    order,
    r,
    numerator_text=None,
    numerator_vis=None,
):
    """Evaluate a selected component set, including any unobserved SVD tail."""
    e_text = S.pow(2) * text_energy
    e_vis = S.pow(2) * vis_energy

    selected = set(order[:r].tolist())
    keep_mask = torch.ones(S.numel(), dtype=torch.bool, device=S.device)
    keep_mask[list(selected)] = False
    # Randomized SVD may store only a prefix. The omitted tail is common to
    # both orderings and must remain in the error denominator.
    tail_text = (
        max(0.0, float(numerator_text) - float(e_text.sum()))
        if numerator_text is not None
        else 0.0
    )
    tail_vis = (
        max(0.0, float(numerator_vis) - float(e_vis.sum()))
        if numerator_vis is not None
        else 0.0
    )
    err_text = float(e_text[keep_mask].sum()) + tail_text
    err_vis = float(e_vis[keep_mask].sum()) + tail_vis
    err_total = (
        err_text / float(denominator_text)
        + rho * err_vis / float(denominator_vis)
    )
    return err_total, err_text, err_vis


def spearman(a, b):
    """Spearman rank correlation。优先用 scipy（C 实现，秒级）；否则手写 O(n log n) 版。"""
    if a.numel() < 2 or torch.all(a == a[0]) or torch.all(b == b[0]):
        return 0.0
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a.numpy().astype(np.float64), b.numpy().astype(np.float64))
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        a = a.double(); b = b.double()
        ra = a.argsort().argsort().double()
        rb = b.argsort().argsort().double()
        ra = ra - ra.mean(); rb = rb - rb.mean()
        denom = (ra.norm() * rb.norm())
        return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def kendall(a, b):
    """Kendall-tau。优先 scipy（C 实现）；否则返回 0 避免爆炸 O(n²) 循环。"""
    if a.numel() < 2 or torch.all(a == a[0]) or torch.all(b == b[0]):
        return 0.0
    try:
        from scipy.stats import kendalltau
        r, _ = kendalltau(a.numpy().astype(np.float64), b.numpy().astype(np.float64))
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--rho_override", default=None,
                    help="覆盖 rho，格式 attn_in=1.5,mlp_out=0.8")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="Device for c_k matrix products. CPU remains the default.")
    ap.add_argument(
        "--allow-legacy-unnormalized",
        action="store_true",
        help=(
            "Analyze old collections that lack D_text/D_vis. This mode is kept "
            "for diagnostics only and is rejected by the adaptive-rank builder."
        ),
    )
    args = ap.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    gmeta = json.loads(Path(args.input_dir, "_global_meta.json").read_text())
    rho_override = {}
    if args.rho_override:
        for kv in args.rho_override.split(","):
            k, v = kv.split("=")
            rho_override[k.strip()] = float(v)

    per_module = []

    for mi in gmeta["module_index"]:
        d = os.path.join(args.input_dir, f"{mi['idx']}_{mi['short']}")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing module statistics directory: {d}")
        print(f"[analyze] {mi['idx']}/{len(gmeta['module_index'])} {mi['short']}", flush=True)
        meta, S, text_energy, vis_energy, statistics_source = load_module_dir(
            d, device=args.device
        )
        rho = rho_override.get(mi["family"], mi["rho"])

        normalized = (
            int(meta.get("metric_version", 0)) >= 2
            and meta.get("denominator_text") is not None
            and meta.get("denominator_vis") is not None
            and meta.get("numerator_text") is not None
            and meta.get("numerator_vis") is not None
            and meta.get("residual_frobenius_sq") is not None
        )
        if not normalized and not args.allow_legacy_unnormalized:
            raise ValueError(
                f"{d}/meta.json lacks required paper-normalized statistics. "
                "Recollect with tools/collect_act_for_ck.py or pass "
                "--allow-legacy-unnormalized for diagnostic-only output."
            )
        denominator_text = float(meta.get("denominator_text", 1.0))
        denominator_vis = float(meta.get("denominator_vis", 1.0))

        c = compute_ck_from_component_energies(
            S,
            text_energy,
            vis_energy,
            rho,
            denominator_text,
            denominator_vis,
        )
        energy = S.pow(2)
        residual_frobenius_sq = float(
            meta.get("residual_frobenius_sq", energy.sum().item())
        )
        if residual_frobenius_sq <= 0:
            raise ValueError(f"non-positive residual energy for {mi['name']}")
        cum_energy = (energy.cumsum(0) / residual_frobenius_sq).cpu().numpy()
        if normalized:
            numerator_text = float(meta["numerator_text"])
            numerator_vis = float(meta["numerator_vis"])
            projection_score = (
                numerator_text / denominator_text
                + float(rho) * numerator_vis / denominator_vis
            )
        else:
            projection_score = float(c.sum().item())
        if projection_score <= 0:
            raise ValueError(f"non-positive projection score for {mi['name']}")
        cum_c = (c.cumsum(0) / projection_score).cpu().numpy()

        sp = spearman(energy.cpu(), c.cpu())
        kd = kendall(energy.cpu(), c.cpu())

        # 各 rank 的覆盖率
        cov_e = {
            r: float(cum_energy[r - 1]) for r in args.ranks if r <= S.numel()
        }
        cov_c = {r: float(cum_c[r - 1]) for r in args.ranks if r <= S.numel()}

        # 判别性证据：top-sigma vs top-c 在各 rank 下的输出残差误差
        order_sigma = torch.argsort(S, descending=True)
        order_c = torch.argsort(c, descending=True)
        judge = {}
        for r in args.ranks:
            if r > S.numel():
                continue
            et_s, ea_s, ev_s = reconstruct_resid_error_from_component_energies(
                S,
                text_energy,
                vis_energy,
                rho,
                denominator_text,
                denominator_vis,
                order_sigma,
                r,
                numerator_text=(float(meta["numerator_text"]) if normalized else None),
                numerator_vis=(float(meta["numerator_vis"]) if normalized else None),
            )
            et_c, ea_c, ev_c = reconstruct_resid_error_from_component_energies(
                S,
                text_energy,
                vis_energy,
                rho,
                denominator_text,
                denominator_vis,
                order_c,
                r,
                numerator_text=(float(meta["numerator_text"]) if normalized else None),
                numerator_vis=(float(meta["numerator_vis"]) if normalized else None),
            )
            # 改善率：top-c 相对 top-sigma 的误差下降比例（>0 说明 top-c 更好）
            improve = (et_s - et_c) / (et_s + 1e-12)
            judge[r] = {
                "err_top_sigma": et_s, "err_top_c": et_c,
                "improve_ratio": improve,
                "err_top_sigma_text": ea_s, "err_top_c_text": ea_c,
                "err_top_sigma_ans": ea_s, "err_top_c_ans": ea_c,
                "err_top_sigma_vis": ev_s, "err_top_c_vis": ev_c,
            }

        per_module.append({
            "idx": mi["idx"], "name": mi["name"], "short": mi["short"],
            "family": mi["family"],
            "bit": mi["bit"], "rho": rho,
            "metric_version": 2 if normalized else 1,
            "coverage_normalized": bool(normalized),
            "mask_semantics": meta.get("mask_semantics", "legacy_answer_text"),
            "statistics_source": statistics_source,
            "denominator_text": denominator_text,
            "denominator_vis": denominator_vis,
            "residual_frobenius_sq": residual_frobenius_sq,
            "score_E": projection_score,
            "available_component_score": float(c.sum().item()),
            "available_component_fraction": float(c.sum().item() / projection_score),
            "weight_shape": meta.get("weight_shape"),
            "factor_prefix_rank": meta.get("factor_prefix_rank"),
            "factor_prefix_file": (
                f"{Path(d).name}/{meta['factor_prefix_file']}"
                if meta.get("factor_prefix_file")
                else None
            ),
            "factor_prefix_basis": meta.get("factor_prefix_basis"),
            "token_subsampled": bool(meta.get("token_subsampled", False)),
            "spearman": sp, "kendall": kd,
            "cov_energy": cov_e, "cov_c": cov_c,
            "judge": judge,
            "cum_energy": cum_energy.tolist(), "cum_c": cum_c.tolist(),
        })

    # ---- 汇总 -----------------------------------------------------------
    if not per_module:
        raise ValueError(f"no module statistics found under {args.input_dir}")
    sp_all = [m["spearman"] for m in per_module]
    kd_all = [m["kendall"] for m in per_module]
    summary = {
        "num_modules": len(per_module),
        "spearman_mean": float(np.mean(sp_all)),
        "spearman_median": float(np.median(sp_all)),
        "spearman_min": float(np.min(sp_all)),
        "spearman_max": float(np.max(sp_all)),
        "kendall_mean": float(np.mean(kd_all)),
        "kendall_median": float(np.median(kd_all)),
        "ranks": args.ranks,
    }
    # 各 rank 下 improve_ratio 的统计
    for r in args.ranks:
        imps = [m["judge"][r]["improve_ratio"] for m in per_module if r in m["judge"]]
        if imps:
            summary[f"improve_ratio_r{r}_mean"] = float(np.mean(imps))
            summary[f"improve_ratio_r{r}_median"] = float(np.median(imps))
            summary[f"improve_ratio_r{r}_pos_frac"] = float(
                np.mean([1 if x > 0 else 0 for x in imps]))

    all_normalized = bool(per_module) and all(
        module["coverage_normalized"] for module in per_module
    )
    mask_semantics = sorted({module["mask_semantics"] for module in per_module})
    factor_bases = sorted({
        module["factor_prefix_basis"]
        for module in per_module
        if module.get("factor_prefix_basis")
    })
    token_subsampling = any(
        module["token_subsampled"] for module in per_module
    )
    out = {
        "metric": {
            "version": 2 if all_normalized else 1,
            "normalized": all_normalized,
            "basis_order": "energy_ordered_svd_prefix",
            "coverage_denominator": (
                "projection_score_E" if all_normalized else "legacy_component_sum"
            ),
            "energy_coverage_denominator": (
                "full_residual_frobenius_energy"
                if all_normalized
                else "available_svd_component_energy"
            ),
            "component_statistics": "input_energy_by_svd_direction",
            "mask_semantics": mask_semantics,
            "factor_basis": factor_bases,
            "token_subsampling": token_subsampling,
        },
        "collection": {
            "source_cache": gmeta.get("scale_path"),
            "config": gmeta.get("config"),
            "reweight_cache": gmeta.get("reweight_cache"),
            "n_samples": gmeta.get("n_samples"),
            "max_tokens": gmeta.get("max_tokens"),
            "svd_full_max_dim": gmeta.get("svd_full_max_dim"),
            "svd_q": gmeta.get("svd_q"),
            "svd_niter": gmeta.get("svd_niter"),
            "factor_prefix_rank": gmeta.get("factor_prefix_rank"),
        },
        "summary": summary,
        "per_module": per_module,
        "rho_override": rho_override,
    }
    with open(os.path.join(args.input_dir, "_ck_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- CSV 汇总表 -----------------------------------------------------
    with open(os.path.join(args.input_dir, "_ck_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["name", "family", "bit", "rho", "spearman", "kendall"]
        for r in args.ranks:
            header += [f"covE_r{r}", f"covC_r{r}", f"improve_r{r}"]
        w.writerow(header)
        for m in per_module:
            row = [m["name"], m["family"], m["bit"], m["rho"],
                   f"{m['spearman']:.4f}", f"{m['kendall']:.4f}"]
            for r in args.ranks:
                if r in m["judge"]:
                    row += [f"{m['cov_energy'][r]:.4f}",
                            f"{m['cov_c'][r]:.4f}",
                            f"{m['judge'][r]['improve_ratio']:.4f}"]
                else:
                    row += ["", "", ""]
            w.writerow(row)

    # ---- 打印关键结论 ---------------------------------------------------
    print("=" * 60)
    print(f"模块数: {summary['num_modules']}")
    print(f"Spearman(σ², c_k): mean={summary['spearman_mean']:.3f} "
          f"median={summary['spearman_median']:.3f} "
          f"[min={summary['spearman_min']:.3f}, max={summary['spearman_max']:.3f}]")
    print(f"Kendall(σ², c_k):   mean={summary['kendall_mean']:.3f} "
          f"median={summary['kendall_median']:.3f}")
    for r in args.ranks:
        key = f"improve_ratio_r{r}_mean"
        if key in summary:
            print(f"  rank={r}: top-c 相对 top-σ 误差改善 "
                  f"mean={summary[key]:+.4f}, "
                  f"中位={summary[f'improve_ratio_r{r}_median']:+.4f}, "
                  f"正改善占比={summary[f'improve_ratio_r{r}_pos_frac']:.2%}")
    print("=" * 60)
    print(f"结果: {os.path.join(args.input_dir, '_ck_analysis.json')}")
    print(f"表格: {os.path.join(args.input_dir, '_ck_summary.csv')}")

    # ---- 曲线图 ---------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(per_module)
            ncols = 6
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
            axes = np.array(axes).reshape(-1)
            for ax, m in zip(axes, per_module):
                ax.plot(m["cum_energy"], label="cum σ² energy", lw=1.5)
                ax.plot(m["cum_c"], label="cum c_k", lw=1.5, ls="--")
                for r in args.ranks:
                    if r < len(m["cum_energy"]):
                        ax.axvline(r, color="gray", ls=":", lw=0.7)
                ax.set_title(f"{m['name'].split('layers.')[-1]}\n"
                             f"sp={m['spearman']:.2f}", fontsize=7)
                ax.tick_params(labelsize=6)
            for ax in axes[n:]:
                ax.axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(os.path.join(args.input_dir, "_ck_curves.png"), dpi=120)
            print(f"曲线: {os.path.join(args.input_dir, '_ck_curves.png')}")
        except Exception as e:
            print(f"绘图跳过: {e}")


if __name__ == "__main__":
    main()
