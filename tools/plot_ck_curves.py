"""为 MARS-VL c_k 分析导出代表性 module 的 cumulative 对比大图（论文用）。

从 _ck_analysis.json 读取 cum_energy / cum_c，挑代表 module（improve
最小/中位/最大），每张图独立导出。颜色遵循 dataviz brand-neutral palette：
  - cum σ² energy : slot 1 blue   #2a78d6
  - cum c_k       : slot 2 aqua   #1baf7a  (light-mode contrast <3:1 → 直接标签)
"""
import json
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# brand-neutral palette (light mode, validated: worst adjacent CVD ΔE 24.2)
COLOR_ENERGY = "#2a78d6"   # slot 1 blue
COLOR_CK = "#1baf7a"       # slot 2 aqua
COLOR_RANK = "#9a9a93"     # neutral gridline gray for rank markers
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def select_representatives(per_module, ranks):
    """挑 improve_r_max 最小 2、中位 1、最大 2 共 5 个代表。"""
    rmax = max(ranks)
    by_imp = sorted(per_module, key=lambda m: m["judge"][str(rmax)]["improve_ratio"])
    picks = []
    picks.append(by_imp[0])          # improve 最小
    picks.append(by_imp[1])          # 次小
    picks.append(by_imp[len(by_imp) // 2])  # 中位
    picks.append(by_imp[-2])         # 次大
    picks.append(by_imp[-1])         # 最大
    return picks


def plot_one(m, ranks, out_path):
    cum_e = np.asarray(m["cum_energy"])
    cum_c = np.asarray(m["cum_c"])
    x = np.arange(1, len(cum_e) + 1)
    rmax = max(ranks)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(x, cum_e, color=COLOR_ENERGY, lw=2.2, label=r"cum $\sigma_k^2$ energy")
    ax.plot(x, cum_c, color=COLOR_CK, lw=2.2, ls="--",
            label=r"cum $c_k$ (cross-modal)")

    # rank 竖线
    for r in ranks:
        if r < len(cum_e):
            ax.axvline(r, color=COLOR_RANK, ls=":", lw=1.0, zorder=0)
            ax.text(r, 0.04, f"r={r}", color=TEXT_SECONDARY, fontsize=8,
                    ha="center", va="bottom", rotation=0)

    # 直接标签（relief rule：aqua 对比度 <3:1，必须有可见标签）
    # 在曲线 90% 高度处标系列名
    idx_e = int(np.argmax(cum_e >= 0.9)) if (cum_e >= 0.9).any() else len(cum_e) - 1
    ax.annotate(r"$\sigma_k^2$", xy=(idx_e + 1, cum_e[idx_e]),
                xytext=(8, 0), textcoords="offset points",
                color=COLOR_ENERGY, fontsize=10, fontweight="bold", va="center")
    idx_c = int(np.argmax(cum_c >= 0.9)) if (cum_c >= 0.9).any() else len(cum_c) - 1
    ax.annotate(r"$c_k$", xy=(idx_c + 1, cum_c[idx_c]),
                xytext=(8, 0), textcoords="offset points",
                color=COLOR_CK, fontsize=10, fontweight="bold", va="center")

    short = m["name"].split("layers.")[-1]
    sp = m["spearman"]
    kd = m["kendall"]
    imp = m["judge"][str(rmax)]["improve_ratio"]
    cov_e = m["cov_energy"][str(rmax)]
    cov_c = m["cov_c"][str(rmax)]
    ax.set_title(f"{short}\nSpearman={sp:.3f}  improve(r={rmax})={imp:+.2%}",
                 fontsize=11, color=TEXT_PRIMARY, pad=8)
    ax.set_xlabel("singular component index k (energy-ordered)", fontsize=10,
                  color=TEXT_SECONDARY)
    ax.set_ylabel("cumulative fraction", fontsize=10, color=TEXT_SECONDARY)

    ax.set_xlim(0, min(len(cum_e), rmax * 2.5))
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, axis="y", color="#e8e8e3", lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#d8d8d2")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    # 副标题：覆盖率
    ax.text(0.98, 0.02,
            f"r={rmax}: energy cov={cov_e:.1%}  c_k cov={cov_c:.1%}",
            transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY,
            ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="act_for_ck/wo_all_w2_w3_48")
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--out_dir", default=None,
                    help="默认 input_dir/figures")
    args = ap.parse_args()

    d = json.load(open(os.path.join(args.input_dir, "_ck_analysis.json")))
    out_dir = args.out_dir or os.path.join(args.input_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    reps = select_representatives(d["per_module"], args.ranks)
    print(f"plotting {len(reps)} representative modules to {out_dir}")
    for m in reps:
        short = m["name"].split("layers.")[-1].replace(".", "_")
        out = os.path.join(out_dir, f"curve_{m['idx']:03d}_{short}.png")
        plot_one(m, args.ranks, out)
        imp = m["judge"][str(max(args.ranks))]["improve_ratio"]
        print(f"  {m['idx']:3d} {short:28s} spearman={m['spearman']:.3f} "
              f"improve={imp:+.4f} -> {os.path.basename(out)}")
    print("done.")


if __name__ == "__main__":
    main()
