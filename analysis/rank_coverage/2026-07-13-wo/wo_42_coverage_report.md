# 42 Line `wo` Residual Coverage

Generated on 2026-07-13 from a new, isolated collection. This report does
not modify `act_for_ck/wo_all_w2_w3_48` or the existing 2026-07-13 coverage
report.

## Scope

- Target cache: `custom_wo_w2_w3_42`
- Target family: 15 two-bit `attention.wo` residual projections
- Calibration: 128 COCO image-text samples, answer and vision tokens only
  (`max_tokens=8192`), matching the existing source collection settings.
- Residual SVD: full 4096-dimensional SVD for every `wo` projection.
- Reweight: `rho_attn_out = 0.1302539007` from the existing reweight cache.
- Source artifacts: `act_for_ck/wo_42_20260713/_global_meta.json` and
  `act_for_ck/wo_42_20260713/_ck_analysis.json`.

All family aggregates below are weighted by the existing activation-aware
projection score `E_m` from the 42-line cache.

## Coverage

`Energy` is the ordinary SVD-energy coverage. `Cross-modal` is the cumulative
`c_k` coverage. Both first use the full residual spectrum as denominator.
`Rel.@128` then normalizes each projection by its own rank-128 coverage before
the `E_m`-weighted family average; it therefore measures fidelity to the
deployed rank-128 residual branch.

| rank | Energy | Cross-modal | Cross minus energy | Energy Rel.@128 | Cross-modal Rel.@128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2.3345% | 2.8936% | +0.5591 pp | 18.6123% | 21.3749% |
| 32 | 3.9143% | 4.5235% | +0.6092 pp | 31.3015% | 33.7547% |
| 48 | 5.4244% | 6.0695% | +0.6451 pp | 43.4541% | 45.6197% |
| 64 | 6.8924% | 7.5394% | +0.6469 pp | 55.2723% | 57.2343% |
| 96 | 9.7267% | 10.3016% | +0.5749 pp | 78.0952% | 79.1226% |
| 128 | 12.4465% | 12.9268% | +0.4803 pp | 100.0000% | 100.0000% |

For the same 42-line rank-128-relative cross-modal metric, the family order is:

| family | Rel.@32 | Rel.@64 |
| --- | ---: | ---: |
| wqkv | 36.2730% | 59.8262% |
| wo | 33.7547% | 57.2343% |
| w2 | 32.6641% | 55.8384% |
| w1 | 30.3190% | 54.7323% |
| w3 | 28.3364% | 52.9236% |

## Component-Order Diagnostic

For the 15 `wo` projections, the rank correlation between ordinary SVD energy
and `c_k` is high: Spearman mean `0.986`, median `0.988`, minimum `0.973`.
Changing from top-sigma to top-`c_k` ordering still reduces activation-aware
output residual error for every projection:

| rank | Mean error reduction | Median | Positive modules |
| ---: | ---: | ---: | ---: |
| 16 | 0.77% | 0.62% | 15/15 |
| 32 | 1.16% | 1.06% | 15/15 |
| 48 | 1.31% | 1.25% | 15/15 |
| 64 | 1.50% | 1.47% | 15/15 |
| 96 | 1.82% | 1.67% | 15/15 |
| 128 | 2.08% | 1.80% | 15/15 |

## Interpretation Boundary

`wo` has the second-highest rank-128-relative cross-modal retention at ranks
32 and 64, after `wqkv`. Its ordinary-energy coverage is lower than its
cross-modal coverage at every measured rank, so the answer/vision activations
do provide additional relevant concentration for this family.

This does not establish that `wo_r32` is independently safe: there is no
single-factor OCRBench result for only `wo:128 -> 32`. The observed 42-line
drop in the `w1_r64 + w3_r64 + wo_r32` combination cannot be attributed to a
poor standalone `wo` spectrum; it remains a cross-module interaction result.
