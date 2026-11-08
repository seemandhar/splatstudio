# BOGausS (arXiv 2504.01844) — reproduction report

**Outcome: not-matched.** T&T mean 23.94 vs paper 24.78 (−0.84 dB). The gain over our gsplat baseline is +0.58, vs the paper's +0.99 over its 3DGS row.

| scene | bogauss | gsplat | #G |
|---|---|---|---|
| truck | 25.75 | 25.21 | 2.58M |
| train | 22.13 | 21.52 | 1.09M |
| mean | 23.94 (SSIM 0.856, LPIPS 0.190) | 23.36 | 1.83M (paper 1.83M) |

Paper "Ours (Big)", without exposure correction: 24.78 / 0.866 / 0.176. Paper 3DGS row: 23.79.
The gap splits into baseline offset −0.43 (the paper's 3DGS row is 0.65 dB above the 3DGS paper) and method delta −0.41. See gap-analysis-1.md.

## Implementation
Native plugin `bogauss` (plugin/ss_bogauss), written from the paper; **no official code exists**.
- Precision Δ_i, with minimum size Δ.
- Unbiased sparse Adam with a per-Gaussian lifespan, Δ-scaled position updates and linear-size updates.
- Split-state inheritance (α_t = 1, α_g = 0.2).
- Effective-opacity pruning (0.02, ≤1 %), computed by a custom CUDA statistics kernel.
- Density-preserving split (±0.3σ) with an offline SplitAdaptation fit.
- P_SNR split priority.
- Densification every 500 steps up to the Taming big budgets.
Checks: ss-verify declared_checks_passed (12 probes); 7 unit tests pass.

## Deviations / fidelity gaps
- η for the Δ-scaled updates is not given in the paper. It is auto-calibrated so the median initial step equals 3DGS's.
- σ_uv = 1 px; Δ uses training cameras whose 1.3× frustum contains the Gaussian.
- S_i = contributing-pixel count. Statistics come only from views rendered during training.
- SplitAdaptation is our own 1-D alpha-profile fit. The growth curve and the 500–15000 window are from Taming. The first split is capped at doubling the count.
- Exposure correction is not ported; the compared number does not use it.
- gsplat runtime differences, shared with the baseline.
- Single seed.
