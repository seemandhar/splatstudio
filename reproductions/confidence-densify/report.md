# Report — Efficient Confidence-Aware Densification for 3DGS (arXiv 2610.13541), plan #102

**Outcome: not-matched.**
- garden: 27.15 vs 27.68 (−0.53)
- kitchen: 31.59 vs 32.39 (−0.80)

Setup: 1M budget, 30k steps, single seed. The 0.5 dB target is not met on either scene.

## What was built
Native plugin `reproductions/confidence-densify/plugin` (`ss_confidence_densify`). No official
code exists; it was written from the LaTeX source plus CoMe's code (github.com/r4dl/CoMe) for the
confidence numerics.
- **`confidence-densify`:** the paper's method.
  - CoMe confidence: Ĉ = Σbγ̄/max(1−T, 1e-3), with its gradient reaching only γ (fixed after
    review-1).
  - CoMe-weighted loss, β 0.075, Ĉ clamped to [1e-3, 5].
  - λ_reg 0.01, active until 25k.
  - 3DGS-MCMC budget and growth.
  - Sources drawn 80 % from Eq. pdf (rank|∇|_abs × confidence factor, η ramp) and 20 % from the
    SSIM term.
  - Relocation of the primitives with the lowest max blend weight every 400 it, with the share
    schedule.
  - New positions sampled from the source Gaussian, no position noise, GT colour init.
  - Custom CUDA kernel for the per-primitive max α·T and the SSIM error attribution.
- **`cad-mcmc-control`:** gsplat MCMCStrategy at the same budget, plus λ_reg and initial opacity
  0.5. This is the paper's baseline row.
- **`confidence-densify-sep`:** a post-hoc diagnostic variant in which the confidence does not
  weight the photometric loss. No effect, so not proposed for benchmarking.

Checks: ss-inspect clean, unit tests pass, ss-verify `declared_checks_passed` (8 probes).

## Results (test PSNR / SSIM / LPIPS)
| run | garden | kitchen |
|---|---|---|
| confidence-densify | 27.15 / 0.856 / 0.152 | 31.59 / 0.930 / 0.154 |
| cad-mcmc-control | 26.64 / 0.842 / 0.172 | 31.63 / 0.928 / 0.164 |
| paper ours (S.3) | 27.68 / 0.867 / 0.120 | 32.39 / 0.935 / 0.117 |
| paper 3DGS-MCMC (S.3) | 26.86 / 0.846 / 0.147 | 31.63 / 0.929 / 0.130 |

- Gain over MCMC: ours +0.51 / −0.04 dB vs paper +0.82 / +0.76.
- SSIM and LPIPS improve over the control on both scenes, but the LPIPS values are higher than
  the paper's throughout. That is a known runtime/LPIPS offset, and the control shows it too.

Gap analysis: `gap-analysis-1.md`.
- Kitchen is a method gap: the backbone matches the paper's MCMC exactly.
- Train-view gain is only +0.18 dB.
- 5 test views lose 0.6–2.5 dB in the far window/plant background.
- The loss-weighting reading was tested and rejected (7k train PSNR identical).

Earlier runs:
- smoke v1 (21.38 at 2k) had the unnormalised confidence bug, fixed after review-1.
- smoke v2 gave 22.71 vs control 22.67 at 2k.

## Deviations / fidelity gaps
- gsplat 1.5.3 rasteriser instead of Faster-GS with fused Adam.
- Runtime conventions: white background and random-with-replacement view sampling.
- Ambiguities resolved by us (paper.md §7):
  - rank ties (min-rank);
  - SSIM-term reduction (averaged over observations, not ranked);
  - one joint source draw for growth and relocation;
  - moments zeroed for sources and destinations;
  - growth-added primitives start with w = 0;
  - initial opacity 0.5 (MCMC code).
- Benchmark covers 2 scenes only (per-scene protocol); no dataset averages are claimed.
