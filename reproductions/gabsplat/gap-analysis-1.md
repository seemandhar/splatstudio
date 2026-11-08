# Gap analysis 1 — gabsplat (room −0.76, garden −0.59 dB)

## Results (30k steps, runs/gabsplat-mipnerf360)
| scene | ours | paper | our gsplat baseline | our gain | paper gain (vs 3DGS* 27.43 / 31.46) |
|---|---|---|---|---|---|
| garden | 27.07 | 27.66 | 26.89 | +0.18 | +0.24 |
| room | 31.51 | 32.27 | 31.52 | −0.01 | +0.82 |
On garden the method gain roughly matches (+0.18 vs +0.24); the absolute gap comes from our runtime offset (baseline −0.52 vs 3DGS*). On room, the paper's gain is missing.

## Diagnostics on the 30k checkpoints (train views only, every 8th; evidence/diag30k-*.json)
| | room | garden |
|---|---|---|
| Gaussians | 1.43M (baseline 1.73M) | 4.06M |
| rows with bit-identical kernels f / w | 29% / 32% | 67% / 68% |
| rows with \|f1 − f2\| < 1e-3 | 89% | 92% |
| \|f\| median / p90 / p99 (cycles per world unit) | 2.9 / 9.3 / 17.6 | 3.0 / 11.8 / 21.9 |
| w median / p90 / p99 | 0.0025 / 0.020 / 0.38 | 0.0020 / 0.024 / 0.67 |
| train PSNR, full model | 34.48 | 29.81 |
| train PSNR, Gabor weights → 0 | 34.37 (−0.11) | 29.57 (−0.24) |

- **The Gabor term does almost nothing.** Zeroing every Gabor weight costs only 0.11 dB on room train views. Weights collapse below their 0.01 init (median 0.0025). Only about 1–2% of rows have Σw > 0.5.
- **The two kernels stay essentially identical (F=2 ≈ F=1), as predicted.** About 90% of rows have nearly identical kernels: the paper's identical init, the child reset and the 3000-step weight reset all re-equalise them.
- Bit-exact equality is only partly broken, by float noise. The weights end up near zero, so the kernels' frequencies hardly matter anyway.
- The room run also ends with 17% fewer Gaussians than the baseline (1.43M vs 1.73M). The paper itself reports fewer primitives (2.74M vs 3.41M on its Mip-NeRF 360 average). Our density settings follow supp. B exactly (prune threshold 0.01, opacity reset value 0.02, opacity lr 0.025).

## Paper-grounded fixes checked
1. **Indoor-specific supp. config: none exists.** Supp. B gives one Mip-NeRF 360 setting (init f 0.05, w 0.01; lr f 0.02, w 0.01; opacity lr 0.025), with no indoor/outdoor split. The resolution (images_2 indoor, official downsampled images) already matches. Nothing to fix.
2. **lr swap: the text does not resolve it.**
   - Main text says lr_f 0.01 / lr_w 0.02; supp. B says 0.02 / 0.01; both call their values "default".
   - Switching would be a choice, not a correction. A 2× weight lr is unlikely to stop the decay below the init value: Adam step size, not lr scale, drives sigmoid weights down when the photometric gradient pushes them lower.
3. **Opacity lr / prune / reset:** already the supp. B values. Not a deviation.
4. **Symmetry breaking:** not specified anywhere in the paper (review-1 §1). Any fix would be an interpretation, and it is unclear whether ≥0.5 dB is recoverable given the weights collapse.
5. **Dilated vs undilated conic:** only affects footprints near 0.3 px². It cannot explain a Gabor term that contributes 0.1 dB overall.

**Conclusion:** no grounded fix is likely to close ≥0.5 dB on room, so no iteration 1. The most likely causes are unpublished implementation details (kernel initialisation and symmetry breaking, how ω is parameterised and whether Σω is bounded, frequency units and scaling).

## Interpretation diagnostics (not run; labelled if ever run)
- `gabsplat-symbreak`: per-kernel sign pattern (+,+,+) / (+,−,+) at init and child reset.
- Main-text lrs (lr_f 0.01 / lr_w 0.02) on room, train-view diagnostics only.
