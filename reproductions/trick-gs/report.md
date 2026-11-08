# Trick-GS (plan #12) — report. Outcome: **not-matched**

Paper: arXiv 2501.14534. **No official code** was found (GitHub search, 2026-11-07), so the port is written from the paper. Each trick follows the official code, or the native port, of the paper it is taken from: Compact-3DGS (GM), LightGaussian (SG, with its CUDA count kernel), RDO-Gaussian (SHM), EAGLES (BL/DS), RAIN-GS (SC) and Taming 3DGS (AT).

The headline "Trick-GS" rows use all eight tricks (Table 3 caption). This is implemented as method `trick-gs`.

The paper reports dataset averages only, so the benchmark is the complete T&T average (2 scenes, 30k steps).

## Results (T&T, PSNR)
| run | truck | train | mean | final #G (mean) | vs our gsplat (23.36) |
|---|---|---|---|---|---|
| iter 0 `trick-gs` (runs/trick-gs-tandt) | 24.26 | 20.77 | **22.51** | 528K | −0.85 |
| iter 1 `trick-gs-it1`, post-hoc: SH masking from 15001 (runs/trick-gs-it1-tandt) | 24.25 | 20.57 | **22.41** | 523K | −0.95 |
| paper Trick-GS / paper 3DGS | – | – | 23.48 / 23.67 | 443K | paper −0.19 |

The gap to the paper is −0.97 dB, which is outside ±0.5. It splits into a method gap of −0.66 and a runtime/protocol offset of −0.31 (our gsplat 23.36 vs the paper's 3DGS 23.67).

## Gap analysis (gap-analysis-1.md)
- **No over-pruning.** Our mean max/final counts are 843K/528K, against the paper's 696K/443K.
- **SH masks close.** At 30k only about 1 % of Gaussians keep any SH band; Fig. 3 suggests about 20 %. A 10k diagnostic gave +0.38 dB without SH masking.
- **Iteration 1 did not help.** It moved the SH-mask start to step 15001 (RDO-Gaussian code), and the T&T mean went from 22.51 to 22.41. **The over-closure hypothesis did not hold at 30k**, and the gap remains unexplained.
- **Other measured effects:**
  - FP16 storage costs −0.09 dB (gsplat truck, train views). It is kept because the paper specifies it.
  - The learned Gaussian masks saturate at σ≈1 after about 1k steps, so they prune only early on. This affects size, not PSNR.
- **Remaining suspects** (all guesses, because there is no author code; see paper.md §7):
  - timing and rates of the significance-pruning rounds (read off Fig. 3);
  - blur decay law;
  - linear vs "logarithmic" resolution schedule;
  - mask initial values;
  - ABE split applied to a non-centred SfM scene;
  - target downsampling without antialiasing.

## Verification
- `ss-inspect` reports no issues; 6 unit tests pass; ruff is clean.
- `ss-verify` passes for `trick-gs` (11/11 probes) and for `trick-gs-it1`.
- In the `trick-cycle` probe the Gaussian-mask learning rate is frozen, because the toy scene collapses under the paper's LR 0.5.
- Smoke run (garden, 2k steps): 18.90 dB vs 23.03 for gsplat. This is expected: at step 2k training runs at about 21 % resolution with blur.

## Fidelity gaps
See plan.md. In short:
- Several schedules are guesses, with no official code to check them against.
- The SfM-error sidecar files in `sfm_errors/` are needed for the 20 % lowest-error init.
- Runtime differences shared with the baseline: white background, random view sampling, valid-convolution SSIM at evaluation.
- Time, FPS and storage numbers are not measured on the paper's RTX 3090.
