# Sort-free GS (WSR), plan id 116: build report (benchmark pending)

- Paper arXiv 2410.18931 (ICLR 2025). No official code. The unofficial reimplementation
  (LiYukeee, which the author links to) was read for learning rates and inits the paper leaves out.
- Plugin `plugin/` adds two methods: `sort-free-gs` (LC-WSR + view-dependent opacity, the variant
  in the paper's per-scene tables) and `sort-free-gs-dir` (DIR-WSR ablation).
- Custom JIT CUDA kernel for order-independent weighted sums, adapted from gsplat 1.5.3 by way of
  the linear-splatting kernel. Checked against a dense float64 reference: forward error 3e-7,
  backward relative error below 1e-5. The output does not change when the Gaussians are permuted,
  and it differs from alpha compositing.
- Smoke test (garden, 2k steps): 22.84 dB against the gsplat baseline's 23.07 dB, all values
  finite. The learned σ rose from 10 to 34.6 and w_B fell from 0.02 to 0.0025. The global rows
  stayed identical.
- Benchmark request: garden (27.78) and truck (25.28) at 30k steps.
- Fidelity gaps: see plan.md. These include non-positive u being culled, the α threshold being
  applied to u·G, the world-size prune being kept, the white background, EXP-WSR not being
  implemented, and learning rates that are not stated in the paper.

## Post review-1 (M1, m1, m2, m6, M2)
Globals are now scalar-Adam-exact: p[0] + straight-through, probe-checked across clone/prune and at N=2e6. c_B is now black. σ, w_B, the culled fraction and the near-zero-denominator fraction are logged every 1k steps. The benchmark moved to bicycle + train.
smoke-3 (garden, 2k steps): 22.86 dB vs 23.07 baseline. σ reached 45.2 and w_B 0.00096. smoke-2 is superseded: it used the mean·N − mean·(N−1) form, which quantised σ and w_B in float32.

## Benchmark results (30k steps, 3DGS recipe)
| scene | ours | paper | gap | our gsplat | Δ ours vs gsplat | paper Δ vs its 3DGS | Gaussians | time |
|---|---|---|---|---|---|---|---|---|
| bicycle | 23.77 | 24.20 | −0.43 | 24.92 | −1.15 | −1.05 | 7.34M | 253 min |
| train | 21.12 | 21.93 | −0.82 | 21.52 | −0.41 | +0.84 | 3.95M | 147 min |

The unofficial port scores bicycle 23.89 (README; it reports no train result). Bicycle matches both
in absolute PSNR and in delta. Train misses by 0.82 dB, and its delta is off by 1.25 dB.

## Gap analysis, train
All diagnostics below use training views of the finished checkpoints. Files are in `evidence/gap/`.
- **σ and w_B.** σ went 10 → 29.5 by step 1k, then drifted to 35.1. Nearly every visible Gaussian
  lies in front of σ (0.1% beyond it), so σ is stable and does not cull. w_B goes negative on both
  scenes (train −0.36, bicycle −0.24). The unofficial code has the same exposure, since its raw
  w_B is unconstrained. With black c_B, a negative w_B only shrinks the denominator, and that
  barely matters: the median S_w is 55 on train. Pixels with S_w+w_B < 0.05 make up 0.07% of
  train pixels and 0% of the MSE. The black-vs-sky background question is therefore immaterial:
  w_B does not model the sky, the Gaussians do.
- **Culled population.** 55% of train Gaussians (23% on bicycle) have a u-DC ≤ 0. They never render
  and are never pruned. This is dead memory and does not change any image, matching the unofficial
  behaviour.
- **Generalisation, not fit.** Mean PSNR over every 10th training view:
  - train: WSR 26.64 vs gsplat 24.68 (+1.96), while test is WSR 21.11 vs 21.52.
  - bicycle: WSR 28.10 vs gsplat 26.28 (+1.82), while test is 23.77 vs 24.92.
  
  WSR fits the training views better and generalises worse. It also ends with 3.1× the gsplat
  Gaussian count on train (3.95M vs 1.26M). The paper instead reports *fewer* Gaussians than 3DGS
  (2.88M vs 3.98M average). Our count grows because there is no opacity prune and no reset, with
  view-dependent opacity on top.
- **α skip threshold (m3).** The unofficial code thresholds u·v·(1−z/σ)·G, which is about 10×
  stricter. That shrinks footprints and the densification gradients of low-weight Gaussians, so
  it could lower N. It is grounded in the unofficial code. However, the unofficial code
  itself lands at bicycle 23.89, no better than ours, and its effect on test generalisation
  cannot be predicted from training views. A ≥0.5 dB train gain is not likely enough to justify
  an iteration.
- **World-size prune.** We keep it, following the paper; the unofficial code removes it. Removing
  it would raise N further, which is the wrong direction for the overfitting seen here.
- **Calibration.** Our gsplat train baseline beats the paper's 3DGS row by 0.43 dB, so part of
  the delta mismatch is baseline-side.

No paper- or code-grounded change is expected to close ≥0.5 dB on train, so iteration 1 is not
attempted. Outcome: **partial**. Bicycle is matched (−0.43 absolute, delta within 0.1 dB of the
paper). Train is not matched (−0.82). The likely cause is over-fitting from a Gaussian population
much larger than the paper reports; why the paper's WSR population is smaller is not specified.

## Fidelity deviations (all runs)
- There is no official code. Learning rates and initial values come from the unofficial code.
- Gaussians with u ≤ 0 are culled; the paper says u is unconstrained.
- The α threshold is applied to u·G inside a 3.33σ footprint.
- The world-size prune is kept, while the opacity prune and reset are removed.
- c_B is black.
- EXP-WSR is not implemented, and the DIR ablation was not run.
- The global σ and w_B are stored as identical per-Gaussian rows. This is exactly equivalent to
  scalar Adam and is checked by a probe.
