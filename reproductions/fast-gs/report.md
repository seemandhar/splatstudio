# FastGS — reproduction report

Paper: *FastGS: Training 3D Gaussian Splatting in 100 Seconds* (arXiv 2511.04283), plan id 23.
Official code: github.com/fastgs/FastGS @ 44e02a5 (rasterizer vendored, Inria licence; research use).
Plugin: `plugin/` (module `ss_fast_gs`), methods `fast-gs` (paper main row) and `fast-gs-big`.

**Outcome: matched (iteration 0).** Both reported scenes are within ±0.02 dB of the paper, and the Gaussian counts match.

## Results (Mip-NeRF 360, `mipnerf360-1600` = official `-r -1` width-1600 PIL resize, 30k, test split, black background)

| scene | method | PSNR | SSIM | LPIPS | #G (M) | paper PSNR / SSIM / LPIPS / #G | ΔPSNR | train min (A6000, shared) |
|---|---|---:|---:|---:|---:|---|---:|---:|
| garden | fast-gs | **27.23** | 0.837 | 0.205 | 0.76 | 27.20 / 0.836 / 0.174 / 0.74 | +0.03 | 11.3 |
| counter | fast-gs | **29.16** | 0.908 | 0.274 | 0.21 | 29.15 / 0.907 / 0.204 / 0.21 | +0.01 | 10.2 |
| garden | fast-gs-big | 27.65 | 0.865 | 0.129 | 2.65 | 27.56 (PSNR only) | +0.09 | 24.2 |
| counter | fast-gs-big | 29.47 | 0.918 | 0.245 | 0.48 | 29.57 (PSNR only) | −0.10 | 13.4 |

Baseline `baseline3dgs-mipnerf360-1600-30000`: garden 26.93, counter 29.04.
- Our FastGS − our gsplat: +0.30 / +0.12 dB. The paper's FastGS − its 3DGS: −0.14 / −0.01 dB. Our gsplat baseline is 0.41 / 0.12 dB below the paper's 3DGS row, so the absolute match does not imply the same relative ranking.
- FastGS-Big over FastGS: +0.42 / +0.31 dB here, vs +0.36 / +0.42 in the paper.
- The efficiency claim reproduces in count: 0.18× / 0.20× the paper's 3DGS Gaussians (paper 0.18× / 0.20×). Wall-clock times are on shared A6000s, not the paper's RTX 4090, and are not comparable to its 2.47 / 1.83 min.
- SSIM matches. LPIPS is higher than the paper by 0.03 (garden) and 0.07 (counter), the known LPIPS-implementation offset of this runtime.

Runs: `runs/fast-gs-1600/{fast-gs,fast-gs-big}/mipnerf360-1600-{garden,counter}`.

## Implementation
- Official FastGS CUDA rasterizer vendored unmodified (CB SnugBox + AccuTile, 4-column screen gradient, per-Gaussian high-error pixel counter).
- Density: multi-view consistent densification (10 cached views, s_d / s_p scores), clone/split rules, probabilistic VCP pruning, opacity clamp 0.8, resets at 3000-step intervals, late VCP prunes at 18k/21k/24k/27k.
- Optimizer: the official step schedule (main parameters every step ≤15k, SH rest every 16; then every 32 / 64), via `ScheduledAdam`.
- Per-scene official arguments selected from the SfM point count.
- Checks: `evidence/verify.json` declared_checks_passed; unit tests pass. Not separately reviewed (official-code port).

## Deviations
1. Runtime: training views sampled with replacement; optimizer step before densification.
2. LPIPS-VGG from the `lpips` package; evaluation on float renders.
3. Single seed; 2 of 9 scenes (per-scene rule).
4. The builder stopped before benchmarking; the benchmark was launched by the orchestrator with the builder's `bench_request.json` unchanged.
