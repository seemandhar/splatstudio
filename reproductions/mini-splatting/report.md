# Mini-Splatting (plan #28) — reproduction report

**Outcome: matched.** Across 5 scenes, mean PSNR gap is −0.09 dB, and every scene is within ±0.5 dB of the paper.

Paper: Fang & Wang, *Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians*,
ECCV 2024, arXiv 2403.14166. Official code: https://github.com/fatPeter/mini-splatting. The port
is written against `ms/train.py`, `ms_d/train.py`, `scene/gaussian_model.py` and
`submodules/diff-gaussian-rasterization_ms`. The plugin is in `plugin/` (`ss_mini_splatting`) and
registers three methods:

- `mini-splatting`: outdoor importance metric I² (Mip-NeRF 360 outdoor scenes, T&T)
- `mini-splatting-indoor`: indoor metric I¹ (Mip-NeRF 360 indoor scenes, Deep Blending)
- `mini-splatting-d`: densification only. It was built and verified but not benchmarked.

## Benchmark (30k steps, `--recipe 3dgs`, test split llffhold8)
The paper numbers come from the supplementary per-scene table (`paper_results.json`). The
paper's 3DGS column is its "3DGS*" row, which the authors retrained with the official 3DGS code.
"Ours gsplat" is the shared `baseline3dgs-*-30000` run, which includes the opacity-reset fix.

| scene | method | PSNR ours / paper | Δ | SSIM ours / paper | LPIPS ours / paper | #G ours | train time ours |
|---|---|---|---|---|---|---|---|
| bicycle | mini-splatting | 24.95 / 25.15 | −0.20 | 0.760 / 0.772 | 0.272 / 0.225 | 0.54M | 39 min |
| garden | mini-splatting | 26.65 / 26.84 | −0.19 | 0.839 / 0.847 | 0.179 / 0.150 | 0.58M | 79 min |
| counter | mini-splatting-indoor | 28.50 / 28.41 | +0.09 | 0.904 / 0.904 | 0.260 / 0.199 | 0.40M | 48 min |
| drjohnson | mini-splatting-indoor | 29.29 / 29.47 | −0.18 | 0.906 / 0.904 | 0.323 / 0.257 | 0.56M | 125 min |
| playroom | mini-splatting-indoor | 30.51 / 30.49 | +0.02 | 0.913 / 0.912 | 0.310 / 0.250 | 0.49M | 36 min |
| **mean** | | 27.98 / 28.07 | **−0.09** | 0.864 / 0.868 | 0.269 / 0.216 | | |

**Gaussian counts.** The paper reports averages only: 0.57M for M360 outdoor and 0.40M for
indoor (Table 3), 0.49M over all of M360 and 0.35M for DB (Table 1). Ours are 0.54/0.58M
outdoor, 0.40M on counter, and 0.56/0.49M on DB, so DB runs about 50% above the paper's
average. Compared with our gsplat baseline the counts drop 12× (bicycle), 8× (garden),
3× (counter), 6.4× (drjohnson) and 4.6× (playroom).

**Time and memory.** The paper (RTX 3090) reports 17m56s for outdoor and 27m2s for indoor, with
about 2.6–2.8 GB of training memory. Our wall times come from 8 A6000s shared with other users,
so they are not comparable. Our peak training memory was 6.8–7.4 GB. That peak is set by the
roughly 4.5M Gaussians present before the 15k simplification, plus the cached uint8 training
images.

**LPIPS is worse than the paper on every scene, by +0.03 to +0.07.** This is the same direction
as our gsplat calibration (`reproductions/_calibration/`); the runtime uses `lpips`-package VGG
and a white background. Even so, LPIPS is slightly better than our own gsplat baseline on
counter, drjohnson and playroom, and worse on bicycle and garden, which has 8–12× fewer Gaussians.

### Baseline-relative comparison (PSNR)
This compares our improvement over our gsplat baseline with the paper's improvement over its
3DGS* row.

| scene | ours gsplat | ours MS | Δ ours | paper 3DGS* | paper MS | Δ paper | Δ ours − Δ paper |
|---|---|---|---|---|---|---|---|
| bicycle | 24.92 | 24.95 | +0.03 | 25.19 | 25.15 | −0.04 | +0.07 |
| garden | 26.89 | 26.65 | −0.24 | 27.41 | 26.84 | −0.57 | +0.33 |
| counter | 28.85 | 28.50 | −0.35 | 29.03 | 28.41 | −0.62 | +0.27 |
| drjohnson | 28.76 | 29.29 | +0.53 | 29.12 | 29.47 | +0.35 | +0.18 |
| playroom | 29.80 | 30.51 | +0.71 | 29.95 | 30.49 | +0.54 | +0.17 |

The relative picture follows the paper's pattern. Mini-Splatting loses PSNR on garden and
counter and gains on Deep Blending, with fewer Gaussians throughout. Measured against our own
baseline, every scene comes out 0.07–0.33 dB better than the paper's relative numbers. That is
because our gsplat baseline sits 0.15–0.36 dB below the paper's 3DGS* on these scenes.

### Density events (from the checkpoints' `density_state`)
On every scene `depth_views_short` is 0: every view had enough pixels with non-zero sampling
probability for the requested sample count.

| scene | reinit 5k (N before→after) | reinit 10k | sampling 15k | CDF prune 20k |
|---|---|---|---|---|
| bicycle | 4.04M→3.50M | 4.58M→3.50M | 4.56M→1.03M | 1.03M→0.54M |
| garden | 4.09M→3.50M | 4.51M→3.50M | 4.56M→0.97M | 0.97M→0.58M |
| counter | 1.48M→3.50M | 3.90M→3.50M | 3.85M→0.77M | 0.77M→0.40M |
| drjohnson | 3.59M→3.50M | 4.52M→3.50M | 4.52M→1.04M | 1.04M→0.56M |
| playroom | 2.68M→3.50M | 4.52M→3.50M | 4.44M→0.98M | 0.98M→0.49M |

The 20k prune removes the Gaussians holding the lowest 1% of total importance. By count that is
40–48% of the population.

## Mechanism → code
gsplat 1.5.3 cannot supply the statistics Mini-Splatting needs. Its rasterization meta includes
the per-view tile lists (`isect_offsets`, `flatten_ids`), but not:

- per-Gaussian sums of blending weights,
- the number of pixels where a Gaussian has the largest blending weight (the max-weight count),
- the per-pixel id of that max-weight Gaussian.

`rasterize_to_indices_in_range` can enumerate (Gaussian, pixel) pairs, but only in batches driven
from Python, which is too slow to run every step.

`csrc/ms_raster.cu` is a custom CUDA kernel adapted from the official `forward.cu` (Inria 3DGS
licence, research use, cited in `csrc/LICENSE-3DGS.md`). It re-traverses gsplat's own tile lists
using gsplat's alpha rules and produces the official per-Gaussian statistics (`accum_weights`,
`area_proj`, `area_max`). It also produces the per-pixel `render_depth` outputs: the mid-point of
the max-weight Gaussian along the pixel ray, and the final transmittance. `csrc/simple_knn.cu` is
the official simple-knn `distCUDA2`, vendored to set scales after re-initialization. The
extension is built JIT on first use (about 1.5 min, then cached).

- **Blur split (Eq. 1):** every step, a Gaussian whose max-weight count exceeds 2e-4·H·W is added
  to the 3DGS split mask. In the 2k smoke run, 1951–4304 Gaussians per densify step were split
  by this rule alone (no gradient trigger).
- **Depth reinitialization:** at steps 5000 and 10000, the whole population is rebuilt from
  `int(3.5M/n_views)` pixels per view. Pixels are sampled in proportion to 1−T without
  replacement, and each new point sits at the ray mid-point of that pixel's max-weight Gaussian
  with the ground-truth pixel colour. Scales come from 3-NN distances, opacity is 0.1, and Adam
  state is reset.
- **Intersection preserving (Eq. 2) and importance sampling, step 15000:** Gaussians that are
  never the max-weight Gaussian for any pixel get importance 0. Of the rest, `int(0.5·nnz)` are
  kept by importance-weighted sampling without replacement, and the population is
  re-initialized from their centres.
- **CDF prune, step 20000:** intersection preserving is applied again, then the 1% CDF prune.
- **Schedules:** SH degree stays 0 until 15k, then rises by 1 every 1000 steps. The means
  learning rate follows `lr(it)` before 15k and `lr(it − 10000)` after. There is no opacity
  reset. Densification is skipped at multiples of 5000 and whenever N ≥ 4.5M.

## Evidence (probes and tests)
`ss-verify` returns `declared_checks_passed` for all three methods (`evidence/verify-*.json`).

| probe | result |
|---|---|
| ms-stats-kernel | Matches an independent brute-force per-pixel blend: max-weight id agreement 1.000, max-weight-count and covered-pixel-count differences 0, weight error 1.7e-5, T error 1.2e-7, alpha identical to gsplat (error 0) |
| ms-depth-midpoint | Matches the independent Σ⁻¹ density-maximum formula to 1.1e-6; points lie up to 2.7 away from the Gaussian centre, so the mid-point is distinguishable from the centre |
| ms-blur-split | A zero-gradient dominant Gaussian is split with blur split on (3→4 Gaussians) and not split with it off (3→3) |
| ms-depth-reinit | 600/600 points on the surface (error 1.3e-5), ground-truth colours, opacity 0.1, Adam state reset |
| ms-simplification | Hidden Gaussians get importance 0; importance matches the reference to 3e-7 (relative); 6 kept = int(0.5·12); CDF prune matches |
| ms-schedule | Learning rate at 14999, 15001 and 30000 matches the official schedule; inactive shN coefficients and their Adam moments are zero |

Other checks:

- Unit tests: 10 pass (CPU). `ruff` is clean.
- Smoke run on garden, 2000 steps: 22.80 PSNR vs gsplat 23.02 (SH degree 0 by design), 252 s vs 230 s.
- A compressed-schedule diagnostic on real data exercised re-initialization, sampling and the prune end to end (`evidence/test.json`).

## Fidelity gaps (all kept in the reported numbers)
1. **Background:** white in the runtime, black in the official M360/T&T/DB code.
2. **View sampling:** random with replacement instead of a shuffled stack, and the stack is not reset after re-initialization.
3. **Step order:** the runtime runs the optimizer step before density control. The official code skips the Adam step for tensors it has just replaced.
4. **Kernel numerics:** the statistics kernels follow gsplat (0.999 alpha cap, +0.5 pixel centres) rather than the official rasterizer (0.99 cap, integer pixel coordinates), so they describe the image gsplat actually renders.
5. **Initial scales:** the SfM initialization uses the runtime's exact 3-NN. Re-initialization uses the official approximate simple-knn.
6. **Importance metric choice:** selected by method name (outdoor/indoor), following the README's per-scene-class rule.
7. **Ground-truth colours for depth points:** these come from a target-image pass-through in the renderer. Only training views are cached, matched by viewmat; re-initialization raises an error if a view has not been rendered by then.
8. **Screen-size pruning:** 3DGS's screen-size prune is omitted because it is inert in the official code (`max_radii2D` is zeroed before the test).
9. **Metrics:** SSIM uses valid convolution and LPIPS uses the `lpips` package. This explains the systematic LPIPS offset.

## Review (reviews/review-1.json): approved, minor items only
- **SH activation leak.** When an SH band activates (16k/17k/18k), it has already received one Adam update and non-zero moments from the previous step. The runtime renders at degree 3, and the optimizer runs before the density hook. The expected effect is below 1e-3 dB, and the schedule probe does not detect it. Not fixed.
- **Silent clipping of depth samples.** If a view has fewer pixels with non-zero probability than requested, fewer points are drawn without an error; the official code would raise. The benchmark logs show `depth_views_short = 0` on every scene.
- **Absolute step constants.** The schedule steps (15000/20000/5000/30000) are absolute, as in the official args. Only benchmark at 30k steps.
- **gsplat alpha numerics.** These are intentional and documented (gap 4).
- **Target image on evaluation renders.** The renderer also forwards the target image during evaluation renders. No leakage is possible, because nothing is cached at evaluation. Optional hardening: forward it only when grad is enabled.
- **Image cache by random sampling.** A run whose first re-initialization comes too early can fail by design. At 5k steps the risk is about 1e-5. Cached uint8 images are lossless for 8-bit sources.

## Reproduce
```bash
cd /home/sejain/splatify/reproductions/mini-splatting/plugin
../../../tools/ssenv pip install -e .
../../../tools/ssenv ss-verify mini-splatting --device cuda
cd /home/sejain/splatify
python3 tools/bench.py launch --exp mini-splatting-mipnerf360 --methods mini-splatting --dataset mipnerf360 --scenes bicycle garden
python3 tools/bench.py launch --exp mini-splatting-indoor-mipnerf360 --methods mini-splatting-indoor --dataset mipnerf360 --scenes counter
python3 tools/bench.py launch --exp mini-splatting-indoor-db --methods mini-splatting-indoor --dataset db --scenes drjohnson playroom
python3 tools/bench.py summarize --exp mini-splatting-mipnerf360 --paper reproductions/mini-splatting/paper_results.json
```
A single run is `tools/ssenv ss-train mini-splatting --data ~/datasets/mipnerf360/native/garden
--output <run> --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000`,
followed by `ss-eval --run <run> --split test`.
