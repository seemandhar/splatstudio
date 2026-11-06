# Speedy-Splat (plan #72): build report

**Status:** ready for the benchmark. The 30k-step benchmark has not been run.

The plugin is in `reproductions/speedy-splat/plugin` and provides three methods:
- `speedy-splat`: the full method.
- `speedy-splat-accutile`: control with tile culling and no pruning (the paper's "+AccuTile" row).
- `speedy-splat-stock`: control with the original Inria rasterizer and the same training (the paper's "Baseline" row).

## What is ported
- **Sparse pixels:** the official rasterizer (speedy-splat-rasterizer b38abb6) is vendored unmodified. The stock Inria rasterizer (59f5f77) is also vendored unmodified, as the unculled reference and the timing control. Both are under the Gaussian-Splatting licence.
- **Sparse primitives:**
  - The pruning score is Σ over views and pixels of (o·∂ΣI/∂α)², taken from the kernel's `dL_dG2` buffer.
  - Soft pruning removes 80 % at steps 6000, 9000 and 12000, after densification and before the opacity reset.
  - Hard pruning removes 30 % at steps 15000 to 27000.
  - A Gaussian is pruned when its score ≤ sort[int(r(N−1))].
- **Density:** the official 3D-GS adaptive density control. The screen-size prune is a no-op, as in the Inria code.

## Probes
All pass; results are in `evidence/verify-*.json`.

| probe | result |
|---|---|
| culling-exact | Max image error 1.2e-6 against a float64 dense unculled compositor; the stock 3σ rasterizer is 1.2e-2 off. 123 tile pairs, matching an exact ellipse-tile count; 3σ squares give 198. |
| score-kernel | 2-view score vs float64 autograd: relative error 1.1e-6. Summing before squaring would be 174× off. |
| prune-schedule / adc-schedule | Step lists, ordering, percentile rule, ties and optimizer alignment all checked. |

## Smoke test (garden, native resolution, 2000 steps)

| run | PSNR | Gaussians |
|---|---|---|
| speedy-splat (no pruning before 6k) | 23.26 | 851k |
| compressed prune schedule (not the paper's schedule) | 23.00 | 142k |
| gsplat baseline | 23.02 | 845k |

Render speed is a secondary measurement on a shared, busy A6000 and is not comparable to the paper's numbers. The culling rasterizer is 1.09–1.14× faster than the stock one and uses 2.1–2.3× fewer tile-Gaussian pairs.

## Benchmark request
- **Scenes:** `mipnerf360-1600` garden and counter, 30k steps, compared against `baseline3dgs-mipnerf360-1600-30000`.
- **Paper targets:** 26.70 (garden) and 28.28 (counter). The paper's 3D-GS scores 27.31 and 29.14 (drops of −0.61 and −0.86 dB) with 10.6× compression.

## Fidelity gaps
1. The runtime takes the optimizer step before density control. Officially, densification and soft pruning run before the step, and hard pruning after it.
2. Views are sampled with replacement. SSIM uses valid convolution. The SH degree goes up one step later.
3. One seed, against the paper's mean over 3 runs.
4. Timing goes through the Python wrapper, on busy GPUs.

*Report saved by the orchestrator; the build subagent could not write it.*

## Review-1 resolution (orchestrator)
Approved, with 0 critical and 0 major issues. The benchmark is launched: `speedy-splat` and `speedy-splat-stock` on `mipnerf360-1600` garden and counter.

**Protocol (confirmed).**
- `train.sh` uses the defaults (`images/`, `-r -1`, which means 1600 px wide).
- The README states this resolution explicitly.
- Our w1600 image sizes match the official ones exactly.

**Reading the results.**
- **Reference:** `speedy-splat-stock` (Inria rasterizer, black background, same training) is the right reference for the count ratio and the PSNR drop.
- **Our gsplat baseline:** reported as context only.
- **Compression target:** the paper's 10.6× is a dataset average.
- **LPIPS:** our absolute LPIPS is not comparable with the paper's, so we judge it by its change against our baseline.

**Correction.** The SH-degree note in `plan.md` was wrong: degree 3 is used at every prune step.

## Benchmark result (speedy-splat-1600, 30k, mipnerf360-1600) — MATCHED
| scene | speedy-splat | paper | Δ | stock control | paper 3D-GS | drop ours / paper | Gaussians speedy / stock (ratio) |
|---|---|---|---|---|---|---|---|
| garden | 26.71 / 0.815 / 0.250 | 26.70 | +0.01 | 27.27 | 27.31 | −0.56 / −0.61 | 520k / 5.91M (11.4×) |
| counter | 28.25 / 0.877 / 0.331 | 28.28 | −0.03 | 29.06 | 29.14 | −0.80 / −0.86 | 97k / 1.22M (12.5×) |

- Both scenes are within ±0.05 dB of the paper.
- The stock control matches the paper's 3D-GS within 0.08 dB.
- The quality drop and the compression (11–12.5× vs the paper's 10.6× Mip-NeRF 360 mean) match.
- The FPS post-step (ss_speedysplat.fps) was not run. It is the secondary speed claim, which the shared GPUs make noisy.
