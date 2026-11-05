# EfficientGS native plan

Method `efficientgs`, module `ss_efficientgs`, package `splatstudio-method-efficientgs`.

## Components
| component | kind | factory | requires | provides / notes |
|---|---|---|---|---|
| `egs-renderer` | renderer | `ss_efficientgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, gradient.screen.signed, render.pixel_gradients, training.target_image. gsplat `rasterization` with precomputed colours (`sh_degree=None`): colours = gsplat `spherical_harmonics` with **per-Gaussian degree** (shN bands above the Gaussian's `sh_order` multiplied by 0) + 0.5, clamp ≥ 0 — the exact gsplat colour path. render_mode RGB. Exposes raw pre-background render / alpha (for gradient retention), per-Gaussian colours and the target image. |
| `egs-density` | density | `ss_efficientgs.density:create_density` | render.pixel_gradients, training.target_image | C1–C3, 3DGS clone/split/prune/opacity reset written on `gsplat.strategy.ops` (same structure as 3DGS `densify_and_prune`). |
| `photometric` | loss | built-in | render.rgb | unchanged (L1 + 0.2 D-SSIM). |

## Fields
| field | shape | lr | init | meaning |
|---|---|---|---|---|
| `sh_order` | (1,) | 1e-12 (never receives a gradient; not in the graph) | 0 | per-Gaussian SH degree (Eq. 9 increments). Copied by duplicate/split, dropped by remove, saved in the checkpoint so evaluation renders with the sparse degrees. |

## Custom CUDA (`csrc/egs_raster.cu`, JIT via `torch.utils.cpp_extension.load`)
On gsplat 1.5.3 tile lists (`isect_offsets`, `flatten_ids`, projected means2d/conics/opacities), gsplat numerics
(α = min(0.999, o·e^{-σ}), skip σ<0 or α<1/255, stop when next T ≤ 1e-4, pixel centres +0.5).
1. `egs_gradnorm`: two front-to-back passes per pixel (pass 1: total colour and T_final; pass 2:
   gsplat backward formula with "behind" buffer = C_total − accumulated) → per-pixel
   v_xy of each Gaussian; accumulates `sqrt((W/2·gx)² + (H/2·gy)²)` per Gaussian with warp
   reduction (Eq. 6, NDC units as official `backward.cu:670`). Inputs: upstream gradients
   of the raw render colours / alphas (retained in `before_backward`).
2. `egs_stats`: forward pass; per pixel top-K weights → dominant flag per Gaussian (Eq. 8),
   and optional Σ_ch |C−C_gt|·w (Eq. 9). Weight convention configurable (`weight_after_T`,
   default True = official code α·T_{after}).
Adapted from the official EfficientGS rasterizer (Inria licence) and gsplat's
`RasterizeToPixels3DGSBwd.cu` (Apache-2.0); structure copied from the mini-splatting plugin
kernel (attribution in file header).

## Density schedule (after_backward(step), runtime calls it after the optimizer step)
- step < 15000: accumulate S_g (kernel 1) and visible counts; step > 500 & %100: clone
  (S ≥ τ_s & small) then split (S ≥ τ_s & large), prune opacity < 0.005, plus world-size > 0.1·extent
  after 3000; step % 3000 == 0: opacity reset to min(o, 0.01).
- step ∈ {15000, 18000, 21000}: top-K prune over all training cameras (`set_training_cameras`).
- next step == 15500: clear shN Adam state (official reallocates f_rest), field already 0.
- next step ∈ {16000, 17000, 18000}: Eq. 9 over all training views (cached GT images from the
  renderer pass-through), +1 degree for `lw >= lw[argsort(-lw)[int(n·0.2)]]`, cap 3.
- initialize: shN lr = 0.0025/5.

## Probes
| probe | contribution | fails when |
|---|---|---|
| `egs-gradnorm-kernel` (CUDA) | C1 | kernel ≠ brute-force per-pixel autograd of a pure-torch blend (sum of per-pixel norms); also checks Σ per-pixel vectors = gsplat means2d.grad and a cancellation fixture (S ≫ ‖Σ∇‖). |
| `egs-selective-densify` (CUDA) | C1 | Gaussian with cancelling gradients is not densified with S_g, or criterion reverted to E_g. |
| `egs-stats-kernel` (CUDA) | C2/C3 | top-K flags / loss weights ≠ brute-force reference (both weight conventions). |
| `egs-topk-prune` (CUDA) | C2 | auxiliary Gaussian (never top-1) survives at K=1, or is removed at K=2 where it is 2nd. |
| `egs-sparse-sh` (CUDA) | C3 | per-Gaussian degree rendering ≠ gsplat render at that degree; top-20% selection/increments/cap wrong. |
| `egs-schedule` (CPU) | all | event steps, shN lr, degree-0-before-15500 wrong. |

## Files
`plugin/pyproject.toml`, `ss_efficientgs/{__init__,renderer,density,kernels,core,probes}.py`,
`ss_efficientgs/csrc/{ext.cpp,egs_raster.cu,LICENSE-3DGS.md}`, `tests/test_efficientgs.py`, `README.md`.

## Fidelity gaps
1. Runtime: white background, random-with-replacement view sampling, valid-SSIM, density after
   the optimizer step (shared with the baseline; expected ≤0.1 dB).
2. Rasterizer numerics gsplat (0.999 alpha cap) instead of official 0.99 cap — affects the rare
   clamped Gaussians and the exact ranking of Eq. 8; negligible.
3. Gradient statistics come from gsplat's backward formula (alpha-clamped Gaussians get zero
   gradient as in gsplat; the official 3DGS backward does not zero them).
4. Datasets: Mill19 / Foothill-Town (the paper's headline large-scale results, 60k iterations)
   are not available; only Mip-NeRF360/T&T/DB averages are reproduced. FPS/storage only indicative.
5. SSIM loss gradient magnitude slightly different (valid conv) → τ_s=0.0007 may select a slightly
   different number of Gaussians. Recorded; not tuned.
