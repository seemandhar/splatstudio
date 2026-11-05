# Mip-Splatting native plan

Method `mip-splatting`, module `ss_mip_splatting`, package `splatstudio-method-mip-splatting`.
(Update: the `mip-splatting-noreset` variant was dropped once the shared baselines
`baseline3dgs-*-30000` gained the 3DGS opacity reset.)

## Can gsplat express the official filters exactly? Yes — no custom CUDA.
- **2D Mip filter**: gsplat 1.5.3 `rasterization(rasterize_mode="antialiased", eps2d=k)` runs
  `add_blur`: `det_orig=|Σ2D|; Σ2D += k I; comp = sqrt(max(0, det_orig/|Σ2D+kI|)); opacity *= comp`,
  differentiable through comp (`add_blur_vjp`). This is the official `computeCov2D` with
  kernel_size k up to its 1e-6 guards. The only issue with the built-in `gsplat-aa` is the
  **value**: SplatStudio passes no `eps2d`, so it uses gsplat's default 0.3; the paper/official
  360 runs use 0.1. Our renderer passes `eps2d=0.1`.
- **3D filter**: pure tensor algebra before rasterization (official does it in Python too):
  scales' = sqrt(s² + f²), opacity' = σ(o)·sqrt(Πs²/Π(s²+f²)), with f constant (detached).
- Differences in the rasterizer itself (alpha clamp 0.999 vs 0.99, tile/radius details) are
  shared with the gsplat baseline.

## Components
| contribution | kind | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| C1+C2 rendering | renderer `mip-gsplat` | `ss_mip_splatting.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, render.antialiased_2d, render.filter_3d | `kernel_size_2d: 0.1` |
| C1 filter update + reset | density `mip-density` | `ss_mip_splatting.density:create_density` | gradient.screen.signed | 3DGS DefaultStrategy defaults (as baseline) + `filter_coef: 0.2`, `near: 0.2`, `frustum_margin: 0.15`, `filter_interval: 100`, `densify_until: 15000`, `total_steps: 30000`, `opacity_reset_every: 3000` |
| baseline | loss `photometric` | splatstudio built-in | unchanged | |

## Fields
| name | shape | lr | init | density ops |
|---|---|---|---|---|
| `filter_3d` | () → [N] | 0.0 (never gets a grad; used detached) | 0.0 | copied by gsplat duplicate/split/remove; then overwritten by the recompute that follows every densification step |

The field lives in `params`, so it is saved in the checkpoint and used verbatim at evaluation —
the trained model carries its 3D filter like the official PLY's `filter_3D` attribute.

## Training cameras
The runtime calls `density.set_training_cameras(cameras)` right after `initialize` (training
split only); the controller computes the initial filter there. `before_backward` raises if the
hook was never called. The built-in renderer-gradient probe is replaced by an equivalent one
that sets the probe camera through the hook.

## Schedules (in density.after_backward, after gsplat's grow/prune of the same step)
- recompute filter at init and at steps with `step % 100 == 0`, `step > 500`, `step != 15000`,
  `step < total_steps - 100` (union of official densify-time and post-densify recomputes).
- filter-aware opacity reset when `opacity_reset_every > 0`, `step % 3000 == 0`, `0 < step < 15000`.
- gsplat 1.5.3's own reset never fires (`step % reset_every == 0 & step > 0` parses as
  `(… == 0) and (0 > 0)`), so there is no double reset; a unit test pins this behaviour. We keep
  `reset_every=3000` because gsplat also uses it to start the big-Gaussian prune.

## Probes
| probe | measures | fails if |
|---|---|---|
| `filter3d-frequency-bound` | plugin `compute_filter_3d` vs an independent per-point/per-camera loop over Eq. 8 + code rules, incl. an analytic case (filter = sqrt(0.2)·d/f) and invisible-point fallback | filter not computed, wrong coefficient, test cameras/visibility rule wrong |
| `filter3d-convolution` (CUDA) | renderer output vs gsplat rendered with explicit `covars = RS²Rᵀ + f²I` and opacity·sqrt(det Σ/det(Σ+f²I)) from `torch.linalg.det`; plus filtered render ≠ unfiltered | 3D filter not applied or not compensated |
| `mip-filter-2d` (CUDA) | single on-axis Gaussian at a pixel centre: rendered alpha = σ(o)·sqrt(det Σ2D/det(Σ2D+0.1 I)), Σ2D = (f/z)² diag(s²) analytic | classic mode, eps2d 0.3, no compensation |
| `filter-aware-reset` | after reset, filtered opacity = min(old filtered, 0.01); moments zeroed | plain gsplat reset or no reset |
| `filter-schedule` | recompute/reset step sets match the official boundaries (600, 14900, 15100, 29800 in; 500, 15000, 29900 out) | schedule changed |

## Unchanged from the gsplat baseline
DefaultStrategy densification (grad 2e-4, 500–15000 every 100, split/clone/prune thresholds),
photometric loss, optimiser and LRs, SH schedule, `--recipe 3dgs` initialisation.

## Fidelity gaps
- **G6 Numerical deviation (reset/coef)**: the official det-ratio coefficient prod(s²)/prod(s²+f²)
  underflows to 0 in float32 for Gaussians much smaller than their filter, and its reset
  logit(min(σ·c, 0.01)/c) then gives 0/0 = NaN (and +inf for saturated σ = 1). This crashed the
  first benchmark at the 3000/6000 reset. We compute the coefficient in log space and evaluate
  the reset as logit(clamp(min(σ, 0.01/c), 1e-6, 1-1e-6)), which is algebraically identical wherever
  the official formula is finite (unit test `test_reset_finite_in_failure_regime`).
- **G1 Opacity reset**: resolved — filter-aware reset every 3000 steps (official), exactly once
  (probe `single-reset`); compare against `baseline3dgs-*-30000`, which also resets.
- **G2 Image protocol**: paper resizes full-res JPEGs with PIL (-r 4/-r 2); we use the
  dataset's `images_4`/`images_2`. Their 3DGS* is 0.49 dB above the 3DGS paper's numbers on the
  same scenes; absolute PSNR will sit ~0.3–1.4 dB below paper on several scenes for this reason
  alone. Compare the delta over our baseline (+0.09 dB paper mean). Closable by building a
  PIL-resized native dataset and re-running the baseline on it.
- **G3 Runtime**: white background (official 360: black), random-with-replacement view sampling,
  densification after (not before) the optimiser step, 0.999 vs 0.99 alpha clamp — shared with
  the baseline.
- **G4 Camera access**: resolved by the runtime hook `set_training_cameras`.
- **G5 Principal point**: projection for the visibility test uses K's cx, cy (official W/2, H/2);
  identical here (cx = W/2).
