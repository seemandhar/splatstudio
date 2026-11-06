# GaussianPro native plan

Method `gaussianpro`, module `ss_gaussianpro`, package `splatstudio-method-gaussianpro`.

## Components
| component | kind | factory | requires → provides | config (paper/code value) |
|---|---|---|---|---|
| `gaussianpro-render` | renderer | `ss_gaussianpro.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, render.depth.accumulated, render.normal.camera, render.alpha, gradient.screen.signed | `st_alpha_clamp: true` |
| `gaussianpro-density` | density | `ss_gaussianpro.density:create_density` | gradient.screen.signed | 3DGS/gsplat baseline densification (grow_grad2d 2e-4, …) + propagation: interval 50, window (1000, 12000), σ 1.0→0.8, patch 20, depth range [0.1, 20], neighbours 4 (shared SfM points), geo thre 2 px / 0.01 / ≥2 views, stride 8, min pixels 100 |
| `gaussianpro-photometric` | loss | `ss_gaussianpro.losses:create_photometric` | render.rgb, render.alpha | λ 0.2, opacity mask 0.999 before step 1000, > 0 after |
| `gaussianpro-normal` | loss | `ss_gaussianpro.losses:create_normal` | render.normal.camera | lambda_l1 0.001, lambda_cos 0.001 (Eq. 8/9 β) |
| `gaussianpro-flatten` | loss | `ss_gaussianpro.losses:create_flatten` | state.gaussian3d | lambda 100 (Eq. 9 γ) |

No new per-Gaussian fields. Component-held state: per-view propagated-normal targets and
propagated-depth caches, propagated flags, neighbour lists (process-local `link.LINK`).

## Mapping (official → native)
- Renderer: gsplat RGB pass identical to the baseline (white bg) + a geometry pass (gsplat,
  colours = [z_i, n_i^cam], black bg) giving accumulated depth (Inria `render_depth`), and
  `F.normalize`d camera normal (Inria `render_normal`). Geometry pass runs with autograd only
  when the view has a stored propagated normal; with no grad when the density asks for depth.
  Straight-through opacity cap 0.999 (gsplat drops gradients where o·G > 0.999; Inria clamps
  alpha in the forward only — matters for the +inf-opacity propagated Gaussians).
- Density: wraps `splatstudio.backends.gsplat.Density` (exactly the baseline densification +
  3DGS opacity reset). In `after_backward(step)` for the view v rendered at this step:
  if 1000 < step < 12000, step % 50 == 0 and v not yet propagated → propagation (below), which
  appends Gaussians and zeroes the densification statistics (Inria `densification_postfix`);
  then the baseline statistics update + densify run (official order: propagation at the start
  of the iteration, densification at its end). The geometry pass' means2d gradient is added to
  the RGB pass' (shared `screenspace_points`).
- Propagation (train.py:113-187): render depth of v; `propagate` (vendored kernel, CPU uint8
  images, w2c poses, K, depth interval [0.1, Δ, 192, 20], patch 20) with v + its 4 neighbours;
  normal → camera; abs-rel error vs rendered depth > σ(step); for each neighbour: cached depth
  or (render + propagate with *v's* neighbour list, cache); `check_geometric_consistency`
  (thre1 2, thre2 0.01); count ≥ 2 → normal target (−10 elsewhere); mask = error & consistent;
  depth := 300 outside mask *in the cached tensor*; if mask.sum() > 100 back-project every 8th
  pixel, colour from the previous step's GT image, scale log sqrt(distCUDA2(new ∪ old)),
  identity quaternion, opacity logit +inf; Adam moments zero.
- Losses: Eq. 10 with the official opacity mask; Eq. 8 (L1 channel-sum mean + 1 − cos mean over
  valid pixels); flatten 100·mean(clamp(min scale, 0, 30)).

## Probes
| probe | contribution | measures / fails when |
|---|---|---|
| `propagation-plane` (CUDA) | C2 | 5-view synthetic textured plane, initial depth = truth × U(0.7,1.3) per pixel and 0 on a band; kernel output depth rel-err median < 1 %, normal angle median < 5°; fails if the kernel is skipped (initial depth error ~15 %) or homography/pose convention wrong |
| `propagation-selection` (CUDA) | C2/C4 | Gaussians cover a plane except a square hole: density's propagation step on one view inserts points only inside the hole (≥ 90 % in hole, all within 2 % depth of the plane), zero when there is no hole, statistics zeroed, Adam state aligned |
| `geometry-render` (CUDA) | C1 | flat Gaussians with known normal: rendered camera normal = R_w2c·(flipped shortest axis), accumulated depth = Σ z α T (independent alpha compositing) |
| `planar-losses` | C3 | normal loss equals an independent formula, zero at target, ignores −10 pixels; flatten loss formula and gradient only on argmin scale |
| `opacity-mask` | C4 | photometric loss before step 1000 ignores pixels with alpha ≤ 0.999 (zero grad), uses them after |
| `consistency-check` | C2 | geometric consistency: identical geometry → all valid; 5 % depth offset → invalid; vs independent reprojection |

## Fidelity gaps
1. Mip-NeRF 360 inputs: paper uses the authors' re-run COLMAP (unreleased); we use the official
   COLMAP and `images_4/_2` (3DGS protocol). Effect: absolute PSNR differs (paper 3DGS garden
   28.77 vs our 26.89); compare gains. Not closable.
2. gsplat rasterizer instead of the Inria rasterizer for RGB/depth/normal (alpha clamp 0.999 vs
   0.99, culling); same as our baseline. Small.
3. White background (runtime) vs black; random-with-replacement view sampling; SSIM valid conv
   (runtime conventions shared with the baseline).
4. Propagation runs at the end of step t (after the optimizer step) instead of the start of t;
   new Gaussians first get gradients at t+1, v's normal loss starts at its next visit. Negligible.
5. Neighbour graph: generated from shared COLMAP points (App. A) — the authors' pair file
   generator is not released.
6. Propagation window end 12000 (code default) — paper silent for 360.
7. Official quirks replicated on purpose (code > paper): u=6 kernel iterations, random initial
   normals, +inf opacity, previous-step colour, stale/overwritten depth caches, stats reset.
