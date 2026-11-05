# RaDe-GS — native SplatStudio plan

Method `radegs`, module `ss_radegs`, package `splatstudio-method-radegs` in `plugin/`.
Official code pinned at `2d4bc08` (paper formulation; see paper.md "Code version").

## Components
| name | kind | factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|---|
| radegs-raster | renderer | `ss_radegs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth (expected z, normalised), render.depth.median, render.alpha, render.normal.camera, render.normal_from_depth.camera (expected+median), render.distortion.radegs, gradient.screen.gof | background [0,0,0] (white_background False) |
| radegs-density | density | `ss_radegs.density:create_density` | gradient.screen.gof | — | GOF densification (identical code in RaDe-GS@2d4bc08 and GOF@5245b20): thr 2e-4, abs quantile, resampled clones, s/1.6 split, min_opacity 0.05, world prune 0.1·extent after 3000, no screen prune (official no-op), filter-aware reset 0.01 / 3000, 3D filter (0.2, near 0.2, margin 0.15), official xyz LR 1.6e-4→1.6e-6·extent |
| photometric | loss | `ss_radegs.losses:create_photometric` | render.rgb | — | (1−0.2)L1 + 0.2(1−SSIM same padding) |
| radegs-reg | loss | `ss_radegs.losses:create_regularizer` | render.distortion.radegs, render.normal.camera, render.normal_from_depth.camera | — | lambda_distortion 100, lambda_depth_normal 0.05, depth_ratio 0.6, regularization_from_iter 15000, edge weighting on |

The vendored kernel (`ss_radegs/csrc`, RaDe-GS `submodules/diff-gaussian-rasterization` @2d4bc08
unmodified + glm 5c46b9c0, Inria/MPII licence) is JIT-built with `torch.utils.cpp_extension.load`
under `tools/ssenv` (name `ss_radegs_raster`).

## Fields
| field | shape | lr | init | density ops |
|---|---|---|---|---|
| filter_3d | () per Gaussian | 1e-8 (never receives grad; used detached) | 0 → recomputed from training cameras | copied by gsplat ops, overwritten at each recompute |

## Probes
| probe | contribution | measures / fails when |
|---|---|---|
| rasterized-depth | C1 | single tilted Gaussian: kernel depth (Σwd/Σw) vs independent float64 exact per-pixel ray intersection (Eq. 6, depth = s*·r_z): equal at the centre, slope equals the finite-difference slope of the exact depth (affine approx.); fails for constant centre depth (slope 0) |
| rasterized-normal | C2 | kernel normal vs normal of the plane P(u,v) = d(u,v)(u,v,1) from the independent exact depth's derivatives, and vs the short axis for a flat Gaussian (facing camera) |
| kernel-gradients | C1,C2 | backward of rgb/depth/median/alpha/normal/distortion vs float64 central finite differences of the forward kernel (means, scales, quats, opacity, sh0) |
| distortion-formula | C3 | kernel distortion = Σ_{j<i} w_i w_j (m_i−m_j)² on NDC-mapped depths for stacked Gaussians; opacity gradient of distortion-only loss is 0 (weights detached), means gradient ≠ 0 |
| regularizer-schedule | C3 | loss = 0 before 15000, = 100·mean(edge·dist/α²) + 0.05·(0.4 err_exp + 0.6 err_med) after, vs an independent re-implementation; depth_to_normal recovers an analytic plane normal |
| densify-metric / density-recipe | inherited GOF/Mip | as in the GOF port: abs-gradient channel, clone/split/prune/reset/filter/LR on a CPU fixture |

## Implementation notes
1. `raster.py`: `extension()` JIT load; `raster_settings(camera)` builds Inria view/proj from
   OpenCV K/viewmat (z-near 0.01/far 100, as official `Camera`), checks |cx−W/2|,|cy−H/2| ≤ 0.51
   (kernel and depth_double_to_normal assume a centred principal point; true for our
   mipnerf360 data, cx = W/2 exactly). `_Rasterize` autograd.Function returns
   (color, radii, depth, middepth, alpha, normal, distortion) and passes all 6 image gradients
   to `rasterize_gaussians_backward` (same order as the official Python binding).
2. `render()`: Mip 3D filter on scales/opacity (`filters.filtered_scales_opacities`), SH from
   sh0/shN; outputs HWC: `rgb`, `alpha`, `depth` = nan_to_num(D/α) [H,W,1], `median_depth`,
   `normal` (renormalised camera normal), `normal_raw`, `surface_normal_expected`,
   `surface_normal_median` (official depth_double_to_normal, camera frame), `distortion`
   (raw), `info` (means2d with retained grad, radii).
3. `losses.py`: train.py lines 138–153 verbatim in torch; edge map computed from `target`
   (official computes it from the clamped GT image at camera construction).
4. `density.py`/`filters.py`: copied from the GOF port (`reproductions/gof/plugin`), since the
   official RaDe-GS densification is a verbatim copy of GOF's (diffed). Self-contained.

## Unchanged from 3DGS
SH degree schedule, Adam LRs of non-means fields (runtime 3dgs recipe), init from SfM points
(3-NN scales, opacity 0.1), llffhold-8 split.

## Fidelity gaps
1. **Mesh extraction / geometry metrics not ported** (TSDF fusion + marching cubes, DTU Chamfer,
   TnT F1). Full-view DTU (2DGS preprocessing) is not available locally (only FatesGS sparse
   views + official GT points); the main contribution is geometric, so NVS is only a
   side-check of fidelity. Could be closed by downloading 2DGS DTU and porting TSDF/eval.
2. Decoupled appearance (DTU/TnT only) not ported; unused for Mip-NeRF 360.
3. Mip-NeRF 360 resolution protocol unknown for RaDe-GS (no script at the pinned commit) →
   images_4/images_2 (GOF protocol whose numbers the table copies). Effect: ±0.1–0.3 dB.
4. Runtime differences shared with the GOF port: views sampled with replacement (official
   shuffled stack), optimizer step before densification, SH degree one step later.
5. 3D filter coefficient computed in log space (same value, NaN-safe).
6. Evaluator SSIM (valid conv) / `lpips` package vs official metric.py — shared with baseline.
7. Paper-vs-code differences resolved to code: w_n 0.05 (paper text 5), NDC/edge/α² distortion,
   0.4/0.6 expected/median normal mix (paper.md "Ambiguities").
