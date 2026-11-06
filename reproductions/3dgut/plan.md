# 3DGUT — native SplatStudio plan

Method name **`3dgut`**, module **`ss_3dgut`**, package `reproductions/3dgut/plugin/`.
Target row: "Ours" (unsorted) of Table 1 / supp. Table 7.

## Renderer decision: gsplat 1.5.3 UT + eval3d kernels, called directly
gsplat's 3DGUT path (`fully_fused_projection_with_ut` + `rasterize_to_pixels_eval3d`) was
contributed to gsplat by the 3DGUT authors and implements the paper's sigma points / weights
(Cameras.cuh `world_gaussian_sigma_points`, λ = α²(3+κ)−3, w₀^Σ = λ/(3+λ)+1−α²+β) and the
3DGRT max-response evaluation (`gro = S⁻¹Rᵀ(o−μ)`, `grd = normalize(S⁻¹Rᵀd)`,
α = min(0.999, σ·exp(−½‖grd×gro‖²))), with OpenCV pinhole/fisheye, F-theta and rolling-shutter
camera models, and gradients that bypass the projection. **But** `gsplat.rasterization()` does
not forward `ut_params`, so it runs with gsplat's defaults α=0.1, require_all_sigma_points_valid
=True — not the paper (α=1, β=2, κ=0, any-valid). The plugin therefore re-implements the thin
Python glue of `rasterization()` (UT projection → SH → isect_tiles → eval3d raster) and passes
`UnscentedTransformParameters(alpha=1, beta=2, kappa=0, in_image_margin_factor=0.1,
require_all_sigma_points_valid=False)`. Vendoring the official 3dgrut kernels (Slang/tcnn/OptiX
build tree) is not justified: the math is the same; remaining kernel-level differences are
listed as gaps. No custom CUDA build.

## Components
| contribution | kind | factory | requires → provides | config (paper values) |
|---|---|---|---|---|
| C1 UT projection, C2 3D eval | renderer | `ss_3dgut.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.world.means, camera.distorted.ut | ut_alpha 1, ut_beta 2, ut_kappa 0, margin 0.1, require_all false, eps2d 0.3, background white |
| C4 3D-gradient ADC | density | `ss_3dgut.density:create_density` | gradient.world.means | grad_threshold 2e-4, densify_every 300, start 500, stop 15000, prune_every 100, prune_opa 0.005, reset_every 3000, reset_value 0.01, percent_dense 0.01, official_init true, init_scale_factor 0.01, init_opacity 0.1 |
| loss | loss | `splatstudio.losses:create_photometric` (unchanged) | render.rgb | 0.8 L1 + 0.2 D-SSIM (code; paper text says L2) |

Camera dict: core loader yields pinhole `K/viewmat/width/height`. The renderer additionally
honours optional keys `camera_model` (`pinhole|fisheye|ftheta`), `radial_coeffs` ([6] OpenCV or
[4] fisheye), `tangential_coeffs` [2], `thin_prism_coeffs` [4], `viewmat_rs` + `rolling_shutter`
(`top_to_bottom|left_to_right|...`). `ss_3dgut.cameras` loads a transforms.json with
OPENCV / OPENCV_FISHEYE parameters into such dicts (plugin-side; the core loader rejects
distortion and cannot be changed) — used by probes / offline eval.

No new fields. Density state: `accum [N]`, `count [N]` (kept aligned through gsplat ops).

## Implementation notes
- render(): `scales=exp`, `opac=sigmoid`; `fully_fused_projection_with_ut(..., ut_params)` →
  radii, means2d, depths, conics; SH: `dirs = means − campos`, `spherical_harmonics(deg, dirs,
  shs, masks=radii>0) + 0.5`, clamp ≥ 0; append camera-z depth channel; `isect_tiles`,
  `isect_offset_encode`; `rasterize_to_pixels_eval3d(means, quats, scales, colors, opac, ...,
  backgrounds=[1,1,1,0], ut_params, camera kwargs)`; depth ED = D/alpha. `info` carries `campos`,
  `radii`, `means` handle.
- density.before_backward: register a one-shot `register_hook` on `params["means"]` that stores
  `grad` (runtime zeroes grads before `after_backward`); store `campos`.
- after_backward (official `GSStrategy`, gs.py): if step<15000: m = grad≠0;
  accum[m] += ‖grad[m]·‖μ[m]−campos‖‖/2, count[m] += 1. `check(step, start, end, f)` = step>start
  ∧ step<end ∧ step%f==0. Densify (500,15000,300): g = accum/count (nan→0); clone g≥thr ∧
  max(exp s) ≤ 0.01·extent (ops.duplicate), then split g≥thr ∧ max s > 0.01·extent on original
  indices (ops.split: N=2, samples ~ N(0, S) rotated, scale/1.6); zero buffers. Prune (500,15000,100):
  σ<0.005 (ops.remove). Reset (0,15000,3000): ops.reset_opa(0.01) (clamp + zero Adam state).
  No screen/world-size prune.
- official init (`set_training_cameras`): scales = log(0.01·min_c‖μ−c‖) isotropic,
  quats = normalise(U[0,1)^4) seeded 42; opacity 0.1 already. Overwrites `--recipe 3dgs` kNN init.
- Scene extent: `--recipe 3dgs` (1.1 × max train-camera distance) ≈ official cameras_extent.

## Probes
| probe | checks | fails if removed |
|---|---|---|
| ut-moments | gsplat UT means2d/cov (from conic, minus eps2d) for anisotropic Gaussians under pinhole, OpenCV-radial and fisheye cameras equal an independent float64 torch UT (Eq. 5–9, α=1); and the α=0.1 gsplat default gives measurably different covariance | reverting to rasterization() defaults / EWA |
| ut-vs-ewa-mc | for a large Gaussian under strong fisheye distortion, UT covariance is closer (KL) to a Monte-Carlo projection than EWA linearisation (Supp. C claim) | swapping UT for EWA |
| eval3d-response | rendered alpha of a tilted anisotropic Gaussian equals the independent ray-space max-response formula α = σ exp(−½‖ô_g×d̂_g‖²) per pixel, and differs from the 2D-conic (EWA) render | replacing with 2D conic eval |
| eval3d-gradient | ∂L/∂(μ, s, q, σ, colour) from the kernel equals autograd of an independent float64 torch compositor (2 Gaussians, pinhole) | — (backward correctness) |
| fisheye-render | fisheye render of a small Gaussian peaks at the analytic fisheye projection of its centre (equidistant θ model), not at the pinhole projection | dropping camera distortion |
| densify-statistic | accum equals ‖∇μ·dist‖/2 computed independently; schedule: densify at 600/900 not 100/700, prune every 100, reset at 3000/…/12000, none ≥15000 | using 2D grads / 3DGS cadence |
| official-init | scales = 0.01·nearest-camera distance, unit random quats | — |
+ base `state-transitions`, `photometric-gradient`.

## Benchmark
Paper per-scene available (supp. Tab. 7). Mip-NeRF 360 **garden** (outdoor, 26.90) + **counter**
(indoor, 29.03), 30k steps, `--recipe 3dgs`, against `baseline3dgs-mipnerf360-30000`; compare
improvement over our gsplat baseline to paper's "Ours − 3DGS-paper per scene" (garden −0.51,
counter +0.33 vs 3DGS paper 27.41 / 28.70). Distorted-camera benchmarks (ScanNet++ fisheye,
Waymo rolling shutter) are licensed and unavailable → not benchmarked.

## Fidelity gaps
1. **Max alpha 0.999 (gsplat) vs 0.99 (official)**, and no min-kernel-response cut (official
   accepts hits only where ρ > 0.0113 ≈ 3.0σ; gsplat bounds at 3.33σ with α ≥ 1/255 only). Small
   effect (≤0.05 dB expected); closable by vendoring and patching gsplat's eval3d kernels.
2. **Tile culling**: official uses StopThePop rect/tight-opacity bounding + tile culling; gsplat
   uses opacity-aware 3.33σ rect bounding. Speed-mostly; minimal quality effect.
3. **Background**: runtime white vs official black. Same as all SplatStudio runs; matters only
   for empty pixels.
4. **Distorted / rolling-shutter training through `ss-train`**: core loader rejects distortion.
   Renderer supports it (probed), but the paper's ScanNet++/Waymo experiments cannot run (data
   licensed and core loader). Closable with a core camera-schema extension (TODO-runtime).
5. **Sorted variant (k-buffer MLAB, C3)** and secondary-ray hybrid rendering not implemented;
   target is the unsorted row.
6. **View sampling** random with replacement (runtime) vs official shuffled epochs; step index
   1-based vs official 0-based (cadence shifted by one step).
7. Scene extent from train cameras vs official all cameras (≤ a few %).
8. Official SSIM/LPIPS implementations differ (runtime: valid-conv SSIM, LPIPS-VGG).
