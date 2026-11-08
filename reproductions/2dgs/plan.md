# Plan — 2DGS native port (`2dgs`, module `ss_twodgs`)

## Decision: vendor the official rasteriser, not `gsplat-surfel`
gsplat 1.5.3 `rasterization_2dgs` (built-in `gsplat-surfel`) shares the ray–splat intersection and the
σ=√2/2 low-pass filter (FILTER_INV_SQUARE 2) but differs from hbb1/diff-surfel-rasterization in ways that
change the objective:
| detail | official 2DGS | gsplat 1.5.3 |
|---|---|---|
| depth in depth/median/distortion maps | per-pixel intersection depth s·Tw + Tw.z | per-Gaussian centre depth (RGB+ED channel) |
| distortion | L2 on NDC m = f/(f−n)(1−n/z), n=0.2 f=100 | L1 on raw depth |
| per-pixel near cut | z < 0.2 skipped | none (projection near plane) |
| alpha clamp | 0.99 | 0.999 |
| densification stat | dL/dT[:,2]·depth·W/2 (projected 3D-centre grad) | `gradient_2dgs` from a dummy tensor |
| built-in normal loss | — | normalises both normals, ×alpha on the error (official: unnormalised Σωn · N·alpha.detach()) |
So the plugin JIT-builds the official CUDA (unmodified, `csrc/`, Inria licence, SHA256SUMS) — same route as
PGSR/GOF/gaussian-surfels ports (raster settings/Inria camera conversion adapted from reproductions/pgsr).

## Components
| name | kind | factory | requires → provides | config (default, source) |
|---|---|---|---|---|
| `2dgs-surfel` | renderer | `ss_twodgs.renderer:create_renderer` | state.surfel2d, camera.pinhole → render.rgb, render.depth.surface, render.alpha, render.normal.world.unnormalized, render.normal_from_depth.world.alpha, render.distortion.ndc_l2, render.depth.median, gradient.screen.projected_center | background [0,0,0] (code), depth_ratio 0.0 (code) |
| `2dgs-density` | density | `ss_twodgs.density:create_density` | gradient.screen.projected_center | densify 500/15000/100, grad 2e-4, percent_dense 0.01, opacity_cull 0.05, reset 3000 → 0.01, world prune 0.1·extent after 3000, split N=2, quat init uniform rand (code) |
| `photometric` | loss | built-in | render.rgb | ssim_weight 0.2 |
| `2dgs-distortion` | loss | `ss_twodgs.losses:create_distortion` | render.distortion.ndc_l2 | lambda_dist 0.0 (code; paper 100), start 3000 |
| `2dgs-normal` | loss | `ss_twodgs.losses:create_normal` | render.normal.world.unnormalized, render.normal_from_depth.world.alpha | lambda_normal 0.05, start 7000 |

Variant `2dgs-paperdist`: identical, lambda_dist 100 (paper §5 unbounded value) — ablation only.

Fields: none new. `scales` keeps 3 columns (runtime contract); only `[:, :2]` are used (kernel takes vec2), the
third column is inert (zero gradient) and copied by density ops. `representation="surfel2d"`.

## Probes (each fails if the mechanism is removed)
| probe | measures |
|---|---|
| `ray-splat` (CUDA) | single oblique splat: rendered surface depth at pixels = analytic ray–plane intersection (≠ centre z); alpha = o·exp(−½ρ) with ρ = min(ρ3d, 2|d|²) vs an independent torch evaluation; normal map = camera-facing world normal; finite grads |
| `distortion-kernel` (CUDA) | two fronto-parallel splats at z1,z2: kernel distortion = ω1ω2(m1−m2)² (NDC); 0 for a single splat |
| `normal-loss` | official formula vs independent; depth_to_normal recovers an analytic plane normal; schedule boundary 7000 |
| `distortion-loss` | λ·mean(dist) only for step > 3000 |
| `density-recipe` | clone copies exactly; split samples lie in the disk plane (zero std on normal), scales /1.6, prune opacity<0.05, world prune only after 3000, reset to 0.01, uniform quat init, requires_grad kept |
| `projected-center-grad` (CUDA) | means2d grad equals dL/dT[:,2]·depth·W/2 hack: nonzero, ∝ image size, zero for invisible splats |

## Fidelity gaps
1. Runtime applies the optimiser step before density (official densifies first) — negligible.
2. View sampling with replacement; SSIM valid-conv; SH degree switch one step later — known calibration gaps.
3. λ_dist: code 0 vs paper 100 for unbounded scenes — benchmark code default; `2dgs-paperdist` available.
4. Released code ≠ paper code (README: low-pass gradient removed from densification → fewer Gaussians,
   slightly lower PSNR). Cannot be closed (paper code unreleased).
5. Metrics on float renders, not 8-bit PNGs (official metrics.py reads saved PNGs) — ~0.01 dB.
6. Eval: runtime evaluates in float; official uses same llffhold 8 split and images_4/images_2 (= `mipnerf360`).

feasible_natively: true.
