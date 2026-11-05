# GOF native plan

Method `gof`, module `ss_gof`, package `splatstudio-method-gof` (plugin/). Representation
gaussian3d. Target: Mip-NeRF 360 NVS (all 9 scenes — paper reports outdoor/indoor averages
only), 30k steps, `mipnerf360` protocol (images_4 / images_2 = official run_mipnerf360.py).

## Components
| component | kind | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| gof-raytrace | renderer | ss_gof.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.median_ray_z, render.alpha, render.normal.world, render.normal_from_depth.world, render.distortion.gof, gradient.screen.gof | background [0,0,0], kernel_size 0.0 |
| gof-density | density | ss_gof.density:create_density | gradient.screen.gof | 3DGS+GOF densify (2e-4, quantile abs, clone-resample, min_opacity 0.05, world prune 0.1, no effective screen prune), 3D filter (0.2/0.2/0.15), filtered reset 0.01/3000, official xyz LR |
| photometric | loss | ss_gof.losses:create_photometric | render.rgb | 0.8 L1 + 0.2 (1−SSIM same-padding, official) |
| gof-reg | loss | ss_gof.losses:create_regularizer | render.distortion.gof, render.normal.world, render.normal_from_depth.world | lambda_distortion 100, lambda_depth_normal 0.05, both from step 15000 |

Renderer = vendored official CUDA (`submodules/diff-gaussian-rasterization` of GOF @5245b20,
JIT-built via torch.utils.cpp_extension, Inria licence copied) + Python side of
`gaussian_renderer.render` (3D-filtered scales/opacities, normals to world with c2w, 2DGS
depth_to_normal). Also exposes `opacity_field(params, points, cameras)` = official
`integrate` kernel + min over views (C2).

## Fields
`filter_3d` [N] (Mip-Splatting 3D filter std; constant, lr 1e-8 to satisfy runtime,
recomputed from training cameras by the density, copied on clone/split).

## Contributions → probes
| contribution | probe | fails if removed |
|---|---|---|
| C1 ray-Gaussian rendering | `ray-intersection` (CUDA): RGB/alpha/median depth/normal of anisotropic tilted Gaussians vs an independent torch ray-tracing reference; plus gradients (means/scales/quats/opacity/sh0) vs autograd of the reference | projection-based splatting (gsplat) differs measurably; wrong backward fails grad check |
| C2 opacity field | `opacity-field` (CUDA): integrate kernel on a single Gaussian: O(t)=αG(t) before t*, αG(t*) after; min over 2 views | constant / projected opacity fails |
| C3 distortion (detached weights) | `distortion-formula` (CUDA): kernel distortion map vs reference NDC formula; d/d opacity of distortion-only loss == 0 while d/d means ≠ 0 | undetached weights → nonzero opacity grad |
| C4 normals + depth-normal | `normal-consistency`: plane depth → analytic normal by depth_to_normal; loss schedule (0 before 15000, weights 100/0.05) | |
| C5 densification metric | `densify-metric` (CUDA+CPU): kernel abs channel ≥ L2 of signed sum; synthetic stats with cancelling gradients get densified via quantile Q only | removing abs path → not densified |
| Inherited 3DGS/Mip recipe | `density-recipe`: clone resamples positions, split 2×/(1.6), prune <0.05, world prune, filtered reset 0.01, 3D filter vs independent loop, schedule boundaries | |

Base probes kept: state-transitions, photometric-gradient (own SSIM).

## Unchanged from 3DGS
SfM init, 3-NN scales, opacity 0.1, LRs, SH schedule, densify intervals, xyz exponential LR.

## Fidelity gaps
1. Random-with-replacement view sampling (runtime) vs official without-replacement stack — small noise.
2. Mesh extraction (C6: CGAL Delaunay tetra + binary-search marching tetrahedra) not ported;
   it does not affect NVS metrics; DTU/TnT geometry not benchmarked. Closable later (tetra-triangulation needs CGAL build).
3. Decoupled appearance (TnT only) not implemented — not used for Mip-NeRF 360.
4. 3D filter coefficient computed in log space (NaN-safe, same value) as in our Mip-Splatting port.
5. Evaluation SSIM uses valid conv (runtime evaluator) vs official same-padding; LPIPS from `lpips` package — shared with the baseline.
6. Runtime white-background evaluation default is irrelevant: the renderer owns the background (black, official).
7. Final-iteration 3D filter update skipped in last 100 steps exactly as official.
