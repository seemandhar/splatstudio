# SuGaR native plan

Method `sugar`, module `ss_sugar`, plus `sugar-smoke` (same code, compressed stage boundaries,
for ≤2000-step-per-stage smoke tests only). One `ss-train --steps 30000` run executes the whole
official pipeline: 3DGS 1–7000 → coarse SuGaR (sdf) 7001–15000 → mesh extraction after step
15000 → mesh-bound refinement 15001–30000 (= R-SuGaR-15K, 1 Gaussian per triangle, 1M triangles
per fg/bg mesh). Protocol: `mipnerf360-1600` (official code loads `images/` with 3DGS `-r -1`).

## Shared state
A `SugarContext` is created by the renderer factory (first component) and bound by the density
and loss factories (module-level "current context"; `build()` instantiates components in order).
It holds: stage, KNN table (16-NN incl. self, scipy cKDTree on CPU, 256 cores), training
cameras/extent, mesh (`verts` nn.Parameter + private Adam, `faces`, edge pairs, thickness), the
refine SH cap, and a per-step cache for the SDF samples shared by the R and R_Norm losses.
At evaluation (`ss-eval`, fresh process) the context has no mesh and the renderer renders the
standard fields, which the density keeps synchronized with the mesh after every refine step, so
checkpoints are self-contained; the mesh itself is stored in `density.state` (checkpointed).

## Components
| component | kind | requires / provides | config (paper value) |
|---|---|---|---|
| `sugar-renderer` | renderer | provides render.rgb, render.depth.expected_z, gradient.screen.signed, render.depth.accumulated, training.camera, sugar.mesh_binding | `alpha_cap` 0.999 (straight-through, gsplat 0.999 backward cut) |
| `sugar-density` | density | requires gradient.screen.signed | 3DGS DefaultStrategy (refine_stop_iter 7000, reset 3000), `coarse_start` 7000, `prune_at` 9000, `prune_opacity` 0.5, `knn` 16, `knn_every` 500, `extract_at` 15000, `surface_level` 0.3, `n_total_points` 1e7, `n_points_in_range` 21, `range_size` 3, `poisson_depth` 10, `density_quantile` 0.1, `fg_bbox_factor` 1, `bg_bbox_factor` 4, `decimation_target` 1e6, `gaussians_per_triangle` 1, `n_vertices_in_fg` 1e6, `vertex_lr_init/final` 1.6e-4/1.6e-6, `vertex_lr_max_steps` 30000, `open3d_path` ~/models/sugar-pylib |
| `photometric` | loss | base (0.8 L1 + 0.2 D-SSIM) | unchanged |
| `sugar-entropy` | loss | render.rgb | weight 0.1, window (7000, 9000) |
| `sugar-sdf` | loss | render.depth.accumulated, training.camera | weight 0.2, start 9000, samples 1e6, scale 1.5, density_factor 1/16, close 2σ, norm E/10, clamp 10E |
| `sugar-normal` | loss | render.depth.accumulated, training.camera | weight 0.2, start 9000 |
| `sugar-normal-consistency` | loss | sugar.mesh_binding | weight 0.1 (refine stage only) |

## Fields
`sugar_rot2d [2]`, lr 0.001 (official rotation_lr), unused until refinement (set to (1,0)).
Refinement reuses base fields: `scales[:,1:]` = learnable in-plane log scales (lr 0.005),
`scales[:,0]` = log thickness (E/1e6, rewritten every step), `opacities`/`sh0`/`shN` learnable
(official lrs equal the runtime's), `means`/`quats` = derived world values (no gradient; synced).

## Probes (each can fail when the mechanism is removed)
| probe | measures |
|---|---|
| `sdf-field` | implemented f(x) vs independent β√(−2 log d); single flat α=1 Gaussian with ρ=1 gives f = plane distance; R = 0 for consistent depth, > 0 when depth offset |
| `normal-regularizer` | loss 0 for co-planar neighbours, equals an independent loop for tilted ones, gradient reaches quats |
| `level-set` (CUDA) | flat opaque Gaussian wall: extracted λ points at the analytic offset s√(−2 ln λ) on the camera side, normals point to the camera |
| `poisson-mesh` | Open3D Poisson + decimation of sphere samples: radius error < 2 %, triangle target honoured (ProbeUnavailable if Open3D absent) |
| `mesh-binding` | means = barycentres; rotation = [n, rotated edge] vs independent Rodrigues rotation by atan2(y,x); scales (ε,s1,s2); vertex gradient flows; normal consistency 0 for a plane, > 0 for a fold; vertex LR schedule values |
| `stage-schedule` (CUDA) | entropy only inside its window; prune removes opacity < 0.5 at prune_at; extraction replaces the population, optimizers stay aligned (`check_state`), renderer switches to mesh-bound path |
plus the base state/photometric/renderer probes.

## Fidelity gaps
1. Mesh-extraction depth: official rasterizes 2σ diamond proxies with PyTorch3D (first-surface
   z-buffer, g* from pix_to_face); we use gsplat expected depth (α > 0.5) and the 16-NN of the
   back-projected point (official `use_gaussian_depth=True` branch). Could shift the level-set
   search window in semi-transparent regions; level-set search ±3σ mostly absorbs it.
2. Background: runtime convention is white; official black. Affects pixels the mesh/Gaussians do
   not cover (sky beyond 4E in outdoor scenes). Expected small, mostly outdoor.
3. View sampling with replacement (runtime) vs official per-epoch shuffle (all stages).
4. SSIM (valid conv) / LPIPS package vs official lpipsPyTorch; PSNR on clamped RGB (same).
5. gsplat vs Inria rasterizer (alpha cap straight-through fix applied; 3DGS stage uses gsplat
   DefaultStrategy, as our baseline).
6. sqrt guard: −2 log d clamped ≥ 1e-12 before sqrt (official would give inf gradient at d = 1).
7. 3DGS stage is part of the same run (official: separate 3DGS run saved at 7000 before that
   step's densification — matched by refine_stop_iter 7000).
8. Paper's 3DGS reference numbers are images_4/2; we compare to our 1600-px gsplat baseline.
