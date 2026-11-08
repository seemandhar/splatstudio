# Vol3DGS native plan

Method `vol3dgs`, module `ss_vol3dgs`, package `splatstudio-method-vol3dgs` in `plugin/`.
Benchmark protocol: `mipnerf360-1600` (official default `-r -1` = full-res resized to width 1600),
scenes bicycle (outdoor) + room (indoor); baselines `baseline3dgs-mipnerf360-1600-30000` (bicycle)
and `baseline3dgs-mipnerf360-1600b-30000` (room).

## Components
| kind | name / factory | requires | provides | config (paper/code values) |
|---|---|---|---|---|
| renderer | `vol-renderer` `ss_vol3dgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.world.means | tile_size 16, eps2d 0.3, near 0.2, radius_sigma 3, softplus_beta 10, sh_every 2000, background white, kappa_norm 2.506 |
| density | `vol-density` `ss_vol3dgs.density:create_density` | gradient.world.means | — | clone 5e-5, split 1e-7, interval 200, from 500, until 15000, percent_dense 0.01, extent_mult 2, min_theta 0.05, prune_scale 0.01, prune_after 3000, max_points 7e6, split_theta 0.99, lrs {opacities 0.03, sh0 0.003, shN 1.5e-4}, means lr 1.6e-4 -> 1e-5 over 30000 |
| loss | base `photometric` (0.8 L1 + 0.2 D-SSIM) | render.rgb | — | unchanged |

No new fields: `opacities` (logit) is the paper's theta (sigmoid, init 0.1 = runtime init).

## Renderer (C1, C2, C4 colour)
1. gsplat 1.5.3 `fully_fused_projection` under no_grad (near 0.2) -> means2d, conics, depths for
   tiling/sorting only; radius = ceil(3 sqrt(lambda_max)) of Sigma2D (+0.3) like `splat_radius`,
   lambda_max with max(0.1, mid^2 - det); gsplat `isect_tiles` / `isect_offset_encode`.
2. Torch (autograd): mu_c = R mu + t; A_c = (R Rq) diag(1/s^2) (R Rq)^T -> 6 entries;
   kappa = -log(1 - 0.99 sigmoid(o)) * mean(1/s) / 2.506 (code `alpha2density`);
   colour = softplus(SH(dir) + 0.5, beta 10).
3. CUDA JIT kernel `csrc/vol_raster.cu` (adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}
   and reproductions/linear-splatting/.../linear_raster.cu): per pixel unit ray d through pixel
   centre, `evaluate_splat_volr` alpha, 0.99 clamp, 1/255 skip, stop when T(1-a) < 1e-4. Backward:
   analytic derivative (envelope theorem: dq/dt = 0 at t = gamma), NaN-gradient skip like the slang code.
4. `means_in = means.view_as(means)` is the graph entry; density retains its grad (world gradient).
5. SH degree: min(runtime degree, next_step // 2000) during training, via a step hint the density sets.

## Density (C3, C4 LRs)
Inria-style controller (gsplat.strategy.ops duplicate/split/remove): accumulate ‖dL/dμ‖ over
visible steps while step < 15000 and N < 7e6; at step > 500, step % 200 == 0: clone, split
(grad or theta >= 0.99), set theta of new Gaussians to halve(theta), prune theta < 0.05 and
(step > 3000) max scale > 0.01 * 2 * extent; reset statistics. Sets LRs in `initialize` and the
Inria exponential means LR every step (overrides the runtime's 0.01^(t) decay).

## Probes
| probe | measures | fails if |
|---|---|---|
| vol-forward-vs-dense (CUDA) | kernel RGB/alpha vs float64 dense volumetric reference; differs from gsplat splat render | alpha formula / blending wrong or replaced by splatting |
| vol-backward-vs-dense (CUDA) | kernel grads (mu_c, A, kappa, rgb) and end-to-end param grads vs autograd of dense ref | backward wrong |
| analytic-transmittance (CUDA) | centre-ray alpha of isotropic Gaussian = 1-exp(-sqrt(2pi) kappa s); stretching along the ray raises alpha (2D splat invariant) | not volumetric |
| density-reparam (CPU) | kappa formula, theta round trip, halve() | reparam missing/wrong |
| densify-rules (CPU) | clone/split/prune rules, theta halving, LRs, means LR schedule | heuristics changed |
| world-gradient (CUDA) | density statistic = ‖dL/dμ‖ from an independent autograd | screen gradient used |
| recipe-colour (CUDA) | softplus colour and SH schedule | clamp colour / 1000-step SH |
Base probes kept: state-transitions, photometric-gradient (renderer-gradient dropped: no screen grads).

## Fidelity gaps
1. White background (runtime convention, shared baseline) vs official black — tiny on 360 scenes; config `background: black` available.
2. Runtime: random-with-replacement view sampling, valid-conv SSIM (official fused_ssim same-padding), optimizer step before density (Inria densifies before the step) — runtime-wide.
3. gsplat EWA projection used only for tile boxes; official uses its own slang projection (same 3σ + 0.3 formula), minor box-boundary differences.
4. Principal point from K instead of W/2, H/2 (identical for centred intrinsics).
5. Kernel float math (`--use_fast_math`) vs slang; NaN skip also skips inf.
6. The paper's numbers were obtained with an older rasterizer NaN-pruning path (commented in train_volr.py); release code handles it in the rasterizer — we follow the release.
