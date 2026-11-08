# HDGS native plan

Methods: `hdgs` (Mip-NeRF 360 outdoor: tex_res 200) and `hdgs-indoor` (tex_res 500, official
run_360.py). Module `ss_hdgs`. Representation `surfel2d` (third scale pinned at -30, unused).
One run = 60000 steps: steps 1–40000 stage 1, Fisher prune + texture init in
`after_backward(step=40000)`, steps 40001–60000 stage 2.

## Components
| component | kind | factory | requires → provides | config (paper/code value) |
|---|---|---|---|---|
| hdgs-raster | renderer | `ss_hdgs.renderer:create_renderer` | state.surfel2d, camera.pinhole → render.rgb, render.depth.expected_z, render.alpha, render.normal.world, render.normal_from_depth.world, render.distortion.hdgs, gradient.hdgs | sort_window 16, background black, depth_ratio 0 |
| hdgs-density | density | `ss_hdgs.density:create_density` | gradient.hdgs | official 2DGS densify (grad 2e-4 /150 its, 500–20000, cull 0.01, reset 3000, percent_dense 0.01, world prune 0.1·extent), expon xyz LR (30000), stage1_steps 40000, prune_ratio 0.1, fisher_probes 1, tex_res, max_tex 500, tex_lr 8e-3, tex_step 8000, tex_gamma 0.1, shn_delay 1000 |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim_weight 0.2 |
| hdgs-geometry | loss | `ss_hdgs.losses:create_geometry` | normal, normal_from_depth, distortion | lambda_dist 100 (>3000), lambda_normal 0.01 (>7000), off after stage1_steps |

Field: `hdgs_textured` (1,), lr 1e-12 (never receives grad), flag set to 1 at the stage switch so
a fresh ss-eval process knows the texture must be loaded (no silent untextured fallback).

## Texture storage (runtime has no global learned fields)
Texture buffer [Ttex,3] + its own Adam (lr 8e-3, eps 1e-8, StepLR) live in a module-level store
shared by density (owner/optimiser) and renderer. The density's `state` dict carries the
texture so `checkpoint.pt["density_state"]["texture"]` persists it. Texture offsets are a
deterministic function of the frozen scales (`texture_index`), recomputed at eval and checked
against a stored checksum. ss-eval: the renderer reads the run directory from `--run` in argv
(same trick as gaussian-surfels' sidecar `--data`) and loads the texture.

## CUDA
Vendor official rasterizers byte-identical except `auxiliary.h` (`#ifndef SORT_WINDOW` guard so
k is a build flag): `csrc/stage1/` (main@b13a359: per-ray k-buffer + per-tile centre-depth sort
+ 5-sample frustum, SH at centre) and `csrc/texture/` (texture@01c6fd4: bilinear texture DC +
SH at intersection). JIT `torch.utils.cpp_extension.load` per (variant, k).

## Fisher pruning (C3, code not released)
At step 40000: for every training camera, render with stage-1 kernel, backward a Rademacher
projection r·I (r ∈ {±1}^{3HW}) w.r.t. activated means and the two scales, accumulate squared
grads → unbiased diag F estimate. U_i = Σ_j log(F_jj + 1e-30); remove the 10 % lowest
(gsplat.strategy.ops.remove).

## Probes
| probe | checks | fails if removed |
|---|---|---|
| raster-vs-reference (CUDA) | stage-1 kernel forward + grads (means, scales, quats, opac, sh) vs dense torch k-sorted 5-sample reference on a random 12-surfel scene; ordering case where centre-depth order ≠ per-ray order | sort, frustum averaging |
| texture-vs-reference (CUDA) | texture kernel forward + grads (texture, shN) vs torch reference with bilinear lookup and intersection-direction SH; non-constant texture changes image | texture |
| fisher-estimator (CUDA) | Hutchinson diag F vs exact per-pixel Jacobian (torch autograd on tiny image); lowest-U removed count = 10 % | pruning |
| stage-schedule (CPU/CUDA) | lr schedule (xyz expon, texture StepLR), freezing at stage 2, loss weights off in stage 2 | two-stage |

## Fidelity gaps
1. Fisher pruning: code unreleased; Hutchinson estimate of diag F (1 probe/view) instead of exact
   per-pixel Jacobians; score Σ log F_jj (paper "L1 norm of log diag"). Small effect (paper
   ablation w/o prune −0.06 dB).
2. Random view sampling with replacement (runtime) vs official shuffled stack.
3. SSIM valid conv (runtime photometric) vs official same-padded SSIM.
4. Stage-2 f_rest: official accumulates grads for 1000 its without zeroing then steps; we keep lr 0
   for the first 1001 stage-2 steps (Adam moments still updated).
5. sh degree increments at step 1001 vs official 1000 (one step).
6. Stage 2 at stage-1 resolution (official script) vs paper "on-demand higher resolution".
7. Texture memory: buffer + Adam ≈ 16 B/texel·4; counter/indoor at tex 500 may need >10 GB.
