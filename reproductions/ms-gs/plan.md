# MS-GS native plan

Method `ms-gs`, module `ss_ms_gs`, package `plugin/` (`splatstudio-method-ms-gs`).
Official code `source/code` @50e5040 + rasterizer @e7a2b4f is the numerical reference (paper.md ⚠ items).

## Components
| kind | name / factory | requires → provides | config (defaults = official) |
|---|---|---|---|
| renderer | `ms-gs-renderer` `ss_ms_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, render.multiscale_selection | `filter_small` true, `filter_large` true, `min_rel` 0.5, `max_rel` 2.0, `abs_threshold` 2.0, `opacity_level` 1/255 |
| density | `ms-gs-density` `ss_ms_gs.density:create_density` | gradient.screen.signed | 3DGS ADC (grad 2e-4, percent_dense 0.01, from 500/until 15000/every 100 1×-its, reset 3000 level-0, prune opa 0.005, world-size prune 0.1 after 3000), `max_scale` 7, `insert_at` [1010,1020,1030], `insert_levels` [2,4,6], `insert_threshold` 1.0, `voxel_base` 0.02, `scale_clip` [0.25,2], `target_px` 2.0, `decay_max` 0.95, `decay_min` 1.05, `px_update_from` 250, `px_skip_after_reset` 250, `p_full_res` 0.75, `position_lr_max_steps` 30000, `seed` 0 |
| loss | `ms-photometric` `ss_ms_gs.losses:create_photometric` | render.rgb | `ssim_weight` 0.2, `low_res_weight` 0.1 (same-padded 3DGS SSIM; target resampled to render size) |

The base `photometric` loss is **replaced** (official multiplies the whole 3DGS loss by 0.1 at non-1× scales).

## Fields (non-trainable metadata, never in the autograd graph; lr declared 1e-8 because the contract needs > 0)
| field | shape | init | meaning |
|---|---|---|---|
| `ms_level` | () | 0 | target resolution index r (scale 2^r) the Gaussian belongs to |
| `ms_max_px` | () | −1 | S^max (−1 = uninitialised) |
| `ms_min_px` | () | −1 | S^min |
Density ops: clone copies; split children get both ÷1.6 (official `/(0.8·N)`); prune removes; insertion appends.

## Shared link (renderer ↔ density, process-local like DashGaussian's)
Density owns the multi-scale sampler (needs #train views from `set_training_cameras`, the step,
and per-level iteration counts). Under `torch.is_grad_enabled()` the renderer asks it for the
level of the step being rendered; evaluation (no_grad) renders the given camera as is.

## Implementation notes
1. `coverage.pixel_coverage(params, cam)` (torch, no_grad): 3DGS computeCov2D (t clamped to
   1.3·tan fov, J, WΣWᵀ), undilated conic, ℓ=max(0,2 ln(255σ)), S=min(√(ℓ/Q11),√(ℓ/Q22)); S=0 for
   z≤0.2 or det≤0.
2. `coverage.select(S, max_px, min_px)`: rel=S/stored if stored>0 else 1; keep unless
   (rel_min<0.5 ∧ S<2) or (rel_max>2) [filter_large nested in filter_small, as in forward.cu].
3. Renderer: training → camera scaled to (int(W/2^r), int(H/2^r)), K·(w/W, h/H); select;
   gsplat `rasterization` on the selected subset (classic, white bg like every native method);
   outputs info `ms_sel`, `ms_px`, `ms_level`.
4. Loss: target → render size by antialiased bicubic (≈ PIL bicubic); (0.8 L1 + 0.2 (1−SSIM_same))·(1 | 0.1).
5. Density after_backward: stats/densify/prune only on 1× steps for level 0 (counter of 1× steps);
   range update (Eq. 6, code rule) from `ms_px` + visibility for Gaussians of the current level;
   level-0 opacity reset; insertion at 1010/1020/1030 (Alg. 1 per code: min S over train cams at 2^r
   where visible at 1×, <1 px, level 0; contraction; voxel floor(p/(0.02 r/4)); mean of raw params;
   scale ×2/clip(S̄,0.25,2); then range update over all train cams at 2^r); means LR = 3DGS expon
   schedule over 30000 then constant (overrides runtime's decay over the full 40000).
6. Multi-scale sampler (train.py:152-194): per "epoch" of n_train draws, level 0 w.p. 0.75 else the
   least-trained level of {0}∪{2,3}∪{4,5}∪{6,7} enabled after 1010/1020/1030.

## Data
New dataset `mipnerf360-msgs` (tools/make_multiscale_msgs.py): train = 1× (width-1600, official -r -1)
frames only (lower training scales generated on the fly, so no RAM blow-up from replicating the
75 % 1× bias); test = llffhold8 views PIL-resized from full res at 1,2,4,8,16,32,64× (128× = 12×8 px
is below the loader's 11 px minimum); validation = hard-linked copies. Scenes garden + counter
(per-scene reported). Comparator: `gsplat` trained 30k on the same dataset (1× only = the paper's
single-scale 3DGS row).

## Probes
| probe | measures / fails when |
|---|---|
| pixel-coverage (CPU) | torch S vs independent float64 formula + analytic fronto-parallel isotropic case; fails if dilation kept, level set or min-axis wrong |
| selection-rule (CPU) | culling truth table incl. uninitialised, abs threshold, 2.0 large ratio |
| range-update (CPU) | Eq. 6 code rule (0.95/1.05, uninit, level gating, coarsest/finest exemptions) |
| aggregation (CPU) | voxel pooling of synthetic Gaussians vs independent loop; contraction; enlargement; level/−1 init |
| ms-schedule (CPU) | sampler: 2× never, enable steps, epoch structure, P(1×)≈0.75; loss ×0.1; LR hold after 30000 |
| densify-split-fields (CPU) | split children ÷1.6 px ranges, only level-0 densified/pruned/reset |
| selective-render (CUDA) | end-to-end: insertion creates coarse Gaussians; at 1× they are culled, at 64× small fine ones culled; rendering differs from no-selection |

## Fidelity gaps
1. Training GT for 2×–128× = antialiased bicubic of the 1600-px frame on the GPU, not PIL bicubic of the
   full-res JPEG (test GT is exact). Small; measured in evidence.
2. Views sampled with replacement (runtime) inside the official per-epoch resolution sequence; densify
   after optimizer step; white background; gsplat rasterizer (near 0.01) instead of the MS-GS fork
   (selection is reproduced in torch: same rule, no custom kernel needed).
3. 128× test scale not evaluable (loader minimum 11 px); compare 1–64×.
4. SH-degree step off by one iteration (runtime). Eval SSIM/LPIPS = runtime (valid SSIM, lpips VGG).
