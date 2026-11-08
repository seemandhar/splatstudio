# AAA-Gaussians — native SplatStudio plan

Method `aaa-gaussians`, module `ss_aaa_gaussians`, package `splatstudio-method-aaa-gaussians`.
Prior art reused: `reproductions/stopthepop` (vendored StopThePop rasteriser + JIT build +
float64 per-pixel-sorted reference, matched), `reproductions/mip-splatting` (3D filter as a
`filter_3d` field computed from `set_training_cameras`, matched), `reproductions/mh-3dgs`
(gsplat MCMCStrategy + per-scene cap from `--data`, matched), `reproductions/3dgut`,
`reproductions/optimal-projection` (ray-based kernels; reference ideas only).

## Components
| kind | name / factory | requires → provides | config (paper/code value) |
|---|---|---|---|
| renderer | `aaa-renderer` `ss_aaa_gaussians.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb | `splatting` = configs/aaa.json (HIER, PTD_MAX, 64/8/4, all culling, load balancing, proper_ewa_scaling, eval_3D, new_aabb), background black |
| density | `aaa-mcmc-density` `ss_aaa_gaussians.density:create_density` | — | cap_max (None → per-scene table of full_eval.py via `--data` basename), noise_lr 5e5, refine 500<s<25000 every 100, min_opacity 0.005, init_opacity 0.5, init_scale_mult 0.1, filter_kernel 0.3, filter_near 0.2, filter_margin 0.15, total_steps (None → ss-train `--steps`, else 30000) |
| loss | base `photometric` (L1 + 0.2 D-SSIM) | | unchanged |
| loss | `aaa-mcmc-reg` `ss_aaa_gaussians.losses:create_reg` | | opacity_reg 0.01, scale_reg 0.01 (train.py:107-108) |

## Fields
`filter_3d` shape (), init 0, lr 1e-8 (never receives a gradient). Zero during training ⇒
kernel uses only the per-view filter k z²/f², exactly like the official `filter_3D=None`.
At the final step the density writes the official `mip_filter_3d` value (min train-view z /
max focal_x · sqrt(0.3), 0 for unseen); test renders then use max(filter², k z²/f²) (Eq. 12).
gsplat relocate/sample_add copy the field from the source Gaussian (same position).

## Renderer implementation
Vendor github.com/DerThomy/AAA-Gaussians-Rasterization @ d03ce2d unmodified into
`plugin/ss_aaa_gaussians/csrc/` (glm, pybind11_json from the StopThePop plugin's third_party),
JIT `torch.utils.cpp_extension.load` (name `ss_aaa_gaussians_rasterizer`). Python wrapper
reproduces `diff_gaussian_rasterization/__init__.py` (filter3D arg, gradient tuple) and
`gaussian_renderer.render` (sigmoid opacity, exp scale, normalised quats, SH concatenated),
camera matrices like the StopThePop plugin (znear 0.01, zfar 100, centred principal point).
The kernel applies the dilation, perpendicular amplitude scaling, view-space bounds, frustum
and tile culling and hierarchical sorting; nothing re-implemented in Python.

## Density
Wrap `gsplat.strategy.MCMCStrategy` (identical relocate/add/noise to 3dgs-mcmc; verified in
mh-3dgs). `initialize`: opacity logit ← logit(0.5), scales ← scales + log(0.1) (runtime
`--recipe 3dgs` already sets log sqrt(mean 3-NN d²)); resolve cap. `set_training_cameras`
stores cameras. `after_backward`: MCMC step with current means lr; if step == total_steps →
compute filter_3d. All ops through gsplat ops (keeps Parameters trainable leaves; runtime
check added).

## Probes (each fails if the mechanism is removed)
| probe | measures |
|---|---|
| `aaa-forward-reference` | CUDA image vs an independent float64 dense torch renderer (3D ray max-contribution, perpendicular-scaled dilation, per-pixel depth sort); fails vs 2D-EWA/no-dilation/volume-scaled variants |
| `aaa-backward-reference` | kernel grads (means, log-scales, quats, opacity, sh0) vs autograd of the dense reference with the kernel's conventions (filter const, SH-dir mean grad dropped) |
| `aaa-perp-scaling` | amplitude factor of a thin disk seen edge-on vs face-on: perpendicular (Eq. 11) ≠ volume (Eq. 8); measured from rendered peak values |
| `aaa-adaptive-filter` | rendered footprint/peak at distance d vs 4d: matches k z²/f² dilation; filter_3d floor enforces max(filter², per-view) when closer than training views |
| `aaa-frustum-bounds` | Gaussian whose mean is behind/outside the frustum but extends into it still renders (matches reference; old HTGS bounding would discard); full-frustum culling removes non-contributing ones |
| `aaa-filter-train` | `filter_3d` equals an independent loop of mip_filter.py (incl. 0 for unseen) and is only written at the final step |
| `aaa-mcmc-schedule` | init opacity 0.5 / scale×0.1, relocation/add cadence 500<s<25000 every 100, cap, noise, reg loss values |

## Fidelity gaps
1. Runtime order: optimizer step before density (official densifies first; on densify steps
   official Adam effectively skips) — negligible (same as stopthepop/mh-3dgs).
2. Views sampled with replacement; SSIM valid conv; LPIPS from `lpips`; SH degree +1 one step
   late (step 1001); means LR without delay multiplier (delay_steps 0 in official ⇒ same).
3. Per-scene cap taken from `--data` basename (bench passes native dir named by scene).
4. Popping (FLIP), large-FOV, multi-resolution evaluations not run (PSNR gate only).
5. Speed not comparable (A6000 vs 4090).
