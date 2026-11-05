# Plan — Improving ADC (arXiv 2503.14274) as a native SplatStudio plugin

- Method name: **`improving-adc`**; module **`ss_improving_adc`**; package `plugin/`.
- Ablation entry points (for probes / optional benchmark, not the paper target):
  none registered; ablations are config switches of the density component.

## Components
| kind | name | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| renderer | gsplat | `splatstudio.backends.gsplat:create_renderer` (stock) | provides rgb, depth, `gradient.screen.signed` | absgrad False |
| density | improving-adc-density | `ss_improving_adc.density:create_density` | requires `gradient.screen.signed` | see below |
| loss | photometric | `splatstudio.losses:create_photometric` (stock L1 + 0.2 D-SSIM) | | |

The renderer is unchanged: the density reads gsplat's `info` (means2d grad, depths, radii,
conics, opacities, isect_offsets, flatten_ids) and runs a **forward-only CUDA statistics
kernel** that re-traverses gsplat's tile lists with gsplat's alpha numerics, returning per
Gaussian the contributing-pixel count (PixelGS `pixels`) and Σ_p α·T (σ_k, Eq. 15). Kernel adapted
from the Taming-3DGS port (`reproductions/taming-3dgs/plugin/ss_taming_3dgs/csrc/taming_stats.cu`).
This replaces the official fake-colour second rasterization (same quantity, cheaper).

Density config defaults (all from official code): `densify_from 500, densify_until 15000,
interval 100, grad_start 1e-4, grad_end 4e-4, threshold_intervals 145 (=(15000-500)//100),
percent_dense 0.01, min_opacity 0.005, reset_every 3000, reset_value 0.01, extent "sfm_mean"
(C1; "camera" = ablation), scale_means_lr True, pixel_gradient True, depth_gamma 0.37,
depth_power 2, revised_opacity True, pruning "significance" (C3; "opacity" = ablation),
threshold "exponential" (C2; "constant" with grad_constant 2e-4 = ablation), size_prune False`.

No new FieldSpecs. Density `state` holds tensors `grad_accum, pix_count, sigma` [N] (kept aligned
by gsplat ops, reset after each densify) plus scalars `extent, camera_extent, calls`.

## Contribution → code
- **C1 extent** (Eq. 13): `initialize` stores the SfM means; `set_training_cameras` computes
  C̄ = mean train camera centre, e_cam = 1.1·max‖C_i − C̄‖ (Eq. 11), e' = mean‖p_i − C̄‖; uses e' for
  clone/split (`percent_dense·e'`) and rescales the means LR by e'/scene_scale (official
  `create_from_pcd(pcd, cameras_extent=e')` → spatial_lr_scale = e'). The runtime captures
  `initial_lr` after `set_training_cameras`, so the schedule follows.
- **C2 threshold** (Eq. 14, code form): call k (1-based) uses `1e-4·4^(k/145)`.
- **C3 pruning** (Eq. 15/16, Alg. 2, code form): σ accumulated per step; after clone+split:
  `scores = σ[old non-split]`, `q = quantile(scores, N_prune/N_old)` (N_prune = #o<0.005 on the
  post-densify set), prune old non-split with σ<q ∧ o<0.005; no size pruning.
- **B1 PixelGS**: accum += min(1,(z/(0.37·e_cam))²)·‖∇μ2D·(W/2,H/2)‖·count; denom += count over
  visible (radii>0); τ' = accum/denom (NaN→0).
- **B2 revised opacity**: clone: parent opacity overwritten in place (moments kept) then
  duplicated; split: gsplat `split(revised_opacity=True)`.
- Opacity reset 3000..12000 via gsplat `reset_opa` (Adam moments zeroed, as Inria).

## Probes (each fails if the mechanism is removed)
| probe | measures |
|---|---|
| adc-stats-kernel (CUDA) | kernel count & σ == brute-force per-pixel loop; σ == autograd fake-colour gradient (official route); Σσ == rendered alpha |
| adc-scene-extent | e' == independent numpy Eq. 13 on a clustered-camera fixture where e' ≫ e_cam; means LR scaled by e'/e; `extent: camera` ablation gives e_cam |
| adc-threshold | per-call thresholds == 1e-4·4^(k/145), monotone, ends < 4e-4; a Gaussian with τ'=2.5e-4 is densified at call 1 and not at call 144; constant ablation differs |
| adc-significance-prune (CUDA) | background preservation: low-opacity, high-σ large background Gaussian survives while low-σ ones are pruned; mask == independent reimplementation; `pruning: opacity` ablation removes the background Gaussian; no size prune of a huge Gaussian |
| adc-pixel-gradient (CUDA) | τ' after a real render == independent Σ f·count·‖g‖ / Σ count with squared depth factor; differs from gsplat's per-view mean |
| adc-revised-opacity | clone/split opacities == 1−sqrt(1−o); new Adam moments zero; parent moments kept |
| adc-schedule | densify calls exactly at 600..14900, resets at 3000/6000/9000/12000, nothing ≥ 15000 |

## Fidelity gaps
1. **Background colour**: runtime renders over white; official black. Small effect; runtime-wide.
2. **Update order**: runtime steps Adam before density; official densifies before the step
   (clones copy post-step params). Expected tiny effect.
3. **Rasterizer**: gsplat (alpha clamp 0.999, T stop 1e-4, gradient units via W/2,H/2) vs Inria
   (0.99). Count/σ follow gsplat so they describe the image actually rendered.
4. **View sampling**: random with replacement vs official shuffled stack; SSIM valid conv.
5. **σ route**: CUDA forward kernel instead of the fake-colour autograd pass — same quantity
   (probe checks equality); faster.
6. **Quantile edge case**: N_prune/N_old clamped to ≤ 1 (official would raise).
7. Dataset protocol: official README/paper → `full_eval.py` images_4/images_2 JPG → our
   `mipnerf360`. Paper reports averages only → benchmark all 9 Mip-NeRF 360 scenes (+ T&T, DB 2
   each, full datasets). Baselines `baseline3dgs-{mipnerf360,tandt,db}-30000`.
8. Paper 3DGS row (27.39) ≠ 3DGS paper (27.21); compare improvements vs our baseline.

Unchanged from 3DGS: renderer, photometric loss, LRs (except means LR scale), SH schedule,
init (recipe 3dgs), opacity reset.

## Files
`plugin/pyproject.toml`, `plugin/ss_improving_adc/{__init__,density,core,kernels,probes}.py`,
`plugin/ss_improving_adc/csrc/{adc_stats.cu,ext.cpp,LICENSE-3DGS.md}`, `plugin/README.md`,
`plugin/tests/test_improving_adc.py`.
