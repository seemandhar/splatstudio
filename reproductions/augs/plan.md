# AuGS native plan

Method `augs` (module `ss_augs`), variant **Ours (3DGS, sh=3)**; smoke variant `augs-smoke`.
One ss-train job = 30000 base steps (built-in gsplat 3DGS components) + 30 × |train| joint steps.

## Variant / benchmark choice
The paper's 3DGS base (Ours − Δ, supp. Tables per_scene_m360 / diff_ps_mipnerf) is close to our
calibrated `gsplat` 3DGS baseline on the PIL `-r 4/2` protocol (`mipnerf360-pilr`) only for room
(31.67 vs ours 31.69); for bicycle it is 25.74 vs ours 25.45 (−0.29 dB). Compare the **gain over
the in-run base** (`<run>/base_test.json`) with the paper's Δ, not only absolute PSNR. The MCMC variant would need a validated
3DGS-MCMC base (opacity/scale regularisers, cap = points/1.1) that we do not have (#25 not
reproduced). Scenes: bicycle (outdoor, target 25.85, Δ +0.11) and room (indoor, 32.39, Δ +0.72).

## Components
| kind | name / factory | notes |
|---|---|---|
| renderer | `augs-renderer` ss_augs.renderer:create_renderer | gsplat 1.5.3 classic `rasterization` exactly as the built-in renderer; opacity of aug Gaussians × lobe (Eq. 1); provides rgb, expected depth, signed screen grads |
| density | `augs-density` ss_augs.density:create_density | base: `splatstudio.backends.gsplat.Density` (DefaultStrategy + 3DGS reset) + 3DGS position LR schedule over base_steps; at base_steps: enhancement pipeline; afterwards: constant LRs, no densification |
| loss | `photometric` (built-in) | (1−0.2) L1 + 0.2 D-SSIM, unchanged (official refine loss) |

## Fields
| field | shape | lr (joint) | init (orig / aug) |
|---|---|---|---|
| lobe_dir | (3,) | 1e-3 (Adam) | 0 / unit vector to the source camera |
| lobe_t | () | 2e-4 (Adam) | 0 / clip(7θ̄/π, 0.01, 1) |
| lobe_beta | () | 2e-3 (Adam) | 0 / 0 |
| aug_flag | () | none (never in the graph) | 0 / 1 |

## Enhancement pipeline (density.augment at step base_steps)
1. Reload train images from `--data`; optional base test metrics → `<output>/base_test.json`.
2. Render train views (RGB, white bg) + T-threshold depth (CUDA `contrib_stats`, thr 0.7, then
   max-response depth along the ray, official MD). Per-view loss → budget ceil(l_i/Σl · 0.1 N).
3. Per view `fit2d.fit_view` (gsplat `rasterize_to_pixels`, per-pixel background, Inria AA
   compensation, 20 % additions / 200 steps, prune opacity ≤ .005 or ≤ 25 px, 1000 finetune).
4. `inverse.inverse_splat` (GPU-batched SLINK = connected components of dist < thr, WPCA, Eq. 6
   scale, frustum-based θ̄, normal to camera).
5. Append Gaussians; optimisers: `SparseRowAdam` (no bias correction; aug rows; opacity also
   used originals = nonzero opacity grad) for base fields, torch Adam for lobe fields.

## Probes
| probe | measures |
|---|---|
| lobe-opacity | renderer opacity vs numpy Eq. 1, zero beyond Tπ, grads to d/T/β |
| median-depth (CUDA) | front/back switch at T threshold vs reference compositing, depth value |
| error-driven-2d (CUDA) | kernels land in the error region; loss drops |
| inverse-splatting | reprojected covariance ≈ target, near surface across discontinuity, normal faces camera, θ̄/T rule |
| joint-optimisation | no-bias-correction step on aug rows; originals frozen except used opacity |

## Fidelity gaps
1. White background (runtime) vs black (official) — runtime-wide, affects both base and method.
2. Random-with-replacement view sampling vs shuffled stack in base and joint phases.
3. SSIM valid-conv in the photometric loss vs fused_ssim (same padding) — runtime-wide; the 2D
   stage uses same-padding SSIM as official.
4. Base model is gsplat 3DGS (opacity_lr 0.05 etc.) rather than the official train_3dgs.py
   (Inria, opacity_lr 0.025): implied base differs per scene (bicycle −0.29, room +0.02 dB).
5. 2D rasterizer: gsplat `rasterize_to_pixels` (alpha clamp 0.999, exclusive termination) vs the
   official Inria-style 2D rasterizer (0.99, inclusive); pixel-centre offset handled exactly.
   pix_covered counts use the Inria semantics.
6. is_used approximated by nonzero opacity gradient (exact up to measure-zero cancellations and
   gsplat's saturated-alpha gradient drop).
7. Cluster tie-breaking (largest cluster
   when the pivot's is too small) by min point index instead of sklearn label order.
8. Per-view budget additions use ≥ 1 kernel (official loops forever for targets ≤ 2).
9. compute_scale assertion failures (official: whole view dropped) → kernel gets a tiny scale; counted.
10. Back-projection/frustum use K (principal point) where official assumes a centred principal
    point; identical for centred cameras.
11. RNG streams differ (multinomial sampling, view order).
