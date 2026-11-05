# AtomGS native plan

Method `atomgs`, module `ss_atomgs`, package `splatstudio-method-atomgs`.

## Components
| name | kind | factory | requires | provides | config (paper/code values) |
|---|---|---|---|---|---|
| atomgs-raster | renderer | ss_atomgs.renderer:create_renderer | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, render.depth.mean_accumulated, render.depth.median, render.alpha, render.normal_from_depth.world, gradient.screen.inria | background [0,0,0] |
| atomgs-density | density | ss_atomgs.density:create_density | gradient.screen.inria | – | densify_from 100, until 15000, interval 100, clone/split thr 2e-4, warm_up_until 7000, atom_proliferation_until 7000, atom_init_quantile 0.01, atom_decay 0.99, prune 0.005, big_world 0.1, reset every 300 to 0.1, split N 2 /1.6, official_clone_index_shift true, xyz lr 1.6e-4→1.6e-6 over 7000 |
| atomgs-photometric | loss | ss_atomgs.losses:create_photometric | render.rgb | – | lambda_ms_ssim 0.1, ms_ssim_scales 3 (replaces base L1+0.2 D-SSIM) |
| edge-aware-normal | loss | ss_atomgs.losses:create_normal_loss | render.normal_from_depth.world | – | lambda_normal 0.1, until 7000, edge_power 500 |

Renderer: official AtomGS rasterizer (`submodules/diff-gaussian-rasterization` @5b9ce904) vendored
unmodified in `ss_atomgs/csrc`, JIT-built with torch.utils.cpp_extension via tools/ssenv.
Inria FoV camera (centred principal point, znear 0.01, zfar 100), black background.
Normals: official `depth_to_normal(mean_depth)` (NDC unprojection with (W-1) normalisation,
zero-padded border) returned as `surface_normal` (world frame).

No extra FieldSpecs (atom scale is a scalar in density state).

## Mapping contributions → code
- C1 atomized proliferation → density.py: `initialize` (S0 = P1 of 3-NN distances, clamp init
  scales), `densify_and_prune` (split-first with warm-up threshold and max s > S, clone from
  index-shifted padded gradients, prune α<0.005 | max s>0.1·extent, stats reset), `atomize`
  (every 100 it < 7000, isotropic reset, Adam moments of scales zeroed, S *= 0.99),
  `reset_opacity` (every 300 it < 15000, min(α,0.1), moments zeroed), xyz LR override.
- C2 edge-aware normal loss → losses.py `edge_aware_normal_loss` (Sobel/4, (|∇I|−1)^500, L2 over
  xyz, mean) on renderer `surface_normal`, active step < 7000.
- C3 feature rasterizer → raster.py / renderer.py (mean depth D=Σz w unnormalised, median, alpha
  with depth/alpha backward).
- C4 MS-SSIM photometric → losses.py `ms_ssim` (3-scale mean of Inria same-padded SSIM).

## Probes
| probe | cuda | fails if |
|---|---|---|
| feature-raster | yes | official kernel RGB / mean depth / alpha / median differ from an independent float64 EWA compositor, or ∂ΣwD/∂μ differs from autograd of the reference |
| normal-from-depth | no | depth_to_normal does not match an independently written ray unprojection (same official normalisation) or misses an analytic plane normal |
| edge-aware-normal | no | loss ≠ loop reference; edge weighting removed (edge-aligned discontinuity not suppressed); schedule boundary 7000 wrong |
| ms-ssim | no | photometric ≠ 0.9 L1 + 0.1(1−mean of 3 SSIM scales) independent formula |
| atomize | no | min s ≤ S rows not isotropic S / others changed / S not ×0.99 / moments not zeroed / runs at 7000 |
| atom-proliferation | no | result of densify differs from a literal re-implementation of the official sequence (incl. index shift), atom-constrained split or warm-up threshold missing, prune rules wrong |
| init-schedules | no | init clamp / quantile wrong, xyz LR not constant after 7000, reset value/interval wrong |

## Unchanged from 3DGS
Inria projection/EWA/0.99 alpha clamp (vendored kernel), SH schedule, Adam(eps 1e-15) and
base LRs (runtime defaults equal official: scales 0.005, opacity 0.05, quats 0.001, sh0 0.0025,
shN /20), init opacity 0.1, 3-NN init (`--recipe 3dgs` required, checked).

## Fidelity gaps
1. Runtime applies optimizer.step before density; official densifies/atomizes/resets first and the
   replaced tensors skip that iteration's step (1 skipped Adam step per 100 it). Small; not closable
   without runtime change.
2. View sampling with replacement (official: shuffled stack). Small/noise-level.
3. SH degree increases at step 1001/2001/3001 instead of 1000/2000/3000. Negligible.
4. Evaluation: runtime SSIM (valid conv) and lpips-package VGG; compare deltas vs our baseline.
5. Resolution protocol not stated in paper; using 3DGS full_eval images_4/images_2 (`mipnerf360`).
6. The gsplat baseline uses gsplat + white background; AtomGS uses the Inria kernel + black
   background, so the delta-vs-baseline also contains a rasterizer difference (~±0.1 dB).
7. DTU chamfer / Poisson mesh extraction not reproduced (PSNR target only).
