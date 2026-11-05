# DN-Splatter native plan

Method `dn-splatter`, module `ss_dnsplatter`, plugin `reproductions/dn-splatter/plugin/`.
Control method `dn-splatter-rgb` (same renderer/density/init, photometric loss only) =
in-house Splatfacto row (paper Table 4 "Splatfacto (L_rgb)" 26.6 on VR room).

## Data (sidecar, `tools/prepare_mushroom.py`)
`~/datasets/mushroom/native/vr_room/`: transforms.json (explicit splits: train 376 long frames,
test = 42 test.txt frames (within/left), validation = 186 short_capture frames (cross/right)),
points.npz (1M TSDF points, official algorithm), `dnsplat/` sidecar keyed by native viewmat:
`depth_mm.npy` uint16 [376,994,738], `normal_omni_u8.npy` [376,994,738,3] (Omnidata v2 png
values), `init.npz` (log_scales, quats, normals), manifest with sha256 (Omnidata ckpt).

## Components
| component | kind | factory | requires → provides | config (defaults = official code) |
|---|---|---|---|---|
| dn-render | renderer | `ss_dnsplatter.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, render.normal.camera, gradient.screen.absolute, prior.sensor_depth, prior.mono_normal | eval_background (0.149,0.1647,0.2157), random_background True, near 0.01, normal_pass True |
| splatfacto-density | density | `ss_dnsplatter.density:create_density` | gradient.screen.absolute | splatfacto 1.1.3 values (paper.md §3), means_lr 1.6e-4, use_sidecar_init True, zero_log_scale −1e4 |
| photometric | loss | splatstudio base | render.rgb | ssim 0.2 (= splatfacto) |
| dn-depth | loss | `losses:create_depth` | render.depth.expected_z, prior.sensor_depth | depth_lambda 0.2, official_double_count True (effective 1.2), tolerance 0.1, rgb_clamp 10/255 |
| dn-normal | loss | `losses:create_normal` | render.normal.camera, prior.mono_normal | l1_weight 1.0, tv_weight 1.0 (official; paper 0.1/0.1) |
| dn-scale | loss | `losses:create_scale` | — | weight 1.0 |

## Mapping to code
- Renderer: gsplat `rasterization(..., render_mode="RGB+ED", absgrad=True, classic, near 0.01,
  tile 16)`; rgb = clamp(render + (1−α)·bg, 0, 1); depth = where(α>0, D, D.detach().max()).
  Normal pass (Eq. 4,7): n = normalize(R(q̂)·onehot(argmin s)); flip if n·(c−μ.detach()) < 0;
  n_cam = R_w2c n (OpenCV); `gsplat.rasterize_to_pixels(means2d.detach(), conics, n_cam,
  meta opacities, isect_offsets, flatten_ids)` + (1−α_n)·(1,−1,−1) (legacy rasterizer's default white
  background in the OpenGL frame, gsplat 1.0 `cuda_legacy/_wrapper.py:339`, mapped to OpenCV by D); N̂ = (normalize(N)+1)/2.
  Convention: official compares Nerfstudio/OpenGL-frame N̂ with diag(1,−1,−1)-converted Omnidata;
  |D a − D b| = |a − b| and TV is sign-invariant, so OpenCV-frame N̂ vs the raw png is
  numerically identical once the background is also mapped through D.
- Density = Splatfacto `after_train` + `refinement_after` with ns = step − 1 (runtime is 1-based):
  stats every step while ns < 15000 (absgrad norm, vis count from 1, max 2D size with the
  gsplat-1.0 eigen radius recomputed from conics), refine when ns % 100 == 0 and ns > 500,
  densify if ns < 15000 and ns % 3000 > num_train + 100 (split: max s > 0.01 or max2D > 0.05
  before 4000; dup computed after the in-place /1.6 of split originals, as official), cull
  (α < 0.1; after 3000 scale > 0.5; 3000 < ns < 4000 max2D > 0.15), cull-only after 15000,
  opacity clamp to 0.2 + Adam moment reset when ns % 3000 == 100 and ns < 15000.
  `initialize` resets the means LR to 1.6e-4 (Splatfacto does not scale by scene extent) and
  loads sidecar log-scales/quats when the means equal the sidecar points.
- Losses: exact numerics of `losses.py` / `regularization_strategy.py` (paper.md §2).

## Probes
| probe | checks | fails if |
|---|---|---|
| depth-loss | independent edge-aware log-L1 on random maps; weight 1.2; constant image → no edge weighting; mask | weighting, log, mask or 1.2 removed |
| normal-loss | L1 + TV vs independent formula | either term removed/reweighted |
| scale-loss | analytic min-scale mean | wrong axis/reduction |
| gaussian-normals (CUDA) | flat Gaussian with known R: rendered N̂ equals R_w2c·R e_min (centre and a partial-alpha pixel vs the official GL-frame white-background formula), flips with camera side, gradient reaches quats and not means2d | normal from wrong axis / no flip / means2d not detached |
| splatfacto-density | synthetic stats: split/dup/cull counts at ns=600, no densify inside the post-reset window, opacity reset at ns=3100 clamps to 0.2 and zeroes Adam moments, means LR reset | schedule or criteria changed |
| sidecar-init | init scales/quats copied, −inf → finite zero-scale | init not applied |
| isotropic-init (splatfacto control) | scales = log(mean 3-NN) ×3 from init.npz col 0, −inf → zero, quats = independent Shoemake formula on the same RNG | oriented init or non-random quats |
| resolution-schedule (splatfacto control, CUDA) | 1/4 for steps 0–2999, 1/2 for 3000–5999, full after; means2d at K/d; eval full res; target = area mean | schedule/intrinsics/target downscale changed |
| renderer-gradient (splatfacto control) | base probe with schedule off + nonzero grads on a scheduled 1/4-res render | — |

## Fidelity gaps
1. gsplat 1.5.3 vs 1.0.0 (projection culling, per-axis radii; max 2D size recomputed from conics
   with the 1.0 formula). Small effect.
2. Init: our Open3D 0.19 TSDF + seeded sampling, not the authors' cloud (random). 35% of the
   1M samples are duplicates with zero 3-NN distance → log(0) = −inf scales in official code;
   we use −1e4 (exp underflows to exactly 0, gradients exactly 0) — same behaviour, finite state.
3. Evaluation: ss-eval PSNR/SSIM (valid-conv SSIM ≈ pytorch_msssim/torchmetrics), LPIPS-VGG
   (official torchmetrics AlexNet on [0,1]); LPIPS not comparable.
4. Evaluation split: headline = test (42 test.txt frames, within). Every single-number MuSHRoom
   NVS table uses the within value (main_sfm_init, ab_2dgs_vs_dn_splatter_mono_sensor caption);
   validation (186 short_capture frames, cross) is secondary. No union (not an official metric:
   dn_pipeline.py averages within_/with_ separately even with eval_mode='all').
8. Splatfacto baseline control `dn-splatter-splatfacto` (= `ns-train splatfacto`,
   eval/batch_run.py:186-207, load_pcd_normals False): same 1M sensor-depth points, isotropic
   log(mean 3-NN) scales and Shoemake random quats (splatfacto.py:214-220), Splatfacto density +
   photometric loss, num_downscales 2 / resolution_schedule 3000 implemented inside the renderer
   (training-render counter, intrinsics/size /d) and the plugin photometric loss (area
   `resize_image` of the target). Residual gaps: base SSIM (valid conv) instead of torchmetrics
   SSIM (same for all three methods); the renderer's schedule counts training renders, which
   equals the runtime step. `dn-splatter-rgb` is a priors-off ablation (oriented flat init, full
   res), not the paper baseline.
9. Normal-pass background: official white (1,1,1) in the OpenGL camera frame = (1,−1,−1) in the
   OpenCV frame used here (fixed after review 1; the probe checks a partial-alpha pixel).
5. View sampling with replacement (Nerfstudio cycles a shuffled list). Small.
6. Meshing / mesh metrics (Poisson, Faro GT) not implemented (geometry claims unverified).
7. Only vr_room benchmarked (only scene with per-scene numbers); 6-scene Table 3 average
   would need 5 more rooms (~1.2 GB, same tool) — optional.
