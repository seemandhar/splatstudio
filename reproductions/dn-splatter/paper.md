# DN-Splatter — implementation spec

Turkulainen, Ren, Melekhov, Seiskari, Rahtu, Kannala. *DN-Splatter: Depth and Normal Priors for
Gaussian Splatting and Meshing*, WACV 2025, arXiv 2403.17822. Official code:
github.com/maturk/dn-splatter @ 97588b4 (Nerfstudio 1.1.3 plugin, gsplat 1.0.0) in `source/code/`;
Nerfstudio v1.1.3 `splatfacto.py` in `source/nerfstudio/`. **Code overrides paper where they differ.**

## 1. Summary and contributions
- Splatfacto (Nerfstudio 3DGS) + per-pixel sensor depth supervision with an RGB-edge-aware log-L1
  loss, rendered normals from the Gaussians' shortest axis supervised by Omnidata monocular normals
  plus TV smoothing, a min-scale (flatness) loss, and a dense sensor-depth initialization with
  normal-aligned flat Gaussians. Meshes by Poisson on back-projected depth+normal maps.
- **C1 gradient-aware depth loss** (§3.1, Eq. 3) on alpha-normalised expected z-depth.
- **C2 normals from Gaussian geometry** (§3.2, Eq. 4–5,7): n_i = R·onehot(argmin s), flipped
  towards the camera, transformed to camera space, alpha-composited.
- **C3 monocular normal loss L1 + TV smoothing** (§3.2, Eq. 8–9).
- **C4 scale (flatness) loss** (§3.2, Eq. 6).
- **C5 sensor-depth initialization** with normal-oriented, flattened Gaussians (§3.3).
- C6 Poisson meshing from back-projected depth/normal maps (§3.5) — evaluation only, not needed
  for PSNR; mesh metrics need the Faro GT and are out of scope here (gap).

## 2. Equations (verbatim from sections/4_method.tex)
- Depth render: $\hat{D} = \sum_i d_i\alpha_i\prod_{j<i}(1-\alpha_j)$, normalised by the final
  accumulation (gsplat `RGB+ED`). Code: pixels with alpha = 0 get `depth.detach().max()`.
- Eq. 3: $\mathcal{L}_{\hat D} = g_\text{rgb}\frac{1}{|\hat D|}\sum\log(1+\|\hat D-D\|_1)$,
  $g_\text{rgb}=\exp(-\nabla I)$.
  Code (`losses.py:EdgeAwareLogL1`, scalar): `l = log(1+|D̂-D|)`;
  `gx = mean_c|I[:, :-1]-I[:, 1:]|`, `gy = mean_c|I[:-1]-I[1:]|` with `I = gt.clamp(min=10/255)`;
  loss = mean(exp(-gx)·l[:, :-1] over mask[:, :-1]) + mean(exp(-gy)·l[:-1] over mask[:-1]);
  mask = D > 0.1 (`depth_tolerance`).
- Eq. 4: $\hat n_i = R\cdot\mathrm{OneHot}(\arg\min(s_1,s_2,s_3))$, Eq. 5 = standard wxyz quat→R.
- Eq. 6: $\mathcal{L}_\text{scale}=\sum_i\|\arg\min(s_i)\|_1$. Code: `min(exp(scales),dim=1).mean()`.
- Eq. 7: $\hat N=\sum_i \hat n_i\alpha_iT_i$; code then normalises per pixel and maps to [0,1]
  ((N̂+1)/2). Normal pass: legacy `rasterize_gaussians(xys.detach(), depths, radii, conics,
  num_tiles_hit, normals_cam, sigmoid(opac))` — means2d detached, conics/opacity/normals not.
- Eq. 8: $\mathcal{L}_{\hat N}=\frac{1}{|\hat N|}\sum\|\hat N-N\|_1$ (code: mean over H,W,3 of the
  [0,1]-encoded maps). Eq. 9: $\mathcal{L}_\text{smooth}=\sum|\nabla_i\hat N|+|\nabla_j\hat N|$
  (code `TVLoss`: mean|ΔW| + mean|ΔH| of the [0,1] map).
- Eq. 10: $\mathcal{L}=\mathcal{L}_{\hat C}+\lambda_d\mathcal{L}_{\hat D}+\mathcal{L}_\text{scale}+
  \lambda_n\mathcal{L}_{\hat N}+\lambda_s\mathcal{L}_\text{smooth}$, paper λd=0.2, λn=0.1, λs=0.1.
- Mono-depth alignment Eq. 11 (scale/shift LSQ to SfM) — only for datasets without sensor depth;
  not used for MuSHRoom.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| depth weight (effective) | **1.2** (paper 0.2) | `regularization_strategy.py:179` `depth_loss += depth_lambda*depth_loss`, depth_lambda=0.2 (README recommended) |
| normal L1 weight | **1.0** (paper 0.1) | `regularization_strategy.py:get_normal_loss` (normal_lambda unused) |
| normal TV weight | **1.0** (paper 0.1) | same |
| scale loss weight | 1.0 | `get_scale_loss` |
| depth valid threshold | D > 0.1 m | `depth_tolerance` |
| edge image clamp | gt ≥ 10/255 | `dn_model.py:get_loss_dict` |
| photometric | 0.8 L1 + 0.2 (1−SSIM) | splatfacto |
| iterations | 30000 | paper §4, dn_config |
| LR means | 1.6e-4 → 1.6e-6 exp, **not** scaled by scene extent | dn_config |
| LR scales/quats/opac/dc/rest | 0.005/0.001/0.05/0.0025/0.000125 | dn_config |
| SH degree | 3, +1 every 1000 | splatfacto |
| warmup / refine_every | 500 / 100 | splatfacto |
| densify grad thresh | 0.0008 on **absgrad**, × 0.5·max(H,W), vis count starts at 1 | splatfacto `after_train`/`refinement_after` |
| densify size thresh | 0.01 (world units, metric) | splatfacto |
| split screen size / stop | 0.05 / 4000 | splatfacto |
| cull alpha | 0.1 | splatfacto |
| cull scale | 0.5 (after 3000) | splatfacto |
| cull screen size | 0.15 (3000 < step < 4000) | splatfacto |
| reset alpha | every 30·100 = 3000 steps, clamp to 0.2, Adam moments zeroed | splatfacto |
| densify window | step%3000 > num_train+100, step < 15000 | splatfacto |
| continue cull after 15000 | True | splatfacto |
| n_split_samples / size_fac | 2 / 1.6 | splatfacto |
| rasterize mode | classic, near 0.01 | dn_model |
| background | random (train), (0.149,0.1647,0.2157) eval | dn_model populate_modules |
| num_downscales | 0 (full res from step 0) | DNSplatterModelConfig |
| init | 1M points, opacity 0.1, SH dc = RGB2SH | mushroom dataparser / dn_model |

## 4. Schedules
All DN losses active from step 0. Normal flip toward camera every step (paper: "at the beginning").
Densification 500 < step < 15000 within the reset window; opacity reset at step%3000 == 100;
SH degree step//1000.

## 5. Builds on
Splatfacto v1.1.3 (gsplat 1.0.0 rasterization, classic). Unchanged: photometric loss, SH, Adam.

## 6. Protocol (MuSHRoom, iPhone)
- Data: zenodo 10230733 `<room>_iphone.tar.gz`; long_capture = training trajectory,
  `transformations_colmap.json` poses (metric, OpenGL), 738×994 images (no downscale, < 1600),
  uint16 mm depth at image res, no distortion.
- Train = long frames not in `test.txt` (vr_room: 376); test "within/left" = test.txt (42,
  every 10th frame); "with/right" = all short_capture frames (vr_room: 186). Official default
  `eval_mode="all"` evaluates the union; the pipeline reports the two subsets separately.
- Normals: Omnidata v2 DPT-hybrid, "low" (384² squash, bilinear back), png uint8.
- Init: 1M points from incremental TSDF of training sensor depths (`generate_iPhone_pointcloud_
  within_sequence`), Open3D normals (r=0.1, 30 nn) → quats; scales log(mean 3-NN) / z ÷10.
- Metrics: torchmetrics PSNR/SSIM; LPIPS = torchmetrics AlexNet on [0,1] inputs (normalize=False).

## 7. Ambiguities
- Table 4 (VR room) eval split not stated — within (test.txt) used as primary, cross + union
  reported too.
- Loss weights: paper 0.2/0.1/0.1 vs code 1.2/1/1 — code used.
- Code flips normals every step (paper: at the beginning).
- Omnidata png is decoded and converted with diag(1,−1,−1) while rendered normals are in the
  Nerfstudio (OpenGL) camera frame; the L1/TV are invariant to applying the same sign flip to
  both, so the native port compares OpenCV-frame rendered normals with the raw png — numerically
  identical to the official loss whatever the convention (see plan.md).
- Duplicate TSDF samples can give zero 3-NN distance → log(0) scale in the official init.
