# PGSR — implementation spec

Paper: *PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface
Reconstruction*, Chen et al., IEEE TVCG 2024, arXiv 2406.06521 (v2, Jan 2025; LaTeX in
`source/latex/`). Official code: github.com/zju3dv/PGSR @ `de24f1a` (`source/code/`), CUDA
rasterizer `submodules/diff-plane-rasterization` is part of the repo (not a git submodule).
Licence: Inria/MPII Gaussian-Splatting licence + ZJU 3DV non-commercial research licence.

## 1. Summary and contributions
- 3D Gaussians are flattened (min-scale loss) and rendered as planes: the rasterizer
  alpha-blends per-Gaussian camera-frame normal and plane distance; depth = distance /
  (normal · ray) ("unbiased depth").
- Single-view regularization: rendered normal vs. normal from the rendered depth (local plane,
  4-neighbour cross product), weighted by (1 − normalized image gradient)^2.
- Multi-view regularization with one random neighbouring training view per step: forward-backward
  reprojection error of the plane depths (geometric) and homography-warped 7×7 grey patch NCC
  (photometric), both weighted by an occlusion weight exp(−φ) (φ < 1 px).
- Exposure compensation: per-image affine (exp(a)·I + b) applied to the L1 term only (TnT only).
- Densification follows AbsGS (+ official extras); multi-view-observation trim.

Contributions:
- C1 Planar flattening loss L_s (§3.1 "Flattening 3D Gaussian").
- C2 Unbiased plane depth rendering: normal map (Eq. render_normal), distance map, depth
  D = 𝒟 / (N K⁻¹ p̃) (§3.1 "Unbiased Depth Rendering"), official CUDA kernel.
- C3 Edge-aware single-view geometric loss L_svgeom (§3.2.1).
- C4 Multi-view geometric consistency L_mvgeom with occlusion weight (§3.2.2).
- C5 Multi-view photometric NCC consistency L_mvrgb on plane-induced homography (§3.2.2).
- C6 Exposure compensation image loss L_rgb (§3.3).
- C7 (code/§4 impl. details) AbsGS densification + official abs-split + multi-view observe trim.

## 2. Equations (verbatim from main.tex)
- $L_s=\parallel \min(s_1, s_2, s_3) \parallel_1$
- $\bm{N}=\sum_{i\in N}\bm{R}_c^T\bm{n}_i\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$; $n_i$ = axis of
  min scale, flipped so that it faces the camera.
- $d_i=(\bm{R}_c^T(\bm{\mu}_i-\bm{T}_c))^T(\bm{R}_c^T\bm{n}_i)$,
  $\bm{\mathcal{D}}=\sum_{i\in N}d_i\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$
- $\bm{D}(\bm{p})=\frac{\bm{\mathcal{D}}}{\bm{N}(p)\bm{K}^{-1}\tilde{\bm{p}}}$
  (code: `All_map[4] / -(N·[ (x-cx)/fx, (y-cy)/fy, 1 ] + 1e-8)`, integer pixel coords; the code's
  local_distance = |n_cam · p_cam| and n faces the camera, hence the minus sign.)
- $\bm{N}_d(\bm{p})=\frac{(\bm{P}_1-\bm{P}_0)\times(\bm{P}_3-\bm{P}_2)}{|\cdot|}$
- $\bm{L}_{svgeom}=\frac{1}{W}\sum_{\bm{p}\in W}(1-\overline{\nabla \bm{I}} )^2 \parallel \bm{N}_d(\bm{p}) - \bm{N}(\bm{p}) \parallel_1$
- $\tilde{\bm{p}_n}=\bm{H}_{rn}\tilde{\bm{p}_r}$, $\bm{H}_{rn}=\bm{K}_n(\bm{R}_{rn}-\frac{\bm{T}_{rn}\bm{n}_r^T}{d_r})\bm{K}_r^{-1}$
- $\bm{L}_{mvgeom}=\frac{1}{V}\sum_{\bm{p}_r\in W}w(\bm{p}_r)\phi(\bm{p}_r)$,
  $w=1/\exp(\phi)$ if $\phi<1$ else 0 (detached), $\phi(\bm{p}_r)=\parallel \bm{p}_r-\bm{H}_{nr}\bm{H}_{rn}\bm{p}_r \parallel$
  (code: not homographies but depth back-projection: p_r → 3D via ref plane depth → neighbour
  camera → depth sampled from the *neighbour plane depth* (bilinear) → 3D → reprojected to ref.)
- $\bm{L}_{mvrgb}=\frac{1}{V}\sum_{\bm{p}_r\in W}w(\bm{p}_r)(1-NCC(\bm{I}_r(\bm{p}_r), \bm{I}_n(\bm{H}_{rn}\bm{p}_r)))$
  (code `lncc`: returns 1 − cc with cc = cross²/(var_r var_n + 1e-8) i.e. *squared* NCC,
  clamped [0,2]; pixels with value ≥ 0.9 masked out; mean over kept pixels.)
- $\bm{L}_{geom}=\lambda_2\bm{L}_{svgeom}+\lambda_3\bm{L}_{mvrgb}+\lambda_4\bm{L}_{mvgeom}$
- $\bm{I}_i^a=exp(a_i)\bm{I}_i^r+b_i$;
  $\bm{L}_{rgb}=(1-\lambda)\bm{L}_1(\tilde{\bm{I}}-\bm{I}_i)+\lambda\bm{L}_{SSIM}(\bm{I}_i^r-\bm{I}_i)$,
  $\tilde I = I^a$ if $L_{SSIM}<0.5$ else $I^r$.
- $\bm{L}=\bm{L}_{rgb}+\lambda_1\bm{L}_{s}+\bm{L}_{geom}$

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| λ1 (scale) | 100 | paper §3.4; code `scale_loss_weight` |
| λ (D-SSIM) | 0.2 | paper; code |
| λ2 single view | 0.015 | paper; code `single_view_weight` |
| λ3 NCC | 0.15 | paper; code `multi_view_ncc_weight` |
| λ4 geo | 0.03 | paper; code `multi_view_geo_weight` |
| single-view start | step > 7000 | code `single_view_weight_from_iter` (paper silent) |
| multi-view start | step > 7000 | code `multi_view_weight_from_iter` |
| patch | 7×7 (`multi_view_patch_size=3`) | paper; code |
| sampled pixels for NCC | 102400 | code |
| φ threshold | 1.0 px | code `multi_view_pixel_noise_th` |
| NCC mask | ncc < 0.9 | code `lncc` |
| neighbours | ≤8, angle < 30°, 0.01 < dist < 1.5 (COLMAP units) | supp. §A; code ModelParams |
| neighbour sort | np.lexsort((angles, diss)) → by distance, ties by angle | code (paper says angle first) |
| ncc_scale | 1.0 (Mip360), 0.5 (DTU, TnT) | run scripts |
| exposure | off (Mip360, DTU), on for TnT; used from step > 1000 | paper §4; code |
| exposure lr | 0.001, Adam betas (0.9, 0.99), init 0 | paper; `scene/app_model.py` |
| iterations | 30000 | paper §4 |
| lr xyz | 1.6e-4·extent → 1.6e-6·extent, exp decay over 30000 | code (3DGS) |
| lr f_dc / f_rest / opacity / scaling / rotation | 0.0025 / 0.0025/20 / 0.05 / 0.005 / 0.001 | code |
| percent_dense | **0.001** (3DGS: 0.01) | code |
| densify from / until / interval | >500, <15000, 100 | code |
| densify_grad_threshold | 0.0002 | code |
| densify_abs_grad_threshold | 0.0008 default; **0.0002 Mip360** (run_mip360.py); 0.00015 TnT | code |
| abs_split_radii2D_threshold | 20 px | code |
| max_abs_split_points | 50000 | code |
| max_all_points | 6,000,000 | code |
| opacity_cull_threshold | 0.005 (TnT 0.05) | code |
| opacity reset | every 3000 while < 15000, to min(o, 0.01) | code |
| world-size prune | max scale > 0.1·extent, after step 3000 | code |
| screen-size prune | max_radii2D > 20 — **never fires** (densification_postfix zeroes max_radii2D first) | code |
| multi-view trim | every 1000 steps < 15000: prune Gaussians with T>0.5-contributions in < 2 training views | code `use_multi_view_trim` |
| background | black (`white_background` False) | code |
| SH | degree 3, +1 every 1000 steps | code |
| seed | 22 | code |

## 4. Schedules
- 1..30000: L_rgb + 100·L_s (L_s on Gaussians with radii > 0, mean of min activated scale).
- step > 1000: exposure image used in L1 if enabled and 1-SSIM < 0.5.
- step > 7000: geometry rendered; L_svgeom; multi-view (one random neighbour from the ≤8 list
  of the reference view, rendered with plane outputs; gradients flow through the neighbour's
  plane depth into L_mvgeom; NCC through ref normal/distance only — the neighbour grey image
  is a training image).
- <15000: densify/prune every 100 (>500), trim every 1000, opacity reset every 3000.

## 5. Builds on
Official 3DGS code (losses, LRs, SH schedule, opacity reset), AbsGS absolute screen gradient
(rasterizer `means2D_abs`, per-pixel |∂L/∂x|, |∂L/∂y| sums), GOF-style resampled clones.
Densification differences vs 3DGS: clones resampled from N(μ, Σ), percent_dense 0.001,
extra abs-gradient split for Gaussians with screen radius > 20 px (≤ 50k per step), opacity
prune 0.005, trim by observation count, cap 6M.

## 6. Protocol
- NVS: Mip-NeRF 360, 9 scenes, `-r 4` outdoor / `-r 2` indoor from full-res `images/`
  (PIL resize), `--eval` llffhold 8, black background, 30k steps; metrics `metrics.py`
  (3DGS ssim, psnr, lpipsPyTorch VGG). Paper reports indoor / outdoor / all averages only
  (Table view_eval); no per-scene NVS table.
- DTU: 15 scans, `-r2 --ncc_scale 0.5`, all 49/64 images (2DGS-preprocessed DTU), TSDF
  (voxel 0.002, max depth 5), Chamfer via DTUeval-python. Per-scene Chamfer reported.
- TnT: 6 scenes, `-r2 --ncc_scale 0.5 --densify_abs_grad_threshold 0.00015
  --opacity_cull_threshold 0.05 --exposure_compensation`, depth filter, F1. Per-scene F1, and
  per-scene ablation PSNR (training-view? — the TnT PSNR in the ablation table, protocol not stated).

## 7. Ambiguities
- Paper sorts neighbours "first by angle then position"; code sorts by distance, ties by angle.
  Follow code.
- Paper NCC vs code: code uses squared correlation 1 − cc² (sign-insensitive), masks ≥ 0.9.
- Paper's homography-based φ vs code depth reprojection through the neighbour depth map. Follow code.
- Paper says depth via K⁻¹p̃; code uses integer pixel coordinates (no +0.5) and cx = W/2.
- README "Code_V1.0" fine-tuned hyper-parameters give better DTU numbers than the paper;
  the v2 paper DTU row equals README "Remove ICP" row. Mip360 numbers come from paper v2 table.
- `knn_f` parameter (random 6-d features, lr 0.01) is unused by any loss — no effect, not ported.
- Abs-split cap: `ratio = limited_num / n_init_points` then quantile over all Gaussians (code quirk,
  ported as is).
