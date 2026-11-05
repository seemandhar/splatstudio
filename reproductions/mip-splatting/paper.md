# Mip-Splatting: Alias-free 3D Gaussian Splatting — implementation spec

arXiv 2311.16493 (CVPR 2024). Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger.
Official code: https://github.com/autonomousvision/mip-splatting (cloned to `source/code_full/`).
**Paper-era code = commit `1ecf7d5` (2024-04-16)**; the later commits `24af724`/`d6ecde3`
(2024-04-18, "add GOF densification") replace 3DGS densification with the Gaussian Opacity
Fields gradient metric, which postdates the paper and is NOT part of the reproduced method.
`source/code/` is the current HEAD (with GOF) for reference only.

## 1. Method summary
- 3DGS, unchanged except for how each Gaussian is turned into a screen-space splat.
- Each Gaussian gets a fixed (non-learned) isotropic 3D low-pass filter whose variance is set
  by the highest sampling rate (focal / depth) at which any training camera sees it.
- The 3D filter is convolved into the Gaussian (covariance + s/ν²·I) with opacity compensated
  so the integrated density is preserved; gradients flow through the filtered Gaussian.
- 3DGS's screen-space dilation (+0.3 to the 2D covariance, no compensation) is replaced by a
  "2D Mip filter": +0.1 to the 2D covariance with det-ratio opacity compensation.
- The filter sizes are recomputed every 100 iterations (training cameras only) and stored with
  the model; at test time they are constant.

Contributions
- **C1 3D smoothing filter** (§4.1, Eq. 7–9): multiview frequency bound per Gaussian + 3D
  Gaussian low-pass convolution with opacity compensation; filter-aware opacity reset (code).
- **C2 2D Mip filter** (§4.2, Eq. 10): box-filter approximation, Σ2D + s I with compensation.

## 2. Equations
Eq. 7 (sampling interval): $\hat{T} = \frac{1}{\hat{\nu}} = \frac{d}{f}$

Eq. 8 (max sampling rate of primitive k):
$\hat{\nu}_k = \text{max}\left(\left\{ \mathds{1}_n(\bp_k) \cdot \frac{f_n}{d_n}\right\}^{N}_{n=1}\right)$
— $\mathds{1}_n$: centre inside the view frustum of training camera n; $d_n$ depth of the centre.

Eq. 9 (3D smoothing):
$\cG_k(\bx)_\text{reg} = \sqrt{\frac{|\bSigma_k|}{|\bSigma_k + \frac{s}{\hat{\nu}_k} \cdot \bI|}} \, e^{-\frac{1}{2} (\bx-\bp_k)^T (\bSigma_k + \frac{s}{\hat{\nu}_k} \bI)^{-1} (\bx-\bp_k)}$

Eq. 10 (2D Mip filter):
$\cG^{2D}_k(\bx)_{\text{mip}} = \sqrt{\frac{|\bSigma^{2D}_k|}{|\bSigma^{2D}_k + s \bI|}} \, e^{-\frac{1}{2} (\bx-\bp_k)^T (\bSigma^{2D}_k + s \bI)^{-1} (\bx-\bp_k)}$

### Exact official numerics (commit 1ecf7d5)
`scene/gaussian_model.py::compute_3D_filter(cameras=trainCameras)` (no grad):
```
for each training camera: xyz_cam = R^T-world-to-cam(xyz);  z = xyz_cam.z
  valid_depth = z > 0.2;  z = clamp(z, 0.001)
  x = x/z*focal_x + W/2;  y = y/z*focal_y + H/2
  in_screen = -0.15W <= x <= 1.15W  and  -0.15H <= y <= 1.15H
  distance[valid] = min(distance[valid], z[valid])          # init 1e5
  focal_length = max over cameras of focal_x
distance[~any_valid] = distance[any_valid].max()
filter_3D = distance / focal_length * sqrt(0.2)              # a std-dev, [N,1]
```
So the filter **variance** is `0.2 · (d_min / f_max)²` — i.e. the paper's s/ν̂² with s = 0.2
(the paper writes s/ν̂; the code squares the std). Note f_max is the max focal over all
cameras (not per camera f_n/d_n); identical for Mip-NeRF 360 (one intrinsics per scene).

`get_scaling_with_3D_filter = sqrt(scales² + filter_3D²)` (axis-wise: R S² Rᵀ + f² I = R (S²+f²) Rᵀ).
`get_opacity_with_3D_filter = sigmoid(o) * sqrt(prod(scales²) / prod(scales² + filter_3D²))`.
The renderer receives the filtered scales and opacity; gradients flow to `_scaling`/`_opacity`.

2D Mip filter (`cuda_rasterizer/forward.cu::computeCov2D`, kernel_size = 0.1):
`det_0 = max(1e-6, |Σ2D|)`, `det_1 = max(1e-6, |Σ2D + k I|)`, `coef = sqrt(det_0/(det_1+1e-6)+1e-6)`,
`Σ2D += k I`, opacity ← opacity·coef; backward differentiates through coef. Alpha clamp 0.99.

Filter-aware opacity reset (`reset_opacity`):
`new_raw = inverse_sigmoid( min(opacity_with_filter, 0.01) / coef3D )` (moments zeroed).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| 2D Mip filter variance `kernel_size` | 0.1 | paper §4.2 "cover a single pixel"; `scripts/run_mipnerf360.py --kernel_size 0.1`; `arguments/__init__.py` default 0.1 (commit 746a17c fixed 0.3→0.1 "to match the paper") |
| 3D filter coefficient s | 0.2 (variance), std factor sqrt(0.2) | `gaussian_model.py:190` |
| near-depth for visibility | z > 0.2 | code |
| frustum margin | 15 % of W/H on each side | code ("similar tangent space filtering") |
| recompute interval m | 100 | `train.py` |
| all 3DGS hyper-parameters | unchanged (lr, densify 500–15000 every 100, grad 2e-4, reset 3000, λ_dssim 0.2, 30k iters) | `arguments/__init__.py` |
| ray_jitter / resample_gt_image / sample_more_highres | False | defaults (not used for 360) |

## 4. Schedules
- `compute_3D_filter` at iteration 0 (after init), after every `densify_and_prune`
  (iterations 600, 700, …, 14900: `iteration > 500 and % 100 == 0 and < 15000`), and every 100
  iterations with `15000 < iteration < 30000 - 100` (15100 … 29800). Not at 15000, not after 29800.
- opacity reset at `iteration % 3000 == 0` while `iteration < 15000` (3000…12000), after the
  densification/filter update of the same iteration, filter-aware.
- SH degree +1 every 1000 iterations; position LR exp decay 1.6e-4→1.6e-6 (3DGS).

## 5. What it builds on
Official 3DGS code (Inria), with the rasterizer modified (kernel_size + compensation, and a
subpixel offset input unused for 360). Densification, losses (L1 + 0.2 D-SSIM), optimiser, SH
schedule all as 3DGS. Densification statistics use unfiltered `get_scaling`/`get_opacity`
(clone/split threshold, min-opacity prune, big-world-scale prune); screen radii come from the
filtered splats.

## 6. Protocol (single-scale train, same-scale test — the one we reproduce)
- Mip-NeRF 360, 9 scenes, `--eval` (llffhold 8), outdoor `-r 4`, indoor `-r 2`, 30k iterations,
  black background (`white_background` False), LPIPS-VGG (`lpips` package), PSNR on [0,1].
- **Images**: `-r 4/-r 2` with default `--images images` → full-resolution JPEGs resized by PIL
  (`Image.resize`, default bicubic) in `utils/general_utils.py::PILtoTorch`, NOT the dataset's
  pre-downsampled `images_4/images_2` that the 3DGS paper used. This explains why their 3DGS*
  retrain (27.70) beats 3DGS's published 27.21.
- Paper numbers: main Table 3 (averages), supplementary Table (per scene, `tab/normal_setting_360_supp.tex`).
- Multi-scale protocols (Blender mtmt/stmt, 360 stmt) out of scope.

## 7. Ambiguities / decisions
- s/ν̂ vs s/ν̂² : code uses variance 0.2·(d/f)² → we follow the code.
- f_n per camera vs global f_max: code uses global max focal_x; we follow the code (identical here).
- Paper: "inside the view frustum" → code uses projected centre within [-15 %, 115 %] of the
  image and z > 0.2; we follow the code. Principal point: code assumes W/2, H/2; we use the
  camera's K (cx, cy), equal to W/2, H/2 for these PINHOLE datasets up to ±0.5 px.
- Gaussians never visible: filter from the largest visible distance (code).
- The 3D filter is constant between recomputations; newly densified Gaussians get a fresh filter
  immediately (recompute right after densify), so copy semantics for new Gaussians do not matter.
