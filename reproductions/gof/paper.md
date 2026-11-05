# Gaussian Opacity Fields (GOF) — implementation spec

Yu, Sattler, Geiger. *Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in
Unbounded Scenes*, SIGGRAPH Asia 2024 (TOG). arXiv 2404.10772.
Official code: github.com/autonomousvision/gaussian-opacity-fields, commit `5245b20`
(cloned to `source/code/`; licence: Inria/MPII Gaussian-Splatting licence, non-commercial
research use). Built on 3DGS + Mip-Splatting (3D filter) + 2DGS regularisers.

## 1. Summary and contributions
- Gaussians are evaluated by explicit **ray–Gaussian intersection** (1D Gaussian along the ray,
  closed-form maximum) instead of 2D projected splats; same tile-based alpha blending.
- This defines an **opacity field** O(x) at any 3D point (min over training views), whose 0.5
  level set is the surface.
- Training adds 2DGS **depth distortion** (weights detached) and **depth–normal consistency**
  with the normal of the ray–Gaussian *intersection plane*.
- **Improved densification**: accumulate per-pixel gradient *norms* (abs) in addition to the
  3DGS signed sum.
- **Tetrahedral mesh extraction**: Delaunay tetrahedra on Gaussian centres + 3σ box corners,
  tile-based opacity evaluation, marching tetrahedra with 8-step binary search.

Contributions
- C1 (§3.1) ray–Gaussian intersection volume rendering (custom CUDA rasteriser).
- C2 (§3.2) Gaussian Opacity Field (opacity of arbitrary 3D points; `integrate` kernel).
- C3 (§3.3) depth distortion on intersection depths with detached weights.
- C4 (§3.3) intersection-plane normals + depth-normal consistency.
- C5 (§3.3 "Improved Densification") gradient-norm accumulation metric M.
- C6 (§3.4) tetrahedral grid + binary-search marching tetrahedra (mesh extraction only;
  irrelevant to NVS metrics).
- Inherited: Mip-Splatting 3D smoothing filter (code: `compute_3D_filter`, filter-aware
  opacity/scale, filter-aware `reset_opacity`), decoupled appearance (TnT only).

## 2. Equations (verbatim from `main/04_method.tex`)
Gaussian: $\cG_k(\bx) = e^{-\frac{1}{2} (\bx-\bp_k)^T \bSigma_k^{-1}(\bx-\bp_k)}$, $\bSigma_k = \bR_k \bS_k\bS_k^T \bR_k^T$.

Ray in Gaussian frame (Eq. 2–4): $\bo_g = \bS^{-1}_k \bR_k (\bo -\bp_k)$, $\br_g = \bS^{-1}_k \bR_k \br$, $\bx_g = \bo_g + t \br_g$.

1D Gaussian (Eq. 5): $\cG_k^{1D}(t) = e^{-\frac{1}{2} (\br_g^T\br_g t^2 + 2\bo_g^T\br_g t + \bo_g^T\bo_g)}$

Intersection depth (Eq. 6): $t^* = -\frac{B}{A}$, $A = \br_g^T\br_g$, $B = \bo_g^T\br_g$;
$\cE(\cG_k, \bo, \br) =\cG_k^{1D}(t^*)$.

Volume rendering (Eq. 7): $\bc(\bo, \br) = \sum^K_{k=1} \bc_k\,\alpha_k\,\cE(\cG_k, \bo, \br) \prod_{j=1}^{k-1} (1 - \alpha_j\,\cE(\cG_j, \bo, \br))$

Opacity along a ray: $\bO_k = \cG^{1D}_k(t)$ for $t\le t^*$, $\cG^{1D}_k(t^*)$ otherwise;
$\bO(\bo, \br, t) = \sum_k \alpha_k\,\bO_k \prod_{j<k} (1 - \alpha_j\,\bO_j)$;
$\bO(\bx) = \min_{(\bo, \br)} \bO(\bo, \br, t)$.

Distortion: $\mathcal{L}_{d} = \sum_{i,j}\omega_i\omega_j|t_i-t_j|$, gradient of $\omega$ detached.

Normal: $\bn_i = -\bR^T_k \, \bS^{-1}_k\br_g$ (normalised). Consistency:
$\mathcal{L}_{n} = \sum_{i} \omega_i (1-\bn_i^\top\bN)$, $\bN$ from finite differences of the depth map.

Final loss: $\mathcal{L} = \mathcal{L}_{c} + \alpha \mathcal{L}_{d}  + \beta  \mathcal{L}_{n}$,
$\mathcal{L}_c = 0.8 L_1 + 0.2 (1-\mathrm{SSIM})$.

Densification metric: $M = \sum_{i} \|\frac{dL}{d\bp_i} \frac{d\bp_i}{d\bx}\|$ (vs. 3DGS
$\|\sum_i \cdot\|$).

### What the official code actually computes (differences from the paper text)
- Kernel (`forward.cu` renderCUDA): ray through pixel centre in camera space
  $(\frac{u+0.5-W/2}{f_x}, \frac{v+0.5-H/2}{f_y}, 1)$; view2gaussian packs the quadric
  ($\Sigma^{-1}$ terms, B, C) so $t$ is **camera-z depth**. Skipped if $t \le 0.2$ (NEAR_PLANE).
  $\alpha = \min(0.99, o\,\cE)$, skip if $<1/255$, stop when $T<10^{-4}$.
- Rendered depth = **median** depth: $t$ of the last Gaussian with $T>0.5$ before blending
  (gradient only to that Gaussian). Rendered normal = blended normalised plane normals
  (camera frame), python normalises and rotates to world with $c2w$.
- Distortion in the kernel = 2DGS form on NDC-mapped depth
  $m = \frac{f t - f n}{(f-n)t}$ ($n=0.2, f=100$): $\sum_i w_i (m_i^2 A_{i-1} + D2_{i-1} - 2 m_i D1_{i-1})$,
  then divided by $(1-T)^2+10^{-7}$. Backward: `dL_dweight = 0` (weights detached; only
  $dm/dt$ path); the normalisation's T-dependence is also dropped.
- Depth normal (`utils/depth_utils.py`, 2DGS): back-project median depth with FoV intrinsics
  at pixel centres (+0.5), central differences, cross(dx, dy), normalised, border = 0.
  $\mathcal{L}_n$ = mean over pixels of $1 - n_{render,world}\cdot N$ (no alpha weighting
  beyond the blending inside the kernel; the rendered normal is renormalised).
- Densification statistics: means2D grad is [N,3]; x,y = 3DGS signed sums (computed with the
  projected 2D conic as a pseudo-gradient, the kernel does not backprop through it), z =
  $\sum_\text{pixels} (|g_x| + |g_y|)$ (L1 per pixel). Stats: `accum += ||g[:2]||`,
  `accum_abs += |g[2]|`, `denom += 1` for radii>0.
- `densify_and_prune(max_grad=2e-4, min_opacity=0.05, extent, max_screen=20 if step>3000)`:
  `ratio = mean(grads >= 2e-4)`, `Q = quantile(grads_abs, 1-ratio)`; candidate =
  `grads >= 2e-4 OR grads_abs >= Q`. Clone (max scale <= 0.01·extent): **new position sampled**
  from N(μ, Σ) (supp. §A), other attributes copied. Split (> 0.01·extent, on padded masks
  computed *after* cloning with zero grads for the clones): 2 samples, scale/(0.8·2), originals
  removed. Prune: opacity < 0.05 (unfiltered sigmoid), and when max_screen set:
  `max_radii2D > 20` OR `max scale > 0.1·extent`. **Quirk**: `densification_postfix` zeroes
  `max_radii2D` before the prune test, so the screen-size prune never fires.
  After every densify step the 3D filter is recomputed.
- Opacity reset every 3000 (step < 15000): filtered opacity clamped to 0.01, unfiltered
  logit recovered by dividing by the filter coefficient.
- 3D filter (Mip-Splatting): $f = \sqrt{0.2}\, z_{min}/f_{max}$ over train cameras
  (z>0.2, inside 1.15× frustum); scales $\sqrt{s^2+f^2}$, opacity × $\sqrt{\prod s^2/\prod (s^2+f^2)}$.
  Recomputed at step 0, after each densification, and every 100 steps for 15000 < step < 29900.
- `kernel_size = 0` (2D Mip filter off for GOF; coef ≈ 1, no 0.3 dilation).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py:83 |
| position lr | 1.6e-4 → 1.6e-6 ·extent, delay_mult 0.01 (no delay steps), 30000 | :84-87 |
| feature / rest / opacity / scaling / rotation lr | 0.0025 / 1.25e-4 / 0.05 / 0.005 / 0.001 | :88-92 |
| lambda_dssim | 0.2 | :96 |
| lambda_distortion α | 100 (unbounded; paper: 1000 bounded) | :97, §4 |
| lambda_depth_normal β | 0.05 | :98 |
| distortion_from_iter / depth_normal_from_iter | 15000 / 15000 | :99-100 |
| densify interval / from / until | 100 / 500 / 15000 | :101,103,104 |
| densify_grad_threshold | 2e-4 (abs threshold adaptive quantile) | :105, gaussian_model.py:689 |
| min_opacity prune | 0.05 (paper §4: "0.05 instead of 0.005") | train.py |
| opacity_reset_interval | 3000, reset value 0.01 (filtered) | :102, gaussian_model.py:468 |
| percent_dense | 0.01 | :95 |
| screen size prune | 20 px after 3000 (never fires, see quirk) | train.py |
| big world prune | 0.1·extent | gaussian_model.py:702 |
| 3D filter | coef 0.2, near 0.2, margin 0.15 | gaussian_model.py:263-311 |
| SH degree | 3, +1 every 1000 | train.py |
| init opacity | 0.1 | create_from_pcd |
| background | black | white_background False |
| appearance model | off for Mip-NeRF 360 (only --use_decoupled_appearance for TnT) | scripts |

## 4. Schedules
SH up every 1000; densify 500<step<15000 every 100; reset every 3000 <15000; regularisers
on from step 15000 (≥); 3D filter every 100 after 15000 until 29900; xyz exponential LR.

## 5. Builds on
3DGS training loop and SfM init; Mip-Splatting 3D filter; 2DGS losses; GOF rasteriser.

## 6. Protocol (Mip-NeRF 360 NVS)
All 9 scenes, images_4 outdoor / images_2 indoor (pre-downsampled JPGs, no PIL resize),
llffhold 8 (`--eval`), 30k iterations, black background, metrics: PSNR, SSIM (3DGS),
LPIPS-VGG (lpipsPyTorch). Reported as outdoor/indoor averages (Table 4).
DTU (Chamfer) and TnT (F1) are geometry benchmarks requiring the tetra mesh pipeline.

## 7. Ambiguities
- The paper's $\mathcal{L}_d$ is $|t_i - t_j|$ but code uses the 2DGS squared form on NDC depth,
  normalised by $(1-T)^2$. Code wins.
- $\mathcal{L}_n$ in the paper is per-Gaussian weighted; code uses the blended normal
  (equivalent up to normalisation).
- Training cameras sampled without replacement (viewpoint stack) in the code.
- The densification "gradient" is the 2D-conic pseudo-gradient, not the true ray-tracing gradient.
