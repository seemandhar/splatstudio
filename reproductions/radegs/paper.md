# RaDe-GS: Rasterizing Depth in Gaussian Splatting — implementation spec

- arXiv 2406.01467 (v1 2024-06-03, **v2 2024-06-24** = LaTeX source in `source/latex/`).
- Official code: github.com/HKUST-SAIL/RaDe-GS (formerly BaowenZ/RaDe-GS), cloned in `source/code/`.
- **Pinned code version: `2d4bc087f1b4bd62c96054fbe89d273490526b81` (2024-06-21)**, see "Code version" below.
  Rasterizer (`submodules/diff-gaussian-rasterization`) last changed at `4e74c25` (same day);
  its glm submodule is g-truc/glm `5c46b9c0` (2023-06-08, provides `gtx/pca.hpp`).

## Code version (fidelity gate)
`git log` of the official repo:

| commit | date | what | paper formulation? |
|---|---|---|---|
| f38844f…2d4bc08 | 2024-06-18…21 | first release ("training and testing code on DTU") | **yes**: centre-cosine depth (Eq. 14 with θ≈θc), depth distortion (w_d=100) + normal consistency |
| de3f67a | 2024-07-24 | README "We have updated the formulation" | no: per-pixel cosine depth, **distortion loss removed**, coordinate maps |
| f96840d…ec9199f | 2024-07…09 | depth backprop, "fix new formulation", 2D filter fix, clean | no (new formulation) |
| 5f5cd3b | 2025-12-26 | compile fix + backward bug (dcoef used dL_dconic.w as opacity) | no |
| c7c56cc, d72f207 | 2026-03 | rewritten kernels + **PGSR-style multi-view regularization** | no |

The arXiv v2 (2024-06-24, numbers in Tables 1–4) predates the July formulation change, so the
paper numbers correspond to the June-2024 code. `2d4bc08` is the last commit of that line (its
last code change is `4e74c25`, "fix CUDA error"; `2cd9fd1`/`2d4bc08` touch README/mesh_extract
only). Later commits are not used.

## Method summary
- Standard 3DGS (local affine / EWA projection), unchanged colour rendering, plus a
  **rasterized spatially-varying depth** per splat: d = z_c + p·(Δu, Δv) (Eq. 3), where p comes
  from the ray–Gaussian maximum-response intersection in "ray space" (Eqs. 5–17).
- **Rasterized normal** per splat: plane of intersections in ray space, n' = −(q, 1), mapped to
  camera space by J^T (here the code uses its own `Jn` matrix) and normalised (Eqs. 18–21).
- Depth and normals are α-blended like colour; median depth (last Gaussian with T>0.5) is also
  rendered.
- Losses (from 2DGS): depth distortion L_d and depth–normal consistency L_n, enabled after
  15k iterations; Mip-Splatting 3D filter and GOF densification are used.

Contributions:
- C1 Rasterized depth for 3D Gaussian splats (§3.2, Eqs. 3–17) — CUDA kernel.
- C2 Rasterized normal for 3D Gaussian splats (§3.3, Eqs. 18–21) — CUDA kernel.
- C3 Geometry regularisation: L_d + L_n on the rasterized depth/normals (§3.4, Eqs. 22–24).
- Inherited (Implementation details §4.1): Mip-Splatting 3D filter, GOF densification
  (abs-gradient quantile, resampled clones), decoupled appearance (DTU/TnT only).

## Equations (verbatim from `source/latex/method.tex`)
- Eq. 3 (`eq:depth_render`): $d = z_c + \mathbf{p} \begin{pmatrix}\Delta u\\ \Delta v\end{pmatrix}$, $\Delta u = u_c-u$, $\Delta v = v_c - v$.
- Eq. 6 (`eq:tstar`): $t^* = \frac{\mathbf{v}^\top \mathbf{\Sigma}^{-1}(\mathbf{x}_c -\mathbf{o})}{\mathbf{v}^\top \mathbf{\Sigma}^{-1} \mathbf{v}}$
- Eq. 10 (`eq:tstar_ray`): $t^* = \frac{\mathbf{v}'^\top \mathbf{\Sigma}'^{-1}(\mathbf{u}_c -\mathbf{u}_o)}{\mathbf{v}'^\top \mathbf{\Sigma}'^{-1} \mathbf{v}'}$, $\mathbf{v}'=(0,0,1)^\top$
- Eq. 12 (`eq:q_hat`): $\hat{\mathbf{q}} = \frac{\mathbf{v}'^\top \mathbf{\Sigma}'^{-1}}{\mathbf{v}'^\top \mathbf{\Sigma}'^{-1} \mathbf{v}'}$
- Eq. 14: $d = cos\theta_c t^* = \frac{z_c}{t_c} t^* = \hat{\mathbf{p}} (\mathbf{u}_c -\mathbf{u}_o)$, $\hat{\mathbf{p}}=\frac{z_c}{t_c}\hat{\mathbf{q}}$; $\hat{\mathbf p}(0,0,t_c)^\top = z_c$, $\mathbf p$ = first two components.
- Eq. 18–20: intersections $\mathbf u = \mathbf u_c + (-\Delta u, -\Delta v, (\Delta u\ \Delta v)\mathbf q^\top)^\top$, $\mathbf q = \frac{t_c}{z_c}\mathbf p$; plane normal $(\mathbf q\ 1)$.
- Eq. 21: $\mathbf{n}' = -\left(\mathbf{q} \quad 1 \right)^\top$; Eq. 22: $\mathbf{n} = \mathbf{J}^\top \mathbf{n}'$, normalised.
- Eq. 23 (distortion): $\mathcal{L}_{d} = \sum_{i,j}\omega_i\omega_j(d_i-d_j)^2$, $\omega_i = \alpha_i \prod_{j<i}(1-\alpha_j)$.
- Eq. 24 (normal consistency): $\mathcal{L}_{n} = \sum_{i} \omega_i (1-\mathbf{n}_i^\top\tilde{\mathbf{n}})$, ñ = finite-difference normal of the depth map.
- Eq. 25: $\mathcal{L} = \mathcal{L}_{c} + w_d \mathcal{L}_{d} + w_n \mathcal{L}_{n}$.

### What the pinned code actually computes (code overrides paper)
`forward.cu:computeCov2D` (per Gaussian, camera frame t, clamped to 1.3·tan(fov)):
- Σ_cam^{-1} via eigen-decomposition (degenerate: outer product of the min eigenvector).
- `uvh = (tx/tz, ty/tz, 1)`, `uvh_mn = normalize(Σ_cam^{-1} uvh)`, `vb = uvh_mn·uvh`,
  `plane = nJ_inv · uvh_mn / max(vb,1e-7)`, `nJ_inv = [[v²+1, −uv, 0], [−uv, u²+1, 0], [−u, −v, 0]]` (glm column-major),
  depth plane `p = plane[:2] · tz/(u²+v²+1)` and divided by (fx, fy) in the render kernel →
  pixel depth `d = z_c + p_x Δx + p_y Δy` (Δ = centre − pixel, pixel coords without +0.5).
- normal: `Jn · (−plane[:2]·l/(u²+v²+1), −1)`, `Jn = [[1/tz,0,−tx/tz²],[0,1/tz,−ty/tz²],[tx/l,ty/l,tz/l]]` (columns), normalised; camera frame.
- `renderCUDA`: skip Gaussian at a pixel if its pixel depth ≤ NEAR_PLANE (0.2) — this also
  removes its colour contribution; α = min(0.99, o·coef·G), α<1/255 skip, stop when T<1e-4.
  Outputs: colour + T·bg, `alpha` = Σ w, `depth` = Σ w d (not normalised), `middepth` = d of last
  Gaussian with T>0.5 (0 if none), `normal` = Σ w n (camera frame), `distortion` 2DGS form on
  NDC-mapped depth m(d) = (f d − f n)/((f−n) d) with n=0.2, f=100:
  Σ_i w_i (m_i² A_{i−1} + D2_{i−1} − 2 m_i D1_{i−1}), weights detached in backward.
- No 2D low-pass dilation (kernel_size = 0, no +0.3); the opacity coefficient `coef` = 1.
- `means2D.grad` [N,3]: x,y = screen gradient (includes the depth-plane term of depth gradients), z = Σ_pixels |∂L/∂x|+|∂L/∂y| of the Gaussian-weight path (GOF abs metric).

`train.py` (iteration ≥ regularization_from_iter = 15000):
```
depth_distortion = where(mask>0, distortion/(mask*mask).detach(), 0)
distortion_loss  = mean(depth_distortion * edge)          # edge = exp(-max 4-neighbour |∇I|), 0 border
depth  = nan_to_num(depth / mask)                          # expected depth (mask not detached)
n_exp, n_med = depth_double_to_normal(cam, depth, middepth) # camera-frame finite differences, W/2,H/2, +0.5 pixel centres
n_r    = normalize(normal)                                  # rendered normal
err    = 1 - <n_r, n_*>                                     # border pixels: n_* = 0 → err = 1
L_n    = 0.4 * mean(err_exp) + 0.6 * mean(err_med)          # depth_ratio = 0.6
loss   = (1-0.2) L1 + 0.2 (1-SSIM) + 0.05 * L_n + 100 * L_d
```

## Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | code arguments/__init__.py:79 |
| position lr | 1.6e-4 → 1.6e-6 ×extent, delay_mult 0.01, 30k | code arguments:80-83 |
| feature / f_rest / opacity / scaling / rotation lr | 0.0025 / 0.0025/20 / 0.05 / 0.005 / 0.001 | code arguments:84-88; gaussian_model.py:333-338 |
| lambda_dssim | 0.2 | code arguments:93 |
| lambda_distortion w_d | 100 | code arguments:94; paper §4.1 |
| lambda_depth_normal w_n | **0.05** (paper text: 5) | code arguments:95 vs paper §4.1 ("following GOF" — GOF uses 0.05) |
| depth_ratio (median vs expected normal mix) | 0.6 | code train.py:150 |
| regularization_from_iter | 15000 | code arguments:100; paper §4.1 |
| densify from/until/interval | 500 / 15000 / 100 | code arguments:96-99 |
| densify_grad_threshold | 0.0002 | code arguments:101 |
| min_opacity prune | 0.05 | code train.py:205 |
| screen-size prune | 20 px after 3000 (never fires: densification_postfix zeroes max_radii2D) | code train.py:204; gaussian_model.py:656 |
| world-size prune | max scale > 0.1·extent (after 3000) | gaussian_model.py:736 |
| opacity reset | every 3000 (< 15000), filtered opacity → min(·,0.01) | train.py:208; gaussian_model.py:493 |
| percent_dense | 0.01 | arguments:91 |
| 3D filter | sqrt(0.2)·min depth / max focal, near 0.2, margin 0.15; recomputed after densify and every 100 it after 15k (until 29900) | gaussian_model.py:173; train.py:206,212 |
| SH degree | 3, +1 every 1000 | train.py:104 |
| background | black (white_background False) | arguments:56 |
| NEAR/FAR for distortion mapping | 0.2 / 100 | auxiliary.h |
| init opacity / scale | 0.1 / sqrt(mean 3-NN dist²) | gaussian_model.py:313-317 |
| decoupled appearance | DTU/TnT only (`--use_decoupled_appearance`) | README; paper §4.1 |

## Schedules
- Steps 1–15000: photometric loss only; densification (500<it<15000, every 100), opacity reset
  every 3000, 3D filter recomputed after each densify.
- Step 15000+: L_d (×100) and L_n (×0.05) on; 3D filter recomputed every 100 its until 29900.
- xyz LR exponential decay over 30000; SH degree up every 1000.

## Builds on
3DGS official code + Mip-Splatting 3D filter (no 2D filter) + GOF densification/reset (identical
code to GOF @5245b20, verified by diff: densify_and_prune/clone/split/stats/reset are the same) +
2DGS losses. Colour rasterisation is 3DGS EWA splatting except the near-plane per-pixel depth test.

## Protocol
- NVS: Mip-NeRF 360 (Table 3; outdoor/indoor averages only) and NeRF-Synthetic (Table 4).
  The repo has no Mip-NeRF 360 script at the pinned commit; a later README says
  `python train.py -s <COLMAP> --eval` (default `-r -1`: full-res images rescaled to 1600 px width).
  The comparison rows (3DGS 24.64/30.41, GOF 24.82/30.79, Mip-Splatting) are GOF's Table 4, whose
  script uses `-i images_4` (outdoor) / `-i images_2` (indoor); RaDe-GS states it follows GOF's
  evaluation. Most likely protocol: images_4/images_2, llffhold 8, 30k its, black background.
- Geometry: DTU 15 scans (2DGS preprocessed, `-r 2`, decoupled appearance, TSDF mesh, Chamfer),
  TnT 6 scenes (GOF preprocessed, F1; TSDF / later marching tetrahedra).
- Metrics: official `metric.py` (lpipsPyTorch VGG, 3DGS SSIM).

## Ambiguities
1. w_n: paper text 5 vs code 0.05 → code 0.05 (paper also says "following GOF", GOF uses 0.05).
2. Paper L_n is Σ_i ω_i(1 − n_i·ñ) per Gaussian; code uses the blended, renormalised normal and a
   0.4/0.6 mix of expected/median depth normals → code.
3. Paper L_d uses depth d; code uses NDC-mapped depth, divides by (Σw)² (detached) and weights by
   the image edge map → code.
4. Paper Eq. 22 says n = J^T n'; code uses a separate `Jn` (unit-ray Jacobian) → code (vendored).
5. Mip-NeRF 360 resolution: images_4/images_2 (GOF protocol) vs 1600-px default; see Protocol.
6. Depth near test (d ≤ 0.2 skip) also affects colour → kept (vendored kernel).
