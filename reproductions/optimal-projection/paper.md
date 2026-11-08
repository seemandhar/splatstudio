# On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy (ECCV 2024)

Huang, Bai, Guo, Li, Guo (Nanjing University). arXiv 2402.00752.
Official code: https://github.com/LetianHuang/op43dgs (cloned into `source/code/`, read-only;
Inria Gaussian-Splatting licence: non-commercial research use). Pinhole rasterizer:
`source/code/submodules/diff-gaussian-rasterization-pinhole/cuda_rasterizer/{forward,backward}.cu`.

## 1. Method summary
- 3DGS's EWA local-affine projection onto the shared z=1 plane has an approximation error that
  grows with the angle between the Gaussian's ray and the optical axis (Sec. 4, Eq. 5-10).
- The error is minimised when each Gaussian is projected onto the plane tangent to the unit
  sphere at the ray through its mean ("optimal projection", Sec. 5.1, Eq. 11-15).
- Rendering becomes a "unit-sphere based rasterizer" (Sec. 5.2, Eq. 16-18): each pixel ray is
  intersected with every Gaussian's own tangent plane and the 2D Gaussian is evaluated in that
  plane's local (Q-rotated) coordinates. So both the 2D covariance AND the per-pixel evaluation
  change; alpha blending, SH, densification and losses are unchanged 3DGS.

Contributions
- C1 Optimal projection: tangent-plane Jacobian J_p (Q J_p, Eq. 14/15/18) replaces the EWA J
  for the 2D covariance (code: `computeCov2D`), with the official analytic gradient w.r.t. the
  camera-space mean (code: `computeCov2DCUDA`).
- C2 Unit-sphere rasterizer: per-pixel tangent-plane coordinates (Eq. 16-17, code `renderCUDA`
  forward and backward, including d(tangent coords)/d(mean2D)).
- (C0, not a contribution) training = unmodified 3DGS (Sec. 6.1 "default parameters of 3D-GS").

## 2. Equations (verbatim from LaTeX, then code form)
Eq. 11 tangent plane: $\mathbf{x_{p}}^\top\cdot\left(\mathbf{x}^{'}-\mathbf{x_{p}}\right)=0$,
Eq. 12 $\mathbf{x_{p}}=\varpi(\boldsymbol{\mu}^{'})={\boldsymbol{\mu}^{'}}({\boldsymbol{\mu}^{'}}^{\top}{\boldsymbol{\mu}^{'}})^{-1/2}$
Eq. 13 $\varphi_{\text{p}}(\mathbf{x}^{'})=\mathbf{x}^{'}(\mathbf{x_{p}}^{\top}\mathbf{x}^{'})^{-1}$
Eq. 14 $\mathbf{J_{p}}=\mathbb{I}\otimes(\mathbf{x_{p}}^{\top}\boldsymbol{\mu}^{'})^{-1}-{\boldsymbol{\mu}^{'}}(\mathbf{x_{p}}^{\top}\boldsymbol{\mu}^{'})^{-1}({\boldsymbol{\mu}^{'}}^{\top}\mathbf{x_{p}})^{-1}\mathbf{x_{p}}^{\top}$
  (= (I - x̂x̂^T)/|μ'|; Eq. 15 is its explicit 3x3 form.)
Eq. 16 $\mathbf{x}_{\text{2D}}=\varphi_{\text{p}}([(u-c_x)/f_x,(v-c_y)/f_y,1]^\top)$
Eq. 17 $G_{\text{2D}}(\mathbf{x_{2D}})=\exp\{-\tfrac12(\mathbf{x_{2D}}-\varphi_p(\boldsymbol\mu'))^\top(\mathbf{J_p}\boldsymbol\Sigma'\mathbf{J_p}^\top)^{-1}(\mathbf{x_{2D}}-\varphi_p(\boldsymbol\mu'))\}$
Eq. 18 Q = rows (e_u; e_v; x̂) with, for x̂=(μx,μy,μz)/r, ρ=sqrt(μx²+μz²):
  e_u = (μz/ρ, 0, -μx/ρ), e_v = (-μxμy/(ρr), ρ/r, -μyμz/(ρr)). Q is applied to x_2D, φ_p(μ'), J_p
  so the third row/column can be dropped.

Code form (forward.cu `computeCov2D`, t = camera-space mean, θ = atan2(-μy, ρ), φ = atan2(μx, μz)):
  J (2x3) = diag(f'x, f'y) · [e_u; e_v] / |t|, f' = max(512, f)  ("for anti-aliasing")
  Σ_2D = J W Σ W^T J^T + 0.3 I; conic = Σ_2D^{-1}.
Per pixel (forward.cu `renderCUDA`): t̂ = normalize((px-cx)/fx, (py-cy)/fy, 1); μ̂ from the
2D mean the same way; skip if μ̂·t̂ < 1e-7;
  u = f'x (e_u·t̂)/(μ̂·t̂), v = f'y (e_v·t̂)/(μ̂·t̂), d = (0-u, 0-v),
  power = -½(a dx² + c dy²) - b dx dy; skip if power > 0; α = min(0.99, o e^power);
  skip α < 1/255; stop when T(1-α) < 1e-4 (Inria rules).
Tile extent (forward.cu preprocess): naive EWA cov (3DGS J with 1.3·tan(fov) clamp, focal f, +0.3)
gives mid = ½(a+c); det from the OPTIMAL cov; λ = mid ± sqrt(max(0.1, mid²-det));
radius = ceil(3 sqrt(max λ)). Near cull z <= 0.2. Depth key = camera z.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| tangent-plane scale floor | f' = max(512, f) | code forward.cu computeCov2D / renderCUDA |
| low-pass | +0.3 on Σ_2D diagonal | code forward.cu |
| α clamp / threshold / T stop | 0.99 / 1/255 / 1e-4 | code forward.cu (Inria) |
| near plane | 0.2 | code auxiliary.h in_frustum |
| iterations | 30000 | code arguments/__init__.py |
| lrs | position 1.6e-4→1.6e-6, feature 2.5e-3, opacity 0.05, scaling 5e-3, rotation 1e-3 | code arguments (3DGS) |
| densify | from 500 until 15000 every 100, grad 2e-4, percent_dense 0.01, opacity reset 3000 | code arguments (3DGS) |
| λ_dssim | 0.2 | code |
| background | black (white_background False) | code |

## 4. Schedules
Identical to 3DGS (SH degree +1 every 1000 its, densification 500-15000, opacity reset every
3000, position lr exponential decay). Nothing method-specific.

## 5. Builds on
Original 3DGS (Inria code, same training script). Only the rasterizer submodule changes.
Backward (code backward.cu): analytic d J/d t (18 coefficients, `computeCov2DCUDA`) and
d(u,v)/d(mean2D) (`renderCUDA`), multiplied by tan_fov to convert to NDC units; opacity /
G gradients are propagated even when α is clamped (Inria behaviour).

## 6. Protocol
Same as 3DGS: 13 scenes (Mip-NeRF 360 ×9, T&T truck/train, DB drjohnson/playroom), llffhold 8,
30k iterations; images_4 outdoor / images_2 indoor (3DGS convention; paper Sec. 6.1 "same
settings as 3D-GS"). PSNR/SSIM/LPIPS (3DGS metrics.py; LPIPS-VGG). 3D-GS rows in the paper's
tables are the published 3DGS numbers (identical to tools/reference_results.json).

## 7. Ambiguities / code findings
- f' = max(512, f) floor appears only in code (no-op for our scenes: focal > 512).
- Official backward `j12_x` contains `2.f * y2 * (x2 * z2)` where the exact derivative (and the
  symmetric `j10_z`) has `(x2 + z2)`: a typo that makes dL/dt_x from dL/dJ12 inexact. The other
  17 coefficients are exact (checked in plugin tests). Code is what produced the numbers.
- Render backward d(u,v)/d(mean) uses `focal_x` where forward uses `max(512, focal_x)`: same for
  f >= 512.
- Tile radius mixes the naive EWA mid with the optimal det (heuristic; kept as in code).
- No per-scene SSIM/LPIPS in the final paper (commented out in the LaTeX; values there kept in
  notes only).
