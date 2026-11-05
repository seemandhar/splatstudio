# Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration

arXiv 2403.11056 (ECCV 2024 oral). Liang, Zhang, Hu, Zhu, Feng, Jia.
Official code: https://github.com/lzhnb/Analytic-Splatting, commit `a905939` (cloned to
`source/code/`; repo MIT licence, rasterizer submodule under the Inria Gaussian-Splatting
licence, non-commercial research use).

## 1. Method summary
- 3DGS evaluates each 2D Gaussian at the pixel *centre*; Analytic-Splatting replaces that
  sample by an approximation of the Gaussian's **integral over the 1×1 pixel window**.
- The 2D covariance is eigen-decomposed; the pixel window is rotated onto the eigen-axes so
  the integral factorises into two 1D window integrals.
- Each 1D integral is a difference of CDFs; the Gaussian CDF is approximated by the
  *conditioned logistic* S(x)=1/(1+exp(-1.6x-0.07x^3)), which has a closed-form derivative.
- Forward and backward (incl. eigen-decomposition chain rule) are implemented in a modified
  Inria CUDA rasterizer. Everything else (losses, ADC densification, schedules) is 3DGS.
  The 3DGS 0.3 px low-pass dilation of Σ̂ is removed (the pixel integral replaces it).

Contributions
- **C1** analytic pixel-area integral response I_g^2D replacing g^2D in α-blending
  (§4.2, Eq. 10–13; supp. §1.1).
- **C2** conditioned logistic approximation of the Gaussian CDF and its derivative
  (§4.1 Def. 1, Eq. 7–9; supp. Eq. 8–12) used in forward and backward.
- (Optional, not in the main "Ours" row) Mip-Splatting 3D filter (`--filter3d`) → "Ours + 3D filter".

## 2. Equations (verbatim LaTeX, main paper)
Eq. 3 (3DGS shading):
`\bm{C}(\bm{u}) = \sum_{i\in N} T_i g^\text{2D}_i(\bm{u} | \hat{\bm{\mu}_i}, \hat{\bm{\Sigma}_i})\alpha_i \bm{c}_i,\quad T_i = \prod^{i-1}_{j=1}(1 - g^\text{2D}_{j}(\bm{u})\alpha_j)`

Eq. 7 (logistic CDF approximation, Definition 1):
`S(x) = \frac{1}{1 + \exp\left(-1.6 \cdot x - 0.07 \cdot x^3\right)}`;  S_σ(x) = S(x/σ).

Eq. 8–9 (1D window): `\mathcal{I}_g(u) = G(u+\frac12) - G(u-\frac12) \approx S(u+\frac12)-S(u-\frac12)`.

Eq. 10 (exact 2D pixel integral, intractable correlation term):
`\mathcal{I}_g^\text{2D}(\bm{u}) = \int_{u_x-\frac12}^{u_x+\frac12}\int_{u_y-\frac12}^{u_y+\frac12}\exp(-\frac a2 (x-\hat\mu_x)^2-\frac c2 (y-\hat\mu_y)^2-b(x-\hat\mu_x)(y-\hat\mu_y))\,dx\,dy`

Eq. 11: `\tilde{\bm u} = [\bm v_1; \bm v_2](\bm u-\hat{\bm\mu})`, eigenvalues λ1 ≥ λ2 of Σ̂.

Eq. 12/13 (approximation, used in the shading):
`\mathcal{I}_{g}^\text{2D}(\bm{u}) = 2\pi\sigma_1\sigma_2[S_{\sigma_1}(\tilde{u}_x + \frac{1}{2}) - S_{\sigma_1}(\tilde{u}_x - \frac{1}{2})][S_{\sigma_2}(\tilde{u}_y + \frac{1}{2}) - S_{\sigma_2}(\tilde{u}_y - \frac{1}{2})]`,
`\bm C(\bm u)=\sum_i T_i \mathcal I^{2D}_{g-i}(\bm u)\alpha_i\bm c_i`, σ_k = sqrt(λ_k).

Derivative (supp. Eq. logistic backward): dS/dx = (1.6 + 0.21x^2) S(x)(1-S(x));
dS_σ(x)/dσ = -(x/σ^2)(1.6+0.21(x/σ)^2) S(1-S).

Code (`cuda_rasterizer/forward.cu`, `auxiliary.h`):
- `approxCdf(x) = 0.5 tanh(0.5(1.6x+0.07x^3)) + 0.5` (= S(x)).
- `intU = σ1 (S(U2) - S(U1))`, U2/1 = (ũx ± 0.5)/σ1; same for V; `G = 2π intU intV`;
  `alpha = min(0.99, o * G)`; skip if alpha < 1/255; stop if T < 1e-4.
- Σ̂ = J W Σ W^T J^T **without** +0.3 I; λ1,2 = tr/2 ± sqrt(max(0,(tr/2)^2 - det));
  σ1 = sqrt(λ1), σ2 = sqrt(|λ2|); v1 = normalize(Σ12, λ1 - Σ11), v2 = normalize(λ2 - Σ22, Σ12).
- tile radius = ceil(3 sqrt(max(λ1', λ2'))) with λ' using sqrt(max(0.1, root)).
- Pixel centre convention: pixel index p ↔ coordinate p; Gaussian centre = ndc2Pix (Inria).

## 3. Hyper-parameters (all 3DGS defaults; code `arguments/__init__.py`)
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments:OptimizationParams |
| position lr | 1.6e-4 → 1.6e-6 (×extent), delay_mult 0.01, max 30000 | idem |
| feature / opacity / scaling / rotation lr | 0.0025 / 0.05 / 0.005 / 0.001 (shN /20) | idem |
| lambda_dssim | 0.2 | idem |
| densify from / until / interval | 500 / 15000 / 100 | idem |
| densify_grad_threshold | 2e-4 (signed screen grad, Inria means2D) | idem |
| percent_dense | 0.01 | idem |
| min opacity prune | 0.005 | train.py:213 |
| big-world prune | max scale > 0.1·extent, after step 3000 | gaussian_model.py:758 |
| screen-size prune | 20 px (no-op: densification_postfix zeroes max_radii2D) | train.py:211 |
| opacity reset | every 3000, < 15000, `min(o·coef, 0.01)/coef` with Mip-Splatting coef | gaussian_model.py:431 |
| 3D filter | computed after each densify (s=0.2 in variance, near 0.2, 15 % margin); **not used in render** unless `--filter3d` | train.py:219, gaussian_model.py:188 |
| SH degree | 3, +1 every 1000 | train.py:134 |
| background | black (`white_background` False for 360/T&T/DB) | train.py:75 |
| `--dense` (GOF abs-grad densify) | off | README / scripts |
| `--filter3d` | off for the "Ours" row | stst_360.sh |
| alpha clamp / threshold / T stop | 0.99 / 1/255 / 1e-4 | forward.cu |

## 4. Schedules
Identical to 3DGS: densification 500–15000 every 100; opacity reset 3000, 6000, 9000, 12000;
SH +1 each 1000; position-lr exponential decay. MTMT (`train_ms.py`, `mtmt_360.sh`): training
set = union of the train cameras at scales 1,2,4,8 (PIL bicubic `round(W/s)` resizes of
images_4/images_2), popped from a without-replacement stack; with probability 0.3 the popped
camera is replaced by a random 1× camera (`--sample_more_highres`, train_ms.py:153-156), so
P(1×)=0.475, P(2×)=P(4×)=P(8×)=0.175. **No radii rescale in MTMT**: the low-res radii
expansion is commented out (train_ms.py:209-211); the 800/W rescale exists only in train.py
(behind a flag) and is irrelevant anyway because max_radii2D is zeroed on every densify. 30k
iterations (default; mtmt_360.sh does not override).

## 5. What it builds on
3DGS training code verbatim (+ Mip-Splatting's 3D-filter bookkeeping copied in, used only for
the opacity-reset quirk unless `--filter3d`). Only the rasterizer is changed.

## 6. Protocol
- **STST (supp. Tab. 7/8/9)**: Mip-NeRF 360 trained/tested at `-i images_4` (outdoor) /
  `-i images_2` (indoor) — the officially provided downsampled images = our native
  `mipnerf360`. Rows marked * use bicubically downsampled full-res images (Mip-Splatting
  protocol) = our `mipnerf360-pilr`. T&T (truck, train) and DB (drjohnson, playroom) at
  `images`, `-r 1`. llffhold 8 (`--eval`). 30k iterations. Metrics: PSNR, SSIM
  (3DGS gaussian-window SSIM), LPIPS-VGG (`lpips` package).
- **MTMT (main Tab. 1/2)**: multi-scale Blender (×1,2,4,8, `convert_blender_data.py`) and
  multi-scale 360 (each training image bicubically downsampled ×1,2,4,8), test at all scales.
- Super-resolution 2× table on multi-scale 360 (main Tab. 3).
- `scripts/stmt_360_r.sh` (train `-r 8` PIL from full-res, test `-r 1/2/4/8`, i.e. zoom-in /
  STMT) exists, but no table in the paper or supplement reports it — not a reproduction target.
- Seeds not stated.

## 7. Ambiguities
- The non-* 3DGS row in the 360 STST table is copied from the 3DGS paper; the * rows
  (3DGS*, Mip-Splatting*, Ours*) are the authors' runs on bicubic images.
- Opacity reset uses the 3D-filter coefficient even when the filter is not rendered
  (code quirk inherited from Mip-Splatting): reset ceiling = 0.01/coef ≥ 0.01.
- `-r 1` is passed explicitly, so no >1600 px auto-downscale (irrelevant at images_4/_2).
- The backward does not propagate through the 0.99 alpha clamp (standard Inria behaviour).
- The paper says "same parameters as 3DGS"; code confirms.
