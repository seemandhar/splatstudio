# 3D-HGS: 3D Half-Gaussian Splatting — implementation spec

arXiv 2406.02720 (v4, CVPR 2025). Li, Liu, Sznaier, Camps (Northeastern).
Official code: github.com/lihaolin88/3DHGS @ dde1bd5 (2025-03-13) in `source/code/` (Inria
3DGS code base + modified `submodules/diff-gaussian-rasterization`, Gaussian-Splatting licence).
LaTeX: `source/latex/sec/{3method,4experiments,X_suppl}.tex`.

## 1. Method summary
- Each 3DGS primitive is a *pair* of half-Gaussians: the ellipsoid is cut by a plane through its
  centre with learned normal **n**; each half has its own opacity (α1 on the n-side, α2 on the
  other). Mean, scale, rotation, SH colour are shared.
- The z-integral of a half-Gaussian is the 2D Gaussian times ½·erfc(·) of a term linear in the
  pixel offset (closed form, Eq. 7–8); the pair's splatted alpha is Eq. 10. α1 = α2 recovers 3DGS.
- Custom CUDA rasterizer (forward + backward through the erf factor, normal and the conditional
  z-statistics of the camera-space covariance), plus an "efficient" half-aware tile bounding
  rectangle used at test time.
- Training = vanilla 3DGS recipe with small changes (normal LR 0.003, opacity LR /1.4 every 5k,
  min-opacity prune 0.01 on max(α1,α2), opacity reset to 0.02, densify until 20k, clone-opacity
  rescale).

Contributions
- **C1 Half-Gaussian kernel / splatting** (§3.2, §3.3, Eq. 7, 8, 10; supp. A.1): two opacities +
  plane normal per primitive; closed-form projected kernel.
- **C2 Half-Gaussian rasterizer** (§3.3–3.4, supp. B.1, Alg. 1): forward/backward CUDA, half-aware
  bounding rectangles (efficient rasterizer; ablation supp. Tab. "Rendering speed").
- **C3 Training settings** (§4.1 Implementation, supp. B.2): normal LR, opacity LR decay, prune
  and reset thresholds.

## 2. Equations (verbatim)
Half-Gaussian (§3.2):
$$HG_\Sigma(x-\mu)=\begin{cases}e^{-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)} & n^T(x-\mu)\ge 0\\ 0 & n^T(x-\mu)<0\end{cases}$$
Eq. 7: $\int_{n^T(x-\mu)\ge0}HG_\Sigma(x-\mu)\,dz=\tfrac12 I(x,y)\hat G_{\hat\Sigma}(\hat x-\hat\mu)$
Eq. 8: $I(x,y)=\mathrm{erfc}\Big(-\frac{(n_1x+n_2y)/n_3+\mu_{z|xy}}{\sqrt2\,\sigma_{z|xy}}\Big)$,
supp.: $\mu_{z|xy}=\Sigma_{zx}\Sigma_{xx}^{-1}[x\ y]^T$, $\sigma^2_{z|xy}=\sigma_{zz}-\Sigma_{zx}\Sigma_{xx}^{-1}\Sigma_{xz}$.
Eq. 9: $C=\sum_i c_i\,H\hat G_i\prod_{j<i}(1-H\hat G_j)$.
Eq. 10: $H\hat G(\hat x-\hat\mu)=\tfrac12\{2\alpha_2+(\alpha_1-\alpha_2)I(x,y)\}\hat G_{\hat\Sigma}(\hat x-\hat\mu)$
  $=\tfrac12\{\alpha_1(1+\mathrm{erf}(a))+\alpha_2(1-\mathrm{erf}(a))\}\hat G$.

**As implemented in the official kernel** (`forward.cu`, `preprocessCUDA` + `renderCUDA`):
- Σ (3×3) in a scaled ray space: T = W·J with J = [[fx/tz,0,−fx tx/tz²],[0,fy/tz,−fy ty/tz²],
  [0,0,fy/tz]] (third row added so z is in pixel-like units); cov = Tᵀ Σw T, +0.3 on *all three*
  diagonal entries. 2D conic from the top-left 2×2.
- normal in camera space: `n_c = transformPoint4x3(n, viewmatrix)` = R·n + **t** (the view
  translation is added — affine point transform applied to a direction; official quirk).
- first_divide = 1/(√2·sqrt(max(1e-8, Σzz − ΣzxΣxx⁻¹Σxz))); x_term,y_term = (Σxx⁻¹Σxz)ᵀ.
- erf argument per pixel, d = μ₂D − pixel: a = ((n_c.x/n_c.z + x_term)·d.x + (n_c.y/n_c.z +
  y_term)·d.y)·first_divide (n_c.z += 1e-6).
- G = exp(power), power = −½ dᵀ conic d; **pixels with power < −4.8 are skipped** (forward only;
  backward only skips power > 0).
- α = min(0.99, ½(α1 G(1+erf a) + α2 G(1−erf a))); skip α < 1/255; T stop 1e-4 (Inria).
- Tile rectangle: half-extent w = ⌊√2·sqrt(cov.x·ln(256·max(α1,α2)))+1⌋ (opacity-aware, not 3σ);
  the half-aware "small" rectangles are active only when `cov3D_precomp_small` is non-empty,
  which the official code does only when a model is loaded from .ply (test/render time).
  radii = ⌈3√λmax⌉ (used for visibility/densification only).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py |
| position lr | 1.6e-4·extent → 1.6e-6·extent, delay_mult 0.01 | arguments (3DGS) |
| feature / rest lr | 0.0025 / 0.0025/20 | arguments |
| opacity lr (both halves) | 0.05, divided by 1.4 at every iteration % 5000 == 0 | train.py:128-132 (`param_groups[4]` = opacity) |
| normal lr | 0.003 (paper: 0.3 on Deep Blending) | train.py `--normal_lr` (unused: training_setup hard-codes 0.003!) ; paper §4.1 |
| normal lr decay | none in code (paper supp. B.2 says normal also /1.4 per 5k) | train.py:128-132 vs supp. |
| scaling / rotation lr | 0.005 / 0.001 | arguments |
| normal init | 0.6·(1,1,1) (unnormalised) | gaussian_model.create_from_pcd |
| opacity init | logit(0.1) both halves | create_from_pcd |
| scale init | log sqrt(mean dist² 3-NN) | create_from_pcd (distCUDA2) |
| densify from / until / every | 500 / **20000** / 100 | arguments |
| grad threshold | 2e-4 | arguments |
| percent_dense | 0.01 | arguments |
| min opacity prune | **0.01** on max(α1, α2) | train.py:136, gaussian_model.densify_and_prune |
| opacity reset | every 3000, both halves → min(α, **0.02**) | gaussian_model.reset_opacity, supp. B.2 |
| world-size prune | max scale > 0.1·extent after step 3000 | densify_and_prune |
| screen-size prune | 20 px — inert (max_radii2D zeroed by densification_postfix) | TODO-runtime.md quirk |
| clone opacity | clone gets α' = 1−sqrt(1−α) per half; original unchanged (`self._opacity[mask].copy_` writes into a temporary) | densify_and_clone |
| split | N=2, samples N(0, s) rotated, scale s/1.6, opacity copied, normal copied | densify_and_split |
| λ_dssim | 0.2 | arguments |
| background | black | train.py |
| SH | degree +1 every 1000 up to 3 | train.py |

## 4. Schedules
opacity LR ÷1.4 at 5k,10k,15k,20k,25k (30k is after the last step). Densify 500<s<20000 every 100;
opacity reset at 3000, 6000, …, 18000; world prune active from s>3000. SH up every 1000.

## 5. Builds on
Vanilla 3DGS (Inria code): loss 0.8 L1 + 0.2 D-SSIM, Adam eps 1e-15, densification (clone/split),
xyz LR schedule. Also Scaffold/Mip-Splatting/MCMC variants in the paper — not ported here.

## 6. Protocol
Datasets: Mip-NeRF 360 (7 scenes in main averages: bicycle, garden, stump, room, counter,
kitchen, bonsai; flowers/treehill reported separately in supp.), T&T (truck, train), DB
(drjohnson, playroom). `full_eval.py`: `-i images_4` outdoor, `-i images_2` indoor, `--eval`
(llffhold 8), resolution −1 (→ ≤1600 px width; images_4/images_2 are below 1600 for 360).
T&T / DB: images at native size, auto-downscaled to 1600 px width when wider. LPIPS VGG
(`lpips` package), SSIM Inria Gaussian 11×11. RTX 3090. Seeds not stated.
Per-scene numbers: supp. Table "tab:all" (PSNR/SSIM/LPIPS for 3DGS and 3D-HGS, all 13 scenes)
and main Table 3 (PSNR).
Baseline 3DGS row: not exactly the 3DGS paper numbers (e.g. room 31.40 vs 30.63, kitchen 30.80
vs 30.32) — partly retrained by the authors.

## 7. Ambiguities / paper-vs-code
1. Normal LR on DB 0.3 (paper) but `training_setup` hard-codes 0.003 regardless of `--normal_lr`.
2. Supp. says opacity and normal LR decay /1.4 every 5000; code decays only opacity.
3. Normal transformed to camera space with the translation added (`transformPoint4x3`).
4. Forward skips power < −4.8 (≈3.1σ); backward does not → gradient/forward mismatch for
   Gaussians with max opacity ≳ 0.48 in their tails.
5. Test-time "efficient rasterizer" culling differs from the training rasterizer (supp. ablation:
   29.56 without vs 29.66 with on Mip-NeRF 360).
6. `README`: test_and_score evaluates a separately split folder; `full_eval.py` uses llffhold 8.
   We use llffhold 8 (the 3DGS convention the paper says it follows).
