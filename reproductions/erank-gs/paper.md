# Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting (NeurIPS 2024)

arXiv 2406.11672 · Hyung, Hong, Hwang, Lee, Kim, Choo · official code
https://github.com/junhahyung/erank_gs (commit 99fd128, cloned to `source/code/`, a WIP release:
README empty, `train.sh` = one DTU command).

## 1. Summary and contributions
- Analyse each Gaussian's shape by the effective rank of its covariance spectrum; 3DGS, SuGaR
  and 2DGS converge to many needle-like Gaussians (erank ≈ 1).
- **C1 (§3.2, Eq. 4) erank regularisation**: penalise erank → 1 with a log barrier, plus a
  penalty on the smallest scale (flatness), from iteration 7000.
- **C2 (§3.2 "ADC algorithm", App. A.4) fixed densification**: densify with the sum of
  per-pixel gradient norms (Bulò et al. 2024 / GOF) instead of the norm of the summed gradient.
  The official code uses exactly GOF's `densify_and_prune` (quantile-matched absolute gradient,
  clone resampling, min opacity 0.05).
- Ablation (Table 2): "+a" = C2 only, "+a+b" = C2 + C1 on top of **3DGS** (= "3DGS+e" in Table 1).
  "+c" (depth distortion + normal consistency, App. A.2) is optional and not part of the
  reported "+a+b" numbers.

## 2. Equations
erank (Eq. 1–3), with $s_1^2\ge s_2^2\ge s_3^2>0$ the covariance eigenvalues (= squared scales):
$$q_i = s_i^2/S,\ S=\sum_i s_i^2,\quad H=-\sum_i q_i\log q_i,\quad \text{erank}(\mathcal G_k)=\exp H.$$
Loss (Eq. 4):
$$\mathcal L_{\text{erank}}=\sum_k \lambda_{\text{erank}}\max(-\log(\text{erank}(\mathcal G_k)-1+\epsilon),0)+s_3,$$
ε = 1e-5 (paper). **Code** (`train.py`): `erank_lambda * clamp(-log(erank-1+1e-7), 0).mean()
+ thin_lambda * sorted_scales[:, 2].mean()` — mean over Gaussians (not sum), ε = 1e-7,
s3 term weighted by `thin_lambda = 1` (train.sh), scales = exp(log-scales) (no 3D filter).
Applied for `erank_from_iter (7000) <= it <= erank_end_iter (30000)`.

Fixed ADC (App. A.4): $\sum_{i\in\mathcal P}\|\partial L/\partial p_i\,\partial p_i/\partial u\|_2>\tau$.
Code: rasterizer backward accumulates `|g_x|+|g_y|` per pixel into `means2D.grad[:,2]`
(`backward.cu:548`), `densify_and_prune_v2`:
```
grads = accum/denom; grads_abs = accum_abs/denom          (nan -> 0)
ratio = mean(||grads|| >= 2e-4); Q = quantile(grads_abs, 1 - ratio)
clone: (||g|| >= 2e-4 | g_abs >= Q) & max(s) <= 0.01 extent; new mean ~ N(mu, Sigma)
split: same on post-clone set (clones padded 0) & max(s) > 0.01 extent; N=2, s/1.6
prune: opacity < 0.05 (v2; 3DGS 0.005); after it 3000 also max(s) > 0.1 extent
       (screen-size prune max_radii2D > 20 is a no-op: postfix zeroes max_radii2D)
```

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| λ_erank | 0.01 | paper §4.1; train.sh `--erank_lambda 0.01` (code default 1.0) |
| erank start / end | 7000 / 30000 | paper §3.2; train.sh `--erank_from_iter 7000`; code default end = iterations |
| ε | 1e-5 paper / **1e-7 code** | Eq. 4 vs train.py |
| s3 weight (thin_lambda) | 1 | train.sh (paper: implicit weight 1) |
| reduction | mean over Gaussians | train.py (paper writes Σ_k) |
| densify grad thr | 2e-4 | arguments |
| densify 500–15000 every 100, reset opacity every 3000 to 0.01 | 3DGS defaults | arguments, gaussian_model |
| min opacity prune | 0.05 | train.py `densify_and_prune_v2(..., 0.05, ...)` |
| LRs | 3DGS defaults (xyz 1.6e-4→1.6e-6 ×extent, f_dc 2.5e-3, f_rest 1.25e-4, opacity 0.05, scaling 5e-3, rotation 1e-3) | arguments |
| iterations | 30000 | train.sh |
| resolution | `-r 2` | train.sh |
| background | black (`white_background=False`) | arguments |
| SSIM weight | 0.2 | arguments |

## 4. Schedules
SH degree +1 every 1000 it; densify 500 < it < 15000; opacity reset every 3000; erank+thin from 7000.
`reg_from_iter 35000` disables the optional "+c" losses (never active in 30k).

## 5. Builds on
Vanilla 3DGS (Inria rasterizer, 0.3 screen dilation, no 3D filter, no AA) + GOF densification.
All other 3DGS training unchanged.

## 6. Protocol
- DTU 15 scans, "downsample to 800×600 following 2DGS/GOF" — in practice 2DGS DTU release
  (1554×1162 undistorted, RGBA with object mask in alpha) with `-r 2` → 777×581. The 3DGS loader
  multiplies RGB by alpha → masked images on black background, black render background.
- NVS split: paper only says "images are split into training and test sets"; train.sh has no
  `--eval` (that run is for CD). **Assumed**: 3DGS `--eval` llffhold8 on sorted image names.
- Mip-NeRF 360: 9 scenes, group averages only (outdoor 24.93, indoor 31.16 PSNR).
- Metrics: PSNR (per image mean); LPIPS not reported for DTU.

## 7. Ambiguities
1. DTU NVS split (assumed llffhold8). 2. Mask: the release's `if image_mask:` on a tensor would
crash, so the opacity-mask BCE loss in train.py cannot have run as released; not part of the
paper → not ported. 3. ε 1e-5 vs 1e-7 (code wins). 4. Σ vs mean (code mean). 5. s3 is the
smallest *scale* (std), not variance (code). 6. Gradient flows through the sort into all scales.
