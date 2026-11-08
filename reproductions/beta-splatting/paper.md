# Deformable Beta Splatting (DBS) — implementation spec

Liu, Sun, Chen, Wang, Feng. SIGGRAPH 2025. arXiv 2501.18630 (LaTeX source in `source/latex/`).
Official code: https://github.com/RongLiu-Leo/beta-splatting (Apache-2.0), cloned at commit
`431d575c5080d996348140644da8b636b0324878` into `source/code/` (read-only). It ships a modified
gsplat 1.3.0 fork in `source/code/submodules/gsplat/` (Beta rasterizer + Spherical Beta CUDA).

## 1. Method summary
- Replace the Gaussian splatting kernel with a bounded **Beta kernel** `(1 - r^2)^{4 e^b}` with a
  per-primitive learnable shape `b` (init 0 ~ Gaussian-like), support r^2 < 1 (1-sigma ellipse).
- Replace SH colour with **Spherical Beta** (SB): base colour + M = 2 specular lobes
  `dot(R_m, V)^{4 e^{b_m}} c_m` (sb=2 default).
- **Kernel-agnostic MCMC**: 3DGS-MCMC relocation/growth but only opacity is adjusted
  (`o' = 1-(1-o)^{1/N}`), no scale adjustment; opacity + scale L1 regularisers keep o small;
  position noise uses the Beta function `(1-o)^{100}` instead of the logit sigmoid.

Contributions
- C1 Beta kernel geometry (Sec. 4.1-4.2, Eq. beta, compositing equation) — custom CUDA rasterizer.
- C2 Spherical Beta colour (Sec. 4.3).
- C3 Kernel-agnostic MCMC (Sec. 4.4): opacity-only relocation/add, Beta noise, regularisers.

## 2. Equations (verbatim from `sec/4_method.tex`)
- Beta kernel: `\mathcal{B}(x; b) = (1 - x)^{\beta(b)}, \beta(b) = 4e^{b}, x \in [0,1]`.
- Distance: `r_i(\bm{x}) = \sqrt{(\bm{x} - \bm{\mu}_i^\prime)^\top \bm{\Sigma}_i^{\prime^{-1}} (\bm{x} - \bm{\mu}_i^\prime)}`.
- Compositing: `\bm{C}(\bm{x}) = \sum_{i=1}^N c_i o_i \mathcal{B}(r_i(\bm{x})^2; b_i) \prod_{j=1}^{i-1} (1 - o_j \mathcal{B}(r_j(\bm{x})^2; b_j))`.
  Code (`rasterize_to_pixels_fwd.cu`): `sigma = conic.x dx^2 + conic.z dy^2 + 2 conic.y dx dy`
  (no 1/2), skip if `sigma < 0 || sigma >= 1`, `alpha = min(0.999, o * (1-sigma)^beta)`,
  **no 1/255 alpha skip**, stop when `T*(1-alpha) <= 1e-4`. Backward gate `o*vis <= 0.999` as gsplat.
  `v_beta = v_alpha * o * (1-sigma)^beta * log(1-sigma)` (w.r.t. activated beta).
- Spherical Beta: `c(\hat{V}) = c_0 + \sum_{m} \mathcal{B}(1-\hat{R}_m \cdot \hat{V}; b_m) c_m`
  i.e. `dot^{4 e^{b_m}}` for dot > 0 else 0. Code (`spherical_beta.cuh`, `rendering.py`):
  `c0 = clamp_min(SH_deg0(sh0) + 0.5, 0)`; `R_m = (sin t cos p, sin t sin p, cos t)` from raw
  (theta, phi); `V = normalize(mu - campos)`; lobe colour `c_m = softplus(raw, beta=10 ln 2)`;
  final `clamp_min(c, 0)`. `sb_params` layout `[N, M, 6] = (r, g, b, theta, phi, b_m)`.
- Noise: `\mu \leftarrow \mu - \lambda_{lr} \nabla_\mu L + \lambda_\epsilon \epsilon`,
  `\epsilon = \lambda_{lr} \mathcal{B}(o_i; b') \Sigma_\eta`, `b' = ln 25` → `(1-o)^{100}`.
  Code: `noise = randn * (1-o)^100 * noise_lr * xyz_lr; noise = Sigma @ noise` with
  `Sigma = R S S R^T` (covariance, not its square root), **applied only at densification steps**.
- Loss: `L = (1-\lambda_{SSIM}) L_1 + \lambda_{SSIM} L_{SSIM} + \lambda_o \sum_i |o_i| + \lambda_\Sigma \sum_i\sum_j |\sqrt{eig_j(\Sigma_i)}|`.
  Code: `opacity_reg * mean(sigmoid(o)) + scale_reg * mean(exp(s))`, both 0.01, only for
  `densify_from_iter < it < densify_until_iter` (500 < it < 25000). SSIM is fused-ssim (same padding).
- Densification (`o' = 1 - \sqrt[N]{1-o}`): every 100 its for 500 < it < 25000: dead = o <= 0.005;
  relocate dead to multinomial(opacity of alive) samples; set sample and copies to
  `clamp(1-(1-o)^{1/(k+1)}, 0.005, 1-eps)` (k = times sampled); no scale change; zero Adam moments
  of the sampled rows. Then add `min(cap, int(1.05 N)) - N` copies of multinomial(opacity)
  samples, same opacity rule, zero moments. Then noise. Then optimizer.step (with the replaced
  Parameters having no `.grad`, so this step is a no-op on densification iterations).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| lr means | 1.6e-4 x extent -> 1.6e-6 x extent, exp decay over 30k (delay_mult 0.01 inactive: delay_steps 0) | supp. Hyperparameters; `arguments/__init__.py` |
| lr sh0 (base colour) | 0.0025 | same |
| lr sb_params (lobe colour, theta, phi, b_m) | 0.0025 | same |
| lr opacity | 0.05 | same |
| lr beta (b) | 0.001 | same |
| lr scaling / rotation | 0.005 / 0.001 | same |
| Adam eps | 1e-15 | `beta_model.py:training_setup` |
| lambda_dssim | 0.2 | code |
| opacity_reg / scale_reg | 0.01 / 0.01 (means) | code; supp. Table psnr_opacity (lambda_o 0.01 best) |
| noise_lr | 5e4 | code |
| densify from / until / interval | 500 / 25000 / 100 | code |
| dead threshold, min opacity | 0.005 | `train.py`, `_update_params` |
| growth | 5% per densification | `add_new_gs` |
| cap_max | per scene: bicycle 6M, flowers 3M, garden 5M, stump 4.5M, treehill 3.5M, room/counter/kitchen/bonsai 1.5M, truck 2.5M, train 1M, drjohnson 3.5M, playroom 2.5M | `benchmark.py`; supp. Table fulltraining |
| sh_degree | 0 (SH only for base colour) | code default |
| sb_number | 2 | paper Sec. 5.1; code |
| init | SfM points; scale = log sqrt(mean 3-NN d^2); opacity 0.5; quats identity; b = 0; SB colour/b_m raw 0, theta ~ U[0,pi], phi ~ U[0,2pi] | `create_from_pcd` |
| background | black | `train.py` (white_background False) |
| eps2d / near / tile | 0.3 / 0.01 / 16 | fork `rendering.py` defaults |
| bounding radius | `ceil(sqrt(lambda_max(Sigma2D)))` (1 sigma) | `fully_fused_projection_fwd.cu` |

## 4. Schedules
means LR decay; regularisers active 500 < it < 25000; densification every 100 its 500 < it <
25000 (relocation + growth + noise); `oneupSHdegree` every 1000 its is a no-op (sh_degree 0).

## 5. Builds on
3DGS-MCMC (densification framework) on gsplat 1.3.0 (projection, tiling). Unchanged from 3DGS:
EWA projection with eps2d 0.3, L1 + 0.2 D-SSIM photometric loss, Adam per-parameter LRs.

## 6. Protocol
Mip-NeRF 360 (outdoor `-r 4`, indoor `-r 2`, PIL resize of full-res `images/`), T&T, DB
(`-r -1` → full 1.6k-width cap), NeRF Synthetic. llffhold 8 on name-sorted images. SfM init.
Metrics: PSNR, SSIM (fused-ssim), LPIPS-VGG (lpipsPyTorch). Two modes: "30K" (30k iterations)
and "full" (patience-based early stopping; the released code selects the best checkpoint by
**test-set PSNR** every 500 its after 15k with patience 20 — a test-set selection). Main table
"Ours (30k)" and supp. Table allmetrics columns are **sb=2**. GPU RTX 6000 Ada.

## 7. Ambiguities / paper-vs-code
- Regulariser: paper `sum_i |o_i|`, `sum sqrt(eig)`; code uses means over primitives and
  `exp(scales)` (= sqrt eig). Code wins.
- Noise: paper presents it as part of every update; code applies it only every 100 its in the
  densification window. Code wins.
- Released `train.py --eval` always runs patience/test-selection ("full"); "30K" mode corresponds
  to plain 30k iterations (no `--eval` in the released code means no test split). We reproduce 30K.
- Spherical-Beta lobe colour is softplus(β=10 ln2) activated; base colour uses SH0 + 0.5 clamp.
- Optimizer step is skipped on densification iterations (Parameter re-wrap drops `.grad`).
