# DNGaussian — implementation spec

Li, Zhang, Bai, Zheng, Ning, Zhou, Gu. *DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance
Fields with Global-Local Depth Normalization*, CVPR 2024, arXiv 2403.06912.
Official code: https://github.com/Fictionarry/DNGaussian (cloned to `source/code/`, read only).
LaTeX: `source/latex/main_arxiv.tex` (main + supplementary in one file).

## 1. Method summary
- 3DGS from a **random** initialization (paper §3.1 "Initialization"; code `--rand_pcd`), trained
  6000 iterations on 3 views with monocular DPT depth of the training views.
- Every iteration runs **three separate forward/backward/optimizer-step passes** on the same view
  (code `train_llff.py:91-168`): hard depth (only centres learn), soft depth (only opacity learns,
  after 1000 iterations), and the photometric pass (all parameters + neural renderer).
- Depth losses compare patches of the rendered depth with the DPT map after **Global-Local Depth
  Normalization** (patch mean removed; divided by patch std (local) or image std (global)).
- Colour comes from a **Neural Color Renderer** (hash grid + MLP, direction SH-encoded) instead of
  per-Gaussian SH.

Contributions
- **C1 Hard Depth Regularization** (§3.2): render depth with all opacities set to τ = 0.95 and
  scale/rotation frozen; only centres μ receive gradients.
- **C2 Soft Depth Regularization** (§3.2): render depth with real opacity and frozen μ, s, q; only
  opacity α receives gradients. Starts after 1000 iterations.
- **C3 Global-Local Depth Normalization** (§3.3): patch-wise local + global normalization, L2 with
  an error tolerance, random patch size in [5, 17] (LLFF).
- **C4 Neural Color Renderer** (§3.4 "Neural Color Renderer", supp. A.2): hash-grid encoder
  (16 levels, 16→512 resolution, 2^19 table) + MLP; inference caches the geometry feature.
- **C5 Training recipe of the official code** (not claimed as a contribution in the paper, but
  needed for the numbers): random init, opacity-gated densification, near-camera pruning,
  invisible-Gaussian pruning, shape/scale/opacity penalties, LR schedule.

## 2. Equations (verbatim from the LaTeX)
Depth (Eq. 4): `\mathcal{D}(x_p) = \sum_{i \in N}{||\mu_i - o||_2} \times \widetilde{\alpha}_i \prod_{j=1}^{i-1}(1-\widetilde{\alpha}_j)`.
**Code**: the ashawkey rasterizer accumulates camera-space z, `D = Σ_i T_i α_i z_i`, not normalized
by alpha and without background (gsplat `render_mode="D"`). Code overrides the paper (ray distance).

Hard depth (Eq. 5): `\mathcal{D}_{hard}(x_p) = \sum_{i \in N}{\tau (1-\tau)^{i-1}\mathcal{G}^{proj}_i(x_p) ||\mu_i - o||_2 }`
and `\mathcal{R}_{hard}(\mathcal{P}) = \mathcal{L}_{similar}(\mathcal{D}_{hard}(\mathcal{P}), \widetilde{\mathcal{D}}(\mathcal{P}))`.
Code `render_for_depth`: opacity `ones * 0.95`, colours ones, scales/rotations `.detach()`, means
with grad → only μ learns.

Soft depth (Eq. 6): `\mathcal{D}_{soft}(x_p) = \sum_{i \in N}{||\check{\mu_i} - o||_2} \times \widetilde{\alpha}_i \prod_{j=1}^{i-1}(1-\widetilde{\alpha}_j)`.
Code `render_for_opa`: means, scales, rotations detached; opacity with grad → only α learns.

Local normalization (Eq. 7): `\mathcal{D}^{LN}(x) = \frac{\mathcal{D}(x) - \text{mean}(\mathcal{D}(\mathcal{P}))}{\text{std}(\mathcal{D}(\mathcal{P})) + \epsilon}`.
Global normalization (Eq. 8): `\mathcal{D}^{GN}(x) = \frac{\mathcal{D}(x) - \text{mean}(\mathcal{D}(\mathcal{P}))}{\text{std}(\mathcal{D}_I)}`.
Code (`utils/loss_utils.py`):
```
patchify(x, p)   = F.unfold(x, kernel_size=p, stride=p) -> [n_patches, p*p]   (remainder cropped)
normalize(x, mean=None, std=None) = (x - mean_dim1(x)) / (std + 1e-2 * std(x.reshape(-1)))
   local : std = unbiased std of each patch (dim 1)
   global: std = input.std().detach()  (whole rendered / mono depth map, unbiased)
margin_l2_loss(a, b, m) = mean((a-b)[|a-b| > m]^2)
patch_norm_mse_loss(_global)(D, D~, p, m) = margin_l2_loss(normalize(patchify(D)), normalize(patchify(D~)), m)
```
So ε = 1e-2·std(all patch values) (scale-equivariant, not a constant).

Regularization (Eq. 9): `\mathcal{R}_{T} = \mathcal{L}_{2}(\mathcal{D}_{T}^{GN}, \widetilde{\mathcal{D}}^{GN}) + \gamma \mathcal{L}_{2}(\mathcal{D}_{T}^{LN}, \widetilde{\mathcal{D}}^{LN})`, γ = 0.1.
Code adds for iteration > 3000 an edge-aware smoothness term with weight 0.1 (not in the paper):
`loss_depth_smoothness(D, D~) = (Σ|∂x D|·e^{-|∂x D~|} + Σ|∂y D|·e^{-|∂y D~|}) / (Σe^{-|∂x D~|} + Σe^{-|∂y D~|})`
with the raw (un-normalized) D and D~ (scale dependent). Local and global use independent
random patch sizes `randint(5, 17)` (inclusive).

Colour (Eq. 10): `\mathcal{L}_{color} = \mathcal{L}_1(\hat{\mathcal{I}}, \mathcal{I}) + \lambda \mathcal{L}_{\mathrm{D-SSIM}}(\hat{\mathcal{I}}, \mathcal{I})`, λ = 0.2
(code: `Ll1 + 0.2 * (1 - ssim)`, L1 weight 1, 3DGS SSIM with same padding).
Full (Eq. 11): `\mathcal{L} = \mathcal{L}_{color} + \mathcal{R}_{hard} + \mathcal{R}_{soft}` —
**code applies them in three separate backward + optimizer steps per iteration**, not summed.

Photometric-pass penalties (code only, `train_llff.py:154-160`):
`shape_pena = mean(max_s/min_s)`, `scale_pena = mean(max_s^2)`,
`opa_pena = 1 - mean(o[o>0.2]^2) + mean((1-o[o<0.2])^2)` over all Gaussians;
`loss += 0.002·shape + 0.001·scale + 0.001·opa` (scale_pena keeps the default 0.001 since
run_llff.sh does not set it). An empty selection makes the official value NaN (logged as such)
while the gradients of the other terms stay finite.

Neural Color Renderer (code `scene/neural_renderer.py`, `gaussian_renderer/__init__.py`):
`enc = HashGrid(x - center; bound)`, 16 levels × 2 features, base 16, desired res `512·bound`,
table 2^19, linear, align_corners False, init U(±1e-4), out-of-box inputs encode to 0;
`h = sigma_net(enc)` (3 bias-free Linear, hidden 64, ReLU, out 1+64; the sigma output is unused
because `combine_opacity` returns `sigmoid(_opacity)`), `geo = h[1:]`;
`d = normalize(x - cam_center)`, `c = sigmoid(color_net([SH4(d), geo]))·1.002 - 0.001`
(color_net: 2 bias-free Linear, 80→64→3). `bound = max(xyz.max-xyz.min)/2·1.2`,
`center = xyz.mean` of the **initial** points. Gradients flow into μ through the hash grid and
the direction. Paper: "5 layer MLP with hidden dim 64" (3 + 2 layers). Inference: geo feature
cached, only color_net evaluated.

## 3. Hyper-parameters (LLFF, `scripts/run_llff.sh` + `arguments/__init__.py`)
| name | value | source |
|---|---|---|
| iterations | 6000 | run_llff.sh; paper §4.1 |
| resolution | `-r 8` (504×378) | run_llff.sh; supp. B.2 |
| init | `--rand_pcd`, num = int(max bbox·50), opacity 0.1, 3-NN scales | dataset_readers.py, gaussian_model.py |
| lambda_dssim | 0.2 (L1 weight 1) | run_llff.sh |
| densify_grad_threshold | 0.0013 | run_llff.sh |
| densify from / until / interval | >500 / <6000 / 100 | arguments, run_llff.sh |
| percent_dense | 0.01 | run_llff.sh |
| split_opacity_thresh | 0.1 (clone and split only if opacity > 0.1) | run_llff.sh, gaussian_model.py:454-492 |
| prune_threshold | 0.01 | run_llff.sh |
| screen-size pruning / opacity reset | none / disabled (commented out) | train_llff.py:194-200 |
| position lr | 0.016 → 0.00016 (×extent), exp over 5500 steps, held until 500 (`max(it-500,0)`) | run_llff.sh |
| scaling / opacity / rotation lr | 0.003 / 0.05 / 0.001 | run_llff.sh, defaults |
| neural grid / net lr | 5e-3 / 5e-4, Adam eps 1e-15 (single Adam) | arguments, gaussian_model.py:219 |
| error_tolerance (margin) | 0.00025 | run_llff.sh |
| hard_depth_start / soft_depth_start | 0 / 1000 | arguments |
| τ (hard opacity) | 0.95 | render_for_depth(value=0.95); paper |
| γ (local weight) | 0.1; global 1.0 | train_llff.py:103-108 |
| smoothness | 0.1 after 3000 | train_llff.py:101 |
| patch size | randint(5, 17) | train_llff.py:66 |
| shape / scale / opa penalty | 0.002 / 0.001 / 0.001 | run_llff.sh, defaults |
| near pruning | prune ‖μ - c‖ < 10 for the 180 spiral cams, every 25 its from 2001 | train_llff.py:202-209, `--near 10` |
| invisible pruning | prune Gaussians invisible from all train views at it 1 and 6000 | clean_views |
| background | black | white_background False |
| SH | unused (neural colour); SH increase commented out | train_llff.py:74 |
| znear | 0.2 culling in the rasterizer (zfar 100 only in the projection) | 3DGS rasterizer |

## 4. Schedules
- it ≥ 1: hard depth pass; it > 1000: soft depth pass; it > 3000: smoothness in both.
- xyz lr constant until 500, then exponential decay over 5500 steps.
- densify at 600, 700, ..., 5900; stats accumulate from it 1 (photometric pass only).
- near pruning at 2001, 2026, ..., 5976; clean_views at 1 and 6000.

## 5. Builds on
Official 3DGS code (Inria rasterizer variant by ashawkey that also outputs depth/alpha).
Kept: photometric L1 + D-SSIM, clone/split/prune densification (with extra opacity gating),
Adam. Changed: random init, no opacity reset, no SH, extra depth passes.

## 6. Protocol (LLFF)
8 scenes, 3 training views = round(linspace(0, n-1, 3)) of the non-held-out images (sorted by
name), test = every 8th image ("eval" cameras; `metrics.py` scores only `eval`), images /8 →
504×378 (PIL default bicubic), no undistortion, black background, 6000 iterations. Metrics:
PSNR, LPIPS (VGG, lpipsPyTorch), SSIM (the paper's table SSIM is not specified; README reports
both skimage and 3DGS SSIM; prior NeRF works use skimage). Render-time `--near 10`: Gaussians
within 10 units of **any eval camera centre** are removed before rendering the eval set.
DPT: `dpt_hybrid_384` on the full-res training images, bicubic-resized to H/8×W/8, min-max to
uint16 PNG, used as `255 - png/255` (relative inverse depth, sign flipped).

## 7. Ambiguities
- Paper Eq. 4 uses ray distance, code uses camera z → code.
- Paper says L = L_color + R_hard + R_soft; code uses three optimizer steps per iteration → code.
- Smoothness term and the three penalties are code-only → code.
- README: released checkpoints use "a lower error tolerance" than the paper; the script value
  0.00025 is used.
- SSIM flavour of the paper table unknown (skimage vs 3DGS); our runtime SSIM differs from both.
- README: "metrics may be unstable in some scenes, especially PSNR" (random init/densification).
- No per-scene numbers in the paper (LLFF average only).
