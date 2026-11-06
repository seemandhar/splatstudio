# WildGaussians: 3D Gaussian Splatting in the Wild — implementation spec

arXiv 2407.08447 (v2), NeurIPS 2024. Kulhanek, Peng, Kukelova, Pollefeys, Sattler.
Official code: github.com/jkulhanek/wild-gaussians (MIT; `source/code/`, HEAD of main, cloned
2026-11-05). Code is read, never run. **Code defaults override the paper; both are recorded.**

## 1. Summary and contributions
- 3DGS + Mip-Splatting (3D smoothing filter + 2D Mip filter, kernel 0.1) + GOF/AbsGS absolute
  screen-gradient densification (§3.1, supp. §A).
- **C1 Appearance modeling (§3.2)**: per-image embedding e_j (32-d), per-Gaussian embedding g_i
  (24-d, Fourier-feature init), MLP f(e_j, g_i, c̄_i) → per-channel affine (β, γ) applied to the
  Gaussian's colour. Two rasterisations per step: raw (no toning) and toned.
- **C2 Split training objective (§3.2 Eq. 6)**: D-SSIM on the raw image, L1 on the toned image.
- **C3 Test-time appearance optimisation (§3.2, supp.)**: unseen image → embedding initialised
  at 0, 128 Adam steps, lr 0.1, objective Eq. 6 (code: masked by the uncertainty), everything else
  frozen. Photo Tourism evaluation = NeRF-W protocol (optimise on left half, metrics on right half).
- **C4 DINO uncertainty (§3.3 Eq. 8–9)**: σ = clip(softplus(affine(DINOv2 ViT-S/14 features of the
  GT image)), 0.1, ∞), trained with the DINO-cosine loss; not back-propagated into the scene.
- **C5 Uncertainty → binary mask (§3.3 Eq. 10–11)**: M = 1(1/(2σ²) > 1) multiplies the per-pixel
  colour loss; disabled during warm-up; uncertainty training paused after opacity resets.
- **C6 Sky initialisation (§3.4, supp.)**: Fibonacci sphere of 100 k points at 10 r_s added to
  the SfM points.

## 2. Equations (paper numbering) and code
Eq. 4 `(β, γ) = f(e_j, g_i, c̄_i)`; Eq. 5 `c̃_i = γ · ĉ_i(r) + β`.
Code (`EmbeddingModel.forward`, method.py:890): input `cat(features[:, :3] (SH DC coeffs, *not*
RGB), g_i, e_j)` (3+24+32 = 59) → Linear 128 → ReLU → Linear 128 → ReLU → Linear 6;
`offset, mul = split(mlp(inp) * 0.01)`; toned SH = `features * mul` (mul repeated over all 16 SH
coefficients) with `offset / C0` added to the DC coefficient only. **No "+1" on γ** (paper supp.
says γ = 0.01γ̂ + 1; code has γ = 0.01γ̂). Toned SH `.clamp_max(1.0)`, evaluated with
eval_sh(active degree), `+0.5`, `clamp_min(0)`.
Raw colours: `features.clamp_max(1.0)` (all SH coefficients clamped at 1), eval_sh +0.5,
clamp_min 0. Both colour sets go through the same rasteriser (two calls).

Eq. 6 `L_color = λ_dssim DSSIM(Ĉ, C) + (1 − λ_dssim) ||C̃ − C||_1`, λ_dssim = 0.2.
Code: `0.8 * mean(L1_map(toned) * M) + 0.2 * mean((1 - ssim_map(raw)) * M)`, ssim = 3DGS same-
padded 11×11 Gaussian SSIM map averaged over channels.

Eq. 7 NLL (background). Eq. 8 `L_dino(D̃, D) = min(1, 2 − 2 D̃·D/(|D̃||D|))`.
Eq. 9 `L_uncertainty = L_dino / (2σ²) + λ_prior log σ`, λ_prior = 0.5 (`uncertainty_regularizer_weight`).
Code (`UncertaintyModel._compute_losses`, mode "dino"):
- σ: GT image normalised by ImageNet mean/std, zero-padded (centered) to a multiple of 14, DINOv2
  `dinov2_vits14_reg` `get_intermediate_layers(n=[num_heads-1]=[5], reshape=True, norm=True)`
  (output of block 5 of 12 + final LayerNorm), Dropout2d(p=0.5, train only) → BatchNorm2d(384)
  (train mode, batch stats) → Conv1×1(384→1) (weight N(0, 0.01), bias 0) → `+ log(e − 1)` →
  softplus → bilinear upsample to padded size → clamp_min 0.1 → crop. Computed at full GT
  resolution (`uncertainty_dino_max_size` None). GT features are cached per training image.
- cosine: GT and *toned prediction (detached)* resized with `dino_downsample(·, 350)` (scale
  min(350/h, 350/w), int, round up to multiple of 14, bilinear), DINO features of both (no grad),
  `F.cosine_similarity(dim=1)`, bilinear upsample to padded size, crop.
  `dino_part = clip(1 − (cos − 0.5)/0.5, 0, 1)`.
- `loss_mult = 1/(2σ²)`; `L_unc = mean(dino_part · dino_downsample(loss_mult, 350)) + 0.5 · mean(log σ)`.
- returned mask: `loss_mult` (full res) `.clamp_max(3)`; train: `M = (loss_mult > 1)`.
Eq. 10 `M = 1(1/(2σ²) > 1)`; Eq. 11 masked colour loss (as above).
Uncertainty loss is *added to the total loss*; only the uncertainty head (BN + conv) receives its
gradient (prediction detached, DINO frozen). It is detached while
`iteration < floor(min(it, densify_until−1)/reset_interval)·reset_interval + protected_iters`.
Warm-up: `M := 1` for it < warmup_start; then `1 + p (M − 1)`, p = (it − start)/warmup_iters.

Fourier init of g_i (`_get_fourier_features`): xyz (SfM + sky points) − mean, / per-axis 0.97
quantile of |·| · 0.5 + 0.5; freqs 2^{0..3} (each twice), offsets {0, π/2}:
`sin(2π f p + offset)` → 3·4·2 = 24 values, ordered [x: f0 sin, f0 cos, f1 sin, …, y: …, z: …];
+ N(0, 1e-4) noise. (Paper: sin/cos(π p 2^m), m = 1..4 — identical.)
Image embeddings: N(0, 0.01). Test embedding: zeros (also the embedding used to render any
non-training camera without optimisation).

Sky (`get_sky_points`): fibonacci sphere (N = 100 000), centre = mean of SfM points, radius
r = 10 · quantile_0.97(|x − mean|₂); keep a point if for some training camera `camera_project`
gives v < 2/3 H (official quirk: points with z ≤ eps project to (cx, cy) and are kept; no u or
v ≥ 0 bound). Colour (237, 247, 252)/255, opacity 1.0 (logit = +inf in code); SfM points 0.1.

## 3. Hyper-parameters
| name | On-the-go | Photo Tourism | source |
|---|---|---|---|
| iterations | 30 000 | 200 000 | supp. §A; phototourism.yml |
| position lr | 1.6e-4 → 1.6e-6 ×extent | 1.6e-5 → 1.6e-7 ×extent | supp.; yml |
| scales lr | 0.005 | 0.0005 | config.py; yml |
| rotation / opacity / sh0 / shN lr | 0.001 / 0.05 / 0.0025 / 0.000125 | same | config.py |
| Gaussian embedding lr | 0.005 | 0.005 | supp.; config embedding_lr |
| image embedding lr | 0.001 | 0.001 | supp.; config |
| appearance MLP lr | 0.0005 | 0.0005 | supp.; config |
| uncertainty lr | 0.001 | 0.001 | supp.; config |
| Adam eps | 1e-15 (all groups) | | method.py:1053 |
| densify from / until / every | 500 / 15 000 / 100 | 4000 / 100 000 / 400 | supp.; yml |
| grad threshold | 2e-4 (+abs-grad quantile) | 2e-4 | config |
| opacity reset every | 3000 | **15 000 (code)**; paper 3000 | yml vs supp. |
| uncertainty warm-up start / ramp | 2000 / 0 | 35 000 / 5000 | yml; supp. |
| uncertainty protected iters | 500 | 1500 | yml; supp. |
| sky points | 100 000 | 100 000 | yml; supp. |
| appearance enabled | **false (code yml)**; paper: true | true | nerfonthego.yml vs Tab. 4/7 |
| percent_dense / min opacity / big-ws prune | 0.01 / 0.005 / 0.1·extent after first reset interval | | method.py |
| SH degree | 3, +1 every 1000 it | | |
| Mip 2D kernel / 3D filter coef | 0.1 / 0.2 | | config kernel_size; compute_3D_filter |
| test-time opt | 128 Adam steps, lr 0.1, "dssim+l1-scaled" | | supp.; config |
| background | black | | method.py |

## 4. Schedules
SH degree +1 every 1000 it (before rendering); xyz lr log-linear over `iterations`; densify
when `from < it < until` and `it % every == 0`, then recompute the 3D filter; opacity reset when
`it % reset == 0` and `it < until` (Mip-Splatting filter-aware: min(filtered α, 0.01)/coef);
3D filter every 100 it after `until` while `it < iterations − 100`; masking warm-up; uncertainty
protected window after resets.

## 5. Builds on / unchanged from 3DGS
Inria 3DGS training loop (shuffled view stack, one view per step, 1-indexed iterations), Mip-
Splatting renderer (3D filter, 2D filter 0.1) with GOF absolute gradients. Densification: clone
(max scale ≤ 0.01·extent) and split (N = 2, scale/1.6) of Gaussians whose mean NDC gradient ≥ 2e-4
**or** whose mean absolute gradient ≥ Q, Q = quantile(abs grads, 1 − ratio), ratio = fraction
selected by the normal criterion. **Gradient statistics are never reset** (densification_postfix
appends zeros for new points and keeps the old accumulators); prune opacity < 0.005 and (after the
first reset interval) world size > 0.1·extent; screen-size prune disabled. Initial features_dc =
**RGB in [0, 1] stored as the SH DC coefficient** (official quirk; RGB2SH commented out); scales
log(sqrt(mean 3-NN dist²)) via distCUDA2; quats identity.

## 6. Protocol
- NeRF On-the-go (undistorted, HF `jkulhanek/nerfonthego-undistorted`): images pre-downscaled
  (patio ×4 → 480×270, patio-high ×8 → 504×378, mountain/fountain/corner/spot ×8); splits from
  `train_list.txt`/`test_list.txt`; no test-time optimisation; metrics on full images.
- Photo Tourism (IMC-2020 `dense/` reconstructions; Brandenburg Gate, Sacré-Cœur, Trevi):
  full-resolution images, NeRF-W split (`nerfw_split.csv`); NeRF-W protocol: embedding optimised on
  the left `W − W//2` columns, full image rendered, metrics on columns `[W//2:]`.
- Metrics (NerfBaselines): predictions and GT converted to uint8 (truncation), PSNR per image
  averaged, SSIM = dm_pix (valid Gaussian 11×11, σ 1.5, clipped variances), LPIPS = **AlexNet**.
- COLMAP world coordinates are used unnormalised. Single seed 0. RTX 4090.

## 7. Ambiguities / paper vs code
1. γ = 0.01γ̂ + 1 (paper) vs γ = 0.01γ̂ (code). Port: code.
2. On-the-go appearance: paper Tab. 7 "Ours" ≠ "w/o app." row, but released `nerfonthego.yml`
   sets `appearance_enabled: false`. Port: both variants; headline = paper (appearance on, the
   numbers we compare against are labelled "Ours").
3. Photo Tourism opacity reset 3000 (paper) vs 15 000 (code). Port: code.
4. Uncertainty predictor described as "single affine transformation"; code adds BN + Dropout2d.
5. Paper: DINO input resized to max 350 for the loss; code also predicts σ at full resolution.
6. NeRF-W half split: odd widths share one column between optimisation and metrics (official).
7. Opacity reset quirk (method.py:1249-1266): `reset_opacity` takes the *already filtered* scales
   s_f = sqrt(s² + f²) from `get_gaussians()` and adds f² again, so the new raw opacity is
   min(σ(o)·coef, 0.01) / coef' with coef' = sqrt(∏ s_f² / ∏(s_f² + f²)) instead of / coef.
   Gaussians at or below the filter size leave every reset with filtered opacity < 0.01 (≈0.0065
   at s = f) and very small ones are pruned at the next densification. Port: official formula.
8. Absolute-gradient statistic (GOF backward.cu:594-595): per pixel |∂L/∂x| + |∂L/∂y| summed into one
   channel (L1), whose norm is accumulated. Port: same (gsplat absgrad (Σ|gx|, Σ|gy|) summed).
