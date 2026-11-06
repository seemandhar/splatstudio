# Deblurring 3D Gaussian Splatting (Lee, Lee, Sun, Ali, Park; ECCV 2024) — implementation spec

arXiv 2401.00834v3 · project https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/ ·
official code https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting (cloned to
`source/code/`, HEAD of default branch, 3DGS-based; read, never executed).

## 1. Method summary
- 3DGS trained on blurry images; during training a small MLP F_θ maps each Gaussian's
  (position, rotation, scale, viewing camera) to per-Gaussian multiplicative factors (δr, δs)
  ≥ 1 (and, for camera-motion blur, M position offsets δx). The transformed Gaussians are
  rasterized to produce the *blurry* training image; the loss compares it to the blurry photo.
- At test time the MLP is not used: the original Gaussians render the sharp image.
- Defocus: one transformed render. Camera motion: M (=5 total) renders with different δx/δr/δs,
  averaged.
- Sparse SfM compensation: at iteration 2500 extra points are sampled uniformly in the point
  bbox, coloured by KNN interpolation (rejecting points far from all neighbours); and
  depth-dependent opacity pruning keeps more far Gaussians.

Contributions:
- **C1 defocus blur MLP** (Sec. 3.2 "Defocus blur modeling", Eq. 7–10): (δr, δs) = F_θ(γ(x), r, s, γ(v)); r̂ = r·clip(λs δr + 1−λs), ŝ = s·clip(λs δs + 1−λs).
- **C2 camera-motion blur** (Sec. 3.2 "Camera motion blur modeling", Eq. 11–12): M sets
  {x + λp δx_i, r·δr_i, s·δs_i}, I_b = mean of the M renders.
- **C3 extra points** (Sec. 3.3, Alg. 1, supp. Eq. N_p): KNN-interpolated colour, distance threshold.
- **C4 depth-based pruning** (Sec. 3.3, Fig. 4): opacity prune threshold t_p / w_p(depth).
- **C5 sharp inference**: render G(x, r, s) without the MLP (Sec. 3.2).

## 2. Equations (verbatim from main.tex)
- Eq. (defocus MLP) `(\delta r_j, \delta s_j) = \mathcal{F_\theta}\Big(\gamma(x_j), r_j, s_j, \gamma(v) \Big)`
- PE: `\gamma(p) = \big(\sin(2^k\pi p), \cos(2^k \pi p)\big)_{k=0}^{L-1}`
- `\hat{r}_j=r_j\cdot \text{min}(1.0, ~\lambda_s \delta r_j + (1-\lambda_s))`,
  `\hat{s}_j=s_j\cdot \text{min}(1.0, ~\lambda_s \delta s_j + (1-\lambda_s))`
  (text: "the minima of them are clipped to 1.0" and "ŝ ≥ s" → it is a max/clamp-min; code
  `torch.clamp(λs·δ + (1−λs), min=1.0, max=max_clamp=1.1)`).
- Motion: `\{(\delta x_j^{(i)}, \delta r_j^{(i)}, \delta s_j^{(i)})\}_{i=1}^{M} = \mathcal{F_\theta}(...)`,
  `\hat{x}_j^{(i)} = x_j + \lambda_p\delta x_j^{(i)}`,
  `I_b = \frac{1}{M}\sum_{i=1}^{M} I_i, \quad I_i = \texttt{Rasterize}(\{G(\hat{x}_j^{(i)},\hat{r}_j^{(i)},\hat{s}_j^{(i)})\})`
- Extra points (supp.): `N_p = \text{min}(\frac{\text{prod}(Q_{max} - Q_{min})}{c^3}, N_{max})`, c = 1.1, N_max = 200000.
- Loss: unchanged 3DGS `(1−0.2) L1 + 0.2 (1 − SSIM)` on the blurred render (code train.py).

### Code-level definitions (source/code)
- `scene/blur_kernel.py` GTnet: inputs `cat[embed_pos(x), embed_view(camera_center), s, q]`;
  NeRF embedder with include_input, log-sampled freqs 2^0..2^(L−1), **no π**; L_pos = 3
  (21 ch), L_view = 10 (63 ch), +3 scale (activated, exp) +4 quaternion (normalised) = 91.
  Trunk: Linear(91,64)-ReLU-[Linear(64,64)-ReLU]×2 (`hidden=3`), heads (no activation):
  defocus s: Linear(64,3), r: Linear(64,4); motion s: 3(M+1), r: 4(M+1), p: 3M with
  `num_moments` M = 4. Init xavier_normal (gain 0.1 if out_features ∈ {2,3}, else 1), bias 0.
  Inputs x, s, q are **detached**; view input is the camera centre (not a direction).
- `gaussian_renderer/__init__.py`: δs = clamp(λs·δs + 1−λs, 1, 1.1), δr likewise;
  defocus: rasterize(means, scales·δs, normalize(q)·δr). Motion: δx·λp reshaped [N,3,M];
  δs [N,3,M+1], δr [N,4,M+1]; render 0 = (x, s·δs[...,M], q·δr[...,M]) (no shift), renders
  1..M = (x+δx_i, s·δs_i, q·δr_i); output = mean of the M+1 = 5 renders (paper's M = 5).
  The Inria rasterizer does **not** normalise the quaternion in computeCov3D, so q·δr gives
  Σ = R(q̃) S² R(q̃)ᵀ with an *unnormalised* q̃ (R not orthogonal). Test: plain 3DGS render.
- Densification stats (motion): each of the 5 renders adds ‖∂L/∂means2D_i‖ and denom += 1/5.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 20000 | paper §4.1; arguments iterations |
| MLP lr | 1e-3 (Adam, eps 1e-15, same optimizer as Gaussians) | paper §4.1; gtnet_lr |
| xyz lr | 1.6e-3 → 1.6e-5 × extent, log-linear over **30000** steps (no delay) | paper (1.6e-3); arguments position_lr_* , position_lr_max_steps=30000 |
| feature / opacity / scaling / rotation lr | 0.0025 (/20 SH rest) / 0.05 / 0.005 / 0.001 | 3DGS defaults (arguments) |
| densify grad threshold | defocus 2e-4, motion 5e-4 | paper §4.1; configs |
| prune opacity t_p | defocus 5e-3, motion 1e-2 | paper §4.1; configs |
| densify from / until / interval | >500 / <15000 / 100 | arguments, train.py |
| opacity reset | **none** (reset_opacity never called) | train.py |
| screen/world size prune | none (size_threshold = None) | train.py |
| percent_dense | 0.01 | arguments |
| λs, λp | 1e-2, 1e-2 (real) | paper §4.1; configs |
| max_clamp | 1.1 | code only (arguments) |
| MLP width / depth | 64 / 3 shared + 1 head | paper §4.1; hidden=3, width=64 |
| PE freqs | pos 3, view 10 | code only (GTnet defaults) |
| M | 5 renders (num_moments 4 + unshifted one) | paper §4.1; code |
| extra points N_st | 2500 | paper; pts_iter |
| K, t_d | 4, 2 (world units) | paper; pts_N_intpl, pts_dist |
| N_p | min(prod(bbox of initial SfM)/1.1³, 200000) | supp.; train.py |
| sampling bbox | per-axis sorted positions [10], [−10] of the current Gaussians | code only (pts_add_bound=10) |
| depth-prune w_p | 3 (prune_range): thr / (1 + 2·(z−z_min)/(z_max−z_min)) | paper; densify_and_prune |
| SH degree | 3, +1 every 1000 | 3DGS |
| background | black | arguments white_background False |
| resolution | -r 4 (600×400) for real sets | configs |

## 4. Schedules
- iter 1–20000: blur render each step; SH up every 1000.
- iter > 500, %100, < 15000: clone/split (thr above) + depth-scaled opacity prune.
- iter 2500 (after that iteration's densification): add points. Implementation detail
  (gaussian_model.add_points): ALL Gaussians re-initialised — scales = log sqrt(mean 3-NN d²)
  over the new cloud, rotation identity, opacity 0.1; positions and SH of old Gaussians kept;
  new points get KNN-weighted SH (weights ∝ distance, normalised; only neighbours with
  distance < t_d; points with none are dropped); a fresh Adam is created (all moments, step
  counts and the MLP's moments reset). The optimizer step of iteration 2500 is skipped (grads
  belong to the replaced tensors).
- xyz lr decays over 30000 although training stops at 20000.

## 5. Builds on
Inria 3DGS (train/densify code), all unchanged except: blur render in training, no opacity
reset, no size pruning, depth-scaled prune, extra points, thresholds above.

## 6. Protocol
Deblur-NeRF real defocus (10 scenes) and real camera motion (10 scenes) sets, as re-released
by the authors (deblur_dataset.zip: same images, COLMAP re-run → denser points, all
SIMPLE_PINHOLE at 2400×1600). Images: full-res `images/` loaded and PIL-resized by −r 4 →
600×400. Split: cameras sorted by name, idx % hold == 0 → test (sharp GT), hold from
`hold=N`. Metrics on test after 20000 iterations: PSNR (torch, mean per image), SSIM
(skimage, multichannel, on [−1,1]), LPIPS (lpips package, AlexNet, CPU). Black background.
RTX 4090. Synthetic sets: resolution 1 and other lrs (not used here).

## 7. Ambiguities
1. Eq. 9/10 say min(1.0, ·) — code uses clamp(min=1, max=1.1) (max_clamp not in paper). Use code.
2. Paper M = 5; code renders num_moments + 1 = 5 (one unshifted). Use code.
3. Paper PE has π; code has none and includes the raw input. Use code.
4. Paper says "linearly interpolate neighbouring colours"; code weights ∝ distance (farther
   neighbours weigh more). Use code.
5. Paper "view direction v"; code feeds the camera centre. Use code.
6. q·δr on a normalised quaternion → unnormalised quaternion that the Inria rasterizer uses as
   is (non-orthogonal R). gsplat normalises quats, so the port builds Σ itself.
7. "total iteration 20000" while position lr decays over 30000 (code). Use code.
8. Depth for pruning = world z (forward-facing LLFF/COLMAP: cameras look along +z).
9. Re-run COLMAP data vs original Deblur-NeRF release: we use the authors' release.
