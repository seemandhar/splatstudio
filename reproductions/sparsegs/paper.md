# SparseGS: Sparse View Synthesis using 3D Gaussian Splatting — implementation spec

arXiv 2312.00206 (v3 LaTeX in `source/latex/`, main.tex + sec/*.tex + supplementary
`sec/X_supp_new.tex`). Official code: https://github.com/ForMyCat/SparseGS (cloned depth-1 to
`source/code/`, Inria gaussian-splatting fork + `submodules/diff-gaussian-rasterization-softmax`).
Authors' preprocessed data (Google Drive folder from the README): `MipNeRF-360 Dataset/{8,10,12,18,24,full}-view`
zips + BMD weights `latest_net_G.pth`, `res101.pth` — downloaded to `~/datasets/sparsegs/raw12/`
and `~/models/bmd/weights/`.

## 1. Method summary
- 3DGS trained from few (12/24) views with a monocular depth prior (BoostingMonocularDepth / LeReS
  in the code; the paper text says Marigold) enforced via Pearson correlation on a
  **softmax-scaled depth** render (custom rasterizer).
- Unseen-viewpoint regularisation (UVR): an SDS loss with Stable Diffusion 2.1-base on random
  views on an elliptic cylinder fitted to the training cameras (last third of training) and a
  depth-warping loss: training images forward-warped with the (range-rescaled) monocular depth to
  rotated pseudo cameras.
- Floater pruning: per training view, the relative difference between alpha-blended and mode
  (max-weight) depth; dip test of its histogram selects an adaptive percentile; Gaussians in front
  of / at the mode of the flagged pixels are removed.

Contributions:
- **C1** Mode-selection & softmax-scaling depth rendering (§3.1, Eq. 1-5; supp. §A gradients).
- **C2** Patch-based (local) + global Pearson depth-correlation loss (§3.2, Eq. 6).
- **C3** SDS loss from SD 2.1 on elliptic-cylinder views (§3.3.1, Eq. 7-8; supp. §B).
- **C4** Depth-warping loss on rotated pseudo views (§3.3.2, Eq. 9-10; supp. §B Eq. 12-14).
- **C5** Advanced floater pruning with the dip test (§3.4; supp. §C Algorithm 1).

## 2. Equations
Alpha-blended depth (Eq. 1): $d^{\text{alpha}}_{x,y} = \sum_{i=1}^N T_i \alpha_i d_i$
(code: normalised, `alpha_depth = AD/(AZ+1e-8)` with `AZ = sum w_i`, `AZ:=1` if 0).

Mode depth (Eq. 2): $d^{\text{mode}}_{x,y} = d_{\arg\max_i(w_i)}$, $w_i = T_i\alpha_i$.

Softmax depth (Eq. 3):
$d^{\text{softmax}}_{x,y} = \log\left(\frac{\sum_i w_i e^{\beta w_i} d_i}{\sum_i w_i e^{\beta w_i}}\right)$,
limits Eq. 4-5 ($\beta\to0$: log alpha depth; $\beta\to\infty$: log mode depth).
Code (forward.cu:377-415): `D += d*w*exp(beta*(w-0.3)); Z += w*exp(beta*(w-0.3))`,
`out = log(D/Z + 1e-8)`, `Z:=1` if 0 (the 0.3 shift cancels). Gradients (supp. §A, backward.cu
564-615): $\partial/\partial d_k = w_k e^{\beta w_k}/D$; $\partial/\partial\alpha_k =
T_k[((1+\beta w_k)e^{\beta w_k}d_k - A^l_k)/D - ((1+\beta w_k)e^{\beta w_k} - A^r_k)/Z]$ with the
back-to-front accumulators $A^l, A^r$ — i.e. the exact derivative of the forward.

Depth loss (Eq. 6): $\mathcal{L}_{\text{depth}} = \frac1N\sum_i^N 1 - \text{PCC}(p^{\text{softmax}}_i, p^{d^{\text{pt}}}_i)$
plus a global Pearson term. Code (`utils/loss_utils.py`):
`pearson(a,b) = 1 - mean(((a-mean a)/(std a+1e-6)) * ((b-mean b)/(std b+1e-6)))` (unbiased std);
local: `n = int(p_corr * floor(H/S) * floor(W/S))` patches of S×S with top-left
`randint(0, H-S)`, `randint(0, W-S)` (random, may overlap; paper says non-overlapping), mean of
per-patch Pearson losses. Rendered depth = softmax depth; target = monocular depth min-max
normalised to [0,1] after cv2 INTER_CUBIC resize.

SDS (Eq. 7-8): $\hat I = \mathcal N(\sqrt{\hat\alpha} I', (1-\hat\alpha)\mathbf I)$,
$\mathcal L_{SDS} = \nabla_G\mathbb E[(\epsilon_\phi(\hat I_p;\tilde I_p)-\epsilon)\partial\hat I_p/\partial G]$.
Code (`guidance/sd_utils.py train_step`): render → bilinear 512×512 → VAE encode (sample ×
scaling factor, grads on) → fixed $t=\text{round}((1-\text{step\_ratio})\cdot1000)$ clipped to
[20, 980] → noise → UNet with CFG scale 100 on empty prompts (neg and pos "") →
`grad = (1-alphas_cumprod[t]) * (eps_pred - eps)`, `target = (latents - grad).detach()`,
`loss = 0.5 * sum((latents - target)^2) / B`. fp16 SD.

Depth warping (Eq. 9-10): $p_j = K_{trg}T(K^{-1}_{src}Z_ip_i)$, $Z$ = monocular depth rescaled
linearly to $[\min d^{\text{alpha}}, \max d^{\text{alpha}}]$ of the source render;
$\mathcal L_{\text{Proj}} = M * \mathcal L_1(\hat I, I_{trg})$.
Code (`scene/cameras.py generate_warp_gt`): per-pixel forward splat with `int(round())`
(banker's rounding), last write wins in raster order, no z-test; mask M = written pixels;
K: `fx/r`, `cx=(W-1)/2`. Loss (`train.py`): `0.8 * sum(|I-Î|·M)/sum(M) + 0.2*(1-SSIM(I, Î))`
(SSIM unmasked, against the warped image with zero holes).

Warp cameras (supp. Eq. 12-14): avg up $\bar Y$ = normalised mean of the camera-to-world
rotations' 2nd column; Rodrigues rotation about $\bar Y$. Code
(`scene/__init__.py`, `generate_new_cam.py`): `R_c2w' = M(Ȳ, θ) R_c2w`, w2c translation `T`
unchanged (⇒ camera centre rotated about the world origin), `M` = `construct_rotation_matrix`
(quaternion with $b,c,d=-\bar Y\sin(θ/2)$), θ ∈ **{-2.5°, +2.5°}** (paper: {-3,-1.5,1.5,3}°).

Elliptic cylinder (supp. §B): PCA-align camera positions (`transform_poses_pca`), centre =
`focus_point_fn`, ellipse semi-axes = 90th percentile of |xy − centre|, z ∈ [10th, 90th]
percentile; code samples θ ~ U(0, 2π), z ~ U(z_low, z_high), look-at centre, up = the world axis
closest to the mean camera up (`setup_ellipse_sampling`, `getRandEllipsePose(idx, 0, 0)`).

Floater pruning (supp. Algorithm 1; code `train.py prune_floaters`, `utils/prune_utils.calc_diff`):
`diff = (d_alpha - d_mode) / (d_alpha + max(min d_alpha, mean d_alpha - 3 std d_alpha))`
(paper: $(d^{mode}-d^{alpha})/d^{alpha}$);
`D̄ = mean_i dipstat(diff_i[diff_i>0])`; `perc = max(80, prune_perc*100*exp(-prune_exp*D̄))`;
per view `τ = percentile(diff[diff>0], perc)`, flagged pixels `diff > τ` (pixels with no
contributor excluded); for each flagged pixel, all tile-list entries from `mode_id[0]` to
`mode_id[1]` (first contributor .. mode) whose `power > power_thresh (-4)` at that pixel are
removed (union over views). **Official bug:** forward.cu sets `startIdx/endIdx = range.x +
progress` with `progress = round*256 + thread_rank` (the fetch index of the thread, not the
Gaussian index `round*256 + j`), so the pruned range is shifted by the pixel's in-tile rank and
collapses to one entry when the first contributor and the mode lie in the same 256-batch.

## 3. Hyper-parameters
| name | paper | official code / README | chosen default |
|---|---|---|---|
| iterations (12 views) | 10k (§4.1) | README 30k (kitchen_12) | 10000 |
| β softmax | 5 (§4.1) | 5.0 (PipelineParams) | 5 |
| λ global Pearson | 0.15 (§4.1) | README 0.05 | 0.15 |
| λ local Pearson | 0.15 | README 0.15 | 0.15 |
| patch S (Mip-NeRF 360) | 64×64 (§4.1); "1/100 of resolution" (§3.2) | README box_p 128 | 64 |
| p_corr (fraction of patches) | – | 0.5 | 0.5 |
| λ warp | 0.05 | README lambda_reg 0.1 | 0.05 |
| warp start | – | warp_reg_start_itr 4999 (GT warped once at 4999, used from 5000) | 4999 |
| warp prob/iter | – | randint(1,10) <= 4 (0.4) | 0.4 |
| warp angles | ±1.5°, ±3° (4 views) | [-2.5, 2.5] (2 views) | [-2.5, 2.5] |
| λ SDS | 5e-4 | README 1e-3 | 5e-4 |
| SDS start | after 2/3 of iterations | `iteration > iterations*2/3` | 2/3 |
| SDS freq | – | SDS_freq 0.1 | 0.1 |
| SDS step_ratio | – | default 0.95, README 0.99 | 0.99 |
| CFG scale / t range / fp16 | – | 100 / [0.02, 0.98] / fp16, SD 2.1-base | same |
| prune a (perc) | 0.97 | prune_perc default 0.98 (README 0.98) | 0.97 |
| prune b (exp) | -7.5 | prune_exp default 7.0 (README 7.5) | 7.5 |
| prune min percentile | – | 80 | 80 |
| power_thresh | – | -4.0 | -4.0 |
| prune when | once after training completes | README `--prune_sched 20000` of 30k | 10000 = after training (paper). The README 20000-of-30000 belongs to a different 30k `-r 2` recipe (densification ends at 18000). Step 6667 exists only as the labelled TUNED variant `sparsegs-tuned6667` (chosen on a test-split smoke; review-1 major). |
| post-prune densify | – | densify_lag 1e6 (disabled) | disabled |
| resolution | 1/4 (§4.1) | README `-r 2` on full-res | 1/4 |
| 3DGS optim. | – | Inria defaults: pos lr 1.6e-4→1.6e-6 over 30k steps, densify 500..18000 every 100, grad 2e-4, reset 3000, SH +1/1000 | Inria defaults |
| background | – | black (`white_background` False) | black |

## 4. Schedules
- SH degree +1 every 1000 steps; densification from 500 to 18000 (> 10k, so throughout), opacity
  reset every 3000; position lr exponential with max_steps 30000 (so at 10k it is at 2.15e-5×extent).
- Pearson losses every iteration except warp iterations.
- Warp: GT generated once at step 4999 from the then-rendered alpha depth min/max; used w.p. 0.4
  per iteration for step ≥ 5000; on warp iterations the densification statistics come from the
  warp render only (train.py "Densification for warp Cam").
- SDS: w.p. 0.1 per iteration for step > 2/3·total.
- Pruning: once (paper: after training completes; code: `prune_sched`).

## 5. What it builds on
Inria 3DGS (photometric 0.8 L1 + 0.2 D-SSIM, adaptive density, opacity reset). Unchanged:
optimisers, learning rates, densification thresholds, SH schedule.

## 6. Protocol
- Mip-NeRF 360, 12 views, "seven scenes"; paper: every 8th image test, 12 views evenly sampled from
  the rest, 1/4 resolution, COLMAP MVS point cloud from the training views (the "w/o MVS" rows use
  SfM points). The released 12-view zips cover **6 scenes** (bicycle, bonsai, counter, garden,
  kitchen, stump; no room) and their training images are *not* llffhold8-compatible (contiguous
  blocks for garden/bicycle/stump, several are every-8th images). The official evaluation renders
  the full scene and averages over **all non-training images** (`metrics.py --exclude_path`).
- Metrics: PSNR, SSIM (Inria SSIM), LPIPS-VGG (lpipsPyTorch).
- LLFF/DTU: RegNeRF protocol (3 views); table values only (no per-scene).
- Only averages are reported (Table 1: 12/24-view 360; Table 2: LLFF/DTU; Table 3 ablation 360
  12-view). No per-scene numbers in the paper, supplementary or README.

## 7. Ambiguities and choices
1. Depth prior: paper says Marigold; code integrates BoostingMonocularDepth (LeReS + merge). We use
   BMD (code). Its output is depth-like (verified: corr(BMD, SfM z) = +0.81 on garden DSC07956).
2. Paper (10k, 0.15/0.15, patch 64, warp 0.05, SDS 5e-4, a=0.97,b=7.5, prune at end) vs README
   (30k, 0.05/0.15, patch 128, warp 0.1, SDS 1e-3, 0.98/7.5, prune at 20k, -r 2): we follow the
   paper where it states a value (its experiments section describes the reported setting) and the
   code/README otherwise.
3. Split: authors' released training images; test = llffhold8 minus training (paper text); the
   official metric set (all non-training) = test ∪ val and is reported alongside.
4. Warp angles: code ±2.5° (2 views) vs paper 4 views — code.
5. Pruning index bug (above): implemented both (`mode_range: official|paper`), default official.
6. Local patches: code samples random, possibly overlapping patches (paper: non-overlapping) — code.
7. Diff sign/normaliser of the pruning statistic — code.


## Resolution ambiguity (review-1 minor)
The paper says 1/4 resolution. The released zips are full resolution, the README trains with `-r 2`, and its render command caps width at 1600 px. Our dataset uses full-res PIL /4 for all scenes (paper text); this is recorded as a fidelity gap.
