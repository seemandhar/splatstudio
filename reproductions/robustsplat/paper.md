# RobustSplat — implementation spec

Paper: *RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS*, Fu et al.,
ICCV 2025, arXiv 2506.02751 (LaTeX source in `source/latex`). Official code:
github.com/fcyycf/RobustSplat @ a130281 (2025-09-16), `source/code` (3DGS-Inria codebase,
non-commercial Inria licence; read only, never executed). **Code values override the paper where
they differ; both are recorded.**

## 1. Method summary
- 3DGS (Inria, 30k iterations, standard LRs) + a per-pixel transient mask predicted by a tiny MLP
  on frozen DINOv2 ViT-S/14-reg features of the *training* image; the mask (thresholded, eroded)
  gates the L1 + D-SSIM photometric loss.
- The MLP is supervised (separately from the Gaussians) by (a) SpotLessSplats' residual hinge
  between two robust residual masks, (b) a DINOv2 feature cosine-similarity map between GT and
  rendered image, (c) an early "everything is static" regulariser that decays as exp(-i/2000).
- **C1 Delayed Gaussian growth** (§4.3): densification starts at 10k (3DGS: 500) and runs to
  20k (3DGS: 15k); opacity reset only from 15k (i.e. 15k, 18k); MLP training paused for 300
  iterations after each reset.
- **C2 Scale-cascaded mask bootstrapping** (§4.4): during early training the MLP supervision
  (residual masks *and* feature similarity) is computed at low resolution (residual: an extra
  ×4-downsampled render/GT pair; features: 16×16 patch grid = 224² input) and later at high
  resolution (residual at training res; features 36×36 = 504² input).
- C3 (adopted, §4.2) mask MLP with residual + feature-cosine supervision and C4 early mask
  regularisation (Eq. 8, §4.3).

## 2. Equations (paper numbering by order; `i` = iteration)
- Eq. 2  $\mathcal L = (1-\lambda)\mathcal L_1 + \lambda \mathcal L_{D\text{-}SSIM}$, λ = 0.2.
- Eq. 3  $M_t = \mathrm{Sigmoid}(\mathrm{MLP}_{mask}(f_t))$, $f_t$ = DINOv2 patch features of
  training image t, bilinearly upsampled to the image resolution.
- Eq. 4  $M_{cos} = \max(2\cos(f_t, f'_t) - 1, 0)$, $f'_t$ = DINOv2 features of the rendering.
- Eq. 5  $\mathcal L_{cos} = \| M_t - M_{cos} \|$ (code: mean absolute difference).
- Eq. 6  $\mathcal L_{MLP} = \lambda_{res}\mathcal L_{residual} + \lambda_{cos}\mathcal L_{cos}$.
- Eq. 7  $\mathcal L_{reg} = e^{-i/\beta_{reg}} \| 1 - M_t \|$ (code: mean).
- Eq. 8  $\mathcal L_{MLP} = \lambda_{res}\mathcal L_{residual} + \lambda_{cos}\mathcal L_{cos} + \lambda_{reg}\mathcal L_{reg}$.
- Supp. Eq. S1  $\mathcal L_{residual} = \max(U - M, 0) + \max(M - L, 0)$ with τ_u = 0.6,
  τ_l = 0.8 (as written; the code is the SpotLessSplats hinge
  `mean(relu(M - upper) + relu(lower - M))`, with `lower` = mask at the 0.6 residual quantile,
  `upper` = mask at the 0.8 quantile — i.e. the MLP output is kept between the two masks).

### Code details (`utils/mask_utils.py`, `train.py`)
- Robust residual mask (SpotLessSplats `robust_mask`): r = mean_c |gt - render| (detached);
  thresholds = residual quantiles from a 10000-bin histogram on [0,1] with decay
  `hist ← 0.95·hist + hist_t` (left bin edge `linspace(0,1,10001)[k]`); inlier = r < thr;
  mask = inlier OR (3×3 box-mean(inlier) > 0.5), zero padding. One shared histogram over both
  scales (coarse and fine residuals feed the same running histogram).
- Masks are bilinearly interpolated to the training resolution before the hinge.
- Cosine map: `cos = cosine_similarity(f_gt, f_render, dim=channel)` at feature grid, map
  `c = clip(1 - (cos - 0.5)/0.5, 0, 1)`, bilinear upsample, `M_cos = 1 - c` (= Eq. 4).
- DINOv2: `torch.hub dinov2_vits14_reg` (ViT-S/14 + 4 registers, same weights as
  `dinov2_vits14_reg4_pretrain.pth`), input = image (no ImageNet normalisation!) bilinearly
  resized (align_corners False) to (14·s, 14·s) square, `forward_features()['x_norm_patchtokens']`
  (last block, final LayerNorm), s = 36 (fine) or 16 (coarse). GT features precomputed once per
  training image at both s; the rendering's features recomputed every iteration from the detached
  **full-resolution** render.
- MLP: Linear(384,16)–ReLU–Linear(16,1)–Sigmoid per pixel on bilinearly upsampled fine (s=36)
  features (input is always the fine features, also during bootstrapping). Adam lr 1e-3.
  `get_regularizer` = 0.5·max|W1|·max|W2| on `.data` → constant, no gradient (no effect).
- Photometric loss for i ≥ mask_beginning (500): `m = erode7x7(M_t.detach() > 0.25)`
  (−maxpool(−x), k=7, pad 3); `L1 = mean(m·|I - I_gt|)`, `Lssim = mean(1 - ssim_map(m·I, m·I_gt))`,
  `loss = 0.8 L1 + 0.2 Lssim`. Before 500: plain L1 + 0.2 D-SSIM, no MLP training.
- MLP loss: `0.5·mean|M_t - M_cos| + 0.5·hinge + 2.0·mean((1 - M_t)·exp(-i/2000))` (+ constant
  regulariser), stepped once per iteration, skipped for reset_start < i < reset_start + 300 when
  i ≥ 15000 (reset_start = 3000·⌊i/3000⌋).
- Bootstrapping switch: coarse supervision for i < `bootstrap_iter` = **20000** (code). Paper
  text: "before the start of Gaussian densification … low-resolution; once densification begins,
  high-resolution" (= 10000). Code default 20000 is used; recorded as ambiguity.
- Coarse render: same camera rendered at the size of the `images_{4·factor}` image (mediapy
  downsampled from full-res); coarse GT = that image.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | §5 / code |
| LRs (xyz 1.6e-4→1.6e-6 ×extent, f_dc 2.5e-3, f_rest 1.25e-4, opacity 2.5e-2, scale 5e-3, rot 1e-3) | 3DGS | code arguments |
| λ_dssim | 0.2 | code |
| densify_from_iter | 10000 | §5 "delayed iteration start 10K", code |
| densify_until_iter | 20000 | code (3DGS 15000) |
| densification_interval | 100 | code |
| densify_grad_threshold | 2e-4 | code |
| percent_dense | 0.01 | code |
| prune min opacity | 0.005; no screen/world-size prune (max_screen_size=None) | code train.py |
| opacity reset | every 3000 for i ≥ reset_iter 15000, i < 20000 → 15000, 18000; reset to min(σ, 0.01) | supp. "Training details", code |
| mask_beginning | 500 | code |
| upper/lower feat res | 36 (504²) / 16 (224²) | §5, code |
| bootstrap_iter | 20000 | code (paper text implies 10000) |
| λ_residual, λ_cos, λ_reg | 0.5, 0.5, 2.0 | §5, code |
| β_reg | 2000 | §5, code |
| τ (residual quantiles) | lower 0.6, upper 0.8 | supp., code |
| histogram | 10000 bins on [0,1], decay 0.95 | code (SpotLessSplats) |
| mask threshold / erosion | > 0.25, 7×7 min-pool | code; supp. ("dilation with kernel 7") |
| MLP | 384–16–1, ReLU, Sigmoid, Adam 1e-3 | §5, code |
| MLP pause after reset | 300 iterations | code |
| coarse residual scale | 1/4 of training res | §5 |
| background | black | code (white_background False) |
| SH | degree +1 every 1000 | 3DGS |

## 4. Schedules
- i < 500: plain 3DGS loss, no mask, no MLP.
- 500 ≤ i: masked loss; MLP trained every iteration (except reset pauses).
- 500 ≤ i < 20000: coarse supervision (residual at 1/4 res, features at 16²); ≥ 20000 fine.
- 10000 < i < 20000, i % 100 == 0: densify (clone/split) + opacity prune. Stats accumulated from
  i = 1 (first refine averages 10100 iterations of gradients).
- i ∈ {15000, 18000}: opacity reset; MLP paused for i ∈ (15000, 15300), (18000, 18300),
  also (21000, 21300), (24000, 24300), (27000, 27300) (the pause condition only checks
  i ≥ 15000, not the reset schedule — code quirk, reproduced).
- exp(-i/2000) regulariser weight decays.

## 5. Builds on
Inria 3DGS (2024 codebase with exposure/depth options unused: no depth maps, train_test_exp
False). Kept unchanged: rasteriser, LRs, SH schedule, clone/split numerics, init from SfM.
Changes: densification window, opacity-reset window, no size pruning, masked loss, mask MLP.

## 6. Protocol
- Datasets: NeRF On-the-go (6 scenes, Table 1) and RobustNeRF (4 scenes, Table 2), per-scene
  PSNR/SSIM/LPIPS(VGG, `lpipsPyTorch`) reported. Also On-the-go II in the supplementary.
- Data: authors' re-run of COLMAP (OPENCV, then `image_undistorter` → PINHOLE) on full-res
  images, released at hf.co/datasets/fcy99/RobustSplat-data; downsample factor 8 (factor 4 for
  patio and arcdetriomphe), images pre-downsampled with mediapy (`images_<f>`, `images_<4f>`).
- Split: train = names with `clutter`, test = names with `extra` (`clean` images unused).
- 3DGS camera: FoV from COLMAP focal; principal point at the image centre (Inria projection).
- Metrics: PSNR, SSIM (Inria Gaussian 11×11 same-padding), LPIPS-VGG. Single run (seed not stated).

## 7. Ambiguities / paper-vs-code
1. Bootstrapping switch: paper "at densification start" (10k) vs code 20000 → code.
2. Supp. Eq. S1 hinge with U/L naming reversed vs τ values; code hinge keeps M between the
   0.6-quantile mask (lower) and 0.8-quantile mask (upper) → code.
3. Paper says "dilation with kernel 7"; code erodes the *static* mask (= dilates the transient
   region) → identical meaning.
4. Paper says "residual further downsampled by an additional factor of 4"; code renders the
   coarse view at images_{4f} size → code.
5. `mask_loss` regulariser `get_regularizer` uses `.data` → no gradient; effectively absent.
6. Masked SSIM: SSIM of masked images (zeros in masked regions on both), not a masked SSIM.
7. LPIPS: paper tables use the official metrics.py (VGG).
