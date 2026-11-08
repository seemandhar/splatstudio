# FSGS — implementation spec

Paper: *FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting*, Zhu, Fan, Jiang, Wang,
ECCV 2024, arXiv 2312.00451. Official code: github.com/VITA-Group/FSGS (cloned read-only into
`source/code/`, Inria Gaussian-Splatting license = non-commercial research use, port with citation).
LaTeX in `source/latex/` (main.tex, sec/3_method.tex, sec/4_exp.tex, rebuttal.tex).

## 1. Summary and contributions
- 3DGS from the SfM/MVS cloud of the sparse training views only; 10k iterations.
- **C1 Proximity-guided Gaussian unpooling** (§3.2): K=3 nearest-neighbour graph over Gaussian
  centres; Gaussians with a large proximity score grow a new Gaussian at the midpoint of each of
  their 3 edges (scale/opacity copied from the destination, rotation identity, SH zero).
- **C2 Pseudo-view augmentation** (§3.3): random virtual cameras near the training cameras,
  rendered online; their DPT depth supervises the rendered depth.
- **C3 Monocular-depth geometry guidance** (§3.3): Pearson-correlation loss between the
  alpha-blended rendered depth and DPT depth, on training views and pseudo views.
- (C4, enabling) differentiable accumulated depth rasterization d = Σ d_i α_i T_i.

## 2. Equations (paper numbering) and official code
- Eq. 3/4: D_i^K = K-min_j d_ij, d_ij = ||μ_i − μ_j||; P_i = (1/K) Σ_j D_i^K, K = 3.
  **Code** (`gaussian_model.py:proximity`, `simple-knn` modified to return the 3 indices):
  `dist` = mean **squared** distance to the 3 NN (self excluded, exact);
  mask = `dist > 5·extent` AND `max(scale) > extent`; for each selected source i and each of
  its 3 neighbours j: μ_new = (μ_i + μ_j)/2, log-scale_new = log-scale_j, opacity-logit_new =
  logit_j, quat_new = (1,0,0,0), f_dc = f_rest = 0. Run inside `densify_and_prune` after clone
  and split, **only while iter < 2000**.
  Also in `densify_and_split`: split mask |= (`dist > dist_thres·extent` AND `max scale >
  extent`), dist_thres = 10 (= paper's t_prox = 10).
- Eq. 5 pseudo pose P' = (t + ε, q), ε ~ N(0, δ). **Code** uses RegNeRF's
  `generate_random_poses_llff` (`utils/pose_utils.py:207`; `scene/__init__.py` calls it for every
  scene because `str.find` returns −1/index, both truthy): 10 000 poses, positions uniform in the
  box of the recentred training positions (radii = max |pos|), looking at the focus point
  `focal = 1/((1−0.75)/(0.9·near) + 0.75/(5·far))` (bounds = poses_bounds.npy of the training
  views, scaled by 1/(0.75·near_min)); FoV of training view 0. Sampled by popping a shuffled stack.
- Eq. 6: Corr(D_ras, D_est) = Cov / sqrt(Var·Var) (torchmetrics `pearson_corrcoef`, clamped to [−1,1]).
- Eq. 7: d = Σ_i d_i α_i Π_{j<i}(1−α_j) (not normalised by alpha; rasterizer `forward.cu:361`).
- Eq. 8: L = 0.8 L1 + 0.2 D-SSIM + λ3 L_reg. **Code** (`train.py`):
  - training view: `L_d = min(1 − corr(−D_dpt, D_ras), 1 − corr(1/(D_dpt+200), D_ras))`,
    weight `depth_weight = 0.05`, set to 0.001 after iteration 9500 (applies from 9502).
    D_dpt = DPT(gt) computed once at load (`estimate_depth`, no grad).
  - pseudo view: every `sample_pseudo_interval` iterations with 2000 < it < 9500:
    `L_p = 1 − corr(D_ras_pseudo, −DPT(render_pseudo))` with **gradient through DPT** into the
    rendered image (`mode='train'`; DPT weights frozen), weight
    `min((it−2000)/500, 1) · depth_pseudo_weight`, skipped if NaN.
  - DPT input: `(img − 0.5)/0.5`, bicubic to 384×512, output bicubic back to H×W.
    Model: `torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")` (dpt_hybrid_384.pt).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 10 000 | §4 Impl.; `arguments/__init__.py` |
| position lr | 1.6e-4 → 1.6e-6 (×extent), max_steps 10 000 | arguments |
| feature / opacity / scaling / rotation lr | 2.5e-3 (rest /20) / 0.05 / 5e-3 / 1e-3 | arguments |
| λ_dssim | 0.2 | arguments; Eq. 8 |
| densify interval / from / until | 100 / >500 / 10 000 | arguments |
| densify grad threshold | 5e-4 | arguments |
| prune opacity threshold | 0.005 (prune only after iter 500) | arguments |
| size pruning | none (`size_threshold=None`) | train.py |
| percent_dense | 0.01 | arguments |
| dist_thres (split) / proximity (unpool) | 10·extent / 5·extent, max scale > extent | arguments; gaussian_model.py |
| unpooling active | iter < 2000 | gaussian_model.py:densify_and_prune |
| opacity reset | min(o, 0.05) at it>2000, (it−2001) % 3000 == 0 → 2001, 5001, 8001 | train.py |
| SH degree | +1 every 500 iterations (max 3) | train.py |
| depth_weight | 0.05 (0.001 after 9500) | arguments; paper λ3=0.05 |
| depth_pseudo_weight | 0.5 (LLFF; README uses 0.03 for Mip-NeRF 360) | arguments, README |
| start/end_sample_pseudo | 2000 / 9500 | arguments; paper "after 2000" |
| sample_pseudo_interval | 1 (LLFF README command; default 10) | README |
| background | black | arguments (`white_background=False`) |
| init opacity / scale | 0.1 / sqrt(mean 3-NN sq. dist) | gaussian_model.py |
| DPT | DPT_Hybrid (paper Table 5 says DPT Large for the main row) | depth_utils.py |

## 4. Schedules
SH up at 500/1000/1500; densify 600..9900 every 100; unpool 600..1900; opacity resets 2001,
5001, 8001; pseudo views 2001..9499 with linear ramp to 2500; train-depth weight 0.05 → 0.001 at 9502.

## 5. Builds on
Inria 3DGS (rasterizer extended with depth/alpha outputs). Unchanged: L1+D-SSIM, Adam, clone/split
criteria, exponential position LR. Changed: no size pruning, 5e-4 threshold, reset to 0.05,
10k steps, SH every 500.

## 6. Protocol (LLFF, the chosen benchmark)
8 scenes, test = every 8th sorted image, train = 3 views `round(np.linspace(0, n_train−1, 3))`
of the rest; images `images_8` (504×378) from the LLFF release; init = COLMAP
`point_triangulator` + `patch_match_stereo` + `stereo_fusion` on the 3 training views
(`3_views/dense/fused.ply`, `tools/colmap_llff.py`). Metrics: PSNR, SSIM (Inria `ssim`), LPIPS-VGG
(lpipsPyTorch) on test views, black background, renders clamped for evaluation. Results are averages
over 8 scenes (no per-scene tables anywhere in the paper/supp./rebuttal). Mip-NeRF 360: 24 views at
1/8 and 1/4 res (all 9 scenes, averages only). Blender: 8 views, 400×400.

## 7. Ambiguities (code choice)
1. Proximity threshold: paper t_prox = 10 on mean distance; code uses mean *squared* distance,
   5·extent for unpooling (and 10·extent for the extra split rule) and also requires max scale > extent.
2. Pseudo views: paper describes interpolation between the two closest training cameras plus
   noise; code samples RegNeRF random poses. Code followed.
3. Paper: depth loss on train and pseudo views with λ3 = 0.05; code: 0.05 (train), 0.5 (pseudo, ramped).
4. Unpooled SH: paper "SH initialized to zero" → code zeroes f_dc too (black, colour 0.5 after SH→RGB offset).
5. Depth estimator DPT Large (Table 5) vs DPT_Hybrid in code → Hybrid.
6. `sample_pseudo_interval` default 10, README LLFF command uses 1 → 1.
