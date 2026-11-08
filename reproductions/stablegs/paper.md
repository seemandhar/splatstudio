# StableGS: A Floater-Free Framework for 3D Gaussian Splatting — implementation spec

arXiv 2503.18458 (v3, AAAI-2026 LaTeX `source/latex/aaai2026.tex`). Official code:
https://github.com/MooreThreads/StableGS @ 397a9c00 (2026-01-06, "init") in `source/code/`.
**The release is incomplete**: `gaussian_renderer/` and `arguments/` are missing (GitHub issue #2),
so `render_rade`, `render_taming` and the argument defaults must be reconstructed. The shipped
`submodules/diff-gaussian-rasterization_radegs` is byte-identical (except `.data<>`→`.data_ptr<>`,
`W/2.`→`W/2.f` and an extra MCMC relocation kernel) to the RaDe-GS rasterizer at
HKUST-SAIL/RaDe-GS commit ec9199f (2024-09-10) → `render_rade` = RaDe-GS@ec9199f
`gaussian_renderer.render` (cloned in `source/radegs_main/repo`) with the opacity swapped for the
dual opacity. `render_taming` / `utils/taming_utils.py` / the densifier are Taming-3DGS
(github.com/humansensinglab/taming-3dgs, already ported in `reproductions/taming-3dgs`).

## 1. Method summary
- Diagnosis (Sec. 3.2): floaters sit in a pseudo-equilibrium where Σ sign(c_f − c_gt) ≈ 0, so the
  L1 colour gradient of their opacity vanishes; depth-based losses give an independent signal.
- **C1 Dual Opacity GS** (Sec. 3.3): extra per-Gaussian auxiliary opacity α'. Geometric path
  renders with σ (α only), appearance path with σ' = α'·σ. Depth losses act only on the geometric
  path, the final image is the appearance path. Colour loss on both paths.
- **C2 Self-supervised depth consistency** (Sec. 3.4): warp the geometric depth of view i into
  view j over selected pairs P (covisible > 30 COLMAP points, relative rotation 16°–60°,
  homography-inlier fraction < 0.8).
- **C3 External DUSt3R prior with global scale optimization** (Sec. 3.5): pairwise DUSt3R
  pointmaps, 7-"equivalent-point" scale alignment, confidence-weighted L1 prior on D^s.
- Training pipeline: RaDe-GS depth/normal rasterization + Taming-3DGS (budgeted score-based
  densification, "final_count" budgets = 3DGS counts, batched SH updates).

## 2. Equations (paper, verbatim)
- Eq. 1  σ_k(x) = α_k exp(−½ (x−μ_k)^T Σ_k^{-1} (x−μ_k)); rendering opacity σ'_k(x) = α'_k σ_k(x).
- Eq. 2  L_3DGS = (1−λ) L_1(C̄, C^gt) + λ L_D-SSIM(C̄, C^gt).
- Eq. 3  L_total = L_color + λ_geo L_geometry;
  L_color = L_3DGS(C̄^o, C^gt) + λ_s L_1(C̄^s, C^gt).
- Eq. 4  L_consis = Σ_{(i,j)∈P} ( ‖W_{i→j}(D̄^s_i) − D̄^s_j‖_1 + ‖W_{j→i}(D̄^s_j) − D̄^s_i‖_1 ),
  over validly projected regions.
- Eq. 5–6 (prior) L_pair-eff = ‖w_ij ⊙ (s_ij T_i x_ij − s_ji T_j y_ij)‖², L_intra, min_S Σ(...).
- Eq. 7  L_prior = Σ_{(i,j)} (D(i,j)+D(j,i)), D(i,j) = 1/‖d_ij‖_0 Σ C^{1,1}_ij ⊙ |D̄^s_i − d_ij|.
- L_geometry = λ_consis L_consis + λ_prior L_prior.

## 2b. What the official code actually does (train_pair.py; code overrides paper)
Per iteration a **pair** (v0, v1) is popped from a stack (= all selected pairs + an equal number
of random pairs whose cameras are drawn with p ∝ 1/(#pairs containing the camera)), both views
are rendered:
- `dual = iteration > 15000`. Appearance path: opacity α·α' if dual else α (render_rade,
  require_depth False).
- Colour loss: (1−0.2)(L1_0 + L1_1) + 0.2 (2 − SSIM_0 − SSIM_0)  ← **SSIM of view 0 counted twice,
  view 1's SSIM never used** (code bug, reproduced; fused-ssim = same-padded SSIM).
- From 15001 (`--dual --single_loss`): geometric path renders of both views (α only,
  require_depth), loss += L1(C^s_0) + L1(C^s_1)  (λ_s = 1).
- `--lambda_depth_normal 0.05` (**not in the paper**): RaDe-GS depth-normal consistency on the
  geometric path, summed over both views:
  E = (1 − n_0·N(D_0)) + (1 − n_1·N(D_1)) for N from (expected, median) depth; loss += 0.05 ·
  (0.4 mean E_exp + 0.6 mean E_med).
- `--lambda_depth_pair 0.05` (= λ_consis) only if the pair is in P:
  X_k = D_k(u) K_k^{-1}(u) (camera points, centred pixel grid), project X_1 into view 0 and X_0
  into view 1 (normalized coords 2u/W), `depth0_1 = grid_sample(D_0, proj(X_1→0))`,
  `depth1_0 = grid_sample(D_1, proj(X_0→1))` (bilinear, zeros padding, align_corners False);
  **L = mean(relu(depth1_0 − D_0)) + mean(relu(depth0_1 − D_1))** — one-sided, compares the
  sampled other-view z with the own-view z (no frame change of z), mean over all pixels.
  Differs from the paper's two-sided L1 with proper warp; both forms implemented, code default.
- `--lambda_depth 0.005` (= λ_prior, full StableGS only): DUSt3R prior, pairs_uni.pth from mvs.py.
- Opacity: sigmoid; `ho_iteration 15000` `modify_functions()` (inside no_grad) replaces `_opacity`
  by a plain tensor → **α frozen from 15000 on** (as in Taming); α' = sigmoid(`_opacity1`), init
  0.99, lr = opacity_lr, copied on clone/split, never reset, not used for pruning.
- Densification: Taming `densify_with_score` with budget `--budget <3DGS count>` (`final_count`),
  10 score cameras, stats from both views every iteration, **plus GOF qualifier** (unless
  `--nogof`, used for Deep Blending): qualifies if ‖grad‖ ≥ 2e-4 OR abs-grad ≥ quantile(abs, 1 −
  frac(‖grad‖ ≥ 2e-4)). Min opacity 0.005; guided prune for calls < 27; reset every 3000 < 15000.
- `--sh_lower`: SH-rest Adam steps every 16 iterations (summed grads).
- Eval: appearance path (dual α·α'), test split `--eval` (llffhold 8).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §4.1, code |
| λ_dssim | 0.2 | 3DGS |
| λ_s (single_loss) | 1.0 (L1 only) | code train_pair.py:138 |
| λ_consis (lambda_depth_pair) | 0.05 | paper §4.1, run.sh |
| λ_prior (lambda_depth) | 0.005 | paper §4.1, run.sh |
| λ_depth_normal | 0.05, depth_ratio 0.6 | run.sh, train_pair.py:149 (not in paper) |
| dual / geometric start | iteration > 15000 | train_pair.py:131 |
| ho_iteration (α freeze) | 15000 | train_pair.py arg default |
| α' init | 0.99 (sigmoid) | gaussian_model.py:195 |
| lrs | xyz 1.6e-4→1.6e-6 ×extent, f_dc 0.0025, f_rest 0.005/20, opacity & opacity1 0.025, scale 0.005, rot 0.001 | Taming defaults (arguments/ missing; f_rest /20 in gaussian_model.py:212) |
| densify | from 500, until 15000, every 100, grad 2e-4, percent_dense 0.01, min opacity 0.005 | Taming/3DGS |
| budgets (final_count) | truck 2584171, train 1085480, garden 5728191 … | run.sh |
| score cams / coefficients | 10 / Taming train.py coefficients | train_pair.py |
| kernel_size (2D filter) | 0.0 (assumed; RaDe-GS default, StableGS model has no 3D filter) | ambiguity |
| background | black (white_background False) | 3DGS default |
| pair criteria | covis > 30, 16° ≤ rot ≤ 60°, H-inlier < 0.8 | paper §3.4 (code uses BoQ retrieval, see §7) |

## 4. Schedules
1–15000: two-view colour loss, α sigmoid, Taming densification (500<it<15000, /100), opacity
reset at 3000·k < 15000, SH degree +1 every 1000. 15001–30000: dual opacity appearance path,
geometric path losses (L1, depth-normal, depth pair), α frozen.

## 5. Builds on
Taming-3DGS pipeline (densification, budgets, sh_lower, high-opacity switch) + RaDe-GS rasterizer
(expected/median depth, normals) ; otherwise 3DGS (init from SfM, LR schedules, losses).

## 6. Protocol
Mip-NeRF 360 (indoor/outdoor groups; paper text says longest edge 1600 px, run.sh uses images_2
for all 9 scenes), Tanks&Temples (truck, train; `images`), Blender. 30k iterations, A100, PSNR /
SSIM / LPIPS (VGG). Table 1 reports **group averages only** (no per-scene numbers, no supplement).
Deep Blending is listed in §4.1 but has no row in Table 1.

## 7. Ambiguities / reconstruction choices
1. `render_rade`/`render_taming`/`arguments` missing → RaDe-GS@ec9199f render with
   opacity = α·α' (dual) or α; kernel_size 0; Taming lrs.
2. Pair selection: paper = COLMAP covisibility/rotation/homography; code (mvs.py) = BoQ
   global-descriptor retrieval + rotation in [min(16°, 2nd-largest per camera), 60°) + |t_z|/|t| <
   0.9, pairs_uni.pth also needed for the w/o-prior runs. We implement the paper criteria
   (BoQ/DINOv2 weights not available offline); homography RANSAC threshold 4 px = COLMAP default
   (the 0.8 ratio is COLMAP's `max_H_inlier_ratio`).
3. Depth pair loss form (paper vs code, §2b).
4. Duplicated SSIM_0 (code bug) reproduced.
5. Paper resolution "longest edge 1600" vs run.sh images_2 (Mip-NeRF 360 only; irrelevant for T&T).
6. Dense TSDF initialization (mesh.ply from mvs.py) is not loaded by the released dataset reader
   (sparse/0/points3D only) → SfM init.
