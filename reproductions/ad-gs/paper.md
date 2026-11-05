# AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting — implementation spec

- arXiv: 2509.11003 (SIGGRAPH Asia 2025). Authors: Patle, Girgaonkar, Somraj, Soundararajan (IISc).
- Official code: https://github.com/gurutvapatle/AD-GS (cloned to `source/code/`, commit a8d12fc).
  Built on CoR-GS (jiaw-z/CoR-GS), itself built on FSGS / 3DGS. `train.py` (refactored, flags) and
  `train_old.py` (hard-coded) implement the same schedule; the run scripts `scripts/run_*_ad.sh`
  give the paper defaults.
- Sources: `source/latex/main.tex` (main + supplementary), `source/paper.pdf`.

## 1. Method summary
- Two 3DGS fields G1, G2 are trained concurrently from the same sparse-view point cloud (CoR-GS);
  only G1 is evaluated.
- Warm-up (first N_W = 1500 iterations): plain 3DGS photometric training with
  densification (τ = 5e-4, ε_α = 0.005).
- Afterwards training alternates in 200-iteration cycles: **high-densification phase** (N_H = 100
  iters, one densification with the loose τ_pos = 2e-4, ε_α = 0.005, photometric loss only) and
  **low-densification phase** (N_L = 100 iters, one densification with strict τ_L = 5e-4 and
  elevated opacity prune threshold ε_L = 0.1, plus geometry regularisation: pseudo-view
  consistency between G1 and G2 and edge-aware depth smoothness + depth-range term).

Contributions
- **C1 Alternating densification** (§3.3 "Alternate Low and High Densification", supp. Table 7):
  phase-dependent (τ, ε_α) per densification step; warm-up / high / low schedule.
- **C2 Alternating (phase-gated) losses** (§3.3.1, Eq. 9): regularisers only in the low phase after
  warm-up.
- **C3 Pseudo-view consistency** (§3.3.1(a), Eq. 6) between two co-trained fields on random
  perturbed (pseudo) views (taken from CoR-GS, gated to the low phase).
- **C4 Edge-aware depth smoothness + depth range** (§3.3.1(b), Eq. 7–8).

## 2. Equations (verbatim from main.tex)
Photometric (Eq. 5): `L_{ph}(v_k, v_{gt}) = (1 - \lambda_{ssim}) \lVert v_k - v_{gt} \rVert_1 + \lambda_{ssim} (1 - SSIM(v_k, v_{gt}))`, λ_ssim = 0.2.

Pseudo-view consistency (Eq. 6): `L_{pseudo}(u_1,u_2) = L_{ph}(u_1, u_2)`; code: target is the
other field's render **detached** (`image_pseudo_co_gs{j}.clone().detach()`), each field receives
its own term (division by gaussiansN-1 = 1).

Edge-aware depth smoothness (Eq. 7):
`L_{ds}(d, v) = \sum_{x,y} \lVert \nabla d(x,y) \rVert_1 \exp(-\lVert \nabla v(x,y) \rVert_1) - \lambda_r (d_{max} - d_{min})`
with d the alpha-composited depth `Σ z_i α_i T_i` (not normalised by accumulated alpha).
Code (`train.py:edge_aware_smoothness`): forward differences in x and y;
`w_x = exp(-mean_c |∇_x I|)` (mean over colour channels, I = **ground-truth** training image);
loss = `mean(w_x |∇_x d|) + mean(w_y |∇_y d|)` (means, not sums); range term
`-(d.max() - d.min())`; combined `depth_smooth_weight * (L_smooth + depth_range_weight * range)`.

Eq. 8: `L_{tds} = \omega_1 L_{ds}(v_k,d_k) + \omega_2 L_{ds}(u_k,d_{uk})` with ω1 = 0.01,
ω2 = 0.05 (supp.). **Code applies only the train-view term** (ω1 = 0.01); the pseudo-view term is
commented out in both `train.py` and `train_old.py` → effective ω2 = 0.

Eq. 9 (low phase): `L_k = \lambda_1 L_{ph}(v_k,v_{gt}) + \lambda_2 L_{tds} + \lambda_3 L_{pseudo}(u_1,u_2)`,
λ1 = λ2 = λ3 = 1. High phase and warm-up: `L_k = L_ph`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 10000 | supp. Table 7; run_llff_ad.sh |
| N_W warm-up | 1500 | Table 7; `--alt_warmup_iters` |
| cycle N_H + N_L | 200, N_H = 100 | Table 7; `in_low_dens_phase = it % 200 >= 100` |
| warm-up τ / ε | 5e-4 / 0.005 | code `warmup_*` |
| low τ_L / ε_L | 5e-4 / 0.1 | Table 7 |
| high τ_pos / ε_α | 2e-4 / 0.005 | Table 7 |
| densify interval / from / until | 100 / >500 / <10000 | arguments/__init__.py; run script |
| prune_from_iter | 500 (prune only if it > 500) | arguments |
| percent_dense | 0.01 | arguments |
| FSGS extra split | split also if mean sq. 3-NN dist > 10·extent and max scale > extent | gaussian_model.py:437 (`dist_thres` 10) |
| max_screen_size pruning | none (`size_threshold = None`) | train.py:370 |
| opacity reset | min(σ(o), 0.05) at it = 3501, 6501, 9501 (`(it-500-1) % 3000 == 0`) | train.py:411 |
| position lr | 1.6e-4·extent → 1.6e-6·extent, exp, max_steps 10000 | arguments, run script |
| feature / opacity / scaling / rotation lr | 2.5e-3 (rest /20) / 0.05 / 5e-3 / 1e-3 | arguments |
| SH degree | +1 every 500 it (max 3) | train.py:191 |
| λ_ssim | 0.2 | arguments |
| depth_smooth_weight (ω1) | 0.01 | Table 7; code |
| depth_range_weight (λ_r) | 0.001 | Table 7; code |
| ω2 (pseudo-view smoothness) | 0.05 paper / **0 code** | Table 7 vs train.py |
| λ1, λ2, λ3 | 1 | Table 7 |
| start_sample_pseudo | 500 (but gated by warm-up 1500) | run script |
| pseudo poses | 10000 random LLFF poses (`generate_random_poses_llff`), drawn without replacement | utils/pose_utils.py:320 |
| background | random per iteration in training (`--random_background`), black at test | run script; render.py |
| gaussiansN | 2 (+`--coreg`), co-pruning off | run script |
| init opacity | 0.1; scales log sqrt(mean 3-NN sq dist) | gaussian_model.py |

## 4. Schedules
- it ≤ 1500: warm-up thresholds, photometric only.
- it > 1500: low phase iff `it % 200 >= 100`. Densification at multiples of 100, so it = 200k
  densifies with high thresholds, it = 200k+100 with low thresholds. Regularisers active on
  iterations with `it % 200 in [100, 199]`.
- Opacity resets at 3501 / 6501 / 9501. Densification stops at 10000 (last at 9900).

## 5. Builds on
3DGS + FSGS densification extra criterion + CoR-GS dual-field co-training (co-reg only, no
co-pruning). Rasterizer `diff-gaussian-rasterization-confidence` (CoR-GS fork: also outputs
accumulated depth and alpha; `use_confidence` False → standard 3DGS).

## 6. Protocol
- LLFF (8 scenes), 3/6/9 views; images downsampled ×4 to 1008×756 from the full 4032×3024
  `images/` by PIL in the loader (`-r 4`).
- Split (FSGS/CoR-GS/DNGaussian convention, `dataset_readers.py:349-360`): sort cameras by image
  name; test = indices `idx % 8 == 0`; train candidates = the rest; train = candidates at indices
  `round(np.linspace(0, len-1, n_views))` (Python banker's rounding).
- Initial point cloud: `<scene>/3_views/dense/fused.ply` produced by `tools/colmap_llff.py`:
  COLMAP SIFT features on the 3 full-res training images (max_image_size 4032, 32768 features,
  affine shape + DSP), exhaustive matching with guided matching, `point_triangulator` with the
  known (full-model) poses and intrinsics, then `image_undistorter`, `patch_match_stereo`,
  `stereo_fusion` → dense fused cloud of the 3 training views only.
- Mip-NeRF 360 12/24 views (-r 4), Tanks&Temples 3/6/9 views also reported.
- Metrics: PSNR, SSIM (3DGS gaussian-window SSIM), LPIPS-VGG (lpipsPyTorch) on test views; G1
  only; test renders on black background.

## 7. Ambiguities
- ω2 = 0.05 pseudo-view depth smoothness: in the paper, disabled in released code → follow code.
- Edge weight: paper "summed over channels"; code averages over channels. Paper Σ over pixels;
  code means. Follow code.
- Paper says the two fields are "initialized independently"; code initialises both from the same
  fused cloud with identical deterministic init; they diverge only through random split sampling
  (and the shared random background is identical for both fields).
- Densification happens before the optimizer step in the official loop (3DGS convention); the
  iteration on which densification happens therefore does not update parameters.
- `densify_and_split_new` branch only active for it ≥ 150000 → dead code here.
