# CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization (ECCV 2024)

arXiv 2405.12110 · project https://jiaw-z.github.io/CoR-GS · official code
https://github.com/jiaw-z/CoR-GS (cloned to `source/code/`, Inria Gaussian-Splatting licence:
non-commercial research/evaluation use; the port re-implements the logic and cites it).
The official code is a fork of FSGS (same dataset reader, pose generator, densification).

## 1. Method summary
- Train **two** 3DGS radiance fields Θ¹, Θ² simultaneously from the same initial point cloud;
  their only source of difference is the randomness of densification (split sampling).
- **C1 Co-pruning (§4.1, Eq. 2-3)**: every 5 densification intervals (500 its) find, for every
  Gaussian of one field, its nearest Gaussian in the other field; prune Gaussians whose nearest
  counterpart is farther than τ = 5 (world units). Both masks computed before either pruning.
- **C2 Pseudo-view co-regularisation (§4.2, Eq. 4-7)**: sample a pseudo view near the training
  views; render both fields there and penalise their disagreement with a photometric loss
  (L1 + D-SSIM), target from the other field detached.
- Only Θ¹ (`gs0`) is kept for inference (`render.py` loads `point_cloud/`, i.e. gs0).

## 2. Equations
Eq. 2 `f(\theta^1_i) = {\rm KNN}(\theta^1_i, \Theta^2)`
Eq. 3 `M_i = 1 if \sqrt{(\theta^1_{xi} - f(\theta^1_i)_x)^2 + ... } > \tau, else 0` ; prune M=1.
Eq. 4 `P' = (t + \epsilon, q)` (pseudo view; code: RegNeRF/FSGS `generate_random_poses_llff`,
  10 000 random poses inside the bounding box of the training camera centres, looking at the
  focus point; see `utils/pose_utils.py:320`).
Eq. 5 `\mathcal{R}_{pcolor} = (1-\lambda)\mathcal{L}_1(I'^1, I'^2) + \lambda\mathcal{L}_{D-SSIM}(I'^1, I'^2)`, λ = 0.2
Eq. 6 `\mathcal{L}_{color} = (1-\lambda)\mathcal{L}_1(I^1, I^*) + \lambda\mathcal{L}_{D-SSIM}(I^1, I^*)`
Eq. 7 `\mathcal{L} = \mathcal{L}_{color} + \lambda_p \mathcal{R}_{pcolor}`, λ_p = 1.0
Code (`train.py:166-176`): for field i, loss_i = photometric(I_i, gt) + photometric(I'_i,
sg(I'_j)) / (N-1). Field 2 gets the symmetric loss. `loss_scale` ramp is computed but unused.
D-SSIM = 1 − SSIM (11×11 Gaussian σ=1.5, same padding). L1 is the mean absolute error.

## 3. Hyper-parameters (LLFF 3-view, `scripts/run_llff.sh` + `arguments/__init__.py`)
| name | value | source |
|---|---|---|
| iterations | 10 000 | run_llff.sh; paper §5.1 |
| position LR | 1.6e-4·extent → 1.6e-6·extent, max_steps 10 000 (delay_steps 0) | run_llff.sh, arguments |
| feature / rest / opacity / scaling / rotation LR | 0.0025 / /20 / 0.05 / 0.005 / 0.001 | arguments |
| densify grad threshold | 5e-4 | run_llff.sh |
| densify interval / from / until | 100 / >500 / <10 000 | arguments, run_llff.sh |
| prune_from_iter | 500 (split parents & pruning only when it > 500) | arguments |
| prune opacity | 0.005; no screen/world-size prune (size_threshold=None) | train.py:262 |
| percent_dense | 0.01 | arguments |
| FSGS far split | split if mean sq. 3-NN dist > 10·extent and max scale > extent | gaussian_model.py:434 |
| proximity unpooling | **not used** (`proximity()` never called) | train.py |
| opacity reset | min(o, 0.05) at it > 500, (it − 501) % 3000 == 0 → 501, 3501, 6501, 9501 | train.py:283 |
| SH degree | +1 every 500 its (before render) → min(t//500, 3) | train.py:121 |
| background | `--random_background`: bg = rand(3) per iteration (train + pseudo renders); eval black | run_llff.sh, train.py:146 |
| gaussiansN | 2 | run_llff.sh |
| co-prune | every 500 its for it > 500 (1000 … 10 000), τ = 5 (`--coprune_threshold 5`) | train.py:292, paper §4.1 |
| co-reg | every iteration (`sample_pseudo_interval 1`) for 500 ≤ it ≤ 10 000 (`end_sample_pseudo` 10 000), λ_p = 1 | run_llff.sh, arguments |
| pseudo poses | 10 000, popped at random from a stack (refilled when empty) | pose_utils.py:322, train.py:182 |
| λ_dssim | 0.2 | arguments |
| seed | 42 | train.py |

## 4. Schedules
SH every 500; densify 600..9900 every 100; opacity reset 501/3501/6501/9501; co-reg from 500;
co-prune 1000..10 000 every 500; position LR log-linear decay over 10k.

## 5. Builds on
Inria 3DGS + FSGS densification (far-split rule; no unpooling, no depth loss). Kept unchanged:
3DGS init (3-NN scales, opacity 0.1), Adam LRs, clone/split, photometric loss.

## 6. Protocol
- LLFF (8 scenes), 3 views: test = every 8th sorted image, train = `round(linspace(0, n−1, 3))`
  of the rest; `-r 8` from full-res `images/` (PIL resize, bicubic) → 504×378. Init = COLMAP MVS
  fused cloud of the 3 training views (`tools/colmap_llff.py`, FSGS protocol).
- Mip-NeRF 360 12/24 views (1/4 res, 30k its), DTU 3/6/9 views (masked eval), Blender 8 views.
- Metrics: PSNR/SSIM/LPIPS(VGG, lpipsPyTorch) on 8-bit PNG renders, black background; AVGE.
- **Only averages are reported** (no per-scene tables in main paper or supplement).

## 7. Ambiguities
- Paper: "co-pruning every 5 optimisation/density-control interleaves"; code: every 500 its,
  also after densify_until (until the last iteration). Code used.
- Paper: co-reg "incorporated from the first densification step"; code: from it 500. Same.
- Paper Eq. 7 has no ramp; code computes `loss_scale` but never applies it → weight 1.
- τ = 5 is in COLMAP world units of the original LLFF reconstruction (scene-scale dependent).
- Official optimizer step comes *after* densification in each iteration (densified tensors have
  no grad that step); native runtime steps before density control.
- Official skips the optimizer step at the final iteration (minor).
- Two fields have identical initialisation; divergence only from split sampling (and shared
  random bg / view order). Official seeds 42.
