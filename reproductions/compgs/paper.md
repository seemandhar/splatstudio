# CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization (Navaneet et al., ECCV 2024)

arXiv 2311.18159 (a.k.a. Compact3D). Official code: https://github.com/UCDvision/compact3d
(MIT, commit dccc07e3ab05f65246be25f0ee416aa9a127b54e, cloned read-only into `source/code/`).
LaTeX: `source/latex/arxiv.tex`.

## 1. Method summary
- Plain 3DGS training (official Inria code, unchanged hyper-parameters) up to the VQ start.
- **C1 K-means vector quantisation-aware training** (§3 "Vector quantization", §4 Impl. details):
  four separate codebooks — DC colour (3-d), SH rest (45-d), scale (3-d, *before* exp),
  rotation (4-d quaternion, *before* normalisation). Forward renders the centroid
  `c[idx]`; straight-through estimator copies its gradient to the non-quantised params.
  Centroids are re-averaged from the current non-quantised params **every iteration**
  (cached assignments); assignments (+ a few Lloyd iterations) every `t=100` iterations
  until 25K, then frozen till 30K. Positions and opacity are not quantised.
- **C2 Opacity regularisation + pruning** (§3 "Opacity Regularization"):
  `L = L_3DGS + λ_reg Σ_i σ_i` (σ = sigmoid opacity) with λ=1e-7 for iterations 15K–20K and
  pruning of Gaussians with σ<0.005 every 1000 iterations in that window.
- **C3 Storage / decoded model**: after training only codebooks + per-Gaussian indices
  (+ fp32 position, opacity) are kept; evaluation renders the decoded model (codebook lookup).
  Paper storage additionally sorts by one index and run-length codes it (§3).

## 2. Equations
- STE (code `kmeans_quantize.py:forward_*`): `x_q = x - x.detach() + C[idx]`.
- Centroid update with cached assignment (`update_centers`): `C_k = Σ_{i:idx_i=k} x_i / (|{i:idx_i=k}| + 1e-6)` (empty cluster → 0).
- Assignment (`cluster_assign`): init `C` = random subset of x (first time only; warm start
  afterwards); repeat `kmeans_iters` times: `idx = argmin_k ||x - C_k||_2`, `C_k = mean`;
  final `idx = argmin_k ||x - C_k||` with the final C (the centroids are *not* recomputed
  after this last assignment within that iteration).
- Loss (§3): $\mathcal{L} = \mathcal{L}_{\textrm{3DGS}} + \lambda_{\textrm{reg}}\sum_i^N \sigma_i$,
  $\mathcal{L}_{3DGS} = 0.8\,L_1 + 0.2\,(1-\mathrm{SSIM})$.

## 3. Hyper-parameters (headline variant = CompGS 32K with opacity reg; supp. Table "Scene-wise metrics")
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §4, code |
| VQ start | after 20000 (first assignment at it 20001) | paper §4 "trained without VQ till 20K" (code run.sh uses `st_iter=15000`) |
| assignment freq t | 100 (iteration % t == 1), freq 5000 after 25000 → last assignment at 25001 | paper §4, code train_kmeans.py:124-127 |
| Lloyd iterations per assignment | 5 | Table perf_vs_time row (5,100,32K)=27.12 = headline; paper text says 1; run.sh 10 |
| codebook DC / SH | 4096 / 4096 | paper §4 "4096 for the color" (run.sh: dc 4096, sh 512) |
| codebook scale / rot | 32768 / 32768 (separate) | "CompGS 32K"; §3 "4 codebooks" |
| λ_reg | 1e-7, active 15000 ≤ it ≤ 20000 | paper §4, code train_kmeans.py:150 |
| prune | σ < 0.005 every 1000 its, 15000 < it ≤ 20000, no size criterion | code train_kmeans.py:205, gaussian_model.py:prune |
| 3DGS densification | unchanged (until 15000, grad 2e-4, reset 3000) | paper §4 |

## 4. Schedules
0–15K 3DGS; 15K–20K + opacity L1, prune @16K,17K,18K,19K,20K; 20001 first k-means
assignment; centroid averaging every iteration 20001–30000; assignments every 100 until 25001;
frozen afterwards; decode at 30000.

## 5. Builds on
Official Inria 3DGS (train.py) — everything else unchanged.

## 6. Protocol
Mip-NeRF 360 (9 scenes, images_4 outdoor / images_2 indoor as 3DGS), T&T (truck, train),
Deep Blending; llffhold 8; 30K iterations; metrics as 3DGS (LPIPS-VGG); black background
(Inria default). Per-scene table: supp. "Scene-wise Metrics" (3DGS retrained by authors and
CompGS 32K, all 13 scenes, PSNR to 3 decimals, Mem MB, #Gauss).

## 7. Ambiguities
- VQ start 20K (paper) vs 15K (run.sh). We follow the paper text; configurable `vq_start`.
- Lloyd iterations: 1 (text) vs 5 (Table perf_vs_time row matching headline 27.12) vs 10 (run.sh). Use 5.
- SH codebook 4096 (paper) vs 512 (run.sh). Use paper.
- Official `save_kmeans` writes indices with ceil(log2 N) bits (len(cls_ids)=N, a bug); paper
  Mem uses log2 K bits + RLE on one index. We report the paper formula and the official on-disk formula.
- Official renders the step-30000 image before the (skipped) last optimizer step; the stored
  centroids are those used at it 30000.
