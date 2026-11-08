# HGS — Pushing Rendering Boundaries: Hard Gaussian Splatting (arXiv 2412.04826, CVPR 2025)

Sources: `source/latex/` (arXiv e-print, sec/3_method.tex, sec/4_experiment.tex, sec/X_suppl.tex, tab/*.tex),
`source/paper.pdf`. **No official code**: `github.com/GhiXu/HGS` (named in the abstract) returns 404 (re-checked
2026-11-06); GitHub search finds nothing. The method is built on **Scaffold-GS** ("We build our methods upon the
Scaffold-GS repository"); its official code is cloned in `source/code/scaffold-gs/` (commit 59c833b) and is the
reference for everything HGS does not change.

## 1. Method summary
- Base: Scaffold-GS (anchors with 32-D feature, K=10 offsets, 6-D scaling; view-dependent MLPs decode neural Gaussians;
  anchor growing from neural-Gaussian view-space gradients every M=100 its, pruning by accumulated opacity).
- Problem: growing uses the *average* view-space positional gradient over the views seen in an interval (Eq. 3),
  which smooths out large gradients from a few hard views and ignores local rendering errors (Eq. 4).
- **C1 PGHGS** (§5.1, Eq. 5–6): a neural Gaussian is also "hard" (a growth candidate) if its k-th largest
  per-view positional-gradient norm in the interval is ≥ λ τ_grad.
- **C2 REHGS** (§5.2, Eq. 7–8): per training view, the max-weight Gaussian of each pixel defines S_i; Gaussians with
  |S_i| > τ_large |P| are potentially over-large; those whose projected centre has local SSIM(I, Î) < τ_SSIM are hard;
  must be found in ≥2 views during the interval; hard Gaussians are grown.
- (C3 Effi-HGS (§5.3) caps PGHGS candidates to the top-Q fraction; a separate variant — not the reproduced row.)

## 2. Equations (verbatim)
Eq. 3 (original criterion): `\frac{\sum_j||\bm{g}_{i,j}||}{n} \geq \tau_\text{grad}`
Eq. 4: `\bm{g}_{i,j} = \frac{\partial L_j}{\partial \bm{\mu}_{i,j}}, L_j = \frac{\sum_{\bm{u} \in \bm{P}} e_{\bm{u}}}{||\bm{P}||}`
Eq. 5: `\{\bm{g}'_{i,1}, \cdots, \bm{g}'_{i,n}\} = \text{sort}_{\downarrow} (\bm{g}_{i,1}, \cdots, \bm{g}_{i,n})`
Eq. 6: `||\bm{g}'_{i,k}|| \geq \lambda\tau_\text{grad}`
idx(u) = argmax_i w_i (w_i = α_i G_i^{2D}(u) Π_{k<i}(1-α_k G_k^{2D}(u)), Eq. 2); S_i = {u | idx(u)=i}
Eq. 7: `||\bm{S}_i|| > \tau_\text{large} ||\bm{P}||` (from Mini-Splatting)
Eq. 8: `\text{SSIM} (\bm{I}_j, \hat{\bm{I}}_j)(\bm{u}_{i,j}) < \tau_\text{SSIM}`, u_{i,j} = projection of G_i in view j.
"such Gaussians should be seen by at least two views during the M iterations".

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| k | 3 | §6.1 |
| λ | 1.0 | §6.1 |
| τ_large | 2e-4 | §6.1 |
| τ_SSIM | 0.7 | §6.1 |
| τ_grad, M | Scaffold-GS: 2e-4, 100 | §6.1 "same as Scaffold-GS"; code:arguments/__init__.py |
| min views (REHGS) | 2 | §5.2 |
| Scaffold: feat_dim 32, n_offsets 10, update_depth 3, update_init_factor 16, update_hierachy_factor 4 | | code:arguments, train_*.sh |
| voxel_size | mip360 0.001, tandt 0.01, db 0.005 | code:train_mip360.sh / train_tnt.sh / train_db.sh |
| appearance_dim | 0 (all three datasets) | train_*.sh |
| start_stat 500, update_from 1500, update_interval 100, update_until 15000 | | code:arguments |
| min_opacity 0.005, success_threshold 0.8 | | code:arguments |
| lrs: feat 0.0075, scaling 0.007, offset 0.01→1e-4 (×extent), mlp_opacity 0.002→2e-5, mlp_cov 0.004, mlp_color 0.008→5e-5, anchor 0 | | code:arguments |
| loss: 0.8 L1 + 0.2 D-SSIM + 0.01 mean(prod(neural scaling)) | | code:train.py:145 |
| iterations | 30000 | §6.1 |

## 4. Schedules
Statistics for start_stat < t < update_until; growing/pruning when t > update_from and t % 100 == 0. PGHGS/REHGS act
only inside growing. All lr schedules are exponential over 30k.

## 5. Builds on
Scaffold-GS unchanged except the growing candidate set. Photometric loss, MLPs, pruning unchanged.

## 6. Protocol
Mip-NeRF 360 (9 scenes): 3DGS-repo downsampling of full-res images, -r 4 outdoor / -r 2 indoor (supp. §B) →
our `mipnerf360-pilr`. T&T: 6 scenes (Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck) at ×2 — we only have
truck (3DGS tandt release). DB: drjohnson, playroom full-res. 30k its, llffhold8 (`--eval`), black bg (Scaffold
default), RTX 3090. All Gaussian baselines retrained by the authors.

## 7. Ambiguities (choices)
1. **How hard Gaussians enter Scaffold's 3-level growing** (unspecified). Choice: at level i (threshold τ_i =
   τ·2^i) a neural Gaussian is a candidate if `avg ≥ τ_i` (OG) **or** `g'_k ≥ λ τ_i` (PGHGS) **or** it is an REHGS
   hard Gaussian (all levels). The level's random keep mask and the voxel dedup apply unchanged.
2. **"n" and the OG success mask**: Scaffold requires `denom > 0.5·0.8·M = 40` views for the OG candidate. Hard
   candidates require only k views (PGHGS) / 2 detections (REHGS) in the current interval, not 40.
3. **"during the M iterations"**: PGHGS top-k buffers and REHGS counters are reset at every growth step (every 100
   its), unlike Scaffold's average accumulators (reset only where denom > 40).
4. **Which Gaussians**: neural Gaussians (the entity Scaffold grows from); argmax over the rasterized neural Gaussians.
5. **SSIM map**: 11×11 Gaussian window σ=1.5, same padding, mean over RGB (3DGS `ssim`), image values clamped [0,1];
   u_{i,j} = rounded gsplat `means2d` (pixel), Gaussians projecting outside the image are not tested.
6. τ_large|P| uses |P| = W·H of the current training view.
