# HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering — implementation spec

arXiv 2412.01823 (Song, Lin, Lei, Liu, Daniilidis; UPenn; CVPR-25 submission). Sources:
`source/latex/` (e-print), `source/paper.pdf|txt`.
**Official code exists** (released Sept 2026, not found by the scout):
github.com/TimSong412/HDGS — `main` branch (commit b13a359, `source/code/`) = stage 1 (2DGS +
k-buffer sort + frustum sampling, *no texture*); `texture` branch (commit 01c6fd4,
`source/code-texture/`) = stage 2 (`train_texture.py`, textured rasterizer). Built on the official
2DGS code (hbb1/2d-gaussian-splatting); rasterizer = a fork of `diff-surfel-rasterization`.
Licence: Inria Gaussian-Splatting licence (non-commercial research use).
**The Fisher pruning step is not in either branch** (README: stage 2 "relies on the pre-trained
and pruned checkpoints"; no pruning script released).

## 1. Method summary
- 2DGS surfels (Huang et al. 2024): ray–splat intersection (u,v), depth/normal/distortion maps.
- Per-ray k-buffer depth sorting + per-tile sort by the depth at the tile centre (StopThePop-like).
- Frustum (5-sample) anti-aliasing: Gaussian weight averaged over pixel centre + 4 corners.
- Fisher-information pruning of 10 % of the surfels after 2DGS training.
- Stage 2: per-surfel texture map (bilinear) replaces the SH DC term; geometry frozen.

Contributions
- **C1 per-ray sort** (§3.3 "Per-Ray Sorting"): k-buffer of size k (24 synthetic / 16 real) per
  pixel + per-tile radix sort keyed by the ray–surfel depth at the tile centre (tile 8x8).
- **C2 frustum sampling** (§3.2, Eq. avg): $\hat G = w_m G(\mathbf x) + \sum w_c G(\mathbf x+(\pm\delta_y,\pm\delta_x))$, $w_m=w_c=0.2$.
- **C3 Fisher pruning** (§3.3, Eq. U_i): prune the 10 % surfels with the lowest sensitivity.
- **C4 per-surfel texture** (§3.4): texture $U=\lceil T s_u\rceil$, $V=\lceil T s_v\rceil$, bilinear
  lookup at $(\frac{u+r}{2r}U,\frac{v+r}{2r}V)$, $r=4.5$; colour $\hat C = C[u,v] + SH(\mathbf d)$ with
  the exact intersection direction $\mathbf d = \mathbf H(u,v,1)^\top-\mathbf O$.
- **C5 two-stage optimisation** (§3.5): stage 1 single-colour 2DGS + prune; stage 2 texture
  fine-tune with centres/rotations/scales frozen.

## 2. Equations (verbatim from sec/3_method.tex)
- 2DGS: $\mathbf{x} = (xz, yz, z, z)^\top = \mathbf{P}\mathbf{H}(u, v, 1, 1)^\top$,
  $\mathbf{H} = [s_u\mathbf{t_u}\; s_v\mathbf{t_v}\; 0\; \mathbf{p}; 0\,0\,0\,1]$, $G(\mathbf{x}) = \exp(-\frac{u^2+v^2}{2})$.
- Blending: $\mathbf{C(x)} = \sum_i \mathbf{c}_i(\mathbf{d}) \alpha_i G_i(\mathbf{x}) T_i$, $D$, $\mathbf N$ likewise.
- Eq. Ld: $L_d = \sum_{i, j} \alpha_i G_i T_i \alpha_j G_j T_j (z_i-z_j)^2$; Eq. Ln: $L_n = \sum_i \alpha_i G_i T_i (1 - \mathbf{n}_i \mathbf{N}_d)$.
- Fisher (Eq. Approx): $\nabla^2_\mathcal{G} L_2 = \sum_{\phi \in P_{gt}} \nabla_\mathcal{G} \textbf{I}_\mathcal{G} \nabla_\mathcal{G} \textbf{I}^T_\mathcal{G}$
- Eq. U_i: $U_i = ||\log(\text{diag}( \sum_{\phi \in P_{gt}} \nabla_{x_i,s_i} \textbf{I}_{\mathcal{G}_i} \nabla_{x_i,s_i} \textbf{I}^T_{\mathcal{G}_i})||_1$ (position + scaling only, following PUP-3DGS).
- Eq. avg: $\hat{G}(\mathbf{x}) = w_mG(\mathbf{x}) + \sum_{i, j \in \{1,-1\}} w_cG(\mathbf{x} + (i\delta_y, j\delta_x))$, pixel size $(2\delta_x,2\delta_y)$ ⇒ corners at ±0.5 px.
- Texture: $\mathbf{\hat{C}(x)} = \mathbf{C}[u, v] + \mathbf{SH}(\mathbf{d})$; Eq. blending $\mathbf{C(x)} = \sum_{i}\mathbf{\hat{C}(x)}_i \alpha_i \hat{G}_i(\mathbf{x}) T_i$ (i in k-sorted order).
- Loss: $L = L_c + \lambda_n L_n + \lambda_d L_d$, $L_c$ = L1 + SSIM (3DGS 0.8/0.2).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations stage 1 | 40000 | code run_360.py `--iteration 40000` (paper silent) |
| iterations stage 2 | 20000 (40001–60000) | code-texture run_360.py `--iteration 60000 --start_checkpoint chkpnt40000` |
| SORT_WINDOW k | 16 real / 24 synthetic | paper §3.3; README "set 16 for nerf360"; output dir `eval360S16`; compiled default 24 |
| tile size | 8x8 | paper §3.3; config.h BLOCK_X/Y 8 |
| frustum weights w_m, w_c | 0.2, 0.2 (plain mean of 5) | paper §4.1; forward.cu `(G00+G10+G01+G11+G_c)/5` |
| corner offsets | ±0.5 px | forward.cu |
| tex_res T | 200 outdoor, **500 indoor (code)** / 400 indoor (paper), 1000 synthetic | run_360.py (texture) / paper §4.1 |
| max texture side | 500 | train_texture.py `init_texture(..., max_tex=500)` |
| tex cutoff r | 4.5 | auxiliary.h TexRange |
| texture sampling | bilinear, `x = (u/r+1)/2 (W-1)` | tex.h (BILINEAR 1) |
| texture init | each surfel's SH DC (features_dc) | texture_manager.py init_texture (paper: "background colour") |
| texture optimiser | Adam lr 8e-3 (eps 1e-8), StepLR 8000 γ 0.1 | train_texture.py |
| stage-2 trainable | texture + SH rest (f_rest) only; xyz, f_dc, scaling, rotation, **opacity** frozen; gaussian optimiser steps only after 1000 stage-2 iters | train_texture.py |
| stage-2 loss | L_c only (dist/normal computed but not added) | train_texture.py `total_loss = loss` |
| prune fraction | 10 % | paper §4.1 |
| lambda_dist | 100 (after 3000) | run_360.py; train.py |
| lambda_normal | 0.01 (after 7000) | run_360.py |
| lambda_dssim | 0.2 | arguments |
| densify_grad_threshold | 0.0002 | run_360.py |
| densification_interval | 150 | run_360.py |
| densify from / until | 500 / 20000 | arguments / run_360.py |
| opacity_cull | 0.01 | run_360.py |
| opacity reset | every 3000 (while < densify_until) to min(o, 0.01) | train.py, gaussian_model.py |
| percent_dense | 0.01 | arguments |
| world-size prune | max scale > 0.1·extent (when step > 3000) | gaussian_model.densify_and_prune |
| screen-size prune | 20 px — no-op (max_radii2D reset in densification_postfix) | gaussian_model.py |
| densify skip | if step > 19000 and N > 8e6 | train.py |
| position lr | 1.6e-4·extent → 1.6e-6·extent over 30000 its, then constant | arguments, get_expon_lr_func |
| other lrs | f_dc 2.5e-3, f_rest 1.25e-4, opacity 0.05, scaling 5e-3, rotation 1e-3 | arguments |
| init | SfM points, scale = log sqrt(3-NN mean d²) on 2 axes, rotation `torch.rand(4)`, opacity 0.1 | gaussian_model.create_from_pcd |
| background | black (360) | `--white_background` not set |
| depth_ratio | 0 (expected depth for normal-from-depth) | run_360.py (default) |
| near/far (distortion) | 0.2 / 100 | auxiliary.h |
| alpha clamp / skip / T stop | 0.99 / 1/255 / 1e-4 | forward.cu |

## 4. Schedules
SH degree +1 every 1000 its; dist loss from 3000, normal loss from 7000; densify 500–20000 every
150 (+opacity reset every 3000); prune at end of stage 1 (paper); stage 2 from 40000: texture
lr 8e-3 → 8e-4 (stage-2 it 8000) → 8e-5 (16000).

## 5. Builds on
Official 2DGS code (losses, densification, data loading). Stage-1 colours: SH evaluated at the
centre direction (preprocess). Stage-2 colours: SH evaluated at the intersection direction with
the DC term from the texture (tex.h computeTexFromSH).

## 6. Protocol
Mip-NeRF 360: outdoor `images_4`, indoor `images_2` (both < 1600 px wide, no further
downscale), llffhold 8 (`--eval`), black background, PSNR/SSIM/LPIPS-VGG (metrics.py).
NeRF-Synthetic (white bg, depth_ratio 1). DTU geometry. Per-scene tables for Mip-NeRF 360
(supp. Tab. 360_full) and NeRF-Synthetic (supp. Tab. nerf_full / ablation_full / prune_full).
Stage 2 is trained at the same resolution as stage 1 in the released script (paper text
mentions "on-demand higher-resolution images"). L40 GPU.

## 7. Ambiguities
1. Fisher score: "‖log diag F‖₁". Literal Σ|log F_jj| inverts the ranking when F_jj<1; we use
   Σ_j log F_jj (log-det of the diagonal, consistent with PUP-3DGS's log|F|) — lowest pruned.
   Pruning code not released; Jacobian per pixel is not computable from a standard backward;
   we estimate diag F with Rademacher (Hutchinson) projections of the image per training view.
2. Stage-2 resolution: paper says higher-resolution; code uses the stage-1 images. We follow code.
3. Indoor texture resolution 400 (paper) vs 500 (code). We follow code.
4. Texture init: paper "background colour", code = surfel DC colour. We follow code.
5. Stage-1 rasterizer (main) inserts surfels with alpha < 0.1 at the buffer head/tail without a
   full insertion sort (approximate); stage-2 rasterizer (texture) sorts every surfel. Both kept.
6. Stage-1 alpha uses only the 3D intersection Gaussian averaged over 5 samples; 2DGS's 2D
   low-pass fallback only affects depth (rho2d branch). Kept as in code.
7. Official texture backward uses the raw texture DC (not the SH colour) inside dL/dalpha —
   irrelevant because opacity/geometry are frozen in stage 2.
