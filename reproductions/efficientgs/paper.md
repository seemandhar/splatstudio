# EfficientGS — implementation spec

Paper: *EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene
Representation* (Liu et al., IJCAI'24 submission, arXiv 2404.12777, LaTeX `source/latex/ijcai24.tex`).
Official code: https://github.com/wenkaier/EfficientGS (commit `1f70babe`, Inria 3DGS licence),
cloned into `source/code/`. Run command (`train.sh`, used for all 13 small-scale scenes):

```
python train.py -s $scene -m $out --eval --sh_degree 3 --use_grad_norm --grad_norm_way s1
  --densify_grad_threshold 0.0007 --prune_by_max_weight --prune_way p4
  --oneupSHdegree_from_iter 15500 --shs_lr_rate 5 --use_sparse_shs --sparse_rate 0.2
```

## 1. Method summary
- 3DGS (Kerbl et al.) training, unchanged except for three mechanisms.
- **C1 Selective densification (Sec. 3.1, Eq. 6)**: the densification statistic is the per-view
  sum over pixels of the *norm* of each pixel's screen-space position gradient (instead of the
  norm of the summed gradient), averaged over views; threshold τ_s = 0.0007. Clone/split decision
  by scale exactly as 3DGS.
- **C2 Gaussian pruning (Sec. 3.2, Eq. 7–8)**: render every training view; a Gaussian survives
  only if it is among the top-K blending weights of at least one pixel (K = 1).
  Code: performed at iterations 15000, 18000, 21000 (`prune_way p4`).
- **C3 Sparse order increment of SH (Sec. 3.3, Eq. 9)**: SH degree 0 for every Gaussian until
  15500 (higher SH coefficients not even allocated); at 16000, 17000, 18000 compute per-Gaussian
  colour error d_k over all training views and increase the SH degree by one for the top
  r_s = 20 % Gaussians (capped at 3). Each Gaussian renders with its own degree.

## 2. Equations (verbatim)
Eq. 5 (3DGS statistic): `\mathbf{E}_g = \frac{\sum_{I\in M}(|\sum(\vec\nabla_{p_I}^{g})|)}{|M|}`
Eq. 6 (C1): `\mathbf{S}_g > \tau_s,\ \mathbf{S}_g = \frac{\sum_{I\in M}(\sum(|\vec\nabla_{p_I}^{g}|))}{|M|}`
  - ∇_{p_I}^g: gradient of the loss wrt Gaussian g's 2D (NDC) mean contributed by pixel p of image I;
    M: views where g is visible (code: `denom` += 1 when radii > 0).
  - Code (`backward.cu:670`): `dL_norm = sqrt(dL_dx^2 + dL_dy^2)` per pixel, `atomicAdd` into the
    third component of `dL_dmean2D`; `dL_dx` includes `ddelx_dx = 0.5 W` (NDC units like 3DGS).
    `add_densification_stats` accumulates `grad[:, 2]`, denominator = visible count.
Eq. 7 (C2): `weight^i_p=\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)`
Eq. 8 (C2): keep g if `weight^g_p \geq sort(\{-weight^j_p\})[K]` for some pixel p of some view.
  - Code (`forward.cu:477-507`): **the weight used is `alpha * T` evaluated after `T` has been
    updated**, i.e. `alpha_i (1-alpha_i) prod_{j<i}(1-alpha_j)`; strict `>` insertion into a
    top-K list; `max_weight_pids[id] = 1`; summed over training views; prune if sum < view_num = 1.
Eq. 9 (C3): `d_k=\sum_{p=1}^{NHW}\mathbbm{1}(k\in\mathcal{N}_p)weight^k_p|C_{p} - C_{p}^{gt}|`
  - Code: `pix_loss = |render - gt|` ([3,H,W], unclamped render, training background);
    `loss_weight += sum_ch pix_loss[ch] * alpha * T` (same post-update T as above), atomic per
    Gaussian, summed over all training views.
  - Selection (`train.py:get_sparse_shs_mask`): `k = int(n * r_s)`; `thr = lw[argsort(-lw)[k]]`;
    `mask = lw >= thr`; `degree[mask] += 1`; `degree = min(degree, 3)`.
  - Rendering (`forward.cu:277`, `backward.cu:452`): `computeColorFromSH(idx, sparse_degree[idx])`,
    the global active degree is ignored once sparse degrees exist.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §4.1, code default |
| τ_s (densify_grad_threshold) | 0.0007 | paper §4.1, train.sh |
| densify from / until / every | 500 / 15000 / 100 | code `arguments/__init__.py` (3DGS) |
| percent_dense, min_opacity, big-world prune | 0.01, 0.005, 0.1·extent after 3000 | code (3DGS) |
| opacity reset | every 3000 while < 15000 | code (3DGS) |
| K (top-K) | 1 | paper §4.1, code `top_k=1` |
| view_num | 1 | code `train.py` |
| prune iterations | 15000, 18000, 21000 | code `prune_way p4`; paper text: "after 15500 iterations" (once) |
| SH memory allocated at | 15500 | code `--oneupSHdegree_from_iter 15500`; paper §4.1 |
| sparse SH increments | 16000, 17000, 18000 (interval 1000, sparse_cnt 3) | paper §4.1, code |
| r_s | 0.2 | paper §4.1, train.sh |
| sparse_way | pix_loss (Eq. 9) | code default |
| max SH degree | 3 | train.sh |
| SH rest lr | feature_lr / shs_lr_rate = 0.0025 / 5 = 5e-4 | train.sh `--shs_lr_rate 5` (3DGS: /20) |
| other lrs | 3DGS (means 1.6e-4·extent → 1.6e-6, sh0 2.5e-3, opac 0.05, scale 5e-3, rot 1e-3) | code |
| loss | (1-0.2) L1 + 0.2 (1-SSIM) | code |

## 4. Schedules
- 1..15000: 3DGS densification with S_g; opacity reset at 3000, 6000, 9000, 12000.
- 15000, 18000, 21000: top-K pruning (end of iteration, after densification block).
- 15500: allocate SH rest coefficients (zeros, fresh Adam), all sparse degrees 0.
- Start of 16000, 17000, 18000: Eq. 9 selection, +1 degree for top 20 %.
  (At 18000 the increment precedes that iteration's pruning.)

## 5. Builds on
Inria 3DGS code base (official rasterizer modified). Unchanged: initialisation from SfM,
loss, LR schedules except SH rest LR, clone/split mechanics, opacity reset, opacity/big-world
pruning. Removed: the global 1000-step SH-degree schedule (degree 0 until 15500).
Unused flags in train.sh path: reg_loss, iso/dis losses, train speedup, sphere mode (all off).

## 6. Protocol
Mip-NeRF 360 (9 scenes), Tanks&Temples (truck, train), Deep Blending (drjohnson, playroom),
every 8th image test (llffhold 8), 30k iterations, black background, 3DGS resolution handling
(images_4 outdoor / images_2 indoor via 1.6K cap in 3DGS loader — the paper states "same datasets
as the vanilla 3DGS"). Metrics PSNR/SSIM/LPIPS (3DGS `metrics.py`, LPIPS-VGG). Single A100.
Only dataset averages are reported (Table 1, ablations Tables 4–6 on Mip-NeRF 360 average).
Large-scale results (Mill19 Building/Rubble at 4608×3456, self-collected Foothill-Town) use 60k
iterations with proportionally scaled schedules; those datasets are not available here.

## 7. Ambiguities
1. Pruning schedule: paper says once after 15500; code prunes at 15000/18000/21000 → use code.
2. Weight in Eq. 7/9: code uses α·T after the T update (α_i(1−α_i)T_i); paper writes α_i T_i.
   Code convention by default, configurable.
3. Rasterizer numerics: official alpha clamp 0.99 and stop at T<1e-4; gsplat clamps at 0.999
   and stops at next_T ≤ 1e-4. The native port uses gsplat numerics (they define the rendered image).
4. Eq. 6 units: code accumulates per-pixel norms of the NDC-space gradient (pixel gradient × W/2,
   × H/2). Loss normalisation (mean over pixels) as in 3DGS.
5. `max_radii2D > 20` screen-size prune is inert in the official code (max_radii2D zeroed by
   densification_postfix before the test), as in 3DGS — only the world-size prune acts.
6. Background: official trains with black background; the native runtime uses white.
