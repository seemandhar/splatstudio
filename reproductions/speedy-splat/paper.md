# Speedy-Splat — implementation spec

Hanson, Tu, Lin, Singla, Zwicker, Goldstein. *Speedy-Splat: Fast 3D Gaussian Splatting with
Sparse Pixels and Sparse Primitives*, CVPR 2025, arXiv 2412.00578.
Official code: https://github.com/j-alex-hanson/speedy-splat (commit 34c45c6), rasterizer
submodule https://github.com/j-alex-hanson/speedy-splat-rasterizer (commit b38abb6, forked from
graphdeco-inria/diff-gaussian-rasterization 59f5f77). Gaussian-Splatting licence (Inria/MPII,
non-commercial research use).

## 1. Summary
- 3D-GS (Inria code) with an unchanged loss, optimizer, ADC densification and opacity resets.
- **Sparse pixels**: the rasterizer assigns a Gaussian only to tiles touched by its true
  alpha ≥ 1/255 ellipse (opacity aware) instead of the 3σ-radius square.
- **Sparse primitives**: a cheap per-Gaussian sensitivity score Σ_pixels (∂I/∂g_i)² accumulated
  inside the backward render kernel; low-score Gaussians are pruned during (soft) and after
  (hard) densification.

Contributions:
- C1 SnugBox (§4.1.1): tight axis-aligned bbox of the alpha=1/255 ellipse → tile rectangle.
- C2 AccuTile (§4.1.2, Alg. 1): exact set of tiles intersected by that ellipse (row/column sweep).
- C3 Efficient pruning score (§4.2.1): Ũ_i = Σ_p (∇_{g_i} I)², computed by the render backward.
- C4 Soft pruning (§4.2.2): prune 80 % lowest-score Gaussians right before the opacity resets at
  6000/9000/12000.
- C5 Hard pruning (§4.2.3): prune 30 % every 3000 iterations from 15000 (15k,18k,21k,24k,27k).

## 2. Equations
Inria rendering (Eq. 6–8): `α_i(p) = σ_i g_i(p)`, `g_i = e^q`,
`q = -½ (p-μ_2D) Σ_2D^{-1} (p-μ_2D)^T`; included if α_i ≥ 1/255;
`C(p) = Σ_i c_i α_i Π_{j<i}(1-α_j)` (forward.cu also clamps α ≤ 0.99 and stops when T < 1e-4).

3D-GS extent (Eq. 9): `r = ⌈3 √λ_max⌉` (square of side 2r).

True extent (Eqs. 10–14): `2 log(255 σ_i) = (p-μ) Σ^{-1} (p-μ)^T`,
`Σ^{-1} = [[a,b],[b,c]]`, `t = 2 log(255 σ_i)`, `x_d = p_x-μ_x`, `y_d = p_y-μ_y`,
`t = a x_d² + 2 b x_d y_d + c y_d²` (Eq. 15).

SnugBox (Eqs. 16–17): `y_d = (-b x_d ± √((b²-ac) x_d² + t c)) / c`,
`x_{d,args} = ± √( -b² t / ((b²-ac) a) )`; symmetric for x with a↔c.
Code (auxiliary.h `duplicateToTilesTouched`): `disc = b²-ac`; reject if `a ≤ 0 || c ≤ 0 ||
disc ≥ 0`; `t = 2 log(255 σ)` (σ = raw sigmoid opacity, before the 0.99 clamp; σ<1/255 gives
t<0 → NaN sqrt → no tiles); tile rect `[int(bbox_min/16), int(bbox_max/16+1))` clipped to the grid.

AccuTile (Alg. 1, `processTiles`): sweep along the shorter side of the SnugBox tile rect;
for each slice, min/max ellipse extent = SnugBox extremum if its arg lies in the slice, else
min/max of the ellipse intersections with the slice's two boundary lines (Eq. 16); convert to
tile range `[int(e_min/16), int(e_max/16+1))` clipped to the rect.

Pruning score (Eqs. 18–23): PUP-3DGS `U_i = log|∇_{μ_i,s_i} I ∇_{μ_i,s_i} I^T|` is replaced by
`Ũ_i = (∇_{g_i} I_G)²`. Code: `backward.cu` renderCUDA `atomicAdd(dL_dG2[i], dL_dG * dL_dG)`
per (pixel, Gaussian) pair, with `dL_dG = σ_i · dL/dα_i` (dL/dα summed over the 3 channels,
includes the background term, ignores the 0.99 clamp), so the score is
`Σ_views Σ_pixels (σ_i dL/dα_i(p))²`. The loss is `L = image.sum()` (train.py `score_func`),
i.e. dL/dC = 1 for every channel and pixel; the gradient is squared per pixel *before*
summation (the paper's "(∇ I)²" is per-pixel, aggregated over pixels and views).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py |
| prune_from_iter | 6000 | arguments:OptimizationParams |
| prune_until_iter | 30000 (exclusive) | arguments |
| prune_interval | 3000 | arguments |
| densify_prune_ratio (soft) | 0.80 | arguments; paper §4.2.2 |
| after_densify_prune_ratio (hard) | 0.30 | arguments; paper §4.2.3 |
| densify_from/until/interval | 500 / 15000 / 100 | 3D-GS defaults |
| densify_grad_threshold | 2e-4 | 3D-GS |
| opacity_reset_interval | 3000 | 3D-GS |
| percent_dense, min opacity | 0.01, 0.005 | 3D-GS |
| alpha threshold | 1/255 | auxiliary.h `t = 2 log(255 σ)` |
| tile size | 16×16 | config.h |
| background | black | white_background False |
| LRs | 3D-GS defaults | arguments |

Percentile rule (`prune_gaussians`): `v = sort(score)[int(ratio·(N-1))]`, prune `score ≤ v`
(ties prune more; Gaussians with zero score — never visible — are all pruned).

## 4. Schedules (official train.py order inside one iteration)
1. render, loss, backward.
2. if it < 15000: accumulate densification stats; if it > 500 and it % 100 == 0:
   `densify_and_prune` (size_threshold 20 after it 3000 — a no-op, see TODO-runtime.md).
3. **soft prune** if it ≥ 6000, it < 30000, it % 3000 == 0 and it < 15000 → 6000, 9000, 12000.
4. opacity reset if it % 3000 == 0 (it < 15000).
5. optimizer step.
6. **hard prune** if it ≥ 15000, it ≥ 6000, it < 30000, it % 3000 == 0 → 15000…27000.
Score: for every training camera, render with current params and a zero `scores` leaf,
`image.sum().backward()`, add `scores.grad` (computed under `torch.enable_grad()`).
Expected remaining fraction ignoring densification: 0.2³ after soft, ×0.7⁵ = 0.168 after hard.

## 5. Builds on
Inria 3D-GS (training loop, loss L1 + 0.2 D-SSIM, ADC densification, opacity reset, SH schedule,
exponential position LR) unchanged; rasterizer changes only tile assignment (forward
`preprocess` + `duplicateWithKeys`) and adds the score buffer to the backward. `radii` returned
are still the 3σ radius (used for max_radii2D), but visibility = tiles_touched > 0.
Forward also times itself with CUDA events (synchronizes every call).

## 6. Protocol
Mip-NeRF 360 (9), T&T train/truck, DB drjohnson/playroom; COLMAP of the dataset authors;
`--eval` llffhold 8; `-r -1` (images/ downscaled to width 1600 when wider: Mip-NeRF 360 →
1600 px; T&T/DB native) — README: "trains at a higher image resolution than the original
3D-GS full_eval.py protocol". 30k iterations, black background, 3 runs averaged, LPIPS from
lpipsPyTorch (VGG). FPS: CUDA events around the forward rasterizer, RTX A5000.

## 7. Ambiguities / notes
- Paper writes `(∇_{g_i} I)²`; code squares per pixel and sums (Σ_p (∂I_p/∂g_i)²), per view
  then over views, with I = sum over RGB channels. We follow the code.
- Score uses raw opacity in dL_dG = σ dL/dα although α is clamped at 0.99 in forward.
- Soft prune happens after densification at the same step (6000 % 100 == 0) and before the
  opacity reset; hard prune after the optimizer step.
- The score backward also accumulates gradients into the model parameters in the official code;
  they are discarded (prune_points rebuilds parameters). No effect on training.
- `radii` used for visibility/max_radii2D: visibility = tiles>0 (opacity-aware), so Gaussians
  with σ < 1/255 are invisible and get no densification stats.
