# Mini-Splatting — implementation spec

Fang & Wang, *Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians*,
ECCV 2024, arXiv 2403.14166. Official code: https://github.com/fatPeter/mini-splatting
(cloned to `source/code/`, built on the Inria 3DGS code; `ms/train.py` = Mini-Splatting,
`ms_d/train.py` = Mini-Splatting-D, `submodules/diff-gaussian-rasterization_ms` = modified
rasterizer with the statistics kernels). Code defaults override the paper where they differ.

## 1. Summary and contributions
- Densification reorganizes Gaussians into a dense, uniform distribution (blur split + repeated
  depth reinitialization); simplification then keeps ~1/7 of 3DGS's count (intersection
  preserving + importance-weighted stochastic sampling).
- Everything is driven by per-Gaussian statistics of the alpha-blending forward pass over the
  training views: S_i (pixels where Gaussian i has the max blending weight), sum of blending
  weights, number of covered pixels, per-pixel argmax Gaussian and its ray mid-point.
- Two settings: **Mini-Splatting** (densify + simplify, `ms/`) and **Mini-Splatting-D**
  (densify only, `ms_d/`). Mini-Splatting-C (compression) is out of scope.

Contributions:
- **C1 Blur split** (§4.1, Eq. 1): split Gaussians whose max-contribution area is large.
- **C2 Depth reinitialization** (§4.1, App. "Calculation of Gaussian Depth"): ray/ellipsoid
  mid-point depth of the max-weight Gaussian per pixel, back-projected, sampled, re-initialize
  all Gaussians from these points every 5000 iterations during densification.
- **C3 Intersection preserving** (§4.2, Eq. 2): only Gaussians that are argmax for some pixel
  of some training view survive.
- **C4 Importance-weighted sampling** (§4.2, App. "Importance Metric", Eq. for I^2): sample
  Gaussians without replacement with probability ∝ importance; outdoor/indoor metrics.
- **C5 Pipeline schedule** (App. Algorithm 1 + code): SH degree 0 during densification,
  re-initialization from centers at 15k, LR restart, second intersection+prune at 20k.

## 2. Equations

Eq. 1 (blur split):
$$\mathcal{G}^{blur}=\{\mathcal{G}_i \mid S_i > \mathcal{T}_{blur}\},\quad
\mathcal{T}_{blur}=\theta_{blur}\cdot H\cdot W,\quad
S_i=\sum_{\mathbf{x}}\delta(i(\mathbf{x})=i_{max}(\mathbf{x})),\ i_{max}=\arg\max_i w_i$$
with $w_i=\alpha_i\prod_{j<i}(1-\alpha_j)$ the blending weight. Code (`ms/train.py`): per
training iteration `mask_blur |= area_max > H*W/5000` (θ_blur = 2e-4); the mask is OR-ed into
3DGS's split mask (`densify_and_split_mask`: `(grad>=thr & maxscale>pd*extent) | mask_blur`)
at every densification step, then reset.

Gaussian depth (App.): ellipsoid $x^2/s_x^2+y^2/s_y^2+z^2/s_z^2=1$ (code uses 3·s), ray
$\mathbf r(t)=\mathbf o+t\mathbf d$ in the Gaussian frame,
$a=\sum d_k^2/s_k^2,\ b=2\sum o_kd_k/s_k^2,\ c=\sum o_k^2/s_k^2-1$,
$$\mathbf p^{mid}=\mathbf o-\frac{b}{2a}\mathbf d,\qquad t^{mid}=t^{opt}=-\frac{B}{2C}$$
(mid-point = maximum of the Gaussian density along the ray). Per pixel, the depth/point is the
one of the max-weight Gaussian: $d^{mid}=d^{mid}_{i_{max}}$. Code (`render_depthCUDA`):
Gaussians with $t^{mid}<0$ are skipped (no transmittance update); returns `out_pts` (the 3D
mid-point of the argmax Gaussian) and `accum_alpha` = final transmittance T.

Depth reinit (code): for every training view, `prob = (1-T)/sum`, sample
`int(H*W * num_depth/(H*W*n_views))` pixels without replacement (Mini-Splatting-D: with
replacement + unique), points = `out_pts`, colors = **ground-truth pixel colors**; then
`reinitial_pts`: 3-NN scales (simple-knn `distCUDA2`), identity rotation, opacity 0.1,
SH rest 0, fresh optimizer (`training_setup`).

Eq. 2 (intersection): $\mathcal{G}^{int}=\{\mathcal{G}_i \mid i\in\mathcal I_{max}\}$,
$\mathcal I_{max}$ = all argmax indexes over all training views (code: `imp[sum_views area_max==0]=0`).

Importance (App.): $I^1_i=\sum_j w_{ij}$ (indoor);
$$I^2_i=\sum_{m}I^{(m)}_i\,\delta(i\in\mathcal I^{(m)}_{max}),\quad I^{(m)}_i=\sum_j\frac{w^{(m)}_{ij}}{S^{(m)}_i}$$
(outdoor) where in code $S^{(m)}_i$ = `area_proj` = number of pixels the Gaussian contributes
to in view m (not the max-area). Sampling probability $\mathcal P_i=I_i/\sum I$.

## 3. Hyper-parameters

| name | value | source |
|---|---|---|
| iterations | 30000 | code `arguments` |
| θ_blur | 2e-4 (`H*W/5000`) | paper §4.1, code `ms/train.py` |
| densify from / until / every | 500 / 15000 / 100, skipped when `iter % 5000 == 0` | code |
| grad threshold, percent_dense | 0.0002, 0.01 | 3DGS defaults in code |
| prune | opacity < 0.005; after 3000: max scale > 0.1·extent (screen-size test inert, see §7) | code `densify_and_prune_split` |
| opacity reset | **none** (not called in `ms/train.py`) | code |
| depth reinit iterations | 5000, 10000 (`iter%5000==0 and iter<15000`) | code |
| num_depth | 3.5M points total | paper §4.1, code |
| num_max | 4.5M (no densify step when N ≥ num_max; MS only) | code |
| simp_iteration1 | 15000: importance sampling, factor 0.5 of the intersected Gaussians, then reinit from centers | code |
| simp_iteration2 | 20000: intersection + prune lowest-importance Gaussians holding 1% of total importance (`init_cdf_mask(thres=0.99)`) | code |
| imp_metric | outdoor (M360 outdoor, T&T), indoor (M360 indoor, DB) | README |
| SH | degree 0 until 15000; +1 every 1000 after (deg 1 at 16000 … 3 at 18000) | code |
| means LR | 3DGS exp. schedule; for iter ≥ 15000 uses `lr(iter-15000+5000)` | code |
| other LRs, loss | 3DGS (feature 0.0025, /20 rest, opacity 0.05, scale 0.005, rot 0.001; L1 + 0.2 D-SSIM) | code |

## 4. Schedules
1–15000: densify (clone + split ∪ blur) every 100 from 500; depth reinit at 5000 and 10000.
15000: (MS) intersection + importance sampling (keep 0.5·|G^int|), reinit from surviving
centers with their DC colors; (MS-D) reinit from all centers. Optimizer reset, SH max → 3,
means LR restarts at the 5000-iteration value. 20000 (MS): intersection + 1% CDF prune +
optimizer reset. 30000 end.

## 5. Builds on
3DGS official code: same loss, LRs, Adam(eps 1e-15), split/clone, sampling of views from a
shuffled stack, black background for M360/T&T/DB, 3-NN scale init.

## 6. Protocol
Mip-NeRF360 (9 scenes; images_4 outdoor, images_2 indoor), T&T (truck, train), DB (drjohnson,
playroom); 3DGS splits (every 8th test), 30k iterations, RTX 3090, LPIPS-VGG. Per-scene
numbers in the supplementary table; `paper_results.json`.

## 7. Ambiguities / code facts
- Screen-size pruning (`max_radii2D > 20`) is inert: `densification_postfix` (clone and split)
  zeroes `max_radii2D` before the prune test, as in 3DGS.
- The outdoor importance divides by the projected area `area_proj` (pixel count), the paper
  writes $S_i^{(m)}$; code wins.
- Depth points use the GT colors of the sampled pixels (training images).
- Depth kernel uses 3DGS's alpha clamp 0.99; the official stats kernel is part of the normal
  forward pass.
- Densification at 3DGS happens before the optimizer step (gradient of that iteration lost
  for new tensors).
- Supplementary dense-init table reports a slightly different MS M360 average (27.22); Table 1
  (27.34) and the per-scene table agree with each other and are used.
