# Mini-Splatting2 — implementation spec

**Paper**: Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification,
Guangchi Fang & Bing Wang (PolyU), arXiv **2411.12788v1** (19 Nov 2024). Pinned to v1: arXiv v2 is
the later journal version ("Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D
Gaussians", sources in `source/latex_v2/`), which is not reproduced here.
**Official code**: https://github.com/fatPeter/mini-splatting2 @ `5ec9202` (2025-02-01), in
`source/code/` (`msv2/train.py`, `msv2_d/train.py`, `scene/gaussian_model.py`, `config/fast`,
`submodules/diff-gaussian-rasterization_ms`). Built on Mini-Splatting (#28) + Taming-3DGS ("3DGS-accel").

## 1. Method summary
- Mini-Splatting pipeline (blur split, depth reinitialisation, intersection preserving, importance
  sampling simplification, SH disabled during densification) compressed to **18k iterations**, with
  densification ending at **3k** instead of 15k.
- **Aggressive densification**: every 250 steps from step 500, *critical Gaussians* (max-blending-weight
  Gaussians, 0.99 CDF of importance) are **cloned** with the opacity/covariance-preserving clone of
  3DGS-MCMC; one **depth reinitialisation at 2k** with as many depth points as there are Gaussians.
- **Visibility culling**: per training view, Gaussians outside the 0.99 CDF of that view's blending
  weights are skipped in rasterisation (steps 500–13k); per-view masks are recomputed at every
  aggressive-clone and simplification pass.
- Training at **half resolution until 3k**, sparse (visibility-masked) Adam from Taming-3DGS.

Contributions:
- **C1 Critical Gaussian identification** (§4 "Critical Gaussian Identification").
- **C2 Aggressive Gaussian clone** (§4 "Aggressive Gaussian Clone").
- **C3 Aggressive densification pipeline**: C1+C2 every 250 its from 500, depth reinit at 2k with
  num_depth = N, densify-until 3k, simplification at 3k/8k, 18k total (§4 "Overall pipeline").
- **C4 Visibility Gaussian culling** with precomputed per-view visibility (§5).
- (Implementation, §6.1) SH disabled + 0.5× resolution during densification; 3DGS-accel (sparse Adam).
- Variant **MSv2-D**: simplification replaced by importance pruning.

## 2. Equations (v1 LaTeX)
- Rendering (Eq. 1): `\bc(\bx)=\sum_i w_i \bc_i, w_i = T_i \alpha_i \cG^{2D}_i(\bx)`, `T_i=\prod_{j<i}(1-\alpha_j\cG^{2D}_j(\bx))`.
- Depth (Eq. 2): `d(\bx)=d^{mid}_{i_{max}}(\bx), i_{max}=\arg\max_i w_i` — mid-point of the ray /
  ellipsoid intersection of the max-weight Gaussian (Mini-Splatting appendix).
- Critical Gaussians (Eq. 3): `\bc(\bx) \approx w_{i_{max}} \bc_{i_{max}}`; G^crit = {G_imax} filtered by
  the 0.99 quantile (CDF) function.
  **Code** (`culling_with_clone`): `imp = Σ_views accum_weights` (Σ_j w_ij), `imp[Σ_views area_max == 0] = 0`
  (never an argmax ⇒ not critical), `clone_mask = init_cdf_mask(imp, 0.99)`.
- Aggressive clone (3DGS-MCMC with N=2): `p_new = p_old`, `α_new = 1 − sqrt(1 − α_old)`,
  `Σ_new = α_old² (2α_new − α_new²/√2)^{-2} Σ_old`. Code applies it to the scale:
  `s_new = α_old / (2α_new − 0.5^0.5 α_new²) · s_old`, then clamps `α_new ∈ [0.0051, 1−eps]` (after the
  scale is computed). The source Gaussian is overwritten with (α_new, s_new) and one copy appended
  (fresh zero Adam moments for the copy, the source's moments kept).
- Visibility (Eq. 4): `I_i^k = Σ_j w_ij^k`, `V_i^k = 1(I_i^k > τ)`, τ from the 0.99 CDF.
  Code: `visibility = init_cdf_mask(accum_weights_k, 0.99)`; `culling[:,k] = ~visibility`.
- `init_cdf_mask(imp, thres)`: sort `imp+1e-6` ascending, cumulative sum / total, first index with
  ratio > 1 − thres gives value v; keep `imp > v`.
- Culling pass side effects (code): `count_rad_i = #views with radii>0`, `count_vis_i = #views visible`,
  `factor_culling = count_vis / (count_rad + 0.1)`; prune `count_vis ≤ 1` OR outside a CDF keep mask
  (clone pass: 0.999 CDF of Σ w; simp1: not importance-sampled; simp2: 0.99 CDF of importance).
- Gradient statistic under culling: `grad_accum += ||∇_{xy}|| · factor_culling` when the current view's
  culling column is non-empty (`add_densification_stats_culling`), else plain 3DGS.
- Importance metric (Mini-Splatting): outdoor `I² = Σ_k w/area_proj` (only views where area_max ≠ 0),
  indoor `I¹ = Σ_k Σ_j w`; `imp[Σ area_max == 0] = 0`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 18000 | config/fast; §4 |
| densify_from / interval / until | 500 / 100 / 3000 (skip 2000) | arguments; config/fast; train.py |
| grad threshold, percent_dense, min opacity | 2e-4, 0.01, 0.005 | arguments; train.py |
| size_threshold (screen/world big prune) | inert (only if it > 3000, but densify stops at 3000) | train.py |
| blur split θ | area_max > H·W/5000 | train.py (`mask_blur`) |
| aggressive clone from / every | 500 / 250, only while it < 3000, not at 2000 | config/fast |
| depth reinit iteration | 2000, num_depth = N·1.0 (before interesction_preserving) | config/fast, `num_depth_factor` |
| depth sampling | per view int(H·W·num_depth/(H·W·V)) pixels ∝ (1−T), without replacement | gaussian_model.depth_reinit |
| interesction_preserving before reinit | 0.99 CDF of importance (imp_metric) | train.py:2000 |
| reinit | 3-NN (simple-knn) scales, identity rot, opacity 0.1, GT colours, fresh Adam | reinitial_pts |
| simp_iteration1 | 3000: culling + importance sampling, factor 0.6 (MSv2) / 0.99 CDF prune of Σw (MSv2-D) | config/fast; train.py |
| simp_iteration2 | 8000: culling + 0.99 CDF of importance (MSv2) / of Σw (MSv2-D) | config/fast |
| culling active | from first clone pass (500) until (8000+18000)//2 = 13000; reset to none at 2000 (reinit) and 13000 | train.py |
| SH | degree 0 until 3000; max 3 from 3000, +1 at 4000, 5000, 6000 | train.py |
| warm-up resolution | 0.5× (train cameras scale 2) for it ≤ 3000, incl. all statistic passes ≤ 3000 | `getTrainCameras_warn_up` |
| position LR | 1.6e-4·extent → 1.6e-6·extent, **max_steps 30000** (not 18000) | arguments (not overridden) |
| opacity LR | **0.025** (0.05/2) | arguments |
| feature / shN / scaling / rotation LR | 2.5e-3 / 1.25e-4 / 5e-3 / 1e-3 | arguments |
| optimizer | SparseGaussianAdam (update only radii>0; bias-corrected lr), eps 1e-15; re-created at 2000 and 3000 | gaussian_model.training_setup |
| loss | (1−0.2)·L1 + 0.2·(1−fused SSIM) | train.py |
| opacity reset | **none** (never called) | train.py |
| imp_metric | outdoor: M360 outdoor + T&T; indoor: M360 indoor + DB | README |

## 4. Schedules (msv2/train.py, per iteration `it`)
1. LR(it); SH +1 if it%1000==0 and it>3000.
2. Render with culling column of the view; loss; backward.
3. If it < 3000: max-radii, grad stats (×factor_culling when culled), blur mask; every 100 (it>500,
   it≠2000): densify_and_prune_mask (clone small, split large∪blur, prune α<0.005); at 2000:
   interesction_preserving → depth_reinit → new optimizer, culling cleared; at it≥500, it%250==0,
   it≠2000: culling_with_clone (visibility masks + prune + aggressive clone).
4. it==3000: culling_with_interesction_sampling, enable SH 3 (zero rest), new optimizer.
5. it==8000: culling_with_interesction_preserving. it==13000: culling cleared.
6. Sparse Adam step (visible = radii>0); skipped implicitly on densification steps (new params have no grad).

## 5. Builds on
Mini-Splatting (#28) inside 3DGS + Taming-3DGS. Kept from 3DGS: L1+D-SSIM, 3DGS clone/split rule,
SfM initialisation (3-NN scales, opacity 0.1), exponential means LR.

## 6. Protocol
3DGS protocol: Mip-NeRF360 (outdoor images_4, indoor images_2), T&T (truck, train), DB (drjohnson,
playroom), llffhold 8, black background (code default), metrics from 3DGS (`lpipsPyTorch` VGG). Single
RTX 3090; time = wall-clock of `training()` incl. stat passes. Per-scene numbers in appendix table
(`tables/appendix/quant_all_arxiv.tex`). No seeds stated.

## 7. Ambiguities / paper ↔ code
- Critical identification: paper says max blending weight w_imax with a 0.99 quantile; code uses
  "has been argmax in some pixel" ∧ 0.99 CDF of **summed** weights (appendix Table "w_imax" row ≈ this).
- Clone covariance in paper is Σ ∝ (...)^{-2}; code applies the unsquared ratio to scales (equivalent).
- Opacity clamp min 0.0051 applied after the scale formula (code).
- Paper says culling "between 500 and 13K": code also clears culling at 2000 (reinit) until 2250 and the
  masks are reset (all visible) during each stat pass.
- Interesction_preserving before depth reinit (code only; comment says it does not affect rendering much).
- Position LR schedule uses max_steps 30000 with 18000 iterations (code default not overridden).
- Viewpoint stack refills only when empty: the first epoch after 3000 continues with half-res views.
- `num_depth` uses N before interesction_preserving.
