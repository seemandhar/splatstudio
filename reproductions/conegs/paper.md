# ConeGS — implementation spec

Baranowski, Esposito, Gschoßmann, Chen, Geiger. *ConeGS: Error-Guided Densification Using Pixel
Cones for Improved Reconstruction With Fewer Primitives.* 3DV 2026, arXiv 2511.06810.
Official code: https://github.com/baranowskibrt/conegs (commit f86df2f, Inria 3DGS licence,
non-commercial research use) — cloned to `source/code/` (read only). It builds on the Inria 3DGS
code, nerfacc (PropNetEstimator) and tiny-cuda-nn hash grids. Reference copies of nerfacc
(57ccfa1) and tiny-cuda-nn were cloned to `source/ref/` to match sampling/hash numerics.

## 1. Method summary
- Train a per-scene iNGP (nerfacc proposal-network variant, tcnn hash grids) for 20k iterations
  on the training images; it is a geometric proxy (median depth along any pixel ray).
- Initialization (C1, §4.1): sample P_init = min(1M, budget) training (image, pixel) pairs
  uniformly; place one Gaussian per pair at the iNGP median depth; colour = iNGP colour rendered
  with a zeroed view direction; isotropic 3-NN scale (Eq. 3); identity rotation; opacity 0.1.
- Error-guided cone densification (C2, §4.2) replaces 3DGS clone/split: every iteration, sample
  N_sample pixels of the current training view without replacement with probability ∝ per-pixel
  L1 error; for each, create a Gaussian on the pixel ray at the iNGP median depth, coloured by
  iNGP, sized by the pixel-cone footprint (C3). Accumulate; every 100 iterations prune
  opacity ≤ 0.005 and merge the accumulated Gaussians (up to the budget).
- Pixel-footprint-aligned isotropic scale (C3, Eq. 11).
- Primitive budgeting (C4, Eq. 12/13): fixed budget (number of rays tied to the number pruned)
  or adaptive growth rate β.
- Pre-activation opacity penalty (C5): λ_o · (opacity logits) with λ_o = 2e-4.
- No opacity reset, no gradient-based densification, all SH bands optimised from step 1.

Contributions: C1 iNGP initialization (§4.1), C2 error-weighted cone densification (§4.2),
C3 pixel-footprint scaling (§4.2), C4 budgeting (§4.2), C5 pre-activation opacity penalty (§4.2).

## 2. Equations (verbatim from `sec/preliminaries.tex`, `sec/method.tex`)
- Cone radius (Eq. 7, Mip-NeRF):
  `r_{\mathrm{cone}}(t) = t \; \frac{\| \mathbf{d}_x - \mathbf{d} \| + \| \mathbf{d}_y - \mathbf{d} \|}{2}`
- Median depth: `t_{\text{med}} = t_k \quad \text{where} \quad \tau_{k-1} > 0.5 \geq \tau_k`;
  centre `\mathbf{p}_j = \mathbf{p}_I + t_{\text{med}} \, \mathbf{d}_I(u, v)`.
- Init scale (Eq. 3): `s_{i} = \frac{1}{3} \sum_{k \in \mathcal{N}_3(i)} \| \mathbf{p}_k - \mathbf{p}_i \|`
  (code: simple-knn `distCUDA2` = sqrt of the mean *squared* 3-NN distance).
- Error sampling: `\{\mathbf{p}_s\} \sim \mathcal{M}(N_{\text{sample}}, E(\mathbf{p})/\sum E)`,
  `E(\mathbf{p}) = |I_j(\mathbf{p}) - I^*(\mathbf{p})|` (code: channel mean of |I−I*|).
- Scale (Eq. 11): `\mathbf{s}_i = \lambda_{\text{scale}} \, r_{\mathrm{cone}}(t_{\mathrm{med}, i}) (1,1,1)`, λ_scale = 2.
- Fixed budget (Eq. 12): `N_{\text{sample}} = \max(0.2 N_{\text{GS}}, 1.2 N_{\text{last}})/100`.
- No budget (Eq. 13): `N_{\text{sample}} = \beta N_{\text{GS}}/100`.
- Opacity penalty: `\mathcal{L}_{\mathbf{o}}^{\text{pre}} = \|\mathbf{o}_{\text{pre}}\|_{1}`, λ_o = 0.0002.
- 3DGS loss (Eq. 4): `(1-\lambda)\text{MAE} + \lambda \mathcal{L}_{D-SSIM}`, λ = 0.2.

## 3. Hyper-parameters (code defaults override paper where they disagree)
| name | value | source |
|---|---|---|
| iterations | 30000 | configs/optimization/defaults.yaml |
| densify (ray casting) window | 0 < it < 25000 | `densify_from_iter: 0`, `densify_until_iter: 25000`, train.py |
| merge interval | 100 | train.py (`iteration % 100 == 0`) |
| prune | opacity ≤ 0.005 every 100 it, only it < 25000 | train.py |
| merge after 25000 | merge still runs (stash from 24901–24999 merged at 25000) | train.py |
| N_sample per iter | `max(int(N·g), int(1.2·N_added))//100 + 1`, min 10 | utils/nerf_utils.py:get_num_rays_to_cast |
| g (`gaussian_percentage_increase`) | **0.02** (paper Eq. 12 says 0.2) | configs/optimization/defaults.yaml |
| N_added term | only when a budget is set | train.py |
| budget truncation | stash truncated to its first `budget − N` entries (chronological) | gaussian_model.py:initialize_new_points |
| new opacity | 0.1 | `opacity_init_value` |
| new scale | std = 1.0 · pixel width at t (code `densification_gaussian_size: 1.0`; Mip-NeRF "radius" = mean neighbour offset = one pixel width, so paper λ_scale=2·r ↔ code 1.0·(dx+dy)/2) | nerf_model.py:select_for_densification, ray_utils.py |
| new SH | DC = RGB2SH(iNGP pixel colour, zero view dir), rest 0 (DC not clamped) | nerf_model.py |
| new rotation | identity | |
| P_init | 1,000,000 (or budget if smaller) | `num_gaussians_initialized` |
| init selection | uniform multinomial over ≥P_init stashed uniform-random pixels | nerf_model.py:pretrain_model |
| init colour clamp | SH DC colour clamped to [0,1] | nerf_model.py |
| opacity penalty | `0.0002 * _opacity.mean()` (signed mean of logits; no abs) | train.py:110-111 |
| SH | all 3 bands from step 1 (`start_sh_degree=3`) | gaussian_model.py:__init__ |
| LRs | 3DGS: means 1.6e-4·extent→1.6e-6·extent, sh0 2.5e-3, shN 1.25e-4, opacity 0.05, scale 5e-3, rot 1e-3 | optimization defaults |
| background | black | gaussian_model defaults |
| rasterizer | Inria diff-gaussian-rasterization (dr_aa branch), antialiasing=False | gaussian_renderer |
| view sampling | shuffled stack (each view once per epoch) | train.py |
| **iNGP** pretraining iters | 20000 | `num_iters_pretrain` |
| rays per iteration | 4096, uniform over all training pixels (stacked images) | `pretraining_num_rays`, `stack_images` |
| loss | smooth_l1 (β=1) on rgb, ×1024 loss scale (GradScaler not unscaled) | nerf_model.py |
| optimizer | Adam lr 1e-2, betas (0.9,0.999), eps 1e-15, wd 1e-5; same for proposal nets | configs/nerf_model |
| LR schedule | LinearLR 0.01→1 over 100 it, MultiStep ×0.33 at 10k/15k/18k | nerf_model.py |
| scene normalisation | nerf-factory: rotate world-up to +z, recentre at median camera-ray focus, scale 1/median cam distance (all cameras' poses) | nerf_utils.get_gs2nerf_transform |
| field | unbounded, contraction to unisphere (ℓ2), aabb [−1,1]³ | ngp.py |
| radiance grid | 16 levels × 2 feat, log2 T=19, res 16→4096, MLP 64×1 → 1+15; head SH4 dir enc + 15 → 64×2 → 3 sigmoid; density trunc_exp(x−1) | ngp.py |
| proposal nets | 5 levels, res 16→128 and 16→256, log2 T=17, MLP 64×1 → 1 | nerf_model.py |
| samples | prop [256, 96], final 48, near 0.2, far 1e6, lindisp, opaque background (last σ=∞) | nerfacc estimator params |
| proposal update | nerfacc `get_proposal_requires_grad_fn(5, 1000)`, pdf loss ×1024 | nerfacc |
| stratified | train yes, inference (init + densification) no | nerf_model.py |

## 4. Schedules
- 0 < it < 25000: per-iteration error-guided ray casting; prune+merge at multiples of 100.
- it = 25000: last merge; no pruning from 25000 on (prune gated by `< densify_until_iter`).
- opacity penalty all 30k iterations. Means LR exponential decay over 30k.
- iNGP: 20k iterations before 3DGS training; then frozen (`nerf_train_during_3dgs: false`).

## 5. Builds on
Inria 3DGS training loop (losses, LRs, rasterizer) with its clone/split/opacity-reset replaced.

## 6. Protocol
Mip-NeRF 360 (7 public scenes: bicycle, garden, stump, room, counter, kitchen, bonsai — no
flowers/treehill), `images_4` outdoor / `images_2` indoor (README + full_eval.py), llffhold-8
(`eval: true`, 3DGS dataset reader). OMMO, T&T (original release, full resolution), DB
(original release; the 3DGS `tandt_db.zip` is a different copy — README warns results differ).
Metrics PSNR/SSIM/LPIPS (Inria metrics.py, lpipsPyTorch VGG). Budgets 100k/500k/1M/2M (fixed) and
β ∈ {0, 0.01, 0.02, 0.04} (no budget). Per-scene tables in the appendix (Tables 6–9).

## 7. Ambiguities
- Eq. 12 says 0.2·N_GS, code default 0.02 — code used.
- Paper calls the penalty an L1 norm ‖o_pre‖₁; code uses the signed mean of logits — code used
  (with Adam the penalty gradient is a constant +λ/N per Gaussian either way for negative logits;
  they differ only for logits > 0).
- λ_scale = 2 × radius in paper vs `densification_gaussian_size: 1.0` in code; both yield a std of
  ≈ one pixel width (see table). Code used.
- Paper: "every 100 iterations ... pruned"; code prunes only before 25k.
- Table→budget mapping: appendix tables follow their captions: lines 449–495 = 100k (avg 27.74,
  matches main Table 1), 500–544 = 500k (29.08, matches Table 1), 549–592 = 1M (29.37, matches
  Table 2 timing table "capped at 1M"), 598–642 = 2M. The scout's bicycle 25.43 / garden 27.59 is
  the **2M** row; the 1M row is bicycle 25.45 / garden 27.42 / room 32.30.
