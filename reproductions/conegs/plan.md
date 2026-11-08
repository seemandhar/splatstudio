# ConeGS native plan

Method `conegs`, module `ss_conegs`, package `splatstudio-method-conegs`. Default budget 1M
(paper's fixed-budget mode, appendix 1M table). `conegs-nobudget` (β = 0.02) is an optional
variant entry point.

## Two stages (EDGS precedent)
1. `ss-conegs-init --source <native scene> --output <copy>` (preprocessing, training views only):
   trains the per-scene iNGP (20k it), samples P_init = 1M uniform training pixels, places
   points at the iNGP median depth with zero-direction iNGP colour, computes exact 3-NN log
   scales, and writes a hard-linked dataset copy (same transforms.json + images ⇒ same
   protocol_id) with `points.npz`, `conegs_init.npz` (points SHA-256, log scales),
   `conegs_ingp.pt` (field weights + normalisation) and `conegs_init.json` (timings: iNGP train
   seconds count toward the method's time).
2. `ss-train conegs --data <copy> --recipe splatstudio --num-gaussians 10000000`.

## Components
| kind | name | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| renderer | conegs-gsplat | `ss_conegs.renderer:create_renderer` | gsplat classic; provides render.rgb, render.depth.expected_z, gradient.screen.signed, `conegs.frame` | `sh_degree: 3` (all bands from step 1) |
| density | conegs-density | `ss_conegs.density:create_density` | requires `conegs.frame` | budget 1e6, growth 0.02, interval 100, densify_until 25000, prune_opacity 0.005, new_opacity 0.1, size_mult 1.0, min_rays 10, added_mult 1.2, camera_extent_lr true |
| loss | photometric | built-in | render.rgb | λ_ssim 0.2 |
| loss | conegs-opacity | `ss_conegs.losses:create_opacity_penalty` | — | λ_o 2e-4, `mean(logits)` (code) |

No new FieldSpecs. Density ops: `gsplat.strategy.ops.remove` for pruning; own `append` via
`_update_param_with_optimizer` (zero Adam moments for new rows; Parameter rebuilt from
`detach()` with the original `requires_grad`).

## iNGP (ss_conegs/ingp.py + hashgrid.py)
- Hash-grid encoding: custom CUDA kernel (JIT `load_inline`), tcnn semantics (scale =
  2^(l·log2 b)·N_min − 1, res = ceil(scale)+1, pos = x·scale+0.5, dense index when res³ ≤ T
  (rounded up to 8), else coherent-prime hash 1/2654435761/805459861, `% size`; params
  U(±1e-4)). Pure-torch reference used in the probe.
- Fields: radiance (16 lv, T=2^19, 16→4096, 64×1 → 16; head SH4 + 15 → 64×2 → 3 sigmoid),
  proposals (5 lv, T=2^17, 16→128 / 16→256, 64×1 → 1), bias-free MLPs (tcnn), trunc_exp(x−1).
- Contraction to unisphere (ℓ2), aabb [−1,1]³.
- Proposal estimator: torch port of nerfacc `importance_sampling` kernel (u = (i+bias)/n,
  bias 0.5 or per-ray U(0,1), upper_bound inverse CDF, midpoint intervals clamped), lindisp
  near 0.2 far 1e6, prop [256, 96] + 48; pdf loss; `get_proposal_requires_grad_fn(5, 1000)`;
  opaque background (last σ = ∞ → 1e10).
- Training: 4096 uniform rays/it over all training pixels, smooth-L1, loss ×1024, Adam(1e-2,
  eps 1e-15, wd 1e-5), LinearLR(0.01, 100) + MultiStep(10k/15k/18k, 0.33).
- Normalisation: nerf-factory `get_gs2nerf_transform` (all camera poses, median scaling).
- Query: inference rays (no stratification, zero view dirs) → first sample with inclusive
  transmittance < 0.5 → point (sample midpoint), colour = volume-rendered pixel colour, std =
  t·(‖d_x−d‖+‖d_y−d‖)/2/‖d‖/scale · size_mult.

## Density schedule (official train.py)
every step 1 ≤ s < 25000: error = mean_c |rgb − I*| (from the renderer's frame passthrough),
multinomial without replacement of n_rays pixels → iNGP → stash. Every s % 100 == 0: prune
opacity ≤ 0.005 (s < 25000), then merge (budget: first `budget − N` stash entries), then
n_rays = max(int(N·g), int(1.2·added if budget))//100 + 1, ≥ 10.

## Probes
| probe | checks | fails if |
|---|---|---|
| conegs-hashgrid (cuda) | CUDA encode == torch reference (fwd + grad), dense vs hashed levels | kernel/indexing wrong |
| conegs-cone-scale | std from pixel rays equals analytic t/(f·‖d‖)·mean offsets at several depths; ×2 fails | footprint sizing removed |
| conegs-median-depth | synthetic σ step profile: picked t is first τ<0.5 sample; colour = rendered colour | depth rule changed |
| conegs-error-sampling | samples concentrate in the high-error region (all-error-in-one-quadrant); uniform fails | error weighting removed |
| conegs-budget | prune+merge keeps N ≤ budget, refills pruned, ray count formula, Adam rows zero & aligned, params trainable | budget/accumulation logic wrong |
| conegs-opacity-penalty | gradient = λ/N on every logit (signed mean), MCMC-style sigmoid differs | penalty removed/changed |
| conegs-proposal (cuda) | importance sampling matches nerfacc docstring example; pdf loss zero when prop = final cdf | sampler wrong |

## Fidelity gaps
1. Runtime view sampling random-with-replacement vs official shuffled stack (runtime; small).
2. White background (runtime convention) vs official black (Mip-NeRF 360: no alpha, negligible
   except in the rendered composite; error map uses the white-composited render).
3. gsplat classic rasterizer vs Inria rasterizer (known calibration offset).
4. iNGP in fp32 PyTorch + own CUDA hash kernel vs tcnn fp16 FullyFusedMLP: different init/
   precision; iNGP quality may differ slightly (paper ablation: 10k vs 20k iNGP iterations ±0.0 dB).
5. 3-NN init scale exact vs simple-knn approximate (`distCUDA2`).
6. Our nerf normalisation uses all poses of the native dataset (same as official, poses only).
7. SSIM valid conv (runtime).
8. Initial-point colours stored in points.npz are clamped to [0,1] (official clamps too).
