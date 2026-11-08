# Plan — MH-3DGS native port

Method name `mh-3dgs`, module `ss_mh_3dgs`; control method `mh-3dgs-mcmc-control`
(gsplat MCMCStrategy at the paper's per-scene MH-3DGS budget + the same regularisers; the paper's
3DGS-MCMC row is at that budget). Port follows the official code (commit 71ef8f9) where it
disagrees with the paper (see paper.md §7).

## Components
| component | kind | factory | requires/provides | config (defaults = official code) |
|---|---|---|---|---|
| mh-raster | renderer | ss_mh_3dgs.renderer:create_renderer | provides render.rgb, render.depth.expected_z, gradient.screen.signed | background [0,0,0] (official full_eval: black) |
| mh-density | density | ss_mh_3dgs.density:create_density | requires {} | densify_from 500, until 15000, interval 100, coarse 10→5, fine 2→1, voxel 0.02→0.005, batches 4500/16000, lambda_v 5, weights 0.8/0.5/0.5, accept_shift 0.5, opacity_threshold 0.005, opacity_lr 0.025, pixel_lookup "world_xy", ssim_kind "similarity", shuffle_cameras true, seed 0, data/downscale (argv fallback) |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim_weight 0.2 |
| mcmc-regularizers | loss | ss_mh_3dgs.losses:create_regularizers | state.gaussian3d | opacity 0.01, scale 0.01 |

No new FieldSpecs. Density keeps a `denom` visibility counter (official `add_densification_stats`
on densification iterations, reset on population append).

## Step-by-step (density.after_backward at step t)
1. if 500<t<15000 and t%100==0: denom[visible]+=1 (visible = radii>0 of this step's render).
2. p=(t−500)/14500; k=max(int(n(1−p)),1); views = round robin start (t·k)%n over the (seeded
   shuffled) train list; render each (no grad), average SSIM-similarity map (11×11 σ1.5, same
   padding, CHW) and |I−Î| map.
3. relocation: dead = σ(opa)≤0.005 → gsplat.strategy.ops.relocate (empty state dict).
4. importance I = σ(0.8 z(σ(opa)) + 0.5 z(ssim@px) + 0.5 z(l1@px)), px=(int(x·(W−1)),int(y·(H−1))) clamped.
5. voxel counts c over pre-step means (exact via torch.unique(dim=0) on floor(x/v) float64).
6. coarse/fine: idx=unique(multinomial(I,B,repl)); std=clamp(s·exp(scales)[idx]/(1+ln(1+denom[idx])),1e-6);
   x'=x+N(0,std); ρ=σ(I[idx]−0.5)/(1+5c(x')); accept U<ρ.
7. append accepted copies (raw params, means=x'), zero Adam moments; denom←0.
8. assert every param is a trainable leaf (requires_grad pitfall).
initialize(): set opacity lr 0.025. set_training_cameras(): reload images (improving-densification /
mvgs precedent: splatstudio.data.load_dataset on --data/--downscale), keep on CPU (pinned), verify
camera alignment.

## Probes
| probe | measures | fails if |
|---|---|---|
| importance-formula | plugin importance vs independent numpy formula on random state + maps (world-xy lookup) | weights/normalisation/lookup changed |
| voxel-acceptance | empirical acceptance in an empty vs c-crowded voxel vs σ(I−0.5)/(1+λ_v c) | crowding prior or acceptance removed |
| birth-proposals | one MH step on fixture: children copy parents except means, offset std ≈ s·scale·denom factor, zero moments, trainable leaves | proposal mechanism wrong |
| relocation | dead Gaussians revived (none ≤ τ), counts preserved, matches MCMC formula | relocation removed |
| schedules | k_t, round-robin start, scales/voxel/batch at several steps; no event outside window | anneals wrong |
| multiview-maps (CUDA) | aggregated maps = independent per-view render+SSIM average | maps not multi-view |
| regularizers | gradient of reg loss = analytic 0.01·σ'(o)/N, 0.01·exp(s)/(3N) | weights wrong |

## Fidelity gaps
1. Renderer gsplat classic vs Inria/MCMC rasterizer (numerically near-identical forward; gsplat drops
   grads at α>0.999) — small.
2. View sampling with replacement vs shuffled stack — small (runtime TODO).
3. SSIM in the photometric loss uses valid conv (runtime) vs same padding — small.
4. Error maps rendered after the optimizer step of iteration t (official: before) — negligible.
5. Round-robin over our seeded shuffle, not the identical 3DGS shuffle — negligible statistically.
6. SH degree of the *training* render increases at step 1001 vs 1000 (runtime); the error-map
   renders use the official `step // 1000` — negligible.
7. LPIPS lpips-package VGG vs lpipsPyTorch — metric only.
8. RNG streams differ (stochastic sampler; paper single run) — ±0.1–0.2 dB expected.
9. Optimizer step on densification iterations (review m1): official computes maps / relocates /
   appends *before* `optimizer.step()`; because those ops rebuild every nn.Parameter, `.grad` is
   None and the official Adam step is effectively skipped on all ~144 densification iterations.
   The runtime steps Adam before `after_backward`, so we take those 144 steps — small effect.
10. Clamp (review m2): error maps now clamp the render to [0,1] as official `render()` does; the
   training loss uses the runtime's unclamped rgb (runtime-wide gap shared with the gsplat baseline).
11. Control (review m5): `mh-3dgs-mcmc-control` caps at the *paper's* MH-3DGS count (matches the
   paper's same-budget row); report our MH run's final count next to it. Otherwise gsplat
   MCMCStrategy defaults (noise_lr 5e5, refine 500->25000 every 100, +5%/event, min_opacity
   0.005) + same regularisers, black background, runtime init opacity 0.1 (3DGS-MCMC uses 0.5)
   and runtime opacity lr 0.05.
