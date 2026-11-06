# Plan — Laplacian-GS native port

Method `laplacian-gs` (module `ss_laplacian_gs`); smoke-only variant `laplacian-gs-smoke`
(iteration plan 1000/1000, densify until 900 per stage — never benchmarked).
Backbone: the official `--backbone taming` path (Ours(Taming)) mapped onto gsplat; it is the
3DGS-equivalent backbone (DashGaussian schedulers off, opacity-only prune, Sparse Adam).

## Representation: one population, a stage field
All levels live in the single native ParameterDict. Field `lap_level` (shape (), initial
L−1, lr 1e-12, never in the graph) stores the pyramid level ℓ a Gaussian belongs to
(L−1 = low-frequency base, 0 = finest residual). gsplat ops copy it on clone/split, so it
survives densification; the checkpoint carries it, so evaluation composes all levels.

| field | shape | lr | init | density treatment |
|---|---|---|---|---|
| lap_level | () | 1e-12 (no grad) | L−1 | copied by duplicate/split; new stage rows get ℓ |

## Components
| component | kind | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| lap-renderer | renderer | ss_laplacian_gs.renderer:create_renderer | provides render.rgb, render.depth.expected_z, gradient.screen.signed, render.laplacian_level | levels 2, iter_plan [10000,20000], blur_up true, base_background white |
| lap-density | density | ss_laplacian_gs.density:create_density | requires gradient.screen.signed | levels, iter_plan, inherit_ratio 1.0, grad 2e-4, percent_dense 0.01, prune_opa 0.005, densify_from 500, every 100, until 15000·T/30000, reset 3000 → 0.01, LRs (opacity .025, sh0 .0025, shN .005/20, scales .005, quats .001), lr_final_factor 0.01 |
| lap-photometric | loss | ss_laplacian_gs.losses:create_loss | requires render.rgb, render.laplacian_level | ssim_weight 0.2, ssim_padding same, levels |

Renderer ↔ density link (as in the DashGaussian port): the density registers itself; the
training render asks for the active level and the stage-local step.

### Renderer (C1)
- Pyramid sizes from the camera: H_{ℓ+1} = ⌊(H_ℓ−1)/2⌋+1 (stride-2 5×5 conv, reflect pad 2).
  Level-ℓ camera = same viewmat, K rows scaled by (w_ℓ/W, h_ℓ/H) (official `render_size`).
- Base (ℓ=L−1): gsplat SH colours (+0.5, clamp ≥0), RGB over background, clamp [0,1] (Eq. 5).
- Residual (ℓ<L−1): colours = gsplat `spherical_harmonics` of (means − campos) with **no**
  +0.5 and no clamp (signed `_pn` rasterizer), background 0, render clamped to [−1,1].
- Training (grad enabled), active level ℓ: only Gaussians with lap_level==ℓ are rendered with
  grad; for ℓ<L−1 the cache Up(Î^{ℓ+1}) is composed under no_grad from the archived levels
  (frozen ⇒ identical to the official per-view cache) and rgb = cache + R̂^ℓ (Eq. 7).
  SH degree = min(⌊i/1000⌋,3) with i the stage-local step (new field restarts at 0).
- Evaluation (no_grad): Î^{L−1} = base render; Î^ℓ = Up(Î^{ℓ+1}) + R̂^ℓ down to ℓ=0; per-level
  SH degree = final stage degree min(⌊T_ℓ/1000⌋,3).
- Up = bilinear (align_corners False) + 5×5 binomial blur (blur_up).

### Density (C2, C3, backbone)
- Per stage s (level ℓ_s, T_s iterations, start step S_s): 3DGS/DashGaussian ADC restricted
  to active Gaussians: accumulate ‖∇means2d‖ (×W/2, ×H/2) of active visible Gaussians while
  i < until_s; at i>500, i%100==0: prune active with σ<0.005, then clone (small, ≤0.01·extent)
  / split (large) active with mean grad ≥ 2e-4 (top-k cap of rate 1 = unbounded); opacity
  reset of active rows to min(σ,0.01) with zeroed moments at i%3000==0, i<until_s.
- Means LR per stage: lr_init·0.01^{i/T_s} (restart at every stage), set for the next step.
- Archive at i == T_s (s < L−1): all current rows become archived (no longer rendered with
  grad, never touched by ADC); new rows = xyz of the archived active field (equidistant subset,
  ratio ρ), log 3-NN scale (cKDTree, mean of 3 squared distances, clamp 1e-7), identity quat,
  opacity 0.1, SH 0, lap_level ℓ−1. All Adam moments zeroed and `step` reset to 0 = fresh
  optimizer (archived rows then get exactly zero updates since their gradients are 0).
- Initial LRs replaced by the official values in `initialize`.

### Loss (C2)
L = 0.8·L1(Î^ℓ, I^ℓ) + 0.2·(1 − SSIM_same(Î^ℓ, I^ℓ)), I^ℓ = Down^ℓ(GT) — reconstructed-image
supervision, not residual fitting.

## Probes
| probe | cuda | fails if |
|---|---|---|
| lap-pyramid | no | Down/Up differ from an independent numpy reflect-pad loop implementation, size rule wrong, or blur_up dropped |
| lap-signed-residual | yes | residual render keeps the +0.5 shift / clamps negatives (checked R = B − 0.5·α on a single Gaussian with negative SH) |
| lap-composition | yes | eval ≠ Up(base@level1)+residual@level0 built from raw gsplat calls; training at level 0 gives archived Gaussians nonzero grad or active zero grad |
| lap-staging | no | at the stage boundary: wrong count (ρ·N), not xyz-only init, moments/step not reset, LR not restarted, archived rows move after an Adam step, ADC touches archived rows |
| lap-loss | no | loss is not L1+0.2 D-SSIM against Down^ℓ(GT) (e.g. residual supervision or full-res target) |
| renderer-gradient, state-transitions | built-in | runtime contract |

## Unchanged from 3DGS / runtime
SfM init, recipe 3dgs (camera extent, 3-NN init scales), llffhold 8, 30K total steps, Adam.

## Files
`plugin/pyproject.toml`, `plugin/ss_laplacian_gs/{__init__,pyramid,link,renderer,density,losses,probes}.py`,
`plugin/README.md`, `plugin/tests/test_laplacian_gs.py`.

## Benchmark
Per-scene numbers exist (supp. Tab. 9–11). Protocol "1K" = width-1600 → dataset
`mipnerf360-1600`, scenes garden (outdoor) + counter (indoor), both in
`baseline3dgs-mipnerf360-1600-30000`. Targets Ours(Taming): garden 27.35, counter 29.22
(paper 3DGS 27.36 / 29.17; Taming 27.48 / 29.43). Ours(FastGS) not attempted (FastGS backbone
absent).

## Fidelity gaps
1. Backbone: gsplat rasterizer + dense Adam instead of Taming fast backward + SparseGaussianAdam
   (Sparse Adam updates only visible Gaussians). Expected ±0.1 dB; closable by sparse Adam.
2. Archived field stays in GPU memory / in the optimizer (zero updates) and the cache is
   re-rendered every step instead of stored ⇒ speed and memory claims not reproduced; PSNR
   unaffected (frozen ⇒ identical cache).
3. White background for the base render (runtime convention) vs black in code. Negligible on
   unbounded scenes.
4. Random view sampling with replacement (runtime) vs shuffled stack.
5. Official last iteration of each stage skips the optimizer step; we step. Negligible.
6. 3-NN init via cKDTree vs simple-knn distCUDA2 (same quantity, exact KNN vs approximate
   grid KNN). Negligible.
7. FastGS backbone (Ours(FastGS)) not ported.
