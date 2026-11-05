# Plan — ImprovedGS (`improving-densification`)

Method name **`improving-densification`**, module **`ss_improving_densification`**.
Port of the official code (source/code, ImprovedGS licence, non-commercial research) with the
official modified Inria rasterizer vendored unmodified (`csrc/`, JIT-built).

## Components
| component | kind | factory | requires | provides | config (paper/code values) |
|---|---|---|---|---|---|
| `improvedgs-raster` | renderer | `renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.inverse, gradient.screen.inria_abs | background [0,0,0] (official black) |
| `improvedgs-density` | density | `density:create_density` | gradient.screen.inria_abs | – | see density DEFAULTS (all of paper.md §3) |
| `photometric` | loss | splatstudio base | render.rgb | – | ssim_weight 0.2 (unchanged) |

No custom FieldSpecs. Per-Gaussian density statistics (`accum_abs`, `denom`) live in the density
state dict and are sliced/reset together with the population.

## Contribution → code
- **C1 EAS** (`core.edge_map`, `core.minmax01`, `density._eas_scores`): edge maps computed once
  from the training images (reloaded from `ss-train --data`, mvgs precedent); per densification
  render the sampled views with `pixel_weights = edge map` through the vendored kernel
  (forward.cu:416 `accum += ω·T·α`), min-max normalise per view, add for radii>0, divide by
  #views. View pool: shuffled train list, pop 10 from the end, refill when empty (exact code
  semantics incl. the short last batch); all views at it ∈ {3400, 6400}. Candidates =
  avg |grad| (means2D grad channels 2:4) ≥ 0.0003; split set = `torch.multinomial(score·qual,
  k, replacement=False)`.
- **C2 LAS** (`core.long_axis_children`, `density._long_axis_split`): ±3·d·s_max along the
  rotated longest axis, long scale ×(1−d), short scales ×√(1−d²), opacity ×0.6, parent
  removed, children moments 0 (`gsplat.strategy.ops._update_param_with_optimizer`). No clone,
  no 3DGS split, no size pruning.
- **C3 RAP** (`density._rap`): prune op<0.02 at 300; reset to ≤0.05 at 3000/6000/9000/12000
  (moments zeroed, gsplat `reset_opa`); remove op < quantile₀.₂ at 3300/6300.
- **C4 MU** (`optim.GatedAdam`, `core.mu_update`): density.initialize swaps every field's Adam
  for a `GatedAdam` (same state/param groups) whose `step()`/`zero_grad()` are no-ops unless
  `core.mu_update(step)` → gradients of skipped iterations sum, exactly like the official loop.
  Also reproduces "no update at the final iteration".
- **C5 GC** (`core.budget`): `int(sqrt(clamp((it−500)/14000,0,1))·N_max)`, N_max per scene
  (scene name from `--data` basename; table = paper Sec. 4.2; config `budget` overrides).
- **impl.** (`density`): position lr 4e-5→2e-6 × extent exponential (overrides runtime schedule
  each step, atomgs precedent); opacity lr 0.025; prune op<0.005 at densify steps < 14900;
  stats window 500<it<15000.

## Probes (each fails if its mechanism is removed)
| probe | contribution | check |
|---|---|---|
| `raster-vendored` (cuda) | renderer | vendored kernel RGB ≈ gsplat classic RGB on a random scene; grad channels 2:4 ≥ |0:2|, nonzero |
| `eas-score` (cuda) | C1 | accum weights = brute-force Σ ω·T·α (independent per-pixel loop) on a tiny scene; edge-weighted score prefers the Gaussian on the edge; `edge_map` == PIL FIND_EDGES-style reference |
| `eas-sampling` | C1 | multinomial split set ⊂ qualifiers and frequency ∝ score; view pool sequence (10,…,10,1) and full set at 3400 |
| `las-geometry` | C2 | children at ±0.45·L0 along the longest world axis, scales (1−d, √(1−d²)), opacity 0.6×, count N+k, moments zero |
| `growth-control` | C5 | budget curve vs independent √ formula; densify never exceeds budget |
| `rap-schedule` | C3 | resets at 3000…12000 to 0.05, 20 % prune exactly at 3300/6300 only, 300 early prune |
| `mu-gated-adam` | C4 | params unchanged on skipped steps, update on k·5 / k·20 equals Adam on summed grads |
| `lr-schedule` | impl | means lr = 4e-5·e·(0.05)^(t) at several steps; opacity lr 0.025 |

## Files
`plugin/pyproject.toml`, `plugin/README.md`, `plugin/ss_improving_densification/{__init__,
core,raster,renderer,density,optim,probes}.py`, `plugin/ss_improving_densification/csrc/`
(vendored), `plugin/tests/test_improving_densification.py`.

## Unchanged from 3DGS / runtime
L1 + 0.2 D-SSIM (runtime SSIM, valid conv), SH +1/1000 its, SfM init with 3-NN scales
(`--recipe 3dgs`), initial opacity 0.1, scale/rot/SH lrs.

## Fidelity gaps
1. View sampling with replacement (runtime `random.choice`) vs official shuffled stack — small,
   could be closed only by a runtime option.
2. SSIM valid convolution vs fused_ssim same padding — small, global runtime difference.
3. Official CUDA Adam ("ours_adam") vs torch Adam: with all-visible mask it is dense Adam with
   the same β/eps; step-count bootstrap differs only for parameters created after step 1 (torch
   Adam keeps one step counter per field, official too) — negligible.
4. Training renders pass empty `pixel_weights` (official passes ones → only fills an unused
   accumulator); outputs identical, faster.
5. Data resolution ambiguity: official reads `images` with `-r -1` (1600 px cap) from an
   uninspectable data package; paper says "as with 3DGS" → we use images_4/images_2. If wrong,
   outdoor PSNR shifts (cf. our 1600 baseline). Can be re-run with `mipnerf360-1600`.
6. MU schedule: code 20000/24000 used (paper text 15000/22500). Variant easy via config.
7. Paper numbers are best-of-3 runs; we run one seed (expect ~0.05–0.1 dB lower).
8. SH degree switch at step 1001 vs official 1000 (runtime) — negligible.
