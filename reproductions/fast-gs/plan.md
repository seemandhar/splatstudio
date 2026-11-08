# FastGS native plan

Method names: `fast-gs` (paper main row, densify every 500), `fast-gs-big` (every 100);
`fast-gs-tdb` / `fast-gs-big-tdb` = same with CB β=0.7 (official T&T / Deep Blending setting).
Module `ss_fast_gs`. Benchmark: `fast-gs` on `mipnerf360-1600` garden (outdoor) + counter
(indoor) vs `baseline3dgs-mipnerf360-1600-30000`.

## Components
| kind | name | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| renderer | fastgs-renderer | `ss_fast_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, gradient.screen.signed, gradient.screen.fastgs4, training.target_image | background [0,0,0], mult 0.5 |
| density | fastgs-density | `ss_fast_gs.density:create_density` | gradient.screen.signed, gradient.screen.fastgs4, training.target_image | see density DEFAULTS (= paper.md §3) + per-scene table |
| loss | fastgs-photometric | `ss_fast_gs.losses:create_photometric` | render.rgb | λ=0.2, same-padded D-SSIM (fused_ssim) |

Renderer = the **official FastGS CUDA rasterizer vendored** (`csrc/`, from
`submodules/diff-gaussian-rasterization_fastgs` @ 44e02a5; Inria licence): CB (SnugBox+AccuTile
with β=`mult`), 4-column screen gradient (signed xy + AbsGS abs xy), per-Gaussian high-error
pixel counter (`metric_map`). Camera conversion follows the speedy-splat port (`raster.py`).
It also forwards `target_image`/`camera` (Taming-3DGS port pattern) so the density can cache
training views for multi-view scoring.

Density (`density.py`), after the runtime's optimizer step:
- s<15000: accumulate ‖g[:, :2]‖, ‖g[:, 2:]‖ and visit count for radii>0.
- s>500, s%interval==0, s<15000: `scores()` on 10 distinct cached views → s_d (floor mean
  count), s_p (min-max of Σ count·E_photo) → clone (grad≥2e-4 ∧ max scale ≤ dense·extent ∧
  s_d>5, gsplat `duplicate`) → split (abs grad≥grad_abs ∧ scale>dense·extent ∧ s_d>5, gsplat
  `split`, N=2, /1.6) → stats reset → prune candidates (o<0.005; s>3000: max scale>0.1·extent)
  ∩ multinomial(int(0.5·#cand), weights 1/(1e-6+1−s_p) positional over first N_old) →
  opacity clamp 0.8 (`reset_opa(0.8)`, moments zeroed).
- s%3000==0, s<15000: opacity reset 0.01.
- s∈{18k,21k,24k,27k}: late VCP prune (o<0.1 ∨ s_p>0.9).
- initialize: lrs (opacity 0.025, sh0 lowfeature_lr, shN highfeature_lr/20), and all six Adams
  replaced by `ScheduledAdam` implementing the official optimizer_step schedule (main every
  step ≤15k, shN every 16; then both every 32 to 20k, every 64 after; grads summed between
  steps; sum dropped when density replaces the Parameter, as the official new nn.Parameter).
- Per-scene overrides looked up from the initial SfM point count (13 scenes, table from the
  Taming port, verified against the dataset); unknown scenes → official argument defaults.
  The density refuses a scene whose official β differs from the configured one.

## Probes
| probe | checks (fails if removed) |
|---|---|
| fastgs-metric-kernel (CUDA) | vendored rasterizer image = independent float64 dense compositor at β large; metric counts = brute-force count of contributing masked pixels; CB β=0.5 strictly reduces tile pairs and counts ⊆ |
| fastgs-scores | `compute_scores` with an injected fake renderer equals an independent formula (Eq. 1–6, floor, min-max, E_photo weighting) |
| fastgs-densify-gate | only s_d>5 ∧ gradient-qualified Gaussians clone/split; boundary dense·extent; abs grad for split; opacities ≤0.8 after |
| fastgs-prune | early: pruned ⊆ candidates, ≤ budget, weighted by s_p; late: exactly o<0.1 ∨ s_p>0.9 |
| fastgs-schedule | densify/reset/late-prune steps for interval 500 and 100 |
| fastgs-optimizer-schedule | ScheduledAdam step pattern and summed gradients (≠ per-step Adam) |
| fastgs-scene-table | per-scene overrides for garden/counter etc. and refusal of β mismatch |

## Fidelity gaps
1. View sampling with replacement (runtime) vs official epoch stack — small.
2. Runtime densifies after the optimizer step (official before) — negligible.
3. Late-stage gradient accumulation ordering identical, but runtime's 30000th step is skipped
   by schedule anyway.
4. Score views come from a cache of views seen so far (≈all after 1000 steps).
5. LPIPS from `lpips` package / valid-conv SSIM at evaluation (runtime metrics) vs official
   metrics.py — affects SSIM/LPIPS, not PSNR.
6. Per-splat parallel backward of 3DGS-accel is in the vendored kernel (no gap); `sparse_adam`
   unused by the official scripts.
7. Mip-NeRF 360 resolution: our `mipnerf360-1600` = official loader protocol.
