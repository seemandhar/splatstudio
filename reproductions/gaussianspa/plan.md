# GaussianSpa — native SplatStudio plan

Method names (one plugin, module `ss_gaussianspa`):
- `gaussianspa` — opacity criterion, `train_opacity.sh` defaults r1 = 0.5, r2 = 0.8.
- `gaussianspa-imp` — importance criterion, `train_imp_score.sh` defaults r1 = 0.5, r2 = 0.7.
- `gaussianspa-<scene>` — the 13 README "recommended" presets (criterion, r1, r2), e.g.
  `gaussianspa-bicycle` (imp, 0.50, 0.72), `gaussianspa-garden` (imp, 0.60, 0.50),
  `gaussianspa-counter` (opacity, 0.60, 0.50). These are what the benchmark uses.

## Prior art reused (copied with attribution, never imported)
`reproductions/mini-splatting/plugin/ss_mini_splatting/` (our matched Mini-Splatting port, −0.09 dB
mean gap): `csrc/` (statistics kernel adapted from official Mini-Splatting `forward.cu` and the
simple-knn kernel), `kernels.py`, `core.py`, `renderer.py` (target-image pass-through),
`density.py` (blur split, depth reinit, 15k simplification, schedules). GaussianSpa's code diffs
against official Mini-Splatting only in the 15k sampling formula, the removed 20k CDF prune and the
added ADMM phase, so the shared stages are copied unchanged. The JIT extension gets its own name
(`ss_gaussianspa_cuda`).

## Components
| kind | name | factory | requires / provides | config |
|---|---|---|---|---|
| renderer | spa-renderer | `ss_gaussianspa.renderer:create_renderer` | provides render.rgb, depth, gradient.screen.signed, training.target_image | absgrad False |
| density | spa-density | `ss_gaussianspa.density:create_density` | requires gradient.screen.signed, training.target_image | Mini-Splatting stages + `criterion`, `prune_ratio1`, `prune_ratio2`, `spa_start` 15200, `spa_stop` 25200, `spa_interval` 50, `imp_metric` indoor |
| loss | photometric | built-in | — | L1 + 0.2 D-SSIM (unchanged) |
| loss | spa-penalty | `ss_gaussianspa.losses:create_penalty` | — | `delta` 5e-4, `spa_start`, `spa_stop`, `spa_interval`, `stop_inclusive` (opacity True / imp False) |

The ADMM state lives in two Gaussian-owned `FieldSpec`s so the loss can read it from `params`,
density ops keep it aligned, and checkpoints carry it (resume needs nothing else):
| field | shape | lr | init | meaning |
|---|---|---|---|---|
| `spa_z` | () | 0 | 0 | auxiliary sparse opacity z (activated-opacity units) |
| `spa_u` | () | 0 | 0 | scaled dual lambda (code `u`) |
Neither gets a gradient (the loss uses them detached), so Adam never touches them.

## Step mapping (runtime calls `after_backward(step)` after the optimizer step)
- `step < 15000`: Mini-Splatting densification + depth reinit (copied).
- `step == 15000`: importance (metric `indoor`) over all training views, `imp[area_max==0]=0`;
  draw `int(N(1-r1))` without replacement, p ∝ imp + 1; reinit from centres (fields z,u → 0).
- `step == 15200`: `z = proj(a + 0)`, u = 0 (no multiplier update).
- `15200 < step <= 25200`, `step % 50 == 0`: loss adds `0.5·delta·||a − z + u||²` (sum over N;
  imp variant `< 25200`); then density: `z = proj(a + u)`, `u += a − z` (imp variant recomputes
  the importance over all views first).
  - opacity proj: `thr = sort(v)[int(r2N)−1]`, `z = v·(v > thr)`.
  - imp proj: `thr = sort(imp)[int(r2N)−1]`, `z = v`, `z[imp < thr] = 0`.
- `step == 25200`: hard prune with gsplat `remove` (survivors keep Adam state, as `prune_points`):
  opacity: keep `topk(a, int((1−r2)N))`; imp: recompute imp, remove `imp <= sort(imp)[int(r2N)−1]`.
- LR: means `lr(it)` before 15k, `lr(it − 10000)` after, max 30000; SH 0 until 15k, +1 per 1000.

## Probes
| probe | checks | fails if |
|---|---|---|
| spa-penalty | loss value = 0.5δ‖a−z+u‖² and d/dlogit = δ(a−z+u)a(1−a) (independent formula); 0 and graph-attached off schedule (step % 50 ≠ 0, ≤ 15200, > 25200) | penalty removed / wrong variable / wrong schedule |
| spa-projection | opacity and importance projections equal an independent numpy sort-based reference; projection uses a+u, not a | projection replaced by L1 / threshold / wrong ranking |
| spa-admm-cycle | density at 15200 sets z=proj(a), u=0; at 15250 z=proj(a+u), u+=a−z; untouched at 15225 / 25250; resume: reloading params (with z,u) into a fresh density reproduces the next update bit-exactly | multiplier update dropped, schedule off, state not resumable |
| spa-hard-prune | at 25200 keeps exactly int((1−r2)N) highest-opacity (or importance) Gaussians, survivors' Adam moments preserved | prune removed / wrong ranking / Adam reset |
| spa-sampling | 15k sampling keeps int(N(1−r1)) with p ∝ imp+1 (zero-importance Gaussians may survive) | Mini-Splatting nnz formula used |
| ms-stats-kernel, ms-depth-midpoint, ms-blur-split, ms-depth-reinit, ms-schedule | copied Mini-Splatting probes | shared stages broken |

## Fidelity gaps
1. Inherited Mini-Splatting gaps (white background, random-with-replacement views, step order,
   gsplat alpha numerics, LPIPS package, SSIM valid conv). Effect: LPIPS offset, ±0.1–0.2 dB.
2. z/u update uses the opacity *after* this step's Adam update (official: before). One Adam step
   of 10000 per 50; negligible.
3. Hard prune at 25200 also runs after the optimizer step (official: before). Negligible.
4. Per-scene presets come from the README "recommended" table, not from the paper (the paper does
   not list them). The repo CSVs reproduce the paper within 0.03 dB on bicycle/garden/counter.
5. Black background in official code; white here (inherited).
6. Observed in the compressed smoke run (garden, importance, r2 = 0.5): once more than r2·N
   Gaussians have zero importance, the importance projection's threshold is 0 and it zeroes
   nothing. That is the official rule, kept unchanged; check the `admm` rows in the benchmark
   checkpoint's `density_state`.
