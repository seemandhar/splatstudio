# Taming 3DGS — native SplatStudio plan

Method names: `taming-3dgs` (main: "Ours (#3DGS)", big mode, final count = 3DGS count) and
`taming-3dgs-budget` ("Ours", multiplier mode). Module `ss_taming_3dgs`.

## Components
| component | kind | factory | requires | provides | config (paper/code values) |
|---|---|---|---|---|---|
| `taming-renderer` | renderer | `ss_taming_3dgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, training.target_image | gsplat classic; forwards the frame's GT image + camera in `outputs` |
| `taming-density` | density | `ss_taming_3dgs.density:create_density` | gradient.screen.signed, training.target_image | — | see `DEFAULTS` in density.py (all §3 hyper-parameters of paper.md) |
| `photometric` | loss | `splatstudio.losses:create_photometric` | render.rgb | — | 0.2 D-SSIM (unchanged 3DGS loss) |

No new FieldSpecs. Density state (gsplat-ops aligned): `grad_accum [N]`, `count [N]`.

## Mapping contributions → code
- C1 growth curve: `core.count_array(S, B, from, until, interval)` (code formula); target of the
  j-th call = `values[j+1]`. Budget per scene: `budget` config, or lookup in
  `budget_table` keyed by the initial SfM point count S (unique per scene; our native point
  clouds hold exactly the COLMAP points; table values from official `full_eval.py`). Unknown S
  and no explicit budget → error (no silent fallback).
- C2 score: `density._score()` — 10 distinct random cached training views; one no-grad gsplat
  render each; saliency = 50·minmax(L1 map) + 50·minmax(PIL FIND_EDGES(gray GT)); photometric
  P_i = 0.8 L1 + 0.2 (1 − SSIM, zero-padded "same" like fused-ssim); per-pixel accumulations
  (count, distance, saliency, blend) by a **custom CUDA kernel** `csrc/taming_stats.cu`
  re-traversing gsplat's tile lists with gsplat alpha numerics (adapted from the Mini-Splatting
  port's kernel). Per-Gaussian terms: grad (3DGS NDC-scaled 2D grad average), opacity, depth
  (camera z, 0 when culled), radius (max of gsplat's 2 radii), prod(scale).
  `core.normalize` = official median normalization. Densify: `core.split_budgets` + multinomial
  without replacement on score·mask, gsplat `duplicate` then `split` (same memory layout as the
  official code). Pruning: official score-guided 50 % sampling with the official (misaligned)
  index weights, only for calls < 27; world-size prune after 3000.
- C3 high opacity: `high_opacity: "freeze"` (official code behaviour): set the opacity Adam lr to
  0 at step 15000. The paper's trainable abs activation is not what the code does → gap.
- C4 SH: `optim.IntervalAdam` replaces `optimizers["shN"]` (accumulates gradients, one Adam step
  every 16 iterations, accumulation dropped when density replaces the parameter, like the
  official new `nn.Parameter`); lr overrides opacities 0.025, shN 2.5e-4 set in `initialize`.
- C5 speed kernels: not ported (gsplat backend); recorded gap, no math change.
- Unchanged 3DGS: init, LR schedule of means, loss, opacity reset (every 3000 < 15000 to 0.01),
  clone/split operators, SH degree schedule.

## Training images
The density needs GT images of arbitrary training views. The renderer forwards
`outputs["target_image"]`/camera; the density caches each seen training view (uint8 CPU + edge
map) in `before_backward`. Score views are sampled among cached views (with 3DGS-style
random-with-replacement sampling ~95 % of views are seen by step 600, all by ~step 2000).

## Probes
| probe | measures | fails if |
|---|---|---|
| `taming-stats-kernel` (cuda) | kernel count/dist/blend/saliency vs brute-force per-pixel loop | kernel wrong/removed |
| `taming-growth-curve` | count array vs independent parabola; endpoints S, B; densify reaches target exactly | growth curve removed |
| `taming-score` (cuda) | `_score` vs independent re-implementation on a tiny scene; opacity term ranks low-opacity lower | score terms/weights changed |
| `taming-sampled-densify` (cuda) | only qualifiers selected, high-score preferred, budgets proportional, zero moments | sampling replaced by threshold densify |
| `taming-guided-prune` | prune ⊆ prune_mask ∧ sampled; none after call 27 | pruning changed |
| `taming-high-opacity` | opacity unchanged after 15000 while changing before | freeze removed |
| `taming-sh-interval` | IntervalAdam = Adam on 16-step summed grads; differs from per-step Adam; lr overrides | SH schedule removed |
| `taming-schedule` | densify steps 600..14900 (144), resets 3000..12000 | schedule changed |

## Fidelity gaps
1. Speed kernels (per-splat backward, fused SSIM, separate SH tensors): not ported; math equal → timing only.
2. Backend: gsplat alpha clamp 0.999 (3DGS 0.99), opacity-aware tile radii, white background, valid-SSIM training loss, random-with-replacement view sampling (official: without replacement per epoch) — shared with our baseline.
3. Score views drawn only from views already seen by the trainer (images not exposed to densities). Small effect early.
4. Optimizer step order: runtime steps the optimizers before density control (official: after). Same as baseline.
5. High opacity = code's freeze, not the paper's trainable abs opacity.
6. (closed) Multinomial sampling matches torch/official: when fewer positive scores than the budget, zero-weight indices fill the remainder (only the all-zero case returns no samples).
7. Radius stat uses gsplat's opacity-aware radii (median-normalized; small effect).
8. LPIPS / SSIM evaluation from the runtime (lpips VGG package, valid SSIM).
