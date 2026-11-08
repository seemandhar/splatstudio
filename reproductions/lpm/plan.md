# Plan — LPM native plugin

Method names: `lpm` (outdoor: reset_interval 100, angle 45°) and `lpm-indoor` (200, 90°; README
note). Module `ss_lpm`. Baseline components kept: photometric loss (L1 + 0.2 D-SSIM), gsplat 3D
rasterizer, all LRs, SH schedule.

## Components
| component | kind | factory | requires / provides | config |
|---|---|---|---|---|
| lpm-renderer | renderer | `ss_lpm.renderer:create_renderer` | provides gsplat caps + `training.target_image` | absgrad False |
| lpm-density | density | `ss_lpm.density:create_density` | requires `gradient.screen.signed`, `training.target_image` | see paper.md §3 |
| photometric | loss | built-in | | |

The renderer is the stock gsplat renderer plus `outputs["target_image"]`, `outputs["viewmat"]`
(the density needs the GT image of the rendered view and the training images for matching).

## Density (`LPMDensity`)
- Statistics: gsplat `DefaultStrategy._update_state` (grad2d = Σ‖∂L/∂μ2d‖ in NDC units,
  count) for steps < 15000; `step_pre_backward` retains means2d grad.
- Training-view index from viewmat (`set_training_cameras`); per-view centre-ray directions →
  angle-neighbour table.
- **Image cache / lazy offline matching** (runtime exposes no training images up front): every
  rendered training target is cached (uint8, CPU). When view i needs LPM, pairs are built from up
  to 3 random angle-neighbours *whose images are cached* (top-up later if fewer were available),
  SuperPoint+LightGlue features cached per view (CPU). Models freed after step 15000.
- At step s (after optimizer step — runtime order):
  1. `s<15000`: update stats.
  2. `500<s<15000, s%100==0`: draw pair, error map (C1), regions (C2), zones (C3);
     if zones: LPM densify+prune (C4) and reset stats; else nothing.
  3. `s%3000==0, s<15000`: global reset (`reset_opa`, value 0.01).
  4. `500<s<15000, s%reset_interval==0`: localized reset (C5) with the same zones recomputed
     from the same pair (official recomputes too).
- All population changes via `gsplat.strategy.ops` (`duplicate`, `split`, `remove`, `reset_opa`);
  localized reset writes `params["opacities"].data` in place (official in-place, moments kept).
  After every population change assert each field is a leaf with `requires_grad` (pitfall).
- Own `random.Random(seed)` so the runtime's view sampling sequence is untouched.

## Probes
| probe | measures | fails if |
|---|---|---|
| lpm-error-map | patch mask of a synthetic error block vs independent numpy reference | C1 thresholds/patching changed |
| lpm-zone-projection | two synthetic cameras viewing a known 3D square: zone contains it and is tight; closest-point formula vs independent least-squares | C3 wrong |
| lpm-localized-densify | Gaussians with τ/2 ≤ g < τ inside zone are cloned/split, outside not; n_extra lowest-opacity pruned; disabled ⇒ identical to 3DGS | C4 removed |
| lpm-localized-reset | high-opacity Gaussian in front of zone reset to 0.01, behind/outside/low-opacity untouched | C5 removed |
| lpm-matching (CUDA) | SuperPoint+LightGlue on an image and its shifted copy recover the shift; region pairing maps a region by the shift | C2 removed (ProbeUnavailable if weights missing) |
| lpm-density-cycle (CUDA) | full after_backward on a tiny scene with two cached views: zones found, population changes, fields still trainable leaves | integration broken |

## Fidelity gaps
1. Lazy pair construction (neighbours restricted to already-seen views at first use) vs offline
   all-pairs. Expected effect: negligible (random choice either way; by step 500 ~85–95% of views seen).
2. Runtime order: optimizer step before density (official densifies before the step, and the
   step after densify is skipped for new Parameters). Shared by all native ports.
3. White background, random-with-replacement view sampling, gsplat rasterizer numerics (runtime).
4. "Add points at zone centre" from the paper is not in the official code → not implemented.
5. Indoor setting choice (README 200/90° vs paper ablation table preferring 100 indoors).
6. K-based rays (our cameras keep cx, cy) vs official symmetric projection — sub-pixel.
