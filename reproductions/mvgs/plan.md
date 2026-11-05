# MVGS native plan

Method names: **`mvgs`** (official indoor Mip-NeRF 360 recipe, CIG 48/24/12/8 views, 39000 steps)
and **`mvgs-outdoor`** (official bicycle/garden/stump recipe: M=3 at w1600, 30000 steps).
Module `ss_mvgs`, package `reproductions/mvgs/plugin/`.

## Runtime constraint and how it is met
The runtime renders one camera per step and gives losses one target. MVGS needs M views per step,
a per-stage resolution, a shuffled-stack view sampler, stage-local schedules and densification
that sees all M views. Everything lives in components (DashGaussian precedent):

- `mvgs-density` (density) owns the schedule. `set_training_cameras` reloads the training images
  (`load_dataset(--data)`, checked against the runtime cameras), keeps a GPU uint8 cache at the
  current stage resolution, and samples the M views of the next step from a shuffled stack
  (official `viewpoint_stack.pop(randint)`; refilled when empty, possibly mid-step).
- `mvgs-renderer` (renderer) asks the density (process-local `LINK`) for the current batch. With
  grad enabled and a batch present it renders all M views at the stage resolution
  (K scaled by w'/W, h'/H = official FoV-preserving resize) and returns `mv_views`
  ([{rgb, target, camera, info}]); `rgb`/`info` are the **last** view (official densification
  statistics view). The runtime's own frame is ignored in this mode (config `sampler: stack`;
  `sampler: runtime` keeps it as view 1 and samples M−1 extras). `no_grad` (evaluation) renders
  the given camera at full resolution.
- `photometric` (loss): Σ_m [(1−λ)L1 + λ(1−SSIM_same)] over `mv_views` (Eq. 4, **sum**); falls back
  to the single given view only when the renderer produced no batch (probes, no data loaded).

## Components
| component | kind | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| mvgs-renderer | renderer | `ss_mvgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb, render.depth.expected_z, gradient.screen.signed, render.multiview_batch | — |
| mvgs-density | density | `ss_mvgs.density:create_density` | gradient.screen.signed | stages [[48,8,3000],[24,4,3000],[12,2,3000],[8,-1,30000]], full_width_ratio 1.946875, sampler stack, grow_grad 2e-4, tau 1.0, window [20,40], densify_from 500, until 15000, every 100, reset_every 3000, white_reset_at 500, min_opacity 0.005, percent_dense 0.01, big_world 0.1, big_after 3000, lr_final_factor 0.01, lr_max_steps 30000, crd true, mvad true, box_index official, stats_views last |
| photometric | loss | `ss_mvgs.losses:create_photometric` | render.rgb, render.multiview_batch | ssim_weight 0.2, ssim_padding same |

No new fields.

## Contributions → code → probes
| C | code | probe (fails if removed) |
|---|---|---|
| C1 MVRL | renderer batch + summed loss | `mvgs-multiview-sum`: CUDA, 4-view batch: loss = Σ of independent single-view losses (≠ mean, ≠ 1 view) and param gradient = Σ per-view gradients; M distinct views from the stack |
| C2 CIG | density stage table, renderer scaling, target cache | `mvgs-cig-schedule`: stage at boundaries 3000/3001/9000/9001, M per stage, sizes vs official `round(W_full/r)`, rendered image = gsplat with hand-scaled K, eval render full-res; stage-local densify/reset/LR restart |
| C3 MVAD | `mvad_threshold` | `mvgs-mvad`: hand-built centres: halved iff some pair's unit-direction angle > 60° (independent arccos formula); disabled → never halved |
| C4 CRD | `max_loss_window`, `corner_rays`, `intersect_lines`, `cross_ray_boxes`, clone | `mvgs-crd`: window = brute-force port of the official loop (incl. `range(0,H−h,h)` edge); two cameras looking at a known point → its cuboid contains it, ray directions equal independent unprojection; in-box Gaussians cloned regardless of size/grad with zero Adam moments, out-of-box untouched; with crd off nothing extra cloned |
| densify stats | last view only | `mvgs-cycle`: CUDA, synthetic dataset through renderer/loss/density for a compressed schedule; count increments by 1 per step on last-view-visible only; population changes; optimizer aligned |

## Implementation notes
- Stage lookup: global step s → (stage k, local it = s − Σ_{<k} len). Densify if
  `densify_from < it < densify_until and it % every == 0` (3DGS `iteration > densify_from_iter`),
  stats accumulated while `it < densify_until`; reset if `it < densify_until and
  (it % 3000 == 0 or it == 500)`; prune world-size only when `it > 3000`.
- Densify (official order): grads = accum/denom (nan→0); β̂ via MVAD; clone mask =
  (‖ḡ‖≥β̂ ∧ max scale ≤ 0.01·extent) ∨ in-cuboid → `ops.duplicate`; split mask on padded grads
  (‖ḡ‖≥β̂ ∧ max scale > 0.01·extent) → `ops.split`; prune opacity<0.005 (∨ max scale > 0.1·extent
  when it > 3000) → `ops.remove`; stats zeroed. Screen-size prune omitted (no-op in Inria 3DGS since
  `max_radii2D` is zeroed by `densification_postfix`).
- Means LR: after every step set lr = lr0·0.01^(min(it_next,30000)/30000) for the next step's
  stage-local iteration (restart per stage, as checkpoint restore does).
- MVAD: t_m = C_m/‖C_m‖ (COLMAP origin, native conversion keeps COLMAP world frame); halve if any
  ordered pair ‖t_i−t_j‖ > τ.
- CRD at every densify step, on the step's M views (sorted by max window loss desc, stable):
  windows on |gt−render| at stage resolution; rays through pixel centres; pairs i<j with
  `box_j = sorted_boxes[j]` (official quirk, `box_index: official`) or `[i+1+j]` (`paper`).
- `--steps` must equal Σ stage lengths (checked from argv); step beyond the schedule raises.

## Unchanged from 3DGS / baseline
gsplat rasterizer (classic), white background, 3DGS LRs and init (`--recipe 3dgs`), SH schedule
(global, = official since stage 1 is the first 3000 steps), opacity reset value 0.01.

## Fidelity gaps
1. gsplat vs Inria rasterizer (saturated-alpha gradient drop, no 0.99 clamp) — shared with baseline.
2. Runtime applies Adam before density control on densify steps (3DGS skips the update there) — shared.
3. Coarse-stage targets are antialiased bicubic resizes of the w1600 images, not PIL resizes of the
   full-res images; sizes from `full_width_ratio` (±1 px in height). Small.
4. One continuous run instead of 4 checkpoint-chained processes; carried state is identical
   (params, Adam moments, SH degree, zeroed stats); schedules restart per stage as in the code.
5. Memory: all M renders share one autograd graph (as official); M=12 at ~1558 px and M=8 at
   1600 px need tens of GB on large scenes.
6. T&T / DB: no official config; not benchmarked.
7. Paper's 3DGS rows are from images_4/images_2, ours from the w1600 baseline — compare gains with care.

feasible_natively: true.
