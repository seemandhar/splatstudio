# ResGS native plan

Method name `resgs`, module `ss_resgs`, package `splatstudio-method-resgs` in `plugin/`.
Target variant: **Ours (AbsGS)** = released official defaults.

## Components
| component | kind | factory | requires | provides | config (paper/code values) |
|---|---|---|---|---|---|
| `resgs-renderer` | renderer | `ss_resgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, gradient.screen.absolute, render.image_pyramid | none |
| `resgs-density` | density | `ss_resgs.density:create_density` | gradient.screen.absolute | – | tau 0.00067, stage_hr_factor 2, stage_split 3, lambda_s 1.6, beta 0.3, min_opacity 0.005, start_stat 500, update_from 700, update_interval 100, update_until 12000, change_iter [2500, 6000], resolution_scales [2,1,0], warm_up_iter 500, opacity_reduce 0.8 / 500 / until 25000, LRs (1.28e-4, 0.002, 0.04, 0.006, 0.001), flags image_pyramid / varying_threshold |
| `photometric` | loss | `ss_resgs.losses:create_photometric` | render.rgb, render.image_pyramid | – | ssim_weight 0.2, ssim_padding "same" (official 3DGS SSIM) |

Renderer/density link: the runtime never passes a step to the renderer, so (as in the
DashGaussian port) the density registers in a process-local `LINK`; under grad the renderer
renders at `pyrDown^level` size of the camera (K scaled by w'/W, h'/H, FoV kept); under
`no_grad` (evaluation) always full resolution. The renderer returns `pyramid_level`; the loss
downsamples the full-res target with an exact torch port of `cv2.pyrDown` that many times and
raises if sizes disagree.

## Fields
No learned fields. The level l_i is a non-learned per-Gaussian tensor kept in the density's
running-stat dict (`levels`) so gsplat `ops.remove` keeps it aligned; residual split extends it.

## Density (C1, C3, schedule, AbsGS opacity reduction)
`ResGSSchedule`: literal port of the official loop counters (warm-up, cur_level, substage
counter, change iters). Per step in `after_backward`:
1. stats (abs grad norm × (W/2, H/2), radii>0 count) if `start_stat < t < update_until` and warm-up 0;
   densify if also `t > update_from and t % 100 == 0`: thresholds Eq. 8 -> `residual_split`
   -> prune opacity < 0.005 (`ops.remove`) -> reset stats.
2. stage change at `t in change_iter` (resolution index +1 → next render smaller pyrDown count).
3. opacity reduction ×0.8 every 500 when res index ≠ 0 and t < 25000 (prune first if t ≥ 12000).
4. end-of-iteration counter updates.
`residual_split` uses `gsplat.strategy.ops._update_param_with_optimizer` twice (opacities with
masked moment zeroing + concat zeros; all other fields concat zeros), exactly Eq. 4-5 / code.
LRs: set in `initialize` (means = 1.28e-4 × scene scale; runtime keeps its 0.01^(t/T) decay,
which equals the official 1.28e-4→1.28e-6 exponential).

## Probes
| probe | contribution | measures / fails when |
|---|---|---|
| resgs-residual-split | C1 | parent kept (mean, scale, SH), parent opacity = β·o, child scale = s/λ, child opacity/rot/SH = parent, whitened offsets ~N(0,I), parent opacity moments zero, other parent moments kept, child moments zero, level+1; fails if split/clone used or β/λ ignored |
| resgs-selection | C3 | mask equals independent Eq. 8 for a level grid; `varying_threshold=False` gives a different mask |
| resgs-schedule | C2, C3 | density state machine equals an independent literal transcription of train.py over 30k steps (resolution, warm-up, level, densify steps), plus hand-checked anchors |
| resgs-pyramid | C2 | torch pyrDown == cv2.pyrDown (odd sizes); loss target = pyramid level |
| resgs-render-scale (CUDA) | C2 | training render at pyramid size with scaled K equals gsplat directly; no_grad full size |
| resgs-opacity-reduce | pipeline | ×0.8 at the right steps, moments zero, prune ≥ 12000 |
| resgs-cycle (CUDA) | all | short native training run on a synthetic scene with a compressed schedule: levels grow, resolution increases, eval full-res |
Built-in state-transitions / photometric-gradient / renderer-gradient kept.

## Unchanged from 3DGS / runtime
SfM init (3dgs recipe), SH schedule, Adam eps 1e-15, L1+0.2 D-SSIM.

## Fidelity gaps
1. White background (runtime) vs black (official) — small effect on unbounded scenes; not closable without core change.
2. Random-with-replacement view sampling vs official shuffled stack (refilled at stage change) — noise-level.
3. Optimizer step precedes density ops in the runtime (official densifies before `optimizer.step()`, which then skips the replaced params for that iteration) — negligible.
4. Pixel-centre convention of the pyramid camera (K scaled by w'/W) vs 3DGS FoV camera with centred principal point — sub-pixel at 1/4 res during the first 2500 steps only.
5. gsplat rasterizer (eps2d 0.3, classic) vs the AbsGS fork of the 3DGS rasterizer; evaluation SSIM valid-padded and white background (runtime-wide calibration gap).
6. Our gsplat baseline ≠ the paper's 3DGS row; compare deltas.
