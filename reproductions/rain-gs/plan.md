# RAIN-GS native plan

Method `rain-gs`, module `ss_rain_gs`, package `splatstudio-method-rain-gs` (MIT code of
cvlab-kaist/RAIN-GS ported, cited). Headline target: "Ours, Random" rows (N=10 random init).

## Components
| component | kind | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| rain-gs-renderer | renderer | ss_rain_gs.renderer:create_renderer | provides render.rgb, render.depth.expected_z, gradient.screen.signed | lowpass_min 0.3, lowpass_max 300, lowpass_every 1000, lowpass_until 25000, eval_lowpass 0.3, sh_start 5000, sh_every 1000, background black |
| rain-gs-density | density | ss_rain_gs.density:create_density | requires gradient.screen.signed | num_init 10, cube_factor 3, init_opacity 0.1, warmup 10000, densify_from 500, densify_until 25000, densify_every 100, grad_threshold 2e-4, percent_dense 0.01, divide_ratio 0.7, abe_factor 0.3, prune_opacity 0.005, prune_big_world 0.1 (after 3000), reset_every 3000, reset_value 0.01, means_lr_final_ratio 0.01, means_lr_max_steps 30000 |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim_weight 0.2 (unchanged 3DGS) |

No new fields.

## Mapping
- C1 SLV init → density `set_training_cameras` (runtime calls it once after `initialize`, before
  step 1): replaces the whole population with N=10 Gaussians in the official cube built from
  camera `tvec` = viewmat[:3,3]; 3-NN log scales; sh0 from U[0,1) with the official uint8
  round trip; opacity 0.1; Adam state rebuilt. Missing call → error at first after_backward.
- C2 progressive low-pass → renderer counts training renders; at render 1 and every 1000 (<25000)
  sets eps2d = clip(HW/(9πN), 0.3, 300) with the current view and N; passes eps2d to gsplat
  `rasterization` (classic mode, same as Inria `cov + low_pass I`). No-grad renders use 0.3.
- C3 ABE-Split → density: clone small, then (step ≤ 10000) append copies of split candidates at
  0.3·extent·μ with unchanged scale, then 2-way split with scale/1.4, prune.
- C4 warm-up → density sets the means LR after the runtime's update (hold, then official
  expon(step−10000) with max_steps 30000); renderer SH schedule clip((step−5000)//1000+1,0,3);
  density schedule extended to 25000; divide ratio 1.4.
- Unchanged from 3DGS: L1+0.2 D-SSIM, other LRs, opacity reset every 3000, prune 0.005 and
  0.1·extent world size after 3000 (screen-size prune is a no-op in official code → omitted).

## Probes
| probe | checks (fails when mechanism removed) |
|---|---|
| slv-init | population = 10, points lie in the official tvec cube, scales = independent 3-NN formula, opacity 0.1 |
| lowpass-schedule | eps2d formula vs independent values at boundaries; recomputed only every 1000; clamps; eval 0.3 |
| lowpass-in-render (cuda) | rendered alpha footprint of one tiny Gaussian grows with eps2d=300 vs 0.3 |
| abe-split | after a densify at step ≤10000, 3 children: two scale/1.4 near μ, one at 0.3·extent·μ with parent scale; none at step >10000 |
| warmup-schedule | means LR constant <10000, matches official expon afterwards; SH degree 0/1/2/3 at 4999/5000/6000/7000 |
| trainable-leaves | every field stays a requires_grad leaf after densify (pitfall) |

## Fidelity gaps
1. Cube from training-camera tvecs only (official: all images incl. test) — tiny change of cube.
2. Runtime-wide: optimizer step before densification, random-with-replacement view sampling,
   valid-conv SSIM, LPIPS package — same for baseline.
3. gsplat per-axis radius (3·sqrt(diag)) vs Inria max-eigenvalue circle; near plane 0.01 vs 0.2.
4. Black background (official) differs from the white runtime baseline — kept official.
5. Random init stochasticity: N=10 points ⇒ high seed variance (paper single run).
