# Mini-Splatting2 — native SplatStudio plan

Methods (module `ss_mini_splatting2`, package `splatstudio-method-mini-splatting2`):

| method | official script | imp_metric | simplification at 3k / 8k |
|---|---|---|---|
| `mini-splatting2` | `msv2/train.py --imp_metric outdoor` | outdoor (I²) — M360 outdoor, T&T | importance sampling (0.6) / CDF 0.99 of I |
| `mini-splatting2-indoor` | `msv2/train.py --imp_metric indoor` | indoor (I¹) — M360 indoor, DB | same |
| `mini-splatting2-d` | `msv2_d/train.py --imp_metric outdoor` | outdoor (only used at 2k) | importance pruning (CDF 0.99 of Σw) at both |
| `mini-splatting2-d-indoor` | `msv2_d/train.py --imp_metric indoor` | indoor | same |

Train with `--steps 18000` (paper/config `fast`).

## Reuse from Mini-Splatting (#28)
Copied (not imported) from `reproductions/mini-splatting/plugin/ss_mini_splatting/` with attribution:
`csrc/ms_raster.cu` (blend statistics + mid-point depth on gsplat tile lists), `csrc/simple_knn.cu`,
`csrc/ext.cpp`, `csrc/LICENSE-3DGS.md`; pure helpers (`cdf_keep_mask`, `weighted_sample`,
`accumulate_importance`, `sampling_count`, `depth_samples_per_view`, `blend_reference`,
`midpoint_reference`). The JIT extension gets its own name `ss_mini_splatting2_cuda`.

## Components
| component | kind | factory | requires / provides | notes |
|---|---|---|---|---|
| `msv2-renderer` | renderer | `renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb, render.depth.expected_z, gradient.screen.signed, training.target_image, render.visibility_culling | gsplat classic; with a link table registered for the params object: identifies the training view (viewmat), renders at 0.5× (floor W/2,H/2; K scaled; target bicubic-antialiased + 8-bit quantised) while warm-up is on, and rasterises only the non-culled subset. No table (evaluation) ⇒ stock gsplat render. |
| `msv2-density` | density | `density:create_density` | gradient.screen.signed, training.target_image, render.visibility_culling | the whole MSv2 schedule; owns the per-view culling table N×V; sparse Adam |
| `photometric` | loss | `losses:create_photometric` | render.rgb | L1 + 0.2 D-SSIM (runtime SSIM) against `outputs["loss_target"]` when the renderer trained at half resolution, else the frame target |

Renderer ↔ density link: `link.py` keeps `{id(params): (weakref(params), LinkTable)}`; the density
registers the table in `initialize`; the renderer looks it up per render. Table fields: training
viewmats, `cull` (bool N×V or None), `half_res`.

## State
No new FieldSpecs. Density buffers (aligned through `gsplat.strategy.ops` state dicts): grad2d,
count, blur, cull [N,V], factor [N]. Optimizers are replaced in `initialize` by
`SparseGaussianAdam` (Taming-3DGS numerics: update rows with radii>0 only, lr·sqrt(1−β2^t)/(1−β1^t),
gsplat fused `adam` kernel on CUDA, same math in torch on CPU); visibility set in `before_backward`.
Opacity LR 0.025 (official), means LR lr(it) with max_steps 30000.

## Schedule (density.after_backward(s), runtime steps optimizer before it)
- s < 3000: grad stats (× factor_culling when the view's cull column is non-empty), blur mask
  `area_max > H·W/5000`; densify every 100 (s>500, s≠2000): clone small / split large ∪ blur /
  prune α<0.005; s≥500, s%250==0, s≠2000: **culling_with_clone** (C1+C2+C4); s=2000: intersection
  preserving (CDF 0.99 of I, with culling) → depth reinit (num_depth = N before prune, with culling,
  half-res, GT colours) → fresh Adam, cull cleared.
- s=3000: culling + importance sampling (MSv2) / importance pruning (MSv2-D); fresh Adam; SH enabled.
- s=8000: culling + CDF 0.99 preserving (MSv2) / importance pruning (MSv2-D).
- s=13000: cull cleared. Every step: lr(s+1), SH zeroing beyond active degree (0 until 4000, +1/1000),
  `half_res = s+1 ≤ 3000`.

## Probes
| probe | contribution | fails if |
|---|---|---|
| `msv2-stats-kernel` | blend statistics | copied kernel disagrees with brute-force blend |
| `msv2-critical-clone` | C1+C2 | critical set ≠ independent (argmax ∧ CDF) reference, or clone values ≠ MCMC formula (independent N=2 MCMC binomial formula), or copy moments not zero |
| `msv2-visibility-culling` | C4 | per-view mask ≠ reference CDF, culled Gaussians reach the rasteriser (render equals subset render, differs from full), count_vis≤1 prune, factor formula |
| `msv2-sparse-adam` | 3DGS-accel | invisible rows move or visible rows ≠ bias-corrected reference Adam |
| `msv2-depth-reinit` | C3 | reinit count ≠ N·1.0 / points off the surface / colours not GT / Adam not fresh |
| `msv2-half-res` | schedule | warm-up render not at floor(W/2) or loss target mismatch; full res after 3000 |
| `msv2-schedule` | C3 | event steps (clone every 250 from 500, reinit 2000, simp 3000/8000, cull off 13000), LR(30000-horizon), SH zeroing, opacity LR wrong |

## Fidelity gaps
1. Optimizer step happens before density control (runtime); official skips the Adam step on
   densification steps (new params have no grad).
2. White background (runtime) vs black (official default).
3. Random view sampling with replacement vs shuffled stack; official also keeps half-res views until
   the stack empties after 3000 (we switch exactly at 3001).
4. gsplat rasteriser (alpha clamp 0.999, 3σ radius) vs Taming rasteriser (0.99, opacity-aware extent).
5. Runtime SSIM (valid conv) vs fused-ssim (same padding); LPIPS from `lpips` package.
6. Half-res target: torch bicubic antialias vs PIL bicubic resize.
7. GT colours for depth points come from the renderer pass-through of training targets; a view never
   drawn before 2000 (random sampling) gets rendered colours instead (logged as `depth_views_fallback`).
8. Training time measured on a shared 8×A6000 host with gsplat; paper: dedicated RTX 3090 + Taming.
