# LP-3DGS native plan

Method name `lp-3dgs`, module `ss_lp_3dgs`. Variant reproduced: **RadSplat score** (paper headline,
Table 1 row "LP-3DGS (RadSplat Score)"). Benchmark: Mip-NeRF 360 garden (outdoor) + room (indoor),
per scene; report PSNR/SSIM/LPIPS and final Gaussian count / pruning ratio (vs count at 19,999).

## Components
| kind | name | factory | requires / provides | config (paper/code default) |
|---|---|---|---|---|
| renderer | lp-renderer | `ss_lp_3dgs.renderer:create_renderer` | provides render.rgb, render.depth.expected_z, gradient.screen.signed | `tau` 0.5 |
| density | lp-density | `ss_lp_3dgs.density:create_density` | requires gradient.screen.signed | stock 3DGS DefaultStrategy config + opacity-reset fix; `mask_start` 19500, `mask_iters` 500, `score_every` 20, `mask_init` 1.0, `weight_decay` 0.01, `tau` 0.5 |
| loss | photometric | stock | render.rgb | λ_ssim 0.2 |
| loss | lp-mask-loss | `ss_lp_3dgs.losses:create_mask_loss` | render.rgb | `weight` 5e-4, `mask_start` 19500, `mask_iters` 500 |

## Fields
| field | shape | lr | init | notes |
|---|---|---|---|---|
| `lp_mask` | () | 0.05 | 1.0 | Adam, eps 1e-15 (runtime); decoupled weight decay 0.01·lr applied by the density inside the window (AdamW). No grad outside window → Adam has no state before 19,500 (= fresh optimiser). Reset to 1.0 in place at 19,500. |

Importance score `S` and the mask phase (`off` / `train` / `frozen`) are not learned: the density
stores them on the `ParameterDict` as plain attributes (`_lp3dgs_score`, `_lp3dgs_phase`), the
renderer reads them. The evaluation renderer gets a plain dict from the checkpoint → phase `off`,
which is the official post-pruning renderer. No population change happens while S is alive
(densification stops at 15k), and S is dropped at pruning.

## Mapping of the official loop onto the native order (render → backward → opt.step → after_backward(s))
- after_backward(19500): fill m=1 (in place, Parameter identity kept), phase stays off.
- loss: 19500 < s < 20000 → 5e-4·mean(sigmoid(m)).
- after_backward(s) with s+1 ∈ {19520, 19540, …, 19980}: recompute S over all training cameras
  (renderer passes `render_score` with the *current* opacity: masked with fresh Gumbel noise
  per view once phase=train), phase := train.
- 19500 < s < 20000 after opt.step: m ← m − lr·wd·m (AdamW decoupled decay).
- after_backward(19999): phase := frozen (render at 20000 masked, m detached = official mask
  optimiser not stepped at 20000).
- after_backward(20000): remove `m + n < 0` (fresh logistic noise, τ>0 does not change the sign)
  via `gsplat.strategy.ops.remove`, phase := off, drop S, log counts.

## Custom CUDA
`csrc/lp_score.cu`: re-traverses gsplat 1.5.3 tile lists (same traversal as the mini-splatting
port's ms_raster.cu, gsplat alpha numerics) and does `atomicMax(score_i, α·T)` (official
`renderCUDA_mw_score`), forward only. JIT via torch cpp_extension.

## Probes
| probe | checks | fails if |
|---|---|---|
| lp-gumbel | gs with fixed uniforms equals sigmoid((x+n)/τ) with official n; P(gs(x)>0.5)=sigmoid(x) over 2e5 draws; d gs/dx analytic | noise/τ removed or wrong |
| lp-score-kernel (cuda) | kernel max α·T equals an independent dense float64 compositor (gsplat numerics) on a random 3-camera scene | wrong reduction (sum vs max), wrong T |
| lp-mask-gradient (cuda) | phase train: lp_mask.grad nonzero, equals autograd of the independently built masked-opacity graph with the same noise; phase off/frozen: grad None | mask not applied / detached |
| lp-mask-loss | value/grad = λ·mean σ(m) inside (start, start+iters), exactly 0 outside | loss or window wrong |
| lp-schedule (cuda) | density on a toy scene with compressed schedule: reset at start, score refreshed at the right steps, phases, decay, prune removes exactly m+n<0 Gaussians, optimiser aligned, lp_mask still trainable leaf | schedule or prune wrong |

## Unchanged from 3DGS
gsplat renderer numerics, DefaultStrategy densification (with the runtime's opacity-reset fix),
photometric loss, LR schedule.

## Fidelity gaps
1. White background / random-with-replacement view sampling / gsplat alpha clamp 0.999 (runtime
   conventions) — affects absolute PSNR slightly; compare deltas vs our baseline.
2. AdamW decay applied after (not before) the Adam update — O(lr²·wd) difference.
3. One extra Gaussian-parameter Adam step at 20,000 before pruning (official prune happens before
   that step and drops the grads). Negligible.
4. Score uses gsplat tile lists (0.999 clamp) instead of Inria (0.99 clamp). Small.
5. Mini-Splatting-score variant not built (needs a changed SH schedule) — Table 1's better
   RadSplat row is the target.
