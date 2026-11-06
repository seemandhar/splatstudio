# splatstudio-method-adamw-gs

Native SplatStudio port of **A Step to Decouple Optimization in 3DGS** (AdamW-GS, Ding et al.,
ICLR 2026, [arXiv 2601.16736](https://arxiv.org/abs/2601.16736)), following the official code
[EliottDJay/3DGS_AdamWGS](https://github.com/EliottDJay/3DGS_AdamWGS) (MIT) on the vanilla-3DGS
host. Paper fidelity unverified.

## Mechanism
- **Sparse Adam** (Sec. 4.1, Eq. 4): only primitives visible in the current view (radii > 0) are
  updated; moments of invisible primitives are frozen; bias correction uses a per-primitive step
  counter `asystep` (official `diff_gauss.SparseGaussianAdam` + `adam.cu`).
- **DAR** (Sec. 4.3, Eq. 7–8): opacity/scale L1 regularisation is not a loss term. The moments see
  only the photometric gradient; the update adds `min(λ/img_scale · ∇R / (sqrt(v̂)+ε), 10)` with
  ∇R = σ(τ)(1−σ(τ)) (logit opacity) and exp(κ) (log scale).
- **RSR** (Sec. 4.2, Eq. 5): every 100 iterations in [1000, 15000] a uniform random fraction of
  all primitives (StSS milestones) gets `exp_avg *= α1`, `exp_avg_sq *= α2` in every optimizer.
- **Recoupling in 3DGS** (Sec. 4.4): 3DGS-MCMC position noise (outdoor), opacity correction
  `1 − sqrt(1 − o)` on clone (source and copy), no opacity reset.

## How the optimizer is installed
The runtime creates one `torch.optim.Adam` per field and calls `step()` without arguments
before `density.after_backward`. `AdamWGSDensity.initialize` replaces every entry of the
runtime's `optimizers` dict in place with `SparseAdamWGS` (a `torch.optim.Adam` subclass with
the same `exp_avg`/`exp_avg_sq` state, so `check_state`, gsplat's `duplicate/split/remove/
reset_opa` and checkpoints keep working). `before_backward` stores this iteration's visibility in
a shared `AdamWGSContext` and increments `asystep`; `asystep` lives in the density state dict, so
the gsplat ops copy it from the parent on clone/split and drop it on prune, like the official
`asystep[mask].repeat(N)`.

## Methods and config (defaults = released configs)
| key | `adamw-gs` (outdoor, `mipnerf_adamwgsp`) | `adamw-gs-indoor` (`mipnerf_adamwgs`) | source |
|---|---|---|---|
| β1, β2, ε | 0.9, 0.999, 1e-15 | same | adam.cu call |
| λo, λs / cap | 1e-3, 1e-5 / 10, 10 | same | rasterize_points.cu, adam.cu |
| img_scale | first two digits of W·H × 10^(len−3) | same | `_img_scale` |
| opacity / scale DAR window | [3000,15000] / [1000,15000] | same | config.yaml |
| RSR ratios @ shifts | 0.1,0.3,0.5 @ 1000,7000,10000 (StSS GS2) | 0.1,0.3 @ 1000,7000 (GS1) | config.yaml |
| α1, α2 | 0.2, **0.01** | same | code reads `rsr_exeabg_sq_fading` (default 0.01); paper text 0.04 |
| noise [begin,end], noise_lr | on [1000,14000], 5e5 × lr_means | off | train.py, arguments |
| opacity reset | off (`opacity_reset_every: 0`) | off | config (300000) |

`adamw-gs-gs8` = `adamw-gs` + `opacity_reset_every: 3000` (reset to 0.01) + 3DGS world-space
big prune (max scale > 0.1·extent, after step 3000): the paper's Table-2 headline outdoor row GS8,
which has no released config. The official 20-px screen test is dead code because
densification_postfix re-zeros `max_radii2D` before `prune()`; it is ported as inactive.
| ADC | 3DGS: grad ≥ 2e-4, every 100 in (500,15000), percent_dense 0.01, split /1.6, prune o<0.005 | same | config |

## Fidelity gaps
1. Runtime-wide: white background, random-with-replacement views, SSIM valid conv.
2. The runtime steps the optimizer before density control; the official code densifies first and
   then skips the update on densification iterations (grads are None): 144 extra updates
   (600..14900). Under `adamw-gs-gs8` the reset iterations (3000/6000/9000/12000) are also
   stepped before the reset, where the official code skips the opacity update.
3. Means LR schedule is applied one step late by the runtime (lr(s−1) at step s).
4. Torch float32 implementation of the fused kernel (same formula, no speed claim).
5. α2 follows the released code (0.01) rather than the paper text (0.04).
6. Paper headline rows "GS8" (outdoor + opacity reset) and "GS7" (StSS GS5) have no released
   config; the defaults reproduce per-scene rows GS9 (outdoor) and GS7/GS8 (indoor).
7. T&T / Deep Blending per-scene configs (noise and StSS differ per scene) are not packaged.
8. Visibility: gsplat 1.5.3 culls opacity < 1/255, uses opacity-aware per-axis radii and near
   plane 0.01 (diff-gauss: no opacity cull, 3σ radius, near 0.2), so near-dead primitives are
   invisible (no Sparse-Adam/DAR update, no asystep increment).

## Commands
```bash
pip install -e .
ss-inspect adamw-gs
ss-verify adamw-gs --device cuda
ss-train adamw-gs --data ~/datasets/mipnerf360/native/bicycle --recipe 3dgs --num-gaussians 10000000 --output runs/x
ss-train adamw-gs-indoor --data ~/datasets/mipnerf360/native/counter --recipe 3dgs --num-gaussians 10000000 --output runs/y
```
