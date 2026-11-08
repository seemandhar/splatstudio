# splatstudio-method-lp-3dgs

Native SplatStudio port of **LP-3DGS: Learning to Prune 3D Gaussian Splatting** (NeurIPS 2024,
[arXiv 2405.18784](https://arxiv.org/abs/2405.18784)), following the official code
[ASU-ESIC-FAN-Lab/LP-3DGS @ b828063](https://github.com/ASU-ESIC-FAN-Lab/LP-3DGS)
(`--prune_method rad_splat --use_importance_mask`, i.e. the "LP-3DGS (RadSplat Score)" row).
Paper fidelity unverified.

## Mechanism
- Steps 1–19,500: stock 3DGS (gsplat renderer, DefaultStrategy densification to 15k, opacity reset
  every 3k).
- 19,500: per-Gaussian mask `lp_mask` reset to 1 (fresh Adam state).
- 19,501–19,999: loss `5e-4 * mean(sigmoid(m))`; AdamW decoupled decay 0.01 on m.
- Before renders 19,520, 19,540, …, 19,980: RadSplat score `S_i = max_views,pixels α_i T_i`
  (CUDA kernel on gsplat tile lists, `csrc/lp_score.cu`, adapted from the official `mw_score`);
  from then on opacity = `sigmoid(o) * sigmoid((m S + n)/0.5)`, n logistic (Gumbel difference),
  fresh per render.
- 20,000: remove Gaussians with `sigmoid((m + n)/0.5) < 0.5` (official `get_prune_mask`),
  then plain 3DGS to 30,000.

## Config (defaults = official code)
| key | value | source |
|---|---|---|
| renderer `tau` | 0.5 | `gumbel_temp` |
| density `mask_start` / `mask_iters` / `score_every` | 19500 / 500 / 20 | `prune_iterations`, `train_mask_iters`, train.py |
| density `mask_init` / `weight_decay` | 1.0 / 0.01 | mask.py (ones, AdamW default) |
| field `lp_mask` lr | 0.05 | mask.py |
| loss `weight` | 5e-4 | `addtional_loss` |

## Fidelity gaps
Runtime conventions (white background, sampling with replacement, gsplat 0.999 alpha clamp also
in the score); AdamW decay applied after the Adam update; one extra Gaussian Adam step at 20,000
before the prune; Mini-Splatting-score variant not built. Code-vs-paper choices (logit `m*S`
instead of `log(m)`, `mean(sigmoid(m))` instead of `mean|m|`, stochastic score-free final prune)
follow the code.

## Commands
```bash
tools/ssenv pip install -e reproductions/lp-3dgs/plugin
tools/ssenv ss-verify lp-3dgs --device cuda
tools/ssenv python reproductions/lp-3dgs/plugin/smoke_schedule.py <scene> <out>   # compressed schedule
```
The first CUDA use JIT-compiles the kernel (~1 min, cached).
