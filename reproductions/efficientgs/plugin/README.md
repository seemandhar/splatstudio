# splatstudio-method-efficientgs

Native SplatStudio port of **EfficientGS: Streamlining Gaussian Splatting for Large-Scale
High-Resolution Scene Representation** (arXiv [2404.12777](https://arxiv.org/abs/2404.12777)),
after the official code https://github.com/wenkaier/EfficientGS (commit `1f70babe`, `train.sh`).
Paper fidelity unverified.

## Mechanisms
| contribution | code | value (source) |
|---|---|---|
| C1 selective densification, S_g = mean over views of Σ_pixels ‖∇_p‖ (Eq. 6) | `csrc/egs_raster.cu:egs_gradnorm_kernel`, `density._accumulate` | τ_s = 0.0007 (paper §4.1, train.sh) |
| C2 top-K dominant pruning (Eq. 7–8) | `egs_stats_kernel`, `density._prune` | K = 1, view_num = 1, steps 15000/18000/21000 (code `prune_way p4`) |
| C3 sparse SH order increment (Eq. 9) | `egs_stats_kernel` (loss weights), `density._sparse_increment`, `renderer.colors` | degree 0 until 15500, +1 for top r_s = 0.2 at 16000/17000/18000, cap 3 |
| SH rest learning rate | `density.initialize` | 0.0025 / 5 (`--shs_lr_rate 5`) |

The per-Gaussian SH degree lives in the field `sh_order` (never in the loss graph, so Adam never
changes it). The renderer evaluates gsplat's SH with bands above each Gaussian's degree zeroed
(identical values and gradients to per-Gaussian degree) and passes the colours to gsplat
(`sh_degree=None`). The runtime's 1000-step global SH schedule is only an upper bound.

Kernels re-traverse gsplat 1.5.3 tile lists with gsplat numerics; `egs_gradnorm` evaluates
gsplat's backward formula per pixel (front to back, buffer = C_total − accumulated) from the
retained gradients of the raw rendered colour/alpha. Weight convention for Eq. 8/9 is the
official code's α·T after the transmittance update (`weight_after_T: true`; `false` = paper Eq. 7).

## Fidelity gaps
- Native runtime: white background (official: black), random-with-replacement view sampling,
  valid-convolution SSIM, density step after the optimizer step.
- gsplat alpha cap 0.999 / gsplat backward (alpha-clamped Gaussians get zero position gradient)
  instead of the official 0.99 cap.
- Large-scale datasets (Mill19, Foothill-Town; 60k iterations) unavailable; small-scale averages only.

## Commands
```bash
tools/ssenv pip install -e reproductions/efficientgs/plugin
tools/ssenv ss-inspect efficientgs
tools/ssenv ss-verify efficientgs --device cuda
tools/ssenv ss-train efficientgs --data ~/datasets/mipnerf360/native/garden --output runs/x \
    --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python reproductions/efficientgs/plugin/smoke_schedule.py efficientgs <data> <out>
```
Ablation configs: `selective: false` (3DGS statistic), `prune: false`, `sparse_sh: false`
(global +1/1000 steps after 15500), `sparse_way: random`, `top_k`, `sparse_rate`.
