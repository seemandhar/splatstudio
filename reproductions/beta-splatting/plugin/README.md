# splatstudio-method-beta-splatting

Deformable Beta Splatting (Liu, Sun, Chen, Wang, Feng; SIGGRAPH 2025; arXiv 2501.18630) as a
native SplatStudio plugin. Ported from the official code (Apache-2.0)
https://github.com/RongLiu-Leo/beta-splatting @ 431d575 — CUDA from its gsplat fork
(`rasterize_to_pixels_{fwd,bwd}.cu`, `spherical_beta.cuh`), MCMC from `train.py` and
`scene/beta_model.py`. Paper fidelity unverified.

## Mechanism → code
| contribution | code |
|---|---|
| Beta kernel `alpha = min(.999, o (1-r^2)^{4e^b})`, r^2 < 1, learnable `b` (field `betas`, lr 1e-3) | `csrc/beta_raster.cu` (JIT, `kernels.py`), `renderer.py` |
| Spherical Beta: `c0 + sum_m dot^{4e^{b_m}} softplus(c_m)`, 2 lobes (field `sb_params [2,6]`, lr 2.5e-3) | `csrc/beta_raster.cu` SB kernels, `renderer.colors_sb` |
| Kernel-agnostic MCMC: opacity-only relocation/5% growth to cap, `(1-o)^100` noise every 100 its in (500, 25000) | `density.py` |
| Regularisers 0.01 mean(o) + 0.01 mean(exp s), 500 < step < 25000 | `losses.py` |

Config (code defaults): sh_degree 0, sb 2, black background, init opacity 0.5, b = 0, lobe
theta/phi uniform, base LRs as 3DGS (means 1.6e-4 x extent decaying 100x). Methods differ only by
`cap_max`: `beta-splatting` 1M, `-garden` 5M, `-room` 1.5M, `-bicycle` 6M (official benchmark.py).

## Validation
`ss-verify beta-splatting --device cuda`: CUDA forward/backward vs a dense float64 torch reference
(rel. err ~3e-7 for means2d, conics, colours, opacities, betas), Beta support/footprint,
Spherical Beta vs torch formula, MCMC opacity rule / no scale change / Adam reset / noise formula,
regulariser schedule.

## Fidelity gaps
- Runtime steps Adam before density control; official densifies first and its Parameter re-wrap
  drops `.grad`, skipping the Adam step on densification iterations (~245 of 30k).
- Loss SSIM is the runtime's valid-conv SSIM (official: fused-ssim, same padding).
- Random views with replacement (official: shuffled stack); means LR updated one step late.
- Projection/tiling from gsplat 1.5.3 (official fork: 1.3.0); 1-sigma box instead of circle
  (same covered pixels).
- Only the "30K" mode; the official "full" mode selects checkpoints on test PSNR.

## Commands
```
tools/ssenv pip install -e reproductions/beta-splatting/plugin
tools/ssenv ss-verify beta-splatting --device cuda
python3 tools/bench.py launch --exp beta-splatting-garden --methods beta-splatting-garden \
  --dataset mipnerf360-pilr --scenes garden --steps 30000
```
