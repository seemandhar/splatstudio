# splatstudio-method-mh-3dgs

Native SplatStudio port of **Metropolis-Hastings Sampling for 3D Gaussian Reconstruction**
(Kim, Jung, Park, NeurIPS 2025, https://arxiv.org/abs/2506.12945) following the official code
https://github.com/SNU-VGILab/MH-3DGS (commit 71ef8f9; Inria Gaussian-Splatting research license).
Paper fidelity unverified.

Methods:
- `mh-3dgs` — MH birth sampler (coarse+fine proposals picked by a multi-view importance score,
  accepted with `sigmoid(I-0.5)/(1+5 c(v'))`), MCMC relocation of opacity<=0.005 Gaussians,
  MCMC opacity/scale regularisers, black background, opacity lr 0.025. No cap.
- `mh-3dgs-mcmc-control` — gsplat MCMCStrategy capped at the paper's MH-3DGS per-scene count
  (the paper's same-budget 3DGS-MCMC row), same regularisers and renderer; gsplat MCMCStrategy
  defaults otherwise, runtime init opacity 0.1 (3DGS-MCMC: 0.5). The cap is resolved from the
  `--data` basename at initialize(), so `ss-eval --run` works without `--data`.

All hyper-parameters (with sources) are in `ss_mh_3dgs/core.py:DEFAULTS`; paper vs code
differences and fidelity gaps: `../paper.md`, `../plan.md`.

The density component reloads the training images from `ss-train --data/--downscale` (or config
`data`/`downscale`) for the multi-view error maps.

```bash
ss-verify mh-3dgs --device cuda
ss-train mh-3dgs --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs \
  --num-gaussians 10000000 --max-gaussians 12000000
```
