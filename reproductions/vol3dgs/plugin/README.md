# splatstudio-method-vol3dgs

Native SplatStudio port of **Volumetrically Consistent 3D Gaussian Rasterization**
(Talegaonkar, Belhe, Ramamoorthi, Antipa; CVPR 2025; https://arxiv.org/abs/2412.03378).
Ported from the official code (MIT) https://github.com/chinmay0301ucsd/Vol3DGS and its slang
rasterizer https://github.com/chinmay0301ucsd/slang-gaussian-rasterization @5eca118 (Apache-2.0).
Paper fidelity unverified.

Methods: `vol3dgs-black` (official black background; primary benchmark) and `vol3dgs`
(white background, runtime convention).

## Mechanism
- `csrc/vol_raster.cu` (JIT, adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu):
  alpha = min(0.99, 1 - exp(-2.506 kappa beta G(gamma d))) per pixel ray (official
  `evaluate_splat_volr`), analytic backward. Tiles/sorting from gsplat's projection with the
  official 3-sigma radius and near plane 0.2.
- kappa = -log(1 - 0.99 sigmoid(o)) mean(1/s) / 2.506 (official `alpha2density`).
- Colour softplus(SH + 0.5, beta 10); SH degree +1 every 2000 steps.
- Density (`density.py`): world-space ||dL/dmu|| statistic, clone 5e-5 / split 1e-7, interval 200
  (500..15000), percent_dense 0.01 x extent_mult 2, halved density for new Gaussians, split
  theta >= 0.99, prune theta < 0.05 and (after 3000) max scale > 0.01 x ext, no opacity reset,
  max 7M points; "halving" and the theta >= 0.99 test use the official density2alpha exactly as
  written (1 - exp(-x)/0.99: halved theta ~0.0102 low, the 0.99 split never fires;
  `density2alpha: inverse` for the intended math); LRs theta 0.03, sh0 0.003, shN 1.5e-4, means 1.6e-4 -> 1e-5 (x extent).

All values are Mip-NeRF 360 values of `scripts/all_datasets.sh`; override via component config.

## Fidelity gaps
White background (runtime convention; official black, `background: black` available);
runtime view sampling with replacement / valid-conv SSIM / optimizer-before-density order;
gsplat EWA projection for tile boxes (same 3-sigma + 0.3 formula); principal point from K;
CUDA built with --use_fast_math and the NaN-gradient skip also drops inf; the runtime steps the
optimizer before densification (Inria densifies first).

## Commands
```
tools/ssenv pip install -e reproductions/vol3dgs/plugin
tools/ssenv ss-verify vol3dgs-black --device cuda
tools/ssenv ss-train vol3dgs-black --data ~/datasets/mipnerf360/w1600/bicycle --recipe 3dgs \
    --num-gaussians 10000000 --max-gaussians 12000000 --steps 30000 --output runs/...
```
