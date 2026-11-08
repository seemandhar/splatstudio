# splatstudio-method-augs

Native SplatStudio port of **AuGS — Augmented Radiance Field: A General Framework for Enhanced
Gaussian Splatting** (Yang, Wu, Zhou, Huang; ICLR 2026, arXiv 2602.19916), variant
"Ours (3DGS, sh=3)". Official code: github.com/XiaoXinyyx/AuGS_Code (read for numerics, not run).

## Mechanism
One `ss-train` job: 30 000 steps of the built-in gsplat 3DGS method, then at step 30 000
(`augs-density.augment`):
1. render all training views (RGB + T-threshold "median" depth; CUDA `csrc/contrib_stats.cu`,
   adapted from gsplat 1.5.3 RasterizeToPixels3DGSFwd.cu, Apache-2.0);
2. error-driven 2D Gaussian fitting over each render (`fit2d.py`), budget 10 % of the base count
   split ∝ view loss, new kernels sampled ∝ loss²;
3. inverse splatting (`inverse.py`): SLINK clustering, weighted PCA, Eq. 6 scale calibration,
   lobe initialisation;
4. joint optimisation for 30 × |train| steps: augmented Gaussians fully optimised, originals
   opacity-only, sparse Adam without bias correction (`optim.py`); view-dependent opacity
   (Eq. 1) in `renderer.py`.

So `--steps` must be `30000 + 30 * |train views|` (bicycle 35070, room 38160); the density
refuses other values. Training images are reloaded from `--data`. `<output>/base_test.json`
holds the test metrics of the base model at step 30 000 (measurement only).

## Config (density `augs-density`, paper / code values)
ratio 0.1; depth T threshold 0.7 (code; paper 0.5); 2D lrs xy 0.001·(H+W)/2, colour 0.01,
opacity 0.02, scale 0.001, rotation 0.02; +20 % kernels / 200 steps, 1000 finetune steps,
prune opacity ≤ 0.005 or ≤ 25 px; T = clip(7θ̄/π, 0.01, 1), β = 0; joint lrs xyz 1.6e-5·extent,
f_dc 1e-3, f_rest 5e-5, opacity 0.02, scale 2e-3, rotation 5e-4, d 1e-3, T 2e-4, β 2e-3.

## Commands
```bash
pip install -e .
ss-inspect augs && ss-verify augs --device cuda
ss-train augs --data ~/datasets/mipnerf360/pilr/bicycle --output runs/x --steps 35070 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
`augs-smoke` is a short schedule for smoke tests only.

## Fidelity gaps
See `../plan.md` (white background, sampling with replacement, gsplat 3DGS base vs official
train_3dgs.py, gsplat 2D rasterization clamp/termination, is_used via nonzero opacity gradient,
float32 clustering distances, and a few guarded edge cases).
