# splatstudio-method-conegs

ConeGS (Baranowski et al., 3DV 2026, arXiv 2511.06810) is a native SplatStudio port of
https://github.com/baranowskibrt/conegs (commit f86df2f). It has not been verified against the
paper's numbers yet.

## Mechanism
1. `ss-conegs-init`: trains a per-scene iNGP (nerfacc proposal variant, 20k iterations) on the
   training views. It then places 1M Gaussians at the iNGP median depth of uniformly sampled
   training pixels, coloured with the iNGP colour for a zeroed view direction, with exact 3-NN
   log scales. The output is a hard-linked dataset copy with `points.npz`, `conegs_init.npz`,
   `conegs_ingp.pt` and `conegs_init.json` (timings).
2. `conegs` training replaces 3DGS clone/split/reset with error-guided cone densification:
   - Every step below 25k, it samples pixels with probability proportional to the L1 error.
   - Each sampled pixel becomes a new Gaussian at the iNGP median depth, sized to one
     pixel-cone footprint, with opacity 0.1.
   - Every 100 steps it prunes opacity ≤ 0.005 and then merges the new Gaussians, up to the
     1M budget.
   - The number of rays per step is `max(0.02 N, 1.2 N_added)//100 + 1`.
   - Training also uses the pre-activation opacity penalty `2e-4 * mean(logit)` and optimises
     all SH bands from step 1.

tiny-cuda-nn and nerfacc are not installed. The hash grid is a JIT CUDA kernel with tcnn
semantics (`hashgrid.py`, checked against a torch reference). The nerfacc PropNetEstimator,
importance sampling and pdf loss are ported to PyTorch (`ingp.py`).

## Config (defaults = official code)
budget 1e6 · growth 0.02 (paper Eq. 12 text says 0.2) · interval 100 · densify_until 25000 ·
prune_opacity 0.005 · new_opacity 0.1 · size_mult 1.0 · opacity penalty 2e-4 signed mean ·
iNGP: 4096 rays, smooth-L1 ×1024, Adam 1e-2 / eps 1e-15 / wd 1e-5, prop [256, 96] + 48 lindisp
0.2–1e6.

## Commands
```bash
ss-conegs-init --source ~/datasets/mipnerf360/native/garden --output ~/datasets/conegs-init/mipnerf360/garden
ss-train conegs --data ~/datasets/conegs-init/mipnerf360/garden --output runs/x --recipe splatstudio \
  --num-gaussians 10000000 --max-gaussians 12000000
```
Use `--recipe splatstudio`. The 3DGS recipe's kNN needs about 32 GB for 1M points, so the
density component restores the 3DGS camera-extent means LR itself.

## Fidelity gaps
- View sampling is random with replacement.
- Background is white; the official code uses black.
- Rasterizer is gsplat classic; the official code uses the Inria rasterizer.
- The iNGP runs in fp32 PyTorch with its own hash kernel, not tcnn fp16 FullyFusedMLP.
- 3-NN distances are exact; the official code uses the approximate simple-knn.
- SSIM uses valid convolution.
