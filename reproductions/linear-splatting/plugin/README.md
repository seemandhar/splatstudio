# splatstudio-method-3dls

Native SplatStudio port of **3DLS — Beyond Gaussians: Fast and High-Fidelity 3D Splatting with
Linear Kernels** (arXiv [2411.12440](https://arxiv.org/abs/2411.12440)). Paper fidelity
unverified. No official code exists (github.com/hche8927/3DLS is the project page only).

## Mechanism → code
| paper | code |
|---|---|
| Linear kernel `max(0, 1 − D'_M)` (Eq. 9–11) | `csrc/linear_raster.cu` forward/backward, adapted from gsplat 1.5.3 `RasterizeToPixels3DGS{Fwd,Bwd}.cu` (Apache-2.0); only the falloff and its derivative change |
| Distribution Alignment `D/λ`, λ = 2.5 (Eq. 12) | renderer `lam`; bounding radius `ceil(λ(1 − 1/(255 o))·σ_axis)` in `renderer.linear_radii` |
| AGS `ω = exp(−D²)` on ∇L' (Eq. 13–14) | backward multiplies dk/dD by `exp(−c D²)` (`ags_coef` c, paper c = 1); opacity/colour gradients unscaled |
| Loss 0.6 L1 + 0.2 L2 + 0.2 (1−SSIM) (supp. Eq. 1) | `losses.py`, three loss components (logged separately) replacing `photometric` |
| Densification grow_scale3d 0.006, prune_scale3d 0.4, prune_opa 0.025 (supp. Table 6) | `splatstudio.backends.gsplat:create_density` config |

Projection (eps2d 0.3), SH, tile intersection and RGB+ED depth are gsplat 1.5.3's. The CUDA
extension is JIT-built with `torch.utils.cpp_extension.load` on first use (needs `tools/ssenv`).

## Methods
- `3dls` — paper-literal (AGS c = 1, ω = exp(−D²) with unscaled D, as Eq. 13 and Fig. 3b).
- `3dls-aags` — interpretation: AGS on the λ-aligned distance, ω = exp(−(D/λ)²) (c = 1/λ²).
  The literal ω suppresses the means2d gradient that drives densification ~100× (garden: 0.09 %
  vs 10.5 % of Gaussians above grow_grad2d); at 2k steps the population shrinks 139k → 101k
  while the paper's AGS model ends with about as many primitives as 3DGS (Table 2). The aligned
  variant halves the count relative to LK+DA (paper: −43 %).
- `3dls-noags` — ablation "LK + DA" (Table 2 row 3).

## Probes (`ss-verify`)
dense float64 torch reference (`reference.py`) vs CUDA: forward (RGB+depth, alpha; max err
~1e-6), backward without/with AGS (relative grad err ~5e-6); single-splat support radius = λσ;
footprint ratio linear/Gaussian = π λ²/3 / 2π = 1.04; loss weights; density thresholds.

## Fidelity gaps
1. No official code: radius rule, AGS scope (geometry path only), keeping gsplat's 0.999
   saturation gate, and the AGS distance (literal vs aligned) are our reading.
2. gsplat 1.5.3 instead of 1.3 (projection per-axis radii; densification equivalent).
3. Runtime: white background (gsplat trainer: black), view sampling with replacement,
   valid-convolution SSIM, Inria camera-extent scene scale.
4. 3DLS+AA (Mip-Splatting 2D filter) not implemented.

## Commands
```bash
cd plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-verify 3dls --device cuda
tools/ssenv ss-train 3dls-aags --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000
```
