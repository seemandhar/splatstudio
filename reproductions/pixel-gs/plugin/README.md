# splatstudio-method-pixel-gs

Native SplatStudio port of **Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian
Splatting** (Zhang et al., ECCV 2024, https://arxiv.org/abs/2403.15530), following the official
code https://github.com/zhengzhang01/Pixel-GS (05c9d55, Inria 3DGS licence). Paper fidelity
unverified.

Mechanism (density component only; gsplat renderer and L1+0.2 D-SSIM unchanged):
- Pixel-aware gradient (Eq. 7): `grad2d += m*|g|`, `count += m`, m = pixels the Gaussian is
  blended into in that view (CUDA kernel `csrc/adc_stats.cu`, reused from the SplatStudio
  Improving-ADC port).
- Scaled gradient field (Eq. 10): `|g| *= clip((z/(0.37*radius))^2, 0, 1)`, radius = 3DGS camera
  extent; applied to the statistic only (official code scales the screen-space dummy gradient).
- 3DGS defaults otherwise: tau 2e-4, densify 500..15000 every 100, reset 3000.

Methods: `pixel-gs`, ablations `pixel-gs-pixel-only`, `pixel-gs-depth-only`.

Fidelity gaps: gsplat numerics (alpha clamp 0.999 vs 0.99) for rendering and pixel counts; white
background; view sampling with replacement; `>` vs `>=` threshold; SSIM/LPIPS implementations.

```
tools/ssenv ss-train pixel-gs --data ~/datasets/mipnerf360/pilr/bicycle --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
