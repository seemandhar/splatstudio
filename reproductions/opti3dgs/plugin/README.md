# splatstudio-method-opti3dgs

Native SplatStudio port of **Opti3DGS** — "Optimized 3D Gaussian Splatting using Coarse-to-Fine
Image Frequency Modulation" (Farooq, Guillemaut, Hilton, Volino; arXiv 2503.14475). Written from
the paper; no official code has been released. Paper fidelity unverified.

## Mechanism
The 3DGS loss (L1 + 0.2 D-SSIM) is computed against a mean-blurred training image (Eq. 2,
stride 1, size-preserving padding). The kernel follows the step schedule of Fig. 3:
K = 15 (steps 1-1000), 9 (-4000), 5 (-8000), 3 (-12000), then original images (Sec. 4).
Renderer, densification logic and evaluation are unchanged 3DGS.

| method | density settings | paper row |
|---|---|---|
| `opti3dgs` | scales lr 0.01, refine_every 500, grow_grad2d 1e-4 (Sec. 4) | Table 1 "3DGS + Ours*" |
| `opti3dgs-filter` | 3DGS defaults (0.005, 100, 2e-4) | Table 2 "Ours-Filtering" |

Loss config: `kernel_sizes` [15,9,5,3], `boundaries` [1000,4000,8000,12000], `padding_mode`
"reflect" (paper unspecified; = cv2.blur default), `ssim_weight` 0.2.

## Fidelity gaps
- Step boundaries read from Fig. 3; padding mode guessed.
- Runtime differences vs Inria 3DGS (white background, view sampling with replacement, valid
  SSIM, gsplat rasterizer): compare Δ vs our gsplat baseline with the paper's Δ.
- "densification_interval" mapped to gsplat `refine_every` only.

## Commands
```
tools/ssenv ss-verify opti3dgs --device cuda
tools/ssenv ss-train opti3dgs --data ~/datasets/tandt_db/native/truck --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000
```
