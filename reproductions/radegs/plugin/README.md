# splatstudio-method-radegs

Native SplatStudio port of **RaDe-GS: Rasterizing Depth in Gaussian Splatting** (Zhang, Fang,
Shrestha, Liang, Long, Tan; arXiv [2406.01467](https://arxiv.org/abs/2406.01467)).
Description: "RaDe-GS native port; paper fidelity unverified".

## Code version
Official code github.com/HKUST-SAIL/RaDe-GS pinned at **`2d4bc08`** (2024-06-21), the last commit
of the paper-v2 formulation (centre-ray cosine depth, depth distortion + normal consistency).
Later commits change the depth formulation (per-pixel cosine, July 2024), drop the distortion
loss, and (2026) add PGSR-style multi-view regularisation; none of them is used.

## Mechanism → components
| paper | component | notes |
|---|---|---|
| rasterized depth d = z_c + p·Δ (Eq. 3, 14) and normal (Eq. 18–22) | `radegs-raster` renderer | official CUDA kernel vendored unmodified in `ss_radegs/csrc` (Inria/MPII Gaussian-Splatting licence, see `csrc/VENDORED.md`), JIT-built on first use (~70 s) |
| L_d, L_n (Eq. 23–25) | `radegs-reg` loss | train.py@2d4bc08: 100·mean(edge·dist/α²) + 0.05·(0.4 L_n(expected depth) + 0.6 L_n(median depth)) from step 15000 |
| photometric | `photometric` loss | 0.8 L1 + 0.2 (1 − SSIM, same padding) |
| GOF densification + Mip-Splatting 3D filter (§4.1) | `radegs-density` | verbatim GOF code in the official repo; copied from the workspace GOF port |

Config values (all in `ComponentSpec.config`): `lambda_distortion` 100, `lambda_depth_normal`
0.05 (paper text says 5; code 0.05), `depth_ratio` 0.6, `regularization_from_iter` 15000,
`densify_grad_threshold` 2e-4, `min_opacity` 0.05, `opacity_reset_interval` 3000, background black.

## Probes (`ss-verify radegs --device cuda`)
- `rasterized-depth`: kernel depth is affine in the pixel offset with the slope of
  cos θ_c · t*(r) from an exact float64 ray/Gaussian maximum (slope error 1e-7); constant
  centre depth (error 1.0) and the later per-pixel-cosine formulation (error 0.06) fail.
- `rasterized-normal`: kernel normal = normal of the intersection surface s*(r)·r (1−cos 1e-14).
- `kernel-gradients`: all six outputs and their gradients vs an independent differentiable
  float64 renderer (≤3e-5 rel.). **Official quirk kept**: the backward of the normal map w.r.t.
  the means is inexact (rel. error ≈3; a finite difference of the kernel forward agrees with the
  reference) — measured, not asserted.
- `distortion-formula`, `regularizer-schedule`, `densify-metric`, `density-recipe`.

## Fidelity gaps
1. No mesh extraction (TSDF / marching tetrahedra) and no DTU Chamfer / TnT F1 evaluation.
2. Decoupled appearance (DTU/TnT only) not ported.
3. Mip-NeRF 360 resolution protocol unstated by RaDe-GS; images_4/images_2 (GOF protocol) used.
4. Runtime: views sampled with replacement, optimizer step before densification, SH degree
   one step later; 3D-filter coefficient in log space; evaluator SSIM/LPIPS shared with the
   baseline.
5. The kernel and `depth_double_to_normal` assume cx = W/2, cy = H/2; the renderer refuses other
   cameras (holds exactly for the local Mip-NeRF 360 conversions).

## Commands
```bash
cd plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-verify radegs --device cuda
tools/ssenv ss-train radegs --data ~/datasets/mipnerf360/native/garden --output out --steps 30000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 20000000
```
