# Native runtime calibration: `gsplat --recipe 3dgs` vs 3DGS paper (Mip-NeRF 360, 30k)

Experiment: `runs/baseline-mipnerf360-30000` (seed 0, llffhold8, outdoor images_4 / indoor
images_2, all SfM points, white background, LPIPS-VGG). Reference: 3DGS supplementary per-scene PSNR.

| scene | ours PSNR | 3DGS paper | gap | SSIM | LPIPS | Gaussians (M) | train min |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 24.79 | 25.25 | −0.46 | 0.754 | 0.239 | 9.25 | 104 |
| flowers | 21.58 | 21.52 | +0.06 | 0.605 | 0.356 | 5.75 | 65 |
| garden | 27.12 | 27.41 | −0.29 | 0.862 | 0.125 | 7.67 | 87 |
| stump | 26.24 | 26.55 | −0.31 | 0.759 | 0.250 | 6.60 | 67 |
| treehill | 22.25 | 22.49 | −0.24 | 0.626 | 0.351 | 7.00 | 72 |
| room | 30.98 | 30.63 | +0.35 | 0.914 | 0.297 | 2.33 | 53 |
| counter | 29.02 | 28.70 | +0.32 | 0.909 | 0.256 | 1.76 | 58 |
| bonsai | 31.96 | 31.98 | −0.02 | 0.942 | 0.256 | 1.94 | 30 |
| kitchen | 30.13 | 30.32 | −0.19 | 0.916 | 0.163 | 2.28 | 68 |
| **mean (9)** | **27.12** | **27.21** | **−0.09** | 0.810 (paper 0.815) | 0.255 (paper 0.214) | | |

Mean gap over all 9 scenes: **−0.09 dB PSNR**, SSIM −0.005, LPIPS +0.041 (LPIPS-VGG from the `lpips` package on full test images; 3DGS's LPIPS numbers are known to be sensitive to implementation, so LPIPS is compared via deltas only). Outdoor scenes are 0.2–0.5 dB low and grow
~1.3–1.5× more Gaussians than 3DGS reports (likely densification statistics differences between
gsplat's DefaultStrategy and the original code); indoor scenes are 0.3 dB high. Single seed.
Use per-scene offsets (`summary.json` → `baseline_offset`) when interpreting a method's gap.

## v2: runtime with the opacity-reset fix (`baseline3dgs-*-30000`) — use this one

gsplat 1.5.3 `DefaultStrategy` never resets opacity (operator-precedence bug); SplatStudio's
density wrapper now performs 3DGS's reset (every 3000 steps before 15000).

| dataset | scene | ours PSNR | 3DGS paper | gap | SSIM | LPIPS | Gaussians (M) | v1 (no reset) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mipnerf360 | bicycle | 24.92 | 25.25 | −0.33 | 0.757 | 0.243 | 6.59 | 24.79 |
| mipnerf360 | flowers | 21.63 | 21.52 | +0.11 | 0.607 | 0.359 | 3.86 | 21.58 |
| mipnerf360 | garden | 26.89 | 27.41 | −0.52 | 0.860 | 0.131 | 4.76 | 27.12 |
| mipnerf360 | stump | 26.23 | 26.55 | −0.32 | 0.761 | 0.258 | 4.51 | 26.24 |
| mipnerf360 | treehill | 22.28 | 22.49 | −0.21 | 0.627 | 0.361 | 5.07 | 22.25 |
| mipnerf360 | room | 31.52 | 30.63 | +0.88 | 0.916 | 0.298 | 1.73 | 30.98 |
| mipnerf360 | counter | 28.85 | 28.70 | +0.15 | 0.907 | 0.260 | 1.22 | 29.02 |
| mipnerf360 | kitchen | 31.34 | 30.32 | +1.02 | 0.925 | 0.162 | 1.42 | 30.13 |
| mipnerf360 | bonsai | 31.89 | 31.98 | −0.09 | 0.941 | 0.258 | 1.37 | 31.96 |
| tandt | truck | 25.21 | 25.19 | +0.02 | 0.881 | 0.179 | 2.49 | |
| tandt | train | 21.52 | 21.10 | +0.42 | 0.818 | 0.235 | 1.26 | |
| db | drjohnson | 28.76 | 28.77 | −0.01 | 0.897 | 0.326 | 3.61 | |
| db | playroom | 29.80 | 30.04 | −0.24 | 0.908 | 0.317 | 2.24 | |

Mip-NeRF 360 mean **27.28 vs 27.21 (+0.08 dB)**. Gaussian counts are now close to 3DGS's
(v1 over-grew outdoors by 1.3–1.5×). Outdoor scenes remain 0.2–0.5 dB low; indoor room/kitchen
are 0.9–1.0 dB high. T&T and DB are within ±0.42 dB. Single seed.
