# 3D Student Splatting and Scooping (SSS) — reproduction report

Plan id 126 · arXiv 2503.10148 (CVPR 2025) · plugin `reproductions/student-splatting/plugin` (`ss_student_splatting`) · **Outcome: matched** (both reported scenes within ±0.5 dB PSNR).

## Results (test split, llffhold 8, `mipnerf360-pilr` = official PIL resize, outdoor -r 4 / indoor -r 2)

| scene | method (official config) | steps | primitives | PSNR | SSIM | LPIPS | paper PSNR / SSIM / LPIPS | ΔPSNR | train time |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | `student-splatting-bicycle` (cap 3M, C_b 5e8, B 15000) | 30k | 3.00M | **25.63** | 0.797 | 0.199 | 25.68 / 0.798 / 0.173 | −0.05 | 733 min |
| room | `student-splatting-room` (cap 1.5M, C_b 5e5, B 7000) | 40k | 1.50M | **32.52** | 0.939 | 0.245 | 32.57 / 0.938 / 0.167 | −0.05 | 695 min |

Runs: `runs/student-splatting-bicycle-pilr`, `runs/student-splatting-room-pilr`. Paper numbers: supp. Tables comparisons_360_psnr/ssim/lpips (full SSS, mean of 3 runs).

### Improvement over 3DGS
| scene | our gsplat baseline (`baseline3dgs-mipnerf360-pilr-30000`) PSNR / SSIM / LPIPS | our gain | paper 3DGS row | paper gain |
|---|---|---|---|---|
| bicycle | 25.45 / 0.772 / 0.236 | +0.18 dB | 25.25 | +0.43 dB |
| room | 31.69 / 0.925 / 0.278 | +0.83 dB | 30.63 | +1.94 dB |

Absolute PSNR matches the paper. The gains are smaller because our gsplat baseline is stronger than the 3DGS row the paper copies from the 3DGS paper (+0.20 dB on bicycle, +1.06 dB on room). The room gain also mixes in iterations: SSS ran 40k steps (official config) and the baseline ran 30k. SSIM matches the paper. LPIPS is worse than the paper by 0.03 (bicycle) and 0.08 (room). We use LPIPS-VGG from the `lpips` package; the paper does not state its LPIPS net or implementation, so this difference is not explained.

## What was ported (official code: github.com/realcrane/3D-student-splating-and-scooping)
- **Student's t kernel** with learnable ν = hardtanh(raw, 1, 1e4) (ν initialised to 100). Rendered by a JIT CUDA rasteriser (`csrc/t_raster.cu`) ported from the official `renderCUDA` forward and backward. It keeps the ±0.99 alpha clamp with gradients passed through unchanged, skips alpha below 1/255 and stops a pixel at T<1e-4. Projection is EWA in torch, matching Inria: no +0.3 low-pass, ±1.3·tan_fov clamp, near plane 0.2, Inria's conic backward, and the official ν critical-value radius table.
- **Scooping:** signed opacity tanh(o).
- **AdamSGHMC for the means** (`utils/sghmc.py`): burn-in without friction and with covariance-shaped noise, momentum reset at B+1, sigmoid gate, lr² schedule. Adam is used for all other parameters (opacity lr 0.005, ν lr 10 halved every 5000 steps).
- **Recycling and growth:** components with |o|<0.005 are recycled (at most 5% per event), and 5% new components are added every 100 steps between steps 500 and 25000, using the t relocation formula (`utils.cu`).
- **Loss:** photometric loss plus 0.01·mean|o| + 0.01·mean(scale).

Verification: `evidence/verify.json` passes all 9 probes. CUDA forward and all 5 gradients match a dense float64 reference to about 2e-6, and the SGHMC update matches the official code to 7e-9. Unit tests pass 8/8.

## Deviations and fidelity notes
- **Partial dν, kept as in the official code.** The official kernel's dL/dν omits the −½ln(1+base/ν)·G term of the exact derivative. We port it unchanged (`exact_nu_grad: false`; `true` gives the exact form). Dropping that term biases ν upwards, which may explain ν drifting into the thousands. Two other official behaviours are also kept: relocation squares the scale factor, where the paper squares the covariance factor; and the radius-table interpolation follows forward.cu exactly, so e(ν) is 9.925 at ν=1.
- **Review 1:** one major issue, fixed before benchmarking. The radius table was interpolated in the wrong direction. Four minor points are documented in `paper.md` §8.
- **Single seed** (seed 0). The paper reports a mean of 3 runs.
- **Room iterations:** room ran 40k iterations (official config); the shared gsplat baseline is 30k.
- **Method label:** `student-splatting` is a generic config label used only for the smoke test. Benchmark rows use the per-scene methods.
- **Runtime differences:**
  - SSIM loss uses valid convolution.
  - Training views are sampled with replacement.
  - The SH degree steps up at step 1001, not 1000.
  - Tile rectangles follow gsplat, not Inria getRect.
  - Evaluation is on float renders, not 8-bit PNGs.
  - Adam steps on the final iteration; the official code skips that step.
  - torch Adam places eps slightly differently (eps = 1e-15, negligible).
- **Not checked:** the truck scene (paper 26.41) was not run.

## Licence
The official repository is GPL-2.0. Its `diff-t-rasterization` submodule is under the Inria Gaussian-Splatting research-only licence. The ported CUDA and Python cite their sources, and this plugin is for research and evaluation use only.
