# Gaussian Surfels (plan #62): build report

**Outcome: matched.** All 15 DTU scans were run.
- Mean Chamfer distance: **0.916 mm, against 0.883** for the paper's per-scan mean (paper Table 1 reports 0.88).
- Test PSNR / SSIM: **32.38 / 0.940, against 32.51 / 0.942** (15-scan average, paper Table 3).
- 14 of 15 scans are within ±0.13 mm of the paper's Chamfer. scan40 is +0.37.

## Results
**Setup**
- 15000 steps on `dtu-gsurfels`, the NeuS2 split with 7–8 test views held out, as the paper states in `04.exp.tex:72`.
- One run per scan gives both the mesh Chamfer and the test PSNR.
- Runs are in `runs/gsurfels-dtu/` and `runs/gsurfels-nogeo-dtu/`. Each run has its own `dtu_cd.json`.

| scan | CD ours / paper (Δ) | test PSNR |
|---|---|---|
| 24 | 0.75 / 0.66 (+0.09) | 29.46 |
| 37 | 0.85 / 0.93 (−0.08) | 26.81 |
| 40 | 0.91 / 0.54 (+0.37) | 29.74 |
| 55 | 0.47 / 0.41 (+0.05) | 31.98 |
| 63 | 0.94 / 1.06 (−0.12) | 33.84 |
| 65 | 1.19 / 1.14 (+0.05) | 32.00 |
| 69 | 0.95 / 0.85 (+0.09) | 29.87 |
| 83 | 1.18 / 1.29 (−0.11) | 36.60 |
| 97 | 1.33 / 1.53 (−0.20) | 28.50 |
| 105 | 0.81 / 0.79 (+0.02) | 33.49 |
| 106 | 0.92 / 0.82 (+0.10) | 34.67 |
| 110 | 1.52 / 1.58 (−0.06) | 32.09 |
| 114 | 0.56 / 0.45 (+0.11) | 31.39 |
| 118 | 0.72 / 0.66 (+0.06) | 37.49 |
| 122 | 0.66 / 0.53 (+0.13) | 37.80 |
| **mean** | **0.916 / 0.883 (+0.03)** | **32.38** (paper 32.51) |

Mean LPIPS is 0.148; the paper reports no LPIPS.

**Control (`gaussian-surfels-nogeo`).** This is the same pipeline trained with only the photometric and mask losses, without the geometry terms (L_c, L_n, L_o):

| scan | CD (mm) | test PSNR |
|---|---|---|
| 24 | 2.15 | 27.74 |
| 37 | 1.41 | 27.14 |
| 106 | 1.84 | 35.02 |

The paper's 3DGS row is 2.89 / 2.65 / 2.54 on the same scans. Removing the geometry terms increases Chamfer 1.7–2.9×, so they are what makes the surfaces accurate.

# Build notes
- **Paper:** arXiv 2404.17774.
- **Official code:** turandai/gaussian_surfels @ 01d1249.

## What was built
- **Plugin** (`plugin/`, package splatstudio-method-gaussian-surfels) provides:
  - `gaussian-surfels`, the full method;
  - `gaussian-surfels-nogeo`, a control trained with only the photometric loss L_p and the mask loss L_m;
  - the CLI `gsurfels-dtu-eval`, which builds the mesh and computes DTU Chamfer.
- **Custom CUDA:** the official rasterizer, vendored unmodified and JIT-built. It renders zero-thickness surfels with per-pixel tangent-plane depth, produces normal and opacity maps, and applies the ×10 normal gradient. `utils/cuda_utils.cu` is also vendored, for the occupancy grid used in volumetric cutting. The licence is Inria Gaussian-Splatting (non-commercial); checksums are in `csrc/SHA256SUMS`.
- **Losses:** the official-code L_p, L_m, L_c, L_n with curvature, and L_o. Where the code and the paper disagree, the code wins (mask loss, schedules, opacity bands; see paper.md §3).
- **Density control:** official prune-then-densify every 100 iterations, clone before masking, in-plane split with scale ÷1.6, opacity reset ×0.12 at 1001 and then every 1000, pruning of points never visible, and no screen-size prune.
- **Training recipe:** random visual-hull initialisation with 1e6 points, official scales and quaternions, resolution /4 → /2 → /1 at training renders 2016 and 5040, random training background.
- **Meshing** (port of `render.py`):
  1. depth/normal points from the training views;
  2. a 512³ cutting grid;
  3. pymeshlab screened Poisson at depth 10;
  4. prune at 3e-5 and close holes up to 300;
  5. Chamfer from the official `eval.py` (DTUeval).
- **Data:** the authors' DTU release from HuggingFace, which includes masks and precomputed Omnidata normals, so no Omnidata weights were needed. `tools/prepare_dtu.py` converts it to `~/datasets/gsurfels_dtu/native` (15 scans, dataset key `dtu-gsurfels`). It uses the NeuS2 test split, with validation as copies of test, and images masked on black (the paper's PSNR protocol).
- **Extra dependencies outside the env:** pymeshlab in `~/models/gsurfels-pylib`; open3d and sklearn in `~/models/dtueval-pylib`; loaded through a sys.path fallback.

## Checks
- `ss-inspect` reports no issues for both methods. 6 pytest tests pass, and ruff is clean.
- `ss-verify` passes all 8 probes across the 7 contributions (`declared_checks_passed`):
  - per-pixel depth error 1.0e-3, against 4.3e-2 with per-pixel depth turned off;
  - normal-gradient ratio −10.01;
  - occupancy grid matches an independent reimplementation to 1e-6.
- **Smoke test** (scan24, 2000 steps, all at 1/4 resolution, evaluated at full resolution):
  - test PSNR / SSIM / LPIPS = 22.17 / 0.691 / 0.395;
  - 44k Gaussians;
  - 267 s on a shared GPU;
  - Chamfer 2.08 mm (the paper's final number is 0.66).
- **Compressed-schedule run** (600 steps): the full-resolution path works (2 GB peak) and Chamfer is 3.47 mm.

## Fidelity gaps
1. **View sampling:** with replacement here; the official code uses a per-epoch stack. As a result, the never-visible prune window is "all training views seen" rather than the last 100 iterations.
2. **Update order:** the optimizer steps before density control.
3. **Normal-prior mask:** the official in-place `mono *= mask` compounding is not replicated.
4. **Images:** masked images are uint8.
5. **SH and seeds:** the SH degree rises one step later. Seeds differ (the official code is unseeded).
6. **Final densify:** the official densify at iteration 15000, after the save, is excluded.
7. **Meshing details:** Poisson runs with threads=16, and neighbour search uses cKDTree instead of pytorch3d knn.
8. **Chamfer training split (settled by review-1):** the paper holds out the NeuS2 test views on DTU for both CD and PSNR (`04.exp.tex:72`: "We follow NeuS2 to leave 7∼8 images for testing on DTU"). Table 4's "Full" row (CD 0.882, PSNR 32.51) agrees with Tables 1 and 3. The official README default (train on all images) is generic usage, not the paper's protocol.
9. **BlendedMVS:** the Table 2 benchmark is not targeted (data is on HuggingFace, 1.9 GB).
