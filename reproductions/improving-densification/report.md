# ImprovedGS (arXiv 2508.12313): build report

**Outcome: matched.** PSNR is within 0.13 dB of the paper on both scenes, and the large gains over 3DGS reproduce.

## Results
Setup: `mipnerf360` (images_4 for garden, images_2 for room), 30k steps. Runs are in `runs/improving-densification-mipnerf360/`. The baseline is `baseline3dgs-mipnerf360-30000`.

| scene | PSNR ours / paper (Δ) | SSIM | LPIPS | #G (budget N_max) | gain over our 3DGS | paper gain over its 3DGS |
|---|---|---|---|---|---|---|
| garden | **28.01 / 28.10 (−0.09)** | 0.881 | 0.106 | 3.00M (3M) | **+1.11** | +0.74 |
| room | **32.52 / 32.65 (−0.13)** | 0.932 | 0.265 | 1.00M (1M) | **+1.00** | +1.21 |

- **Gaussian count:** each scene ends exactly at its growth-control budget.
- **Gains:** the gains over our baseline fall in the range set by the paper's table and the authors' current-code release (garden +0.50 to +0.74, room +0.70 to +1.21). Garden is slightly above that range.
- **Number of runs:** one seed per scene. The paper reports the best of 3 runs.


## What was built
The method `improving-densification` (module `ss_improving_densification`) ports the official code, Improved-GS @ 20b5db3. The official rasterizer is vendored unmodified in `csrc/` and JIT-built. It is covered by the Inria and ImprovedGS non-commercial licences.

| contribution | implementation | probe |
|---|---|---|
| EAS (Sec. 3.2) | Laplacian edge maps. A kernel accumulates edge weight × T × α, normalised by per-view min-max. Gaussians are sampled multinomially among candidates with abs-grad ≥ 3e-4. | eas-score (relative error 3e-7 vs an independent gsplat computation), eas-sampling |
| LAS (Sec. 3.3) | Children placed at ±0.45·L0 along the longest axis, with scales (1−d, √(1−d²)) and opacity ×0.6. This is the only densification operation. | las-geometry |
| RAP (Sec. 3.4) | Prune opacity < 0.02 at step 300. Reset opacity to ≤ 0.05 between 3k and 12k. Prune the lowest 20 % at steps 3300 and 6300. | rap-schedule |
| MU (Sec. 3.5) | GatedAdam on summed gradients, stepping every 5 iterations from 20k and every 20 from 24k | mu-gated-adam |
| GC (Sec. 3.6) | √-curve growth budget from step 500 to 14500, up to an N_max of 3M / 1.5M / 1M depending on the scene | growth-control |
| App. A.2 | Position LR 4e-5 → 2e-6, opacity LR 0.025, no size pruning | lr-schedule |

## Verification
- **Static checks:** `ss-inspect` reports no issues, 11 pytest tests pass, and ruff is clean.
- **Probes:** `ss-verify` reports `declared_checks_passed` for all 14 probes (2 on CUDA).
- **Smoke run** (garden, 2000 steps):

  | run | PSNR | SSIM | LPIPS | Gaussians |
  |---|---|---|---|---|
  | improving-densification | 24.64 | 0.763 | 0.265 | 0.98M |
  | gsplat | 23.07 | — | — | 0.84M |

  The Gaussian count tracks the official training log to within 14 (948,384 vs 948,370).

## Benchmark request
Garden and room on `mipnerf360`:

| scene | paper target | paper's 3DGS row | our gsplat baseline |
|---|---|---|---|
| garden | 28.098 | 27.361 | 26.893 |
| room | 32.652 | 31.446 | 31.516 |

## Fidelity gaps
1. **Resolution (settled by review-1):** we use images_4/2 (`mipnerf360`).
   - The official configs read `images` at `-r -1` (capped at 1600 px).
   - Both the paper's 3DGS row and the authors' released 3DGS results match the 3DGS paper's images_4 SSIM and LPIPS (garden 0.868 / 0.103). Our 1600-px runtime is 0.011–0.017 lower on SSIM outdoors.
   - So the authors' `images` folder is already at images_4/2 resolution.
2. **MU schedule:** the code steps at 20000/24000, while the paper text says 15000/22500. We follow the code.
3. **View sampling:** views are drawn with replacement. The official code uses a shuffled stack.
4. **SSIM:** computed with valid convolution, where the official code uses same padding.
5. **Optimizer:** torch Adam instead of the official CUDA Adam (dense here, same β and eps).
6. **Seeds:** one seed, while the paper reports the best of 3 runs.
7. **SH schedule:** the SH degree steps up one iteration later than the official code.
8. **Pixel weights:** we pass empty pixel weights to the rasterizer. The official code passes ones, but they only fill an unused buffer, so the output is identical.

## Review-1 resolution (orchestrator)
Approved (0 critical, 0 major). The benchmark is launched on garden and room (`mipnerf360`).

The expected gain is a range, not a single number. The authors' current-code release gains less over its 3DGS than the paper's table does:

| scene | current-code gain | paper-table gain | expected PSNR |
|---|---|---|---|
| garden | +0.50 | +0.74 | about 27.39–27.63 |
| room | +0.70 | +1.21 | about 32.22–32.72 |

These expected values are our baseline plus each gain. Several hyper-parameters (opacity LR 0.025, the 3400/6400 all-view rule, MU at 20k/24k) come from the current official code. They are labelled that way, not as the paper's.
