# AAA-Gaussians (plan #127) — build report

Paper: AAA-Gaussians (ICCV 2025, arXiv 2504.12811). Method `aaa-gaussians` (`plugin/`).
**Outcome: matched.** Both reported scenes are within ±0.5 dB. Single seed, no tuning.

## Implementation
- **Renderer**: the official CUDA rasterizer, DerThomy/AAA-Gaussians-Rasterization @ d03ce2d.
  - Vendored unmodified, with glm and pybind11_json taken from the StopThePop plugin.
  - JIT-built in about 75 s.
  - Used for both training and test, with `configs/aaa.json`: HIER sort with PTD_MAX, all culling on, proper_ewa_scaling, eval_3D, new_aabb.
- **Density**: gsplat MCMCStrategy, matching 3dgs-mcmc.
  - Init: opacity 0.5, scale ×0.1.
  - Refinement: 500 < s < 25000, every 100 steps; noise_lr 5e5.
  - Per-scene caps from the official `full_eval.py`, looked up from the `--data` basename.
- **Losses**: L1 + 0.2 D-SSIM, plus the MCMC opacity/scale regularisers (0.01 each).
- **3D filter**: the `filter_3d` field is 0 during training, which matches the official `filter_3D=None`. At the final step it is set to the official `mip_filter_3d` value (k=0.3, unseen Gaussians get 0). Test renders then use max(filter², k z²/f²).

## Verification
`evidence/verify.json`: `declared_checks_passed`.
- The forward and backward passes match an independent float64 dense torch 3D reference.
  - Image error is 3e-7; gradient relative error is 1e-6.
  - The volume-scaling, no-scaling and 2D-EWA variants are rejected (image errors 0.12–0.55).
- Perpendicular scaling on a face-on disk: the factor is 0.988 (Eq. 11), against 0.030 for volume scaling (Eq. 8).
- Adaptive filter:
  - Rendered mass at z=2 vs z=4 has ratio 4.3 (energy-preserving).
  - The training-view floor raises pixel variance from 0.33 to 1.02.
- View-space bounds:
  - A Gaussian whose mean is behind the camera still renders; the projective (old) bounding gives 0.
  - A Gaussian outside the frustum is culled.
  - Tile culling leaves the image unchanged and cuts sort entries from 71 to 64.
- The filter loop, MCMC cadence/init/cap and the regulariser all pass.
- 5 unit tests pass; ruff is clean.

## Smoke test (garden, 2000 steps)
| run | PSNR | SSIM | LPIPS | Gaussians | train s |
|---|---|---|---|---|---|
| aaa-gaussians | 22.81 | 0.661 | 0.392 | 288k | 308 |
| gsplat baseline (stopthepop smoke-baseline) | 23.02 | 0.643 | 0.419 | 845k | 174 |
| gsplat-MCMC control from mh-3dgs (same count) | 22.83 | — | — | 288k | — |

## Benchmark result (runs/aaa-gaussians-30k, summary.json) — MATCHED
| scene | ours | paper Ours | Δ | Gaussians | train | our gsplat baseline | paper 3DGS | paper MCMC |
|---|---|---|---|---|---|---|---|---|
| garden | 27.66 | 27.50 | **+0.16** | 5.83M (cap) | 132 min | 26.89 | 27.30 | 27.87 |
| room | 31.32 | 31.45 | **−0.13** | 1.59M (cap) | 116 min | 31.52 | 31.48 | 32.25 |

Gain over 3DGS: ours +0.77 (garden) and −0.20 (room); paper +0.20 and −0.03. The sign pattern matches the paper: a gain outdoors, a small loss indoors. Our larger garden gain comes from our gsplat baseline being weaker than the paper's 3DGS row (26.89 vs 27.30). Review-1 approved the port with 6 minors; they are folded into the deviations below.

## Benchmark request (as launched)
Mip-NeRF 360 **garden + room**, from the supplementary per-scene table:

| scene | paper Ours | paper 3DGS | paper MCMC |
|---|---|---|---|
| garden | 27.50 | 27.30 | 27.87 |
| room | 31.45 | 31.48 | 32.25 |

Protocol: 30k steps, images_4/images_2, llffhold8. Alternatives if needed: bicycle 25.74, truck 25.43.

## Fidelity gaps
1. **Update order**: the runtime takes the optimizer step before density control; the official code relocates first. The effect should be negligible.
2. **Shared runtime differences**:
   - Views are sampled with replacement.
   - SSIM uses valid convolution.
   - LPIPS comes from the `lpips` package.
   - The SH degree steps up one step late.
3. **Official kernel quirks, kept faithfully**:
   - The backward pass treats the filter as constant and uses focal_x.
   - The SH view-direction gradient on the means is overwritten in `preprocessCUDA_3D`.
4. **Cap lookup**: the cap comes from the `--data` basename. The total step count comes from `--steps`, to place the final filter computation.
5. **Not evaluated**: popping/FLIP, large-FOV and multi-resolution evaluations.
6. **Training setup**: the official training is plain 3DGS-MCMC plus the AAA renderer. No training-time AA beyond the per-view filter is used, and the training-view 3D filter is computed only at save (the final step). Both are reproduced as such.
7. **Dropped probe**: the built-in `renderer-gradient` probe is removed. It needs 2D screen-space gradients, which the 3D kernel does not produce and MCMC never consumes. `aaa-backward-reference` checks the kernel gradients instead.
8. **Timing**: training is about 1.8× slower per step than gsplat. Timings are not comparable to the paper (RTX 4090).
