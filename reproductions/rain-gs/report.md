# RAIN-GS — native SplatStudio reproduction report

Paper: *Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting* (arXiv 2403.09413).
Official code: github.com/cvlab-kaist/RAIN-GS (MIT), path `--ours_new --train_from random`; ported to
`plugin/ss_rain_gs` (method `rain-gs`). **Outcome: matched** (both reported scenes within ±0.5 dB).
Not separately reviewed (no splat-reviewer pass).

## Results (Mip-NeRF 360, 30k steps, random SLV init N=10, test split llffhold8)
| scene | ours RAIN-GS | paper RAIN-GS (random) | Δ | paper 3DGS (SfM) | paper 3DGS (random) | our gsplat (SfM) baseline | Gaussians | train time |
|---|---|---|---|---|---|---|---|---|
| garden (outdoor, images_4) | 26.83 | 26.884 | −0.05 | 27.410 | 23.217 | 26.89 | 7.60M | 88 min |
| counter (indoor, images_2) | 28.52 | 28.529 | −0.01 | 28.700 | 23.608 | 28.85 | 1.45M | 39 min |

Paper columns are from Table 1 (tab:main_mip360). Runs: `runs/rain-gs-mipnerf360`. The paper finds RAIN-GS from
random init 0.53 / 0.17 dB below 3DGS-SfM. We see 0.06 / 0.33 dB below our gsplat SfM baseline:
the same sign and size, with one seed per scene. Smoke test (2k steps, garden): 16.26 dB vs baseline 23.02.
That gap is expected during the 10k warm-up because training starts from 10 Gaussians.

## Deviations from the paper / official code
- **SLV init cube from training cameras only.** The official code uses the min/max over all COLMAP
  image tvecs, test views included. The runtime only exposes training cameras, so the cube is
  slightly smaller.
- **ABE-Split follows the code:** the third Gaussian is pushed from the **world origin**
  (`mu * 0.3 * extent`), not from the bounding-box centre B_0 given in Alg. 2.
- **Black background**, as in the official default. Our gsplat baseline runtime uses white.
- Initial opacity 0.1 (code) instead of InverseSigmoid(1) (Alg. 1).
- Native-runtime differences shared with the baseline:
  - the optimizer step runs before densification;
  - views are sampled randomly with replacement;
  - SSIM uses valid convolution;
  - LPIPS comes from the `lpips` package.
- gsplat rasterizer differences: per-axis 3σ radius instead of the Inria max-eigenvalue circle; near plane 0.01 instead of 0.2.
- The Inria screen-size prune is omitted because it is a no-op in the official code. The world-size prune (>0.1·extent after step 3000) is kept.
- One seed per scene. With N=10 random points, seed variance is likely high.
- Only 2 of the 9 M360 scenes were benchmarked (scene rule). We make no claim about the T&T/DB or SfM-subset ("cluster", "<10%") variants.

## Evidence
`evidence/verify.json` (8 probes, declared_checks_passed), `evidence/test.json` (smoke), `paper.md`,
`paper_results.json`, `plan.md`.
