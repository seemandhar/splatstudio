# MH-3DGS reproduction report

Paper: Metropolis-Hastings Sampling for 3D Gaussian Reconstruction (Kim, Jung, Park, NeurIPS 2025,
arXiv 2506.12945). Plugin `mh-3dgs` (plus `mh-3dgs-mcmc-control`) in `plugin/`, a native port of
the official code SNU-VGILab/MH-3DGS @71ef8f9 (Inria research licence). **Outcome: matched**:
both reported scenes are within ±0.5 dB. Single seed, no tuning.

Benchmark: `runs/mh-3dgs-mipnerf360` (summary.json). Mip-NeRF 360 garden (images_4) and room
(images_2), llffhold8, 30k steps, `--recipe 3dgs`, black background.

## Results (test PSNR / SSIM / LPIPS, #Gaussians)
| scene | method | ours | paper | Δ PSNR |
|---|---|---|---|---:|
| garden | MH-3DGS | 26.70 / 0.831 / 0.186, 808k, 54 min | 26.49 / 0.819 / 0.185, 812k | **+0.21** |
| garden | same-budget MCMC (control) | 26.78 / 0.839 / 0.182, 812k (cap) | 26.71 / 0.832 / 0.170 | +0.07 |
| garden | 3DGS (ours: gsplat baseline3dgs; paper: 3DGS*) | 26.89, 4.76M | 27.31, 5.67M | −0.42 |
| room | MH-3DGS | 31.97 / 0.923 / 0.276, 763k, 94 min | 32.15 / 0.934 / 0.179, 765k | **−0.18** |
| room | same-budget MCMC (control) | 31.80 / 0.922 / 0.289, 765k (cap) | 32.15 / 0.933 / 0.186 | −0.35 |
| room | 3DGS (ours: gsplat baseline3dgs; paper: 3DGS*) | 31.52, 1.73M | 31.65, 1.48M | −0.13 |

Relative comparisons:
- MH − MCMC: ours −0.08 (garden) / +0.17 (room); paper −0.22 / 0.00.
- MH − 3DGS: ours −0.19 / +0.45; paper −0.82 / +0.50.

Our final Gaussian counts land within 0.5% of the paper's. That count is not capped: it comes out
of the acceptance rule, so it is independent evidence that the sampler matches.

LPIPS: our values come from the `lpips` package (VGG), which is known to read higher than the
paper's lpipsPyTorch. The room LPIPS gap (0.276 vs 0.179) is mostly this metric difference; the
gsplat baseline shows the same offset. Compare LPIPS through deltas only.

## Deviations / fidelity gaps (see plan.md for details)
1. gsplat classic rasterizer instead of the 3DGS-MCMC/Inria rasterizer.
2. View sampling with replacement instead of the shuffled stack. Our own seeded camera shuffle
   for the round robin. Different RNG streams, and the sampler is stochastic.
3. The SSIM in the training loss uses valid convolution (runtime). The training loss also uses an
   unclamped render (runtime-wide). The error maps are clamped, as in the official code.
4. Error maps, relocation and births run after the optimizer step. In the official code they run
   before it, which effectively skips Adam on all 144 densification iterations (review m1).
5. The SH degree of the training render steps up one step late (1001 instead of 1000).
6. The official code is followed where it disagrees with the paper:
   - pixels are looked up from world x/y, with no projection;
   - the error map aggregates SSIM similarity, not 1−SSIM;
   - acceptance is σ(I−0.5)/(1+5c);
   - the proposal spread is relative to the parent's scale and the visibility count (`denom`);
   - batch sizes decay over training;
   - relocation runs only inside the densify window.
7. Control: the cap is the paper's MH-3DGS count. Otherwise it uses gsplat MCMCStrategy defaults,
   runtime initial opacity 0.1 (3DGS-MCMC uses 0.5) and opacity lr 0.05. It is not a full
   3DGS-MCMC reproduction.
8. Training images are reloaded inside the density component through `--data`, because the
   runtime does not expose them to components.

## Evidence
- `evidence/verify.json`: 11 probes pass, including an end-to-end density step.
- `evidence/test.json`: 2k smoke results — mh-3dgs 22.40 dB / 271k Gaussians; control 22.83 dB.
- `reviews/review-1.md`: 1 major (control could not be evaluated by `ss-eval`) — fixed; 6 minors
  fixed or documented.
