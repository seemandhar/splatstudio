# 3DGS-MCMC (plan id 25) — reproduction report

Paper: 3D Gaussian Splatting as Markov Chain Monte Carlo (Kheradmand et al., NeurIPS 2024,
arXiv 2404.09591). Official code: github.com/ubc-vision/3dgs-mcmc @7b4fc9f.

**Outcome: matched.** Both benchmarked scenes are within ±0.5 dB of the paper. This is a single seed with no tuning; the paper reports the mean of 3 runs.

## Results (mipnerf360-pilr, 30k steps, method `mcmc-3dgs`, run `runs/mcmc-3dgs-pilr`)

| scene | cap | ours PSNR / SSIM / LPIPS | paper "Ours (SfM)" | Δ vs paper | our gsplat baseline | our gain | paper gain over its 3DGS run | time |
|---|---|---|---|---|---|---|---|---|
| garden | 5.2M | 28.30 / 0.886 / 0.103 | 28.16 / 0.89 / 0.10 | +0.14 | 27.35 | +0.95 | +0.43 (27.73) | 86 min |
| room | 1.5M | 32.07 / 0.935 / 0.253 | 32.48 / 0.94 / 0.25 | −0.41 | 31.69 | +0.38 | +0.78 (31.70) | 33 min |

The final Gaussian counts equal the caps. Paper numbers come from Supp. Tab. allres. The paper's Mip-NeRF 360 average covers 7 scenes (no flowers or treehill); only 2 scenes were run here, so no average is claimed. Our gains over our baseline differ from the paper's gains over its own 3DGS run: larger on garden, smaller on room. The garden figure partly reflects our weaker baseline (27.35 vs the paper's 27.73).

## Why the built-in `gsplat-mcmc` was not used
gsplat's MCMCStrategy matches the official relocation and birth step. That step runs every 100 steps in (500, 25000) and adds 5% more Gaussians each time. It also matches the official SGLD noise (noise_lr 5e5, gated by opacity, shaped by the covariance). The built-in config differs from the official recipe in five ways, all fixed in plugin `mcmc-3dgs` (`plugin/ss_mcmc_3dgs`):
1. **Cap:** the built-in uses 1M on every scene. The plugin uses the per-scene `cap_max` from the official `configs/<scene>.json`, looked up from the `--data` scene name.
2. **Regularisers:** the built-in has none. The plugin adds the L1 opacity regulariser (0.01; 0.001 on Deep Blending) and the scale regulariser (0.01).
3. **Initial opacity:** 0.5 (runtime default 0.1).
4. **Initial scale:** 0.1 × sqrt(mean 3-NN squared distance). The 3dgs recipe uses 1×.
5. **Background:** black (runtime default white).

The protocol also changed: the official configs use Inria `-r 4` / `-r 2` PIL resizing of the full-resolution images, so the benchmark used the `mipnerf360-pilr` dataset and the `baseline3dgs-mipnerf360-pilr-30000` baseline.

The plugin reuses pieces from the mh-3dgs control: the cap lookup pattern, the renderer and the regulariser form. It was not reviewed separately.

Checks:
- ss-inspect is clean.
- 4 pytest tests pass.
- ss-verify returns `declared_checks_passed`.
- Probes cover the cap table, the initialisation against brute-force numpy, the regulariser gradients, the black background, and the relocation/birth/noise schedule.
- A 2000-step smoke run on room finished (`evidence/smoke-room`).

## Remaining gaps
- Training views are sampled with replacement; the official code draws from a shuffled stack.
- Relocation runs after the Adam step (official: before), and the noise uses the learning rate after that step's decay.
- The SH degree goes up at steps 1001/2001/3001 instead of 1000/2000/3000.
- We use gsplat's rasterizer, not the official diff-gaussian-rasterization.
- Single seed against the paper's mean of 3 runs. SSIM and LPIPS are given to only 2 decimals in the paper. Our runtime has a known LPIPS offset.
- Only SfM initialisation was run; the paper's random-initialisation column was not reproduced.
