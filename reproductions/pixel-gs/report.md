# Pixel-GS (ECCV 2024, arXiv 2403.15530) — reproduction report

**Outcome: matched** (both reported scenes within ±0.5 dB of the paper's PSNR). The gain over our own baseline is smaller than the paper's gain over its 3DGS*. See below.

- Plugin: `plugin/` (methods `pixel-gs`, ablations `pixel-gs-pixel-only`, `pixel-gs-depth-only`). Port of the official code
  github.com/zhengzhang01/Pixel-GS (05c9d55): the pixel-count-weighted, depth-scaled 2D gradient statistic in a gsplat
  `DefaultStrategy` subclass. Pixel counts come from a CUDA kernel reused from `reproductions/improving-adc`.
- Protocol: `mipnerf360-pilr` (official `-r 4` outdoor / `-r 2` indoor = PIL resize), llffhold8, 30k steps,
  `--recipe 3dgs --num-gaussians 10000000`. Max-Gaussian cap raised to 20M for the run (per review-1). Run: `runs/pixel-gs-pilr-30000`.
- Evidence: `evidence/verify.json` (all declared probes pass), `evidence/test.json` (smoke), `reviews/review-1.md` (approved, 4 minors).

## Results (test split, 30k)

| scene | metric | Pixel-GS ours | Pixel-GS paper | Δ vs paper | our gsplat pilr baseline | paper 3DGS* |
|---|---|---|---|---|---|---|
| bicycle | PSNR | **25.47** | 25.739 | −0.27 | 25.45 | 25.634 |
| bicycle | SSIM | 0.785 | 0.793 | −0.008 | 0.772 | 0.778 |
| bicycle | LPIPS | 0.202 | 0.173 | +0.029 | 0.236 | 0.204 |
| bonsai | PSNR | **32.52** | 32.697 | −0.18 | 32.18 | 32.184 |
| bonsai | SSIM | 0.953 | 0.951 | +0.002 | 0.948 | 0.947 |
| bonsai | LPIPS | 0.218 | 0.162 | +0.056 | 0.234 | 0.174 |

Gain over the baseline (PSNR): bicycle **+0.02** (paper +0.105), bonsai **+0.34** (paper +0.513). LPIPS improves on both
scenes (−0.034 / −0.016) and SSIM rises too, matching the paper's main claim (largest gains in LPIPS). Our LPIPS level is
higher overall because of the runtime's LPIPS implementation, and the baseline has the same offset.

| scene | Gaussians ours | baseline | ratio | train time | peak mem |
|---|---|---|---|---|---|
| bicycle | 8.92M | 6.25M | 1.43× | 120 min | 28.9 GB |
| bonsai | 2.41M | 1.38M | 1.75× | 49 min | 8.5 GB |

The paper reports ~1.67× memory over 3DGS* on Mip-NeRF 360 (1.2 vs 0.72 GB), which is consistent with these ratios.

Smoke (garden native, 2k steps): 23.10 PSNR vs gsplat 23.07, 1.09M vs 0.85M Gaussians.

## Deviations / fidelity gaps
1. Renderer numerics: gsplat (α clamp 0.999, stop when next T ≤ 1e-4, per-axis 3σ tile boxes) vs Inria (0.99, < 1e-4, circular radius).
   Pixel counts follow gsplat, so Inria may count a few more edge pixels for elongated Gaussians.
2. White background (runtime) vs black (official). The official code would also reset opacity at iteration 500 on white; we correctly don't.
3. Random-with-replacement view sampling vs the Inria shuffled stack. SSIM uses valid convolution; LPIPS comes from the `lpips` package.
4. Densify threshold `>` (gsplat) vs `>=` (official). Negligible.
5. Max-Gaussian cap 20M (a safety cap only; never reached).
6. Only 2 of 9 reported scenes were benchmarked (batch rule). There is no 9-scene average, and T&T was not run (the paper uses 21 scenes with its own processing).
7. Gain shortfall: bicycle +0.02 vs +0.105 (outdoor gain mostly absent), bonsai +0.34 vs +0.513. Partly explained by baseline calibration:
   our bicycle baseline is 0.18 dB below paper 3DGS*, while bonsai matches. Not tuned; no hyper-parameters changed.
