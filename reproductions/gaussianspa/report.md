# GaussianSpa: reproduction report

**Paper:** *GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting* (arXiv 2411.06019).

**Method:** `gaussianspa-<scene>`, which uses the README's recommended presets for each scene. The plugin is in `plugin/` (`ss_gaussianspa`) and is frozen at `85fe12ab2d39961e`. It includes custom CUDA: the extension `ss_gaussianspa_cuda`, which renders the importance score.

**Outcome: partial. 2 of 3 scenes are within ±0.5 dB.**

| Scene | PSNR difference | Within ±0.5 dB? |
|---|---|---|
| bicycle | −0.20 | yes |
| counter | −0.15 | yes |
| garden | −0.74 | no |

The mean over these 3 scenes is 26.95 against the paper's 27.31 (−0.36). That is a subset mean, not the paper's 9-scene average.

## Results
Protocol: 40k steps on `mipnerf360-1600`, the official `-r -1` setting, which resizes images to width 1600 with PIL. Split: llffhold 8.

| scene (preset) | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | #G ours / paper | our gsplat baseline (1600) |
|---|---|---|---|---|---|
| bicycle (imp 0.50/0.72) | 25.24 / 25.44 (**−0.20**) | 0.755 / 0.769 | 0.301 / 0.246 | 0.59M / 0.66M | 24.93 |
| garden (imp 0.60/0.50) | 26.52 / 27.26 (**−0.74**) | 0.841 / 0.848 | 0.178 / 0.151 | 0.77M / 0.73M | 26.93 |
| counter (opacity 0.60/0.50) | 29.08 / 29.23 (**−0.15**) | 0.920 / 0.919 | 0.226 / 0.176 | 0.76M / 0.39M | 29.04 |

The paper targets match the official repo's `results/MipNeRF360/*.csv` within 0.03 dB.

**Comparison with our own baseline.**
- Bicycle: +0.31 dB, against the paper's +0.31 dB (25.44 vs its 3DGS row 25.13). This is an exact match.
- Garden: −0.41 dB. The paper's comparison is −0.06 dB, against a 3DGS row it borrowed from EAGLES, which used a different protocol.
- Counter: +0.04 dB. The paper reports +0.12 dB.
- Our runtime already sits 0.39 dB below that 3DGS row on garden (26.93 vs 27.32). This offset accounts for about half of garden's gap. The rest matches what our Mini-Splatting port shows on garden: −0.24 dB against our baseline (`reproductions/mini-splatting/report.md`). GaussianSpa's first 15k steps are Mini-Splatting.

## Gaussian-count path

| scene | at 14.5k | after 15k sampling | after 25.2k hard prune (final) |
|---|---|---|---|
| garden | 4.50M | 1.80M | 0.77M |
| counter | 3.79M | 1.52M | 0.76M |

Counter's final count is about twice the paper's. Under the official counter preset, the final count is (1−0.6)(1−0.5)·N₁₅ₖ. For the paper's 0.39M, that means N₁₅ₖ ≈ 1.96M. Our Mini-Splatting stage reaches about 3.8M, the same as our standalone Mini-Splatting port (3.85M). The excess therefore comes from the inherited densification stage, not from the sparsifier. Counter's PSNR still matches.

## Why no iteration
The batch rule allows 1–2 fix iterations, but only with a fix grounded in the paper or the official code. None was found:
- The presets, `imp_metric` (`indoor` in the official scripts for every scene), `num_max` (4.5M) and the resolution all match the official code.
- The garden count matches the paper.
- The remaining gap is about 0.4 dB of runtime offset plus about 0.3 dB that the Mini-Splatting stage shows too.

Closing either part would mean changing the shared runtime: background colour, view sampling, update order. None of these was tuned.

## Fidelity gaps (from plan.md)
1. Gaps inherited from Mini-Splatting:
   - white background instead of black;
   - random-with-replacement view sampling;
   - Adam steps before density;
   - gsplat alpha numerics, including the backward that skips saturated alpha (see `TODO-runtime.md`);
   - the `lpips` package;
   - valid-convolution SSIM.
2. The z/u update and the hard prune at 25.2k use opacities after the Adam step. The official code uses them before it.
3. Presets come from the README, not the paper.
4. Importance ties at zero: removing `imp <= thr` can prune more than r2·N Gaussians. This is the official rule and is kept. It makes garden end at 0.77M instead of 0.90M.

## Review
`reviews/` has no critical or major findings and 4 minors, all documented.

## Reproduce
```bash
python3 tools/bench.py launch --exp gaussianspa-garden --methods gaussianspa-garden --dataset mipnerf360-1600 --scenes garden --steps 40000
# likewise gaussianspa-bicycle (bicycle), gaussianspa-counter (counter)
```
