# SafeguardGS: reproduction report

**Paper:** *SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding Catastrophic Scene Destruction*, arXiv 2405.17793 v2 / WACV 2026.

**Method:** `safeguardgs` (Prune_SG). This is vanilla 3DGS with a single top-1 Score_SG prune at step 20000. The tile top-k follows the official per-tile kernel semantics. The plugin is in `plugin/`.

**Outcome: matched.**
- The 9-scene Mip-NeRF 360 mean PSNR is **26.87 against the paper's 27.07 (−0.20 dB)**.
- The mean Gaussian count is **340k against 350k**.
- About 90 % of Gaussians are pruned, against the paper's 88.9 %.

## Results
Setup:
- 30k steps.
- Dataset `mipnerf360-1600`: the released sweep's automatic downscale to width 1600. This closes plan gap 3, so counts are comparable.
- Split: llffhold 8.
- Runs: `runs/safeguardgs-mipnerf360-1600/`.

| scene | PSNR | #G | our baseline PSNR | Δ vs baseline | baseline #G |
|---|---|---|---|---|---|
| bicycle | 24.41 | 451k | 24.93 | −0.52 | 6.17M |
| flowers | 20.79 | 320k | 21.51 | −0.72 | 3.64M |
| garden | 26.01 | 479k | 26.93 | −0.92 | 4.55M |
| stump | 25.62 | 316k | 26.38 | −0.76 | 4.23M |
| treehill | 22.72 | 339k | 22.40 | +0.32 | 4.60M |
| room | 31.31 | 292k | 31.52 | −0.21 | 1.67M |
| counter | 28.64 | 250k | 29.04 | −0.40 | 1.19M |
| kitchen | 30.91 | 359k | 30.93 | −0.02 | 1.40M |
| bonsai | 31.40 | 251k | 31.75 | −0.35 | 1.37M |
| **mean (9)** | **26.87** | **340k** | **27.27** | **−0.40** | 3.65M |
| paper (9) | 27.07 | 350k | paper 3DGS 27.43 | −0.36 | 3.16M |

The mean SSIM is 0.776 against the paper's 0.788, and the mean LPIPS is 0.335 against 0.269. The LPIPS gap reflects the runtime's metric convention: the `lpips` package with a white background.

**Against our own baseline.** Across all 9 scenes, the mean ΔPSNR is **−0.40 dB**. The paper reports −0.36 dB against its own retrained 3DGS. Our runs prune 90.7 % of all Gaussians (86.2 % per scene on average), against the paper's 88.9 %. Both the trade-off and the compression reproduce. Our absolute PSNR is lower mostly because our gsplat baseline sits below the authors' 3DGS on this protocol: 27.27 against 27.43.

## Fidelity gaps (kept)
1. The official cross-tile slot race (the "Gaussian 0" artefact) is not reproduced. We use deterministic, tile-private slots. The official behaviour is nondeterministic.
2. gsplat numerics: the alpha cap is 0.999 against 0.99, and the cutoff is `next_T <= 1e-4` against `< 1e-4`. gsplat also drops gradients at saturated alpha (see `TODO-runtime.md`).
3. Shared runtime differences:
   - white background;
   - view sampling with replacement;
   - SSIM with valid convolution;
   - LPIPS from the `lpips` package;
   - density step after the optimizer step.
4. The baseline is our gsplat 3DGS, not the authors' retrained 3DGS.

## Review
`reviews/review-1.json` lists 5 minors and no majors. The official tile top-k semantics and the Score_SG_36 formula were checked against independent references.

## Reproduce
```bash
python3 tools/bench.py launch --exp safeguardgs-mipnerf360-1600 --methods safeguardgs --dataset mipnerf360-1600 --steps 30000
```
