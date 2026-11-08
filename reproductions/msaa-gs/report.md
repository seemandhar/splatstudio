# MSAA-GS — reproduction report
Paper: *Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting* (arXiv 2508.10507), plan id 4. Variant: "Ours(3DGS)", the only one with per-scene numbers (Table 2). The Pixel-GS/AbsGS rows (msaa-gs-plus) are averages only and were not implemented.

**Outcome: matched (iteration 1, post-hoc).**
- Garden: 26.93, 0.48 dB below the paper, inside the ±0.5 dB target. Our gsplat garden is 0.39 below the paper's 3DGS, which explains most of it.
- Kitchen: 31.48, +0.03 dB.
- Improvement over our gsplat baseline is +0.04 / +0.14; the paper's improvement over its retrained 3DGS is +0.13 / +0.13.

## Implementation
- **No official code.** Nothing found on arXiv, in the LaTeX source, by web search or by GitHub search. Built from the paper.
- **C1 MSAA** (Sec. 3.2, Eq. 4-5, 14): JIT CUDA rasterizer `plugin/ss_msaa_gs/csrc/msaa_raster.cu`.
  - Adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd} (Apache-2.0), via reproductions/linear-splatting.
  - Each pixel keeps 4 independent alpha-blending states at the standard rotated-grid sub-pixel positions; the 4 colours are averaged, at train and test time.
  - The backward pass sums the 4 sample paths, so the screen-space mean gradient is that of one fused rasterizer.
  - Projection, SH and tiling are gsplat's; the installed gsplat is not modified.
- **C2 AWS** (Eq. 6-9): 0.8 · weighted L1 + 0.2 D-SSIM, with w = α + (1−α) e/(max e + ε), α 0.5, ε 1e-6. w is held constant (no gradient through it).
- **C3 GDC** (Eq. 10-13): 0.1 · (mean|Δ∇x| + mean|Δ∇y|), forward differences.
- **Density and learning rates:** 3DGS (gsplat DefaultStrategy with the runtime's opacity reset).
- **Methods:** `msaa-gs` (iteration 0), `msaa-gs-it1` (iteration 1), and ablations `msaa-gs-msaa-only` / `msaa-gs-aws-gdc-only` (Table 3 rows).
- **Checks:** 8 probes pass in ss-verify (evidence/verify.json, verify-it1.json).
  - CUDA forward and backward match a float64 dense 4-sample reference to about 2e-7.
  - The 1-sample kernel equals gsplat's rasterize_to_pixels.
  - AWS, GDC and the Eq. 13 weights each match an independent formula.
  - 7 unit tests pass; ruff is clean.

## Results (Mip-NeRF 360, 30k, test split)
| scene | paper 3DGS (T2) | paper Ours | ours gsplat | iter0 msaa-gs | **iter1 msaa-gs-it1** |
|---|---|---|---|---|---|
| garden PSNR | 27.28 | 27.41 | 26.89 | 26.90 | **26.93** |
| garden SSIM/LPIPS | 0.856/0.122 | 0.863/0.109 | 0.860/0.131 | 0.859/0.135 | 0.861/0.129 |
| garden #G | – | – | 4.76M | 4.14M | 5.32M |
| kitchen PSNR | 31.32 | 31.45 | 31.34 | 30.66 | **31.48** |
| kitchen SSIM/LPIPS | 0.931/0.116 | 0.934/0.111 | 0.925/0.162 | 0.924/0.166 | 0.927/0.160 |
| kitchen #G | – | – | 1.42M | 1.07M | 1.53M |

PSNR improvement over the baseline:
- Paper (vs its retrained 3DGS): garden +0.13, kitchen +0.13.
- Iteration 0 (vs our gsplat): garden +0.01, kitchen −0.68.
- Iteration 1 (vs our gsplat): garden +0.04, kitchen +0.14.

Training time for iteration 1 at 30k: 55 min (garden) and 68 min (kitchen). MSAA costs about 1.3× per step.
Runs: runs/msaa-gs-mipnerf360 (iter 0), runs/msaa-gs-it1-mipnerf360 (iter 1), runs/baseline3dgs-mipnerf360-30000.

## Iteration history
**Iteration 0** matched garden. It missed kitchen by −0.79 dB and was below our own baseline.

**Gap analysis 1** (gap-analysis-1.md; diagnostics on train views only, ≤7k steps):
- **One floater view.** Kitchen test view DSCF0672 lost 10.9 dB: a few large semi-transparent Gaussians darkened the frame, costing about −0.30 dB of the kitchen mean.
- **MSAA is not the cause.** Kitchen 7k train-view PSNR: gsplat 29.26, MSAA only 29.37, AWS+GDC only 28.58, full 28.62. MSAA helps.
- **The losses under-densify.** With α=0.5 the weights average 0.52, so the weighted L1 is about 0.54× 3DGS's L1. The densification statistic shrinks with it: 1.08% of Gaussians above the 2e-4 threshold vs 1.70% for gsplat. Result: 24% fewer Gaussians.

**Iteration 1 (`msaa-gs-it1`, post-hoc, not in the paper):** the AWS weight map is divided by its image mean. This keeps Eq. 7's relative emphasis at 3DGS's L1 scale.
- **Grounding:**
  - The paper keeps the 3DGS hyper-parameters (Sec. 4.1), whose densification thresholds are calibrated to that L1 scale.
  - The paper uses "a similar number of Gaussians as in 3DGS" (Sec. 4.2). Iteration 0 had 13–24% fewer; iteration 1 has 6–12% more.
  - α=0.5 is not from the paper; it was inherited from the old Nerfstudio port and left untouched. Nothing was tuned, and no test images were used.
- **7k check (train views):** +0.80 dB on kitchen and +0.44 dB on garden over iteration 0.

## Deviations / fidelity gaps
1. AWS mean-normalisation (iteration 1, post-hoc).
2. Sub-sample positions are the standard 4× rotated grid. The paper gives none; read literally it would shift the image by half a pixel.
3. α 0.5 and ε 1e-6 are not given in the paper.
4. Loss reductions are per-channel means (the literal ℓ1 norm is the channel sum); GDC is the mean form (the literal sum is 3× larger); w is held constant.
5. λ3 is a constant 0.1; the paper says it "can be gradually increased" but gives no schedule.
6. Alpha and depth are averaged across samples like the colour.
7. Runtime protocol differs from the Inria code: white background, sampling with replacement, valid-convolution SSIM, LPIPS-VGG.
8. Paper tables are inconsistent: Table 1's 3DGS row is 27.21, while Table 2's retrained 3DGS averages 27.49. Deltas use Table 2.
9. Single seed, 2 of 9 scenes (scene rule); not a dataset average. Per-view kitchen deltas range from −1.31 to +1.38, so about ±0.1–0.15 dB noise applies.

## Evidence
evidence/ (verify-iter0/verify/verify-it1 json, test.json, gap1/ ablations, densification statistics, floater renders), bench_request-iter0.json, bench_request.json. Iteration-0 runs cannot be re-evaluated with ss-eval because the plugin source changed; their test.json files are kept.
