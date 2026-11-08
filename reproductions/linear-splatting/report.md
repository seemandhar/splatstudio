# 3DLS (arXiv 2411.12440) — reproduction report

**Outcome: not matched.** Primary method `3dls-aags` (a labelled interpretation) lands 0.64 dB (garden) and 0.61 dB (kitchen) below the paper, and 0.13 / 0.17 dB below our own gsplat baseline. The paper reports 3DLS at +0.0 / +0.38 dB over its gsplat-3DGS. No paper-grounded fix closing ≥0.3 dB was found.

No official code exists; github.com/hche8927/3DLS is the project page only. The plugin is `plugin/` (`ss_linear_splatting`). The linear-kernel rasteriser is a JIT CUDA forward/backward adapted from gsplat 1.5.3 `RasterizeToPixels3DGS{Fwd,Bwd}.cu`. It matches a dense float64 torch reference to about 1e-6 in the forward pass and about 5e-6 relative error in the backward pass, with and without AGS.

## Results (Mip-NeRF 360, 30k steps, test views, llffhold8, images_4/images_2)

| scene | paper 3DLS | paper gsplat-3DGS | paper Δ | our gsplat baseline | ours | our Δ | Gaussians: ours (baseline) | train time |
|---|---|---|---|---|---|---|---|---|
| garden, `3dls-aags` | 27.40 | 27.41 | −0.01 | 26.89 | **26.76** | −0.13 | 1.89M (4.76M) | 36 min |
| kitchen, `3dls-aags` | 31.78 | 31.40 | +0.38 | 31.34 | **31.17** | −0.17 | 0.60M (1.42M) | 18 min |
| garden, `3dls` literal ω | 27.40 | 27.41 | −0.01 | 26.89 | **21.53** | −5.36 | 216k | — |

- LPIPS for `3dls-aags` is 0.169 / 0.169 vs baseline 0.131 / 0.162. SSIM is 0.842 / 0.922 vs 0.860 / 0.925.
- **Runtime offset:** our gsplat baseline is 0.52 dB below the paper's gsplat-3DGS on garden and 0.06 dB below on kitchen.
  - Known causes: white background (the gsplat trainer uses black), view sampling with replacement, valid-convolution SSIM, and the camera-extent scene scale.
  - So most of garden's absolute −0.64 is this offset. The meaningful gaps are against the paper's improvement over its own baseline: 0.12 dB short on garden and 0.55 dB short on kitchen.
- Runs: `runs/linear-splatting-mipnerf360/` and `runs/linear-splatting-literal-mipnerf360/`. Baseline: `runs/baseline3dgs-mipnerf360-30000/`.

## The AGS question
See reviews/review-1.md §1 and evidence/reviewer_scratch/.

1. **Fig. 3b confirms the literal ω = exp(−D²) on the unaligned D.** The reviewer digitised the curve: 0.4·exp(−D²) matches the readings 0.0074 / 0.042 / 0.145 / 0.306 at D = 2 / 1.5 / 1 / 0.5. The alternatives exp(−D²/2) and exp(−(D/λ)²) are ruled out.
2. **The literal ω fails at any scope.**
   - It cuts the mean densification gradient 8.5× (`evidence/ags_gradstat.json`: 1.06e-5 vs 8.96e-5). The fraction of Gaussians above the grow threshold drops from 10.5% to 0.09%.
   - Densification is not the main problem. With an unscaled densification statistic (910k Gaussians at 2k steps) it still reaches only 21.1 dB vs 23.07, and the same with ω on conics only.
   - Its loss is already worse at step 400, before densification starts.
   - The cause is spatial: ω < 0.02 beyond D = 2, so the outer 84% of each splat's footprint gives almost no geometry gradient.
   - At 30k on garden it reaches 21.53 dB with 216k Gaussians.
3. **`3dls-aags` is an inferred implementation detail, not a reading of the text.**
   - It uses ω = exp(−(D/λ)²), i.e. the distance the kernel actually uses after distribution alignment.
   - It trains normally.
   - Its Gaussian count is 0.50× that of LK+DA at 2k steps (paper Table 2: 0.57×).

## Gap analysis (7k-step diagnostics, train views only, every 4th train view)

Results are in `evidence/diag/diag_7k_train.json`, script `evidence/diag/diag.py`, runs in `runs/linear-splatting-diag/`. "3DLS thresholds" means grow_scale3d 0.006, prune_scale3d 0.4, prune_opa 0.025, and opacity reset to 0.05.

| variant | garden PSNR | garden Gaussians | kitchen PSNR | kitchen Gaussians |
|---|---|---|---|---|
| gsplat (base loss, base thresholds) | 26.58 | 2.61M | 29.24 | 0.85M |
| gsplat + 3DLS loss | 26.47 | 2.40M | 29.22 | 0.76M |
| gsplat + 3DLS thresholds | 27.08 | 2.51M | 30.40 | 0.89M |
| `3dls-noags` (LK+DA, 3DLS loss and thresholds) | 26.68 | 2.30M | 30.26 | 0.90M |
| `3dls-aags` | 26.18 | 1.08M | 30.01 | 0.51M |
| aags + base loss | 26.26 | 1.20M | 30.08 | 0.56M |
| aags + base thresholds | 25.70 | 1.16M | 28.20 | 0.45M |

**Caveat:** step 7000 is only 1000 steps after the last opacity reset. Variants with the 3DLS thresholds reset opacity to 0.05 instead of 0.01 and recover faster. Threshold effects at 7k are therefore inflated and do not predict 30k. Comparisons with the same thresholds are reliable.

- **(a) AGS and the Gaussian count — the largest effect.**
  - Holding loss and thresholds fixed, aags is 0.50 dB (garden) and 0.25 dB (kitchen) below noags, with about half the Gaussians.
  - The paper reports the opposite: AGS adds +0.03 dB on kitchen and +0.42 dB on train over LK+DA.
  - At 30k our aags has 0.40× / 0.43× the baseline's Gaussians. The paper's 3DLS has 26% more than 3DGS on average (Table 3) and about the same on kitchen (Table 2).
  - So our AGS under-densifies and under-fits compared with the authors' implementation.
  - The only form the paper grounds is the literal one, and it fails. Keeping AGS out of the densification statistic would contradict Table 2, where AGS lowers the count. Further changes to the AGS distance or coefficient would be tuning.
- **(b) Distribution alignment — ruled out.** λ = 2.5 is exact. The linear/Gaussian footprint ratio is (πλ²/3) / 2π = 1.04. A single splat's support radius equals λσ (probe `alignment-footprint`).
- **(c) Loss swap — small.** With the Gaussian kernel the 3DLS loss costs 0.11 / 0.02 dB. With aags, switching back to the base loss gains 0.08 / 0.07 dB. The loss is exactly as specified (supp. Eq. 1–4). The small residual effect probably comes from the runtime's valid-convolution SSIM.
- **(d) gsplat v1.3 vs 1.5.3 defaults — no grounded difference.**
  - The paper's "3DGS" threshold row equals gsplat's DefaultStrategy defaults.
  - absgrad and antialiasing are off (3DLS+AA is a separate row), and packed mode does not change the maths.
  - v1.3 resets opacity to 2·prune_opa = 0.05, which our runtime reproduces.
  - The 2D grow/prune thresholds are inactive in both versions (refine_scale2d_stop_iter = 0).
  - The real difference is the runtime's white background versus gsplat's black.
- **(e) Radius rule — ruled out.** The rule D ≤ λ(1 − 1/(255·o)) is exactly the α ≥ 1/255 support. On large splats the rendered alpha matches the dense reference, which does not use tiles, to 1.5e-7, so nothing is clipped.

**Conclusion:** the shortfall comes from AGS.
- The literal form cannot train.
- The inferred form roughly halves the Gaussian count and costs 0.25–0.5 dB against LK+DA at 7k.
- The paper's version of AGS adds quality and keeps the count near 3DGS's. The paper text does not give enough to close this gap.

An optional datapoint (not a fix): benchmark `3dls-noags` against the paper's LK+DA row (kitchen 31.75, Table 2). On train views at 7k it is 0.25–0.5 dB above aags.

## Deviations and fidelity gaps
1. **AGS distance:** `3dls-aags` uses ω = exp(−(D/λ)²), which is inferred. The paper and Fig. 3b give exp(−D²); that literal form is reported above as a failed row.
2. **Inferred because there is no code:**
   - the radius rule;
   - AGS applied to the geometry gradients only (opacity and colour unscaled);
   - gsplat's `o·k ≤ 0.999` backward saturation gate kept. It only affects o > 0.999 near the splat centre; see TODO-runtime.
3. gsplat 1.5.3 instead of v1.3.
4. **Runtime protocol:** white background, sampling with replacement, valid-convolution SSIM in loss and evaluation, Inria camera-extent scene scale.
5. 3DLS+AA is not implemented.
6. The 7k diagnostics are confounded by opacity-reset timing (see the caveat above).
