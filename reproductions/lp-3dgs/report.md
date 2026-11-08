# LP-3DGS (NeurIPS 2024, arXiv 2405.18784) — reproduction report

**Outcome: matched** (both reported scenes within ±0.5 dB). Not independently reviewed.
Method `lp-3dgs` (RadSplat-score variant), native port of the official code
github.com/ASU-ESIC-FAN-Lab/LP-3DGS @ b828063. Benchmark `runs/lp-3dgs-mipnerf360`, 30k steps,
`--recipe 3dgs`, test split (llffhold 8).

## Results (Mip-NeRF 360, per scene)
| scene | ours PSNR | paper LP-3DGS (RadSplat) | gap | our gsplat baseline | paper 3DGS | Δ ours vs baseline | Δ paper | SSIM / LPIPS (ours) | paper SSIM / LPIPS |
|---|---|---|---|---|---|---|---|---|---|
| garden | 26.99 | 27.290 | −0.30 | 26.89 | 27.254 | +0.10 | +0.04 | 0.8586 / 0.1364 | 0.8548 / 0.1270 |
| room | 31.30 | 31.490 | −0.19 | 31.52 | 31.500 | −0.22 | −0.01 | 0.9139 / 0.3049 | 0.9243 / 0.2032 |

Pruning (density log at step 20,000; learned, not preset):
| scene | before | after | ours ratio | paper ratio | final Gaussians | train time |
|---|---|---|---|---|---|---|
| garden | 4,772,264 | 2,024,625 | 0.576 | 0.59 | 2.02M | 128 min |
| room | 1,746,113 | 462,042 | 0.735 | 0.74 | 462k | 89 min |

Both learned pruning ratios are within 0.015 of the paper's. On garden our result sits slightly above our own baseline, as the paper's sits above its 3DGS; room loses 0.22 dB against our baseline where the paper loses 0.01. Room's LPIPS (0.305) is much worse than the paper's 0.203. Our gsplat baseline on room should be checked for the same offset (white background, LPIPS implementation) before treating it as a method gap.

## Code-over-paper choices (official code followed)
1. Gumbel-sigmoid input is the logit `m·S` (paper Eq. 7 writes log m); noise is the official `-log(log(u1)/log(u2))` logistic sample, fresh per render, τ = 0.5.
2. Mask loss `5e-4 · mean(sigmoid(m))` (paper Eq. 9: mean |m|), steps 19,501–19,999.
3. The final prune at 20,000 is random and does not use the score: it removes `sigmoid((m+n)/τ) < 0.5`, i.e. m + n < 0.
4. The mask uses AdamW (lr 0.05, eps 1e-15, default weight decay 0.01). The mask starts at 1.0 at step 19,500 with a fresh optimizer state.
5. The score is the max over all training views of α·T, recomputed before renders 19,520…19,980 using the current, already-masked opacity (as the official `prune_list` does).

## Deviations from the official setup
- SplatStudio runtime conventions: white background (official: black), random-with-replacement view sampling, SSIM with valid convolution, LPIPS-VGG from the `lpips` package.
- gsplat rasterizer with a 0.999 alpha cap, used in rendering and in the score kernel (`csrc/lp_score.cu`); the official Inria kernel caps at 0.99.
- AdamW weight decay is applied after the Adam update instead of before (O(lr²·wd)).
- One extra Gaussian optimizer step at 20,000 before the prune (the official code prunes first and drops those gradients).
- Single seed; only 2 of the 9 reported scenes were run (per the batch scene rule). The Mini-Splatting-score variant was not built (it also needs a changed SH schedule).

## Evidence
`evidence/verify.json` (all 7 contributions' probes passed), `evidence/test.json` (smoke + compressed-schedule smoke: 33% pruned at 2k), `paper_results.json` (Table 1 + T&T), `plan.md`.
