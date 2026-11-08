# GaussianPOP — reproduction report

**Outcome: matched** (both reported scenes within ±0.5 dB of Supp. Tab. 8, P=0.8).
Paper: arXiv 2602.06830 (Fraunhofer HHI). No official code (project page "Coming Soon"); written from the paper text.
Plugin: `plugin/` (`gaussianpop`, `gaussianpop-p07`, `gaussianpop-p09`). Benchmark: `runs/gaussianpop-mipnerf360`.

## Setup
Post-training scenario (Supp. Tab. 8: P=0.8, C=8 cycles, 5k fine-tune) as one 35k-step run:
3DGS steps 1–30000 (same recipe and means-LR schedule as `baseline3dgs-mipnerf360-30000`), 8 cycles of
quantify-then-prune at step 30000 (CUDA Alg. 1 over all training views, compound split), then 30001–35000
fine-tune (fresh Adam, means LR = lr0·0.01, no densification). Scenes: bicycle (outdoor), counter (indoor).

## Results (test split, llffhold8)
| scene | ours PSNR / SSIM / LPIPS | #G after prune | paper (P=0.8) PSNR / SSIM / LPIPS | paper #G | gap |
|---|---|---|---|---|---|
| bicycle | 25.01 / 0.754 / 0.267 | 1.31M | 25.12 / 0.747 / 0.252 | 0.973M | −0.11 |
| counter | 28.87 / 0.901 / 0.278 | 0.243M | 29.00 / 0.902 / 0.219 | 0.216M | −0.13 |

Change relative to the unpruned model:
| scene | our unpruned 30k (PSNR / SSIM / LPIPS, #G) | our change | paper 3DGS row (Tab. 7) | paper change |
|---|---|---|---|---|
| bicycle | 24.92 / 0.757 / 0.243, 6.59M | +0.09 | 25.13 | −0.01 |
| counter | 28.85 / 0.907 / 0.260, 1.22M | +0.02 | 29.11 | −0.11 |

Pruning to 20% of Gaussians costs no PSNR, as in the paper. LPIPS gets worse by +0.024 (bicycle) and +0.018 (counter), while the
paper's post-training LPIPS stays close to its 3DGS row. Train time 77 min (bicycle) / 91 min (counter).
Final counts are exactly 0.2× the pre-prune counts (1,308,505 ≈ 0.2·6.54M; 242,609 ≈ 0.2·1.21M), consistent with
the baseline's 6,590,201 / 1,215,973 at 30k.

## Verification
- `delta-se-oracle`: kernel ΔSE vs brute-force delete-and-re-render, max rel. err 1.2e-6; C_render vs gsplat 2.4e-7.
  The reviewer also tested fixtures with termination and N_max truncation: median rel. err ≤1.6e-5, Spearman ≥0.99999.
- `prune-cycles`: 8 quantifications; per-cycle counts follow the compound schedule; the lowest-error Gaussians are removed first;
  Adam state is cleared; the LR schedule is correct.
- Smoke (garden, accelerated 2k): 22.97 dB / 97k G vs gsplat 23.04 dB / 839k G.
- Review 1: approved, 0 critical / 0 major / 6 minor (`reviews/review-1.md`).

## Deviations / fidelity gaps
1. Our base model is the gsplat 3DGS (white bg, random views with replacement, valid-SSIM). It is 0.21/0.26 dB below the paper's
   3DGS row and **larger**: bicycle 6.59M vs 4.87M. At fixed P we therefore keep 35% more Gaussians on bicycle
   (1.31M vs 0.97M), which may explain why our change is slightly more positive than the paper's.
2. The paper does not say how P is split over C=8 cycles. We use compound r = 1−(1−P)^{1/C}, the authors' commented supp. formula.
   Linear split is available (`split: linear`).
3. N_max=64 interpreted as "only the first 64 contributors per pixel are scored; C_render uses the full blend". The
   alternative (truncated C_render) is not offered; the measured effect is negligible.
4. Fine-tune LR and optimizer are unstated. We chose a 3DGS continuation (means LR held at lr0·0.01) with fresh Adam. A restart at lr0
   is a plausible alternative and was not run.
5. C_render includes the white training background (equals the paper exactly for black bg).
6. The pre-prune model is not evaluated in-run; the reference is `baseline3dgs-mipnerf360-30000`, which uses the same recipe.
7. Not run: post-prune-only (`--steps 30000`, Tab. 3 bicycle 20.71 dB), P=0.7/0.9 variants, and the on-training mode
   (needs a GaussianSpa/Mini-Splatting base).
8. Open minor review items: the probe fixture has no high-opacity/termination case (reviewer verified it separately).
   The plan.md option name is now fixed to `lr_final_ratio`.
