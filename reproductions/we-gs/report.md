# WE-GS reproduction report (plan #113, arXiv 2406.02407)

**Outcome: not-matched.**
- brandenburg-gate: 25.21 vs 27.74 (−2.53).
- sacre-coeur: 22.98 vs 23.62 (−0.64). The sacre gap is explained by the baseline offset.

## Setup
- Native plugin `we-gs` (`plugin/ss_wegs`), written from the paper text. Values the paper omits
  come from the authors' released WE-GS *inference* model (takinglangsplatw, TVCG); no training
  code has been released.
- Photo Tourism with the NeRF-W split, downscale 2, 150k steps.
- Test appearance is encoded feed-forward from the test image and the full image is scored
  (`python -m ss_wegs.evaluate`). Single seed.

## Results (`metrics.full`; LPIPS-Alex)
| scene | PSNR | SSIM | LPIPS | paper | our gsplat (ds2, 10k, white bg) | paper 3DGS |
|---|---|---|---|---|---|---|
| brandenburg-gate | 25.21 | 0.922 | 0.114 | 27.74 / 0.933 / 0.128 | 19.63 | 20.72 |
| sacre-coeur | 22.98 | 0.878 | 0.136 | 23.62 / 0.890 / 0.148 | 16.74 | 17.57 |

- Gain over our gsplat: brandenburg +5.58 vs the paper's +7.02; sacre +6.24 vs the paper's +6.05.
- Gaussians: 284k / 272k. Paper storage of 66.4 MB corresponds to about 280k.
- Training time: 5.1 h / 6.9 h on a shared A6000 (paper: 1.8 h on a 4090).
- Static-mask mean on test: 0.92 / 0.85.

## Gap analysis (`gap-analysis-1.md`)
- **Brandenburg** is −1.09 dB baseline offset and −1.44 dB method. The method part is test-time
  appearance generalisation of the encoder.
  - Diagnostic only, not used for selection: a global per-channel colour fit to the GT recovers
    +2.62 dB on test views but only +0.36 on train views.
  - Two of the 10 test views (84113133, 86628129) dominate.
- Per-image appearance is active: the mean-embedding render is about 4 dB worse on both splits.
- No paper- or released-code-grounded fix for this was identified.

## Deviations and assumptions (all listed in `paper.md` §7–8 and `plan.md`)
1. M is detached inside the D-SSIM term of Eq. 19. The paper is silent; the literal form
   collapses to M ≡ 0 (`evidence/smoke-1`, `reviews/review-1.md`).
2. L_regM is the sum over M_k at W/4×H/4 with λ = 1e-5. L_regSH is a mean over Gaussians (the
   paper writes Σ). ε for L_regTS is 0.1; it is not stated in the paper.
3. CAM and U-Net widths are assumed. Encoder conv shapes, G_φ width 256 + tanh, PE frequencies
   and learning rates come from the TVCG inference code. G_φ uses the paper's 4 layers (the code
   has 8).
4. Black background. Runtime differences: random-with-replacement view sampling, gsplat
   rasteriser, valid-conv SSIM.
5. Trevi Fountain was not run (data not prepared), so the paper's 3-scene average is not
   comparable.

## Failed or intermediate runs (kept)
- `evidence/smoke-1`: literal losses; the mask collapsed.
- `evidence/smoke-2`: full-res L_regM; the mask was inert, 19.50 at 2k.
- `evidence/smoke-3`: final configuration, 19.18 at 2k.
