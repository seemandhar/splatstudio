# DET-GS gap analysis 1

## Benchmark (30k, `runs/det-gs-mipnerf360`, test split, same protocol_id as baseline)
| scene | det-gs | our gsplat | Δ vs gsplat | paper DET-GS | paper 3DGS row | paper Δ |
|---|---|---|---|---|---|---|
| garden | 26.51 | 26.89 | −0.38 | 28.09 | 27.73 | +0.36 |
| counter | 28.42 | 28.85 | −0.43 | 30.07 | 29.12 | +0.95 |
Absolute gap: −1.58 / −1.65 dB. Gap relative to the paper's claimed gain: −0.74 / −1.38 dB.

## Diagnostics (garden, 7k steps, train views only)
Split: `~/datasets/detgs-depth/mipnerf360-diag/garden`. Of the 161 llffhold8 train views, 141 train and 20 are held out
(every 8th, offset 3). No original test view is used. `tools/diag_train.py`; one seed each. These are
diagnostics, not method rows.

| variant | held-out PSNR | SSIM | LPIPS | Gaussians |
|---|---|---|---|---|
| gsplat | **25.18** | 0.787 | 0.258 | 2.56M |
| det-gs (full) | 24.82 | 0.773 | 0.278 | 2.29M |
| − depth | 25.08 | 0.778 | 0.273 | 2.30M |
| − TV | 24.99 | 0.779 | 0.267 | 2.37M |
| − edge | 25.03 | 0.776 | 0.276 | 2.51M |
| depth in disparity space (1/z vs DA-V2) | 24.91 | 0.773 | 0.278 | 2.29M |
| DNGaussian-style separate depth backward + Adam step | 24.95 | 0.775 | 0.276 | 2.31M |

The 7k gap (full − gsplat = −0.36) reproduces the 30k gap. Removing any one loss recovers only 0.1–0.26 dB, and no
variant reaches gsplat. Each of the three regularisers is neutral-to-harmful here. Single-seed noise is probably
~0.1 dB, so the per-loss attributions are uncertain; the overall sign is not.

## Findings
1. **Depth (C1).** The normalised depth loss stays at 0.25–0.43 throughout 30k on both scenes: the DA-V2 prior never
   agrees with the photometric geometry, so the term pulls against it for the whole run. On depth steps its
   mean-gradient is 3.6–5.1× the photometric one. The Adam second moment of the means is about 1.3× the baseline's
   (median sqrt(v) 1.10e-6 vs 0.85e-6), so the photometric step on the means is ~25% smaller.
   - γ applies to the patch term only; that is already the case (Eq. 7, DNG `train_llff.py`).
   - The paper gives no warm-up or stop. Its line "refines … Gaussians in the early training stages" describes the
     effect, not a schedule, so a stop iteration would be invented.
   - Disparity space: +0.09. Separate step: +0.13. Both are within noise and far below 0.5 dB.
2. **TV (C3)**: −0.17 at 7k. **Edge (C2)**: −0.21 at 7k, although its gradient is only ~1% of the photometric one.
   This is most likely noise or a densification-count effect (2.51M vs 2.29M Gaussians), not a real mechanism.
3. **No grounded fix ≥0.5 dB.** The best variants only move the method back toward our own baseline (at most
   +0.36 dB). The paper's +0.36 / +0.95 gains over 3DGS are not reproduced by any component.
   - The paper's 3DGS row is itself 0.3–0.4 dB above the 3DGS paper, so the authors' code very likely contains
     unreported changes (the code is unreleased).
   - The unknowns (error tolerance, ω, δ, schedule) were checked where they could be grounded: tolerance in
     review-1, space and step structure here. None of them is decisive.

## Outcome
not-matched. No iteration 1: no paper- or DNGaussian-grounded change is likely to close ≥0.5 dB.
