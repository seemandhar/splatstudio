# DecompGS (plan #6, arXiv 2507.20239) — not matched

- **T&T (averages only):** 23.32 vs paper 24.06 (−0.74). Truck 25.19 / gsplat 25.21; train 21.45 / 21.52. Δ vs our gsplat −0.04, paper +0.44 (vs 3DGS-accel +0.48).
- **Quality:** SSIM 0.836 / LPIPS 0.243 vs gsplat 0.849 / 0.207.
- **Efficiency reproduces:** 0.96M vs 1.88M Gaussians (0.51×; paper 0.867M vs 1.576M = 0.55×); 30.9 vs 72.5 min (2.3×, shared A6000s).
- **Plugin `decomp-gs`:** written from the paper (no official code); renderer and loss target adapted from our DashGaussian port. 11 probes pass. Not separately reviewed.
  - G2L: split only before T_2, clone candidates skipped (Fig. 2).
  - C2F: T_r = round(E_r/E_1 · 25K), r = 1..8, FFT energy summed over views; position LR held to T_2, then log-linear decay.
  - Adaptive prune: τ = min(α[⌊0.03N⌋], 0.05).
- **Gap:** runtime offset −0.26, missing method gain −0.48 (see gap-analysis-1.md). The paper's 3DGS row is DashGaussian's retrain (identical numbers), so its +0.44 compares across code bases. 3DGS-accel (Taming per-splat backward + sparse Adam) is −0.04 dB vs 3DGS on T&T in the paper's own table, so it does not explain the paper's higher absolute numbers. Paper-grounded variants were ≤+0.2 dB on training views (10k diagnostics, compressed schedule).
- **Deviations:**
  - dense Adam instead of 3DGS-accel's sparse Adam;
  - global-phase skip vs split is ambiguous;
  - LR cut form and energy pooling are unstated (T_2 is insensitive to pooling: 13187 vs 13153);
  - runtime-wide differences: white background, sampling with replacement, valid-convolution evaluation SSIM;
  - single seed.
- **Evidence:** `evidence/`, `diag/`. Runs: `runs/decomp-gs-tandt`.
