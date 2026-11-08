# BOGausS gap analysis 1 (bench runs/bogauss-tandt, 30k, T&T)

| scene | ours bogauss | ours gsplat | Δ ours | #G ours (budget) |
|---|---|---|---|---|
| truck | 25.75 | 25.21 | +0.54 | 2,584,171 (2,584,171) |
| train | 22.13 | 21.52 | +0.61 | 1,085,480 (1,085,480) |
| **mean** | **23.94** | **23.36** | **+0.58** | 1.83M (paper 1.83M) |

Paper (no per-scene table, averages only): BOGausS 24.78 / its 3DGS 23.79 → +0.99.
- abs_gap = 23.94 − 24.78 = **−0.84**
- baseline_offset = 23.36 − 23.79 = **−0.43** (the paper's 3DGS T&T row is 0.65 dB above the 3DGS paper's 23.14; protocol/runtime, not the method)
- delta_gap = 0.58 − 0.99 = **−0.41** (method fidelity)
Classification: **mixed, about half protocol and half method**. SSIM 0.856 vs 0.866, LPIPS 0.190 vs 0.176.

## Checked items
1. **Exposure correction**: does not apply. Our 24.78 comes from the "Without Exposure Correction" half of Table 2. The exposure-corrected T&T number is 25.99.
2. **Counts**: they hit the Taming big budgets exactly at 15k and hold to 30k. The means equal the paper's #G column (T&T 1.83M).
3. **Growth curve / first-split cap**: the cap only binds at step 1000 on truck (269k vs 302k target). From step 1500 on, every target is met. Negligible.
4. **Pruning**: it hits the 1 % cap at every call (e.g. truck 25.8k per call at 15k). This is paper-faithful (τ = 0.02, cap 1 %).
5. **Δ-update learning rates**: the paper defines η_θ (3_preliminary.tex:75) but gives no values anywhere: no supp., no figures, no reference to an earlier optimizer. Our auto calibration gives truck η_means 1.62 and η_scales 0.124; train 2.52 and 0.147. Any other value would be a tuned choice, not a paper-grounded fix.
6. **σ_uv / frustum for Δ**: with the auto calibration, σ_uv cancels out of the step sizes. It only sets the minimum size: about 3 % of Gaussians sit at the floor at 2k, median Δ 5.8e-4 vs median σ 1.7e-2. Unlikely to be worth ≥0.1 dB.
7. Training loss is below the baseline (truck 0.030 vs 0.036, train 0.040 vs 0.042), so the optimizer and densification do work. The gain is smaller on test views.

## Hypotheses (ranked)
1. Protocol offset −0.43: the authors' 3DGS/Taming code base vs our gsplat runtime (white bg, 0.999 alpha clamp, sampling with replacement, valid SSIM). Not fixable in the plugin.
2. Unspecified η / optimizer hyper-parameters (Δ-scaled lrs, other-field lrs if the base is Taming "fast"): plausibly 0.1–0.4 dB. Not paper-grounded.
3. Unspecified SplitAdaptation, statistics window (training views only vs all cameras), densification window (500–15000 assumed): each ≤0.1–0.2 dB, unknown sign.

## Recommendation
No paper-grounded fix is likely to close ≥0.5 dB. No iteration-1 code change. Outcome **not-matched**: abs −0.84; the delta-based gap is −0.41, and about half of the abs gap is protocol offset. An η sweep judged on training views could be run later, labelled "tuned"; it is not part of this reproduction.
