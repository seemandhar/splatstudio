# DecompGS gap analysis 1 (T&T, 30K, runs/decomp-gs-tandt)

## Gap table (test PSNR; paper reports averages only)
| | truck | train | mean | paper |
|---|---|---|---|---|
| decomp-gs (ours) | 25.19 | 21.45 | **23.32** | 24.06 |
| gsplat baseline (baseline3dgs-tandt-30000) | 25.21 | 21.52 | 23.36 | 3DGS row 23.62 / 3DGS-accel 23.58 |
| Δ method − baseline | −0.02 | −0.07 | **−0.04** | **+0.44** (vs 3DGS), +0.48 (vs accel) |

abs_gap −0.74 = baseline_offset −0.26 (ours gsplat 23.36 vs paper 3DGS 23.62) + delta_gap **−0.48**.
Classification: **method / unexplained delta** (not protocol). SSIM 0.836 vs gsplat 0.849, LPIPS 0.243
vs 0.207 (paper also trades LPIPS: +0.012; ours +0.036). Per-view Δ vs gsplat is mixed (42–44 % of
views improve, median −0.06/−0.17 dB): no single-view failure.

## Efficiency matches the paper
| | ours | gsplat | ratio | paper ours | paper 3DGS | paper ratio |
|---|---|---|---|---|---|---|
| Gaussians (mean) | 0.96 M | 1.88 M | 0.51 | 0.867 M | 1.576 M | 0.55 |
| time (min, mean) | 30.9 | 72.5 | 2.3× faster | 4.10 | 15.80 (accel 7.73) | 3.9× (1.9× vs accel) |
Gaussian-count curve: ~120–315 k during the low-res split-only phase (truck: 150 k@1k → 316 k@13k),
explosive cloning after T_2 ≈ 13.2 k (774 k@14k, 1.31 M@17k), flat after 25 k. Our wall-clock is on shared
A6000s (not comparable in absolute terms).

## "3DGS-accel" and the paper's baseline rows
- 3DGS-accel (footnote 1) = original 3DGS + Taming-3DGS per-splat backward + sparse Adam — i.e. the
  `--optimizer_type sparse_adam` accelerated mode of the official Inria repo; same hyper-parameter
  defaults as 3DGS. In the paper's own table it is −0.04 dB vs 3DGS on T&T, so it cannot explain a
  +0.44 dB method gain or our missing gain.
- The paper's 3DGS row (T&T 23.62/0.8471/0.1772; M360 27.72/0.8263/0.2016) is **identical to
  DashGaussian's Table 2 3DGS retrain** (reproductions/dashgaussian/paper_results.json: 23.62/0.847/0.177),
  i.e. borrowed, not run in the DecompGS code base. It is 0.48 dB above the 3DGS paper (23.14). The paper's
  "+0.44 over 3DGS" therefore mixes code bases; vs its own 3DGS-accel it is +0.48 (single seed, 2 scenes).
- No official DecompGS code exists, so the code base's further defaults cannot be checked.

## Ablations (train scene, ×1/3 compressed schedule: 10k steps, T_densify 8333, reset 1000;
## metrics on TRAINING views only — diag/train-*.json)
| variant | train-view PSNR | SSIM | LPIPS | #G |
|---|---|---|---|---|
| A base (benchmarked config) | 22.05 | 0.773 | 0.338 | 372 k |
| B global_mode = split_all (split clone candidates in global phase) | 22.25 (+0.20) | 0.780 | 0.326 | 419 k |
| C LR: hold, then jump to the 3DGS value | 21.77 (−0.28) | 0.759 | 0.354 | 338 k |
| D G2L off | 22.24 (+0.19) | 0.781 | 0.326 | 418 k |
| E C2F off | 21.79 (−0.26) | 0.776 | 0.332 | 371 k |
| F 3DGS reference (G2L, C2F, adaptive prune off; densify to 5000) | 22.96 (+0.91) | 0.810 | 0.290 | 521 k |
Training-view PSNR measures fit, and fit follows the Gaussian count (F has 40 % more Gaussians).
The full method fits ~0.9 dB worse than the 3DGS-like reference while the 30K test PSNR ends equal.

## Hypotheses (ranked)
1. **The paper's +0.44 dB is not reproducible from the text alone / is partly a baseline artefact**
   (borrowed 3DGS row, single seed, 2 scenes). For: borrowed row; our method equals our baseline on both
   scenes; efficiency numbers (−45–50 % Gaussians, ≥2× speed) do reproduce. Against: none measured.
2. **Global-phase rule (skip vs split clone candidates)**. split_all = G2L-off ≈ +0.2 dB train-view fit
   with +13 % Gaussians; the paper's G2L ablation costs −0.10 dB, so the test effect is likely ≤0.2 dB. Not
   ≥0.5 dB; Fig. 2 supports the current choice.
3. **LR cut form**: the alternative (jump to the 3DGS value at T_2) is worse (−0.28 train-view). Keep.
4. **Energy pooling** (sum over views vs per-image mean of ratios): T_r differs by a few hundred steps at
   most for T&T (ratios nearly identical on truck and train: T_2 13187 vs 13153); negligible.
5. **Sparse Adam** of the accel base: speed feature, paper shows ±0.04 dB; negligible.

## Recommendation
No paper-grounded change is expected to close ≥0.5 dB, so there is **no iteration 1**. Report as
**not matched**: absolute −0.74 dB, of which −0.26 dB is the runtime offset and −0.48 dB is the method's
missing gain. The efficiency claims reproduce: about half the Gaussians and a ≥2× faster run. Optional and
low priority: if the DB average is wanted as a second data point, run the current config unchanged.
