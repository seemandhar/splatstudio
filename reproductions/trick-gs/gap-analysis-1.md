# Trick-GS gap analysis 1 (T&T, 30k, runs/trick-gs-tandt)

| scene | ours trick-gs | ours gsplat | paper Trick-GS | paper 3DGS | delta_gap |
|---|---|---|---|---|---|
| truck | 24.26 | 25.21 | – | – | ours −0.95 |
| train | 20.77 | 21.52 | – | – | ours −0.75 |
| mean | 22.51 | 23.36 | 23.48 | 23.67 | −0.85 vs paper −0.19 → **−0.66 method gap** |

baseline_offset = −0.31 (protocol). Classification: **method** (delta_gap −0.66).

## Evidence (train views only)
- Gaussian counts: ours max/final (truck 1072K/704K, train 613K/352K; mean 843K/528K) vs paper
  T&T mean 696K/443K → **no over-pruning** (we keep ~20 % more).
- **SH masks over-mask**: final checkpoints keep any SH band on only 1.1 % (truck) / 0.6 % (train)
  of Gaussians (already 1.3 % at 10k); paper Fig. 3 (bicycle) shows ~80 % masked, i.e. ~20 % kept.
  Cause: masking starts at 3500 (our reading of Fig. 3) while the SH coefficients are still ~0
  (degree 3 at 3000, shN Adam stepped only every 16 its by AT). Mask gradient = Σ c·dL/dĉ ≈ 0, so
  the λ_sh term closes the mask, after which c gets zero gradient → permanent (clones inherit it).
- 10k diagnostic on truck (evidence/gap1-*-truck-10k, full-res train views): trick-gs 22.23 vs
  without SH masking 22.61 (+0.38 dB at 10k; SH matters more late in training).
- GM masks saturate (σ(m) ≈ 1 after ~1k steps) → GM prunes only in the first ~1k steps; affects
  size, not PSNR.
- FP16 storage costs 0.09 dB on gsplat truck (train views 27.61 → 27.52); the paper specifies it, kept.

## Ranked hypotheses
1. SH over-masking (above): expected +0.4–0.7 dB. Fix grounded in the cited source: RDO-Gaussian
   official code starts SH masking at `sh_prune_start_iter` = 15001 (after densification).
2. Guessed schedules (SG rounds, blur decay, linear vs "logarithmic" DS): counts match the paper,
   so no sign of over-pruning; effect unknown, probably ≤0.2 dB. Not changed.
3. FP16 (−0.09, paper-specified), GM saturation (size only).

## Recommended fix → iteration 1 (post-hoc), implemented
Method `trick-gs-it1` = `trick-gs` with `sh_mask_start=15001` (RDO-Gaussian code). `trick-gs` is
unchanged (benchmark evidence remains valid). Checks: ss-inspect issues [], 6 unit tests pass,
ruff clean, ss-verify trick-gs-it1 declared_checks_passed (evidence/verify-it1.json).
Launch: `bench.py launch --exp trick-gs-it1-tandt --methods trick-gs-it1 --dataset tandt` (30k).
