# Turbo-GS (plan #117) — reproduction report

**Outcome: matched.** On the complete Deep Blending average at 30k steps, the port scores
**29.11 PSNR**, against the paper's Turbo-3DGS **29.21 (−0.10 dB)**.

- Relative to baselines, our change versus our gsplat baseline is **−0.17 dB** (29.28 →
  29.11). The paper reports **−0.54 dB** versus its own 3DGS (29.75 → 29.21).
- **Paper**: Lu et al., *Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance
  Fields*, CVPR 2026, arXiv 2412.13547. The target is the Turbo-3DGS 1K row of Table 2.
- **Official code**: github.com/inspirelt/Turbo-GS (aeddd12) holds **only a README and a
  teaser**, with no code and no license. The port is written from the paper and supp.
  Algorithm 1.
- **Plugin**: `plugin/` (`splatstudio-method-turbo-gs`, method `turbo-gs`). No custom CUDA:
  dilated rendering is exact on gsplat 1.5.3.
- **Runs**:
  - Method: `runs/turbo-gs-db/turbo-gs/db-<scene>`
  - Baseline: `runs/baseline3dgs-db-30000/gsplat/db-<scene>`

## Why the Deep Blending average, not per-scene numbers
The supp. per-scene table lists **identical** rows for Turbo-Scaffold-GS and Turbo-3DGS. Its
means (DB 30.49 PSNR / 0.911 SSIM / 0.233 LPIPS / 4.0 min, M360 27.43) equal the
**Turbo-Scaffold-GS** averages in Table 2, not Turbo-3DGS (29.21 / 27.39). The per-scene numbers
(e.g. drjohnson 29.85, playroom 31.12) therefore belong to the Scaffold-GS backbone (10k steps,
anchor MLP). Turbo-3DGS has averages only. Review-1 confirmed this from all four per-scene
tables. We compare the complete 2-scene DB average.

## Results (test split, 30k steps, 3DGS recipe, seed 0)
| scene | PSNR | SSIM | LPIPS | final #G | peak #G | M (budget) | peak/M | time (min) | gsplat PSNR | gsplat SSIM | gsplat LPIPS | gsplat #G | ΔPSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drjohnson | 28.489 | 0.8950 | 0.3261 | 3,669,958 | 3,705,590 | 3,273,600 | 1.13 | 104.8* | 28.760 | 0.8972 | 0.3256 | 3,613,268 | −0.27 |
| playroom | 29.724 | 0.9073 | 0.3091 | 2,586,433 | 2,596,516 | 2,326,100 | 1.12 | 55.6 | 29.803 | 0.9082 | 0.3167 | 2,237,750 | −0.08 |
| **DB mean** | **29.107** | **0.9011** | **0.3176** | 3.13 M | **3.15 M** | 2.80 M | 1.12 | | 29.282 | 0.9027 | 0.3212 | 2.93 M | **−0.17** |
| paper Turbo-3DGS | 29.21 | 0.905 | 0.240 | | 2.82 M | | 1.01 | 12.1 (A100) | 3DGS: 29.75 | 0.902 | 0.241 | 2.46 M | −0.54 |

\* The drjohnson job was retried once after a transient "CUDA error: unspecified launch
failure" at step 28.5k on a shared GPU. The failed attempt is kept as `db-drjohnson.stale-*`.
Times are on shared A6000s and are not comparable with the paper's. Speed claims are out of
scope.

Absolute LPIPS differs from the paper by about 0.08 for both the method and our baseline. This
is the known protocol offset (lpips-package VGG, white background, valid SSIM). Relative LPIPS
improves by 0.004.

## Budget behaviour (convergence-aware control)
- Over the full runs, the cap bound at 30 of 144 (drjohnson) and 45 of 144 (playroom)
  refinement events, starting at step 3100.
- M_adaptive first rose to its 1.5·M cap. After the opacity resets the loss rose, the rate went
  negative, and M_ad shrank (final 3.22 M and 2.83 M). Growth then stopped when B(t) fell
  below n.
- Peak #G is 1.12–1.13·M. The paper's DB peak is 1.01·M (its M360 peak is 1.10·M). Our peak is
  1.08× our baseline's final count, against the paper's 1.15× its 3DGS.
- The colour branch fired at 24 of 144 events per scene.

## Deviations and interpretations (every one is a possible fidelity gap)
1. **No official code.** All the readings below are ours, checked against the paper by
   review-1.
2. **t in B(t)**: t = clamp(100·step/total_steps, 1, 100), using the "Total steps" input of
   Alg. 1. Densification stops at t = 50. The first build mapped t over the densify window
   (500–15000). That let B reach 1.5·M and the cap never bound in the smoke run, so review-1
   M1 asked for this fix. The old mapping remains as `t_mode: "densify_window"`.
3. **Deviation sign / α**: we use positive decay exponents. α_base is the mean of historical
   fits, deviation = α_base − rate, and α = clip(α_base + 0.5·tanh(dev), 0.1, 2). This matches
   the prose (slower convergence → slower densification). Under the literal Alg. 1, α_base
   would be a negative slope and α would almost always clip to 0.1.
4. **Unspecified budget parameters**: window = 500 records, refit at every densification event,
   ×1.1/×0.9 applied per event. M is Taming-3DGS's `big_budgets` (= 3DGS counts), inferred from
   the paper's peak #G; the paper says only "manually set".
5. **Colour branch**: one Bernoulli(0.2) draw per densification event (per event vs per
   Gaussian is unstated). τ_col = 0.01·τ_pos = 2e-6.
6. **Colour gradient attribute**: ‖∂L/∂sh0‖ (SH DC coefficients), averaged over visible steps.
   A gradient w.r.t. RGB would be 3.5× smaller and would select fewer Gaussians.
7. **Dilation offset**: p = d·p_ds + o, o ∈ {0,1}² (chessboard sub-lattice), one uniform
   offset per step. Paper Eq. 3 is written with the block centre ½(d−1). Our reading is the
   only one consistent with the paper's 4-offset merge check. Our renders use exact sub-pixel
   intrinsics and eps2d = 0.3/d². The merged 4-offset render equals the full render at 151 dB
   (paper: > 70 dB).
8. **Densification schedule** for the 3DGS backbone is unstated. We use the 3DGS schedule
   (refine 500–15000 every 100, opacity reset every 3000; the paper says the reset is kept).
9. **Not ported**:
   - batched training and "better initialization" (README TL;DR only, undocumented);
   - the Taming-3DGS speed kernel (speed only);
   - sensitivity pruning (only the † rows use it).
10. **Shared runtime differences**: white background, random-with-replacement view sampling,
    valid-SSIM loss/eval, lpips-package VGG LPIPS, gsplat alpha clamp 0.999. Single seed (0).
11. **Run retry**: the drjohnson job was retried once after a transient CUDA launch failure
    (see above).

## Review and evidence
- `reviews/review-1.md`: 1 major issue (M1, t-normalization, fixed) and 5 minors (m1–m4
  documented, m5 addressed with the new probe `turbo-refine-allowance`).
- `evidence/verify.json`: declared_checks_passed, 9 probes.
- Smoke runs on drjohnson, 2k steps:
  - smoke-1 (old t): 24.95
  - smoke-2 (fixed t): 25.09
  - gsplat: 25.00
- `evidence/colcheck-p1`: an 800-step run with colour probability 1. The branch is selective:
  +14k Gaussians on 90k.
