# Unified Loss-Aware Density Control (plan #101): reproduction report

**Outcome: MATCHED.** PSNR on both benchmarked scenes is within ±0.5 dB of the paper. The gain
over 3DGS is weaker than the paper's, and the bicycle Gaussian count is high.

- **Method:** `loss-aware-dc`, module `ss_loss_aware_dc`, in `plugin/`.
- **Paper:** arXiv 2611.02044, NeurIPS 2026.
- **Official code:** none. The author's repository contains only a README (commit 267f5ef), so the
  port is written from the paper's LaTeX: Sec. 4 and App. A.1–A.5.
- **Custom CUDA:** `plugin/ss_loss_aware_dc/csrc/stats.cu`.
- **Checks:**
  - `ss-verify`: `declared_checks_passed`, 10 probes.
  - pytest: 4 tests passed. ruff: clean.
  - Review 1 approved with 0 critical, 0 major and 6 minor findings (`reviews/review-1.md`).

## Results
Mip-NeRF 360, 30k steps, `--recipe 3dgs` (images_4 outdoor, images_2 indoor, llffhold 8). Run:
`runs/loss-aware-dc-mipnerf360`.

| scene | ours PSNR / SSIM / LPIPS | paper ours | Δ vs paper | our gsplat baseline | our gain | paper gain vs its 3DGS | Gaussians (ours) | train |
|---|---|---|---|---|---|---|---|---|
| bicycle | 24.98 / 0.759 / 0.247 | 25.31 / 0.770 / 0.213 | −0.33 | 24.92 | +0.06 | +0.12 (25.19) | 3.85M | 44 min |
| room | 31.71 / 0.916 / 0.298 | 32.07 / 0.921 / 0.218 | −0.36 | 31.52 | +0.19 | +0.69 (31.38) | 1.12M | 31 min |

- The paper reports a 1.9M Mip-NeRF 360 average for this method and 3.4M for 3DGS, but no
  per-scene counts.
- Our LPIPS is from the `lpips` VGG package and is systematically higher than the paper's (the
  baseline shows the same offset).
- The 2k-step garden smoke run gave 23.91 against 23.09 for the baseline, with 679k Gaussians
  against 846k (`evidence/smoke-4`).

## Contributions → implementation
- **C1, joint (p, log s) splitting matrix.**
  - `surrogate_stats` replays gsplat 1.5.3's forward pass and accumulates g_z and H_z.
  - `joint_split_accumulate` pulls these back to (p, log s) with hyper-dual derivatives of
    gsplat's projection, dropping the mixed p–s term (Eq. app_Sps_impl).
  - The matrix is normalised by J = blockdiag(R diag s, I/√2), and the minimum eigenvector gives
    the split direction.
  - Children are placed at p ± L v_p, log s ± v_s/√2, each with opacity o/2.
- **C2, split selection (Eq. 8):** stationary and λ_min < −1e-8. The screen-gradient filter is
  ANDed in; see gap 1.
- **C3, surrogate clone:** η = −g/h in (1e-6, 1e-2], child opacity η·o.
- **C4, selective opacity decay:** η = g/h in (1e-4, 1e-2]. There is no global opacity reset.
- **C5, proxy curvature ω:** Charbonnier plus the diagonal SSIM proxy, γ = 1e-4.

Probes:
- The kernel statistics match an independent float64 compositor to about 1e-7.
- The CUDA splitting matrix matches the torch reference to 1e-6.
- The torch reference matches a brute-force Hessian to 1e-15.
- A first-order re-render confirms the clone and decay surrogates (within 3%).

## Fidelity gaps (main gap first)
1. **Main gap: the split filter is borrowed from SteepGS.**
   - Eq. 8 lists only the stationarity and curvature tests. The prose also requires "unresolved
     view-space variation".
   - We AND in SteepGS's "uncertain" filter, a mean per-view screen gradient ≥ 2e-4 (the
     3DGS threshold), switchable with `require_uncertain`.
   - In practice about 98% of uncertain Gaussians pass the stationarity test and about 88% of
     those have λ < −1e-8. Whether a Gaussian splits is therefore set almost entirely by this
     borrowed threshold. The paper's loss-aware terms mainly set how it splits.
   - The literal Eq. 8, without the filter, would split nearly every visible Gaussian every 100
     steps.
   - This is the most likely cause of the high bicycle count (3.85M against an average of 1.9M)
     and of the weaker gain over the baseline.
2. **Clones are removed by the same-interval prune.**
   - With Alg. 1 read literally, a clone's opacity is η·o ≤ 0.01·o, and the o < 0.005 prune
     runs in the same interval.
   - Total clones were 72.0M on bicycle and 25.7M on room; total prunes were 73.4M and 26.0M.
   - The logs record only per-interval clone and prune totals, so the number of surviving clones
     cannot be separated out exactly. It lies between 0 and the clone count. Net growth (+3.7M)
     is fully explained by splits (5.13M in total), so the number of surviving clones is
     evidently small.
   - The off-by-default switch `exempt_new_clones_from_prune` is a diagnostic only; it is not
     the paper's behaviour.
3. **Unspecified details.**
   - ε_ssim is not given; we use 1e-8.
   - The splitting matrix is a sum over pixels averaged over the views where the Gaussian is
     visible (the SteepGS convention). This affects only τ_split.
   - ∂L/∂I is the gradient of the per-image mean loss, while ω is un-normalised. This is
     consistent with the paper's very small η thresholds.
4. **Optimizer state at decay.** Decay changes opacity in place and keeps its Adam moments. The
   paper is silent on this, and (1−η) ≥ 0.99, so the effect is minor.
5. **Pruning.** We prune o < 0.005, plus the world-size prune (> 0.1·extent) after step 3000.
   The Inria screen-size prune is not applied because it is a no-op in the official 3DGS code.
6. **Update order.** Density control runs after the optimizer step and the statistics use
   pre-step snapshots. J and the children use post-step scales and quats, a one-step mismatch
   (review minor 1).
7. **Runtime and rendering differences shared with the baseline.**
   - Background is white; the paper uses black.
   - The α clamp is 0.999 (gsplat) instead of 0.99 (Inria).
   - SSIM uses valid convolution.
   - Views are sampled with replacement.
   - LPIPS comes from the `lpips` package.
8. **Float32 round-off.** Tb = (C − acc)/(1−α) amplifies float32 round-off near α = 0.999. The
   effect on Δ is about 1e-4 relative (review minor 4).
9. **Probe coverage (review minor 5).** No probe asserts that parameters stay trainable leaves
   with aligned Adam state after `_densify`. The reviewer checked this by hand.

## Kept failed or aborted runs
- `evidence/smoke-1-oom-gpu1`: out of memory, because the GPU was full.
- `evidence/smoke-1`: trained with the earlier slow torch splitting-matrix code. It cannot be
  evaluated because the source manifest has changed since.
- `evidence/smoke-2-aborted-codechange`: aborted because the code changed during the run.
- `evidence/smoke-3-oom`: out of memory, because another user took the GPU.
