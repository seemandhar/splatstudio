# CRiM-GS (plan #114) — native port, benchmark and gap report

**Outcome: partial.** ball (real) matched within ±0.5 dB; cozy2room (synthetic) is 1.10 dB short.

| scene | ours PSNR / SSIM / LPIPS-VGG | paper PSNR / SSIM / LPIPS-Alex | gap |
|---|---|---|---|
| ball (deblur3dgs-motion) | 27.80 / 0.822 / 0.298 | 28.25 / 0.8064 / 0.1074 | −0.45 |
| cozy2room (deblur3dgs-synmotion) | 31.91 / 0.951 / 0.069 | 33.01 / 0.9409 / 0.0209 | −1.10 |

- Runs: `runs/crim-gs-realmotion`, `runs/crim-gs-synmotion`; 40k steps, single seed, about 270 min each.
- Final Gaussian counts: 105k (ball) and 98k (cozy2room).
- ss-eval renders the test views sharp at the COLMAP test pose, with no test-pose optimisation (paper Table 1 protocol).
- SSIM and LPIPS conventions differ from the paper's (valid-conv SSIM, VGG LPIPS), so compare PSNR only.
- There is no gsplat baseline on these datasets, and the paper's baseline (Mip-Splatting on blurry inputs) is reported as averages only, so the gain over baseline cannot be decomposed per scene.

## Build
- **No official code.** Written from arXiv 2407.03923v2.
- **Omitted numerics** come from CoMoGaussian (MIT, cd6d536): kernel lr, ODE tolerances, CNN norm layers, warm-up mean of N+1, no opacity reset, densify until 20k.
- **Verification:** ss-inspect clean, 5 unit tests pass, ss-verify `declared_checks_passed` (9 probes).
- **Reviewer:** approved (`reviews/review-1.md`).
- **Components:**
  - kernel owned by `crimgs-density`: per-image embedding → two latent ODEs (own torch dopri5) → screw-exp T_r and A+I T_a, composed as inv(c2w·T_r·T_a);
  - renderer: N=9 batched gsplat renders, CNN softmax weights + scalar mask, Mip-Splatting 3D/2D filters, black background;
  - losses: 0.7·L1 + 0.3·D-SSIM + 1e-4·L_o + 1e-3·L_M;
  - schedule: trajectory from step 1k, weights and mask from 3k.
- **Data:** new dataset `deblur3dgs-synmotion`, built by `tools/prepare_synthetic.py` from the Deblurring-3DGS release: full res, hold 8, test views are the sharp ones.

## Gap analysis (gap-analysis-1.md; training-view diagnostics only)
- **Densification stall** between 1k and 3k: statistics come from the sharp render only, which carries 1/(N+1) of the gradient. This matches the CoMo code, and growth resumes after 3k.
  - cozy2room: 19.9k / 26.9k / 80.2k / 95.5k / 98.4k Gaussians at 1k / 3k / 5k / 10k / 20k.
  - ball: 49.7k / 50.1k / 75.3k / 89.8k / 104.8k.
  - Neither paper reports Gaussian counts.
- **Blur reconstruction of the blurry training images:** 41.5 dB (cozy2room) and 41.4 dB (ball). A plain mean of the 9 renders gives 35.1 / 36.5; the sharp render alone gives 26.9 / 26.7. The mask is about 0.62 everywhere.
  - The fit is the same on both scenes, and ball matches the paper. Over-absorption by the blur model therefore does not single out cozy2room.
- **Trajectories and ODE:** trajectories are small and smooth (path length about 7% of the camera spread, rotation ≤0.4°). The ODE is well-conditioned, at about 12.7 dopri5 steps per solve, so review m1's stiffness concern did not occur.
- **Classification:** no method defect found. The most likely cause is a protocol/data offset: our test poses come from the Deblurring-3DGS COLMAP release, not the authors' own SfM. At about 33 dB, sub-pixel pose error costs about 1 dB, while the paper's post-optimisation gain on synthetic is only +0.2. Single-seed noise may add to it.
- **Not attempted:** the CoMo-grounded changes (centred ±4 integration, zero embedding init, time embedding). The evidence does not predict ≥0.5 dB from them.
- **Suggested diagnostics:** a gsplat/Mip baseline on cozy2room; the paper's appendix test-pose post-optimisation, run as a pose-alignment diagnostic only; CoMo's dataset/SfM for cozy2room.

## Fidelity gaps (all deviations)
1. **CoMo-code substitutes** for numerics the paper leaves out (listed above). CRiM-GS's own values are unknown.
2. **ODE time grid:** τ = 0..8, integrated forward from τ0, with an autonomous f (paper writes f(z, τ)). CoMo integrates ±4 from the middle, with a time embedding and rk4.
3. **Embedding init:** N(0,1), where CoMo uses zeros plus an Rt encoding.
4. **L_o:** Frobenius norm, where CoMo uses mean |RRᵀ−I|. The weight is 1e-4, so the effect is negligible.
5. **Solver:** our dopri5 hits the output times exactly and uses detached step control. Poses use a clean c2w right-multiplication convention, where CoMo uses a mixed [R_c2w|t_w2c].
6. **CNN:** weights have 1 channel broadcast over RGB (the paper writes N×H×W×3). The mask is a per-image 1×1 conv averaged over N, then sigmoid.
7. **Runtime differences:** views sampled with replacement, valid-conv SSIM in loss and metric, LPIPS-VGG, config records a white background while the renderer uses black (harmless), gsplat in place of the iComMa pose-backprop rasterizer.
8. **Data:** COLMAP from the Deblurring-3DGS release, not the authors' own SfM.
9. **Single seed**, and 2 reported scenes only (no dataset average claimed).
