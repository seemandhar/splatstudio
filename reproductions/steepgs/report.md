# SteepGS (plan #68): build report

**Status:** ready for benchmark.
- **Methods:** `steepgs`, plus the control `steepgs-adc`, which is the same code with the official `--densify_strategy adc`.
- **Official code:** commit 63eee45, run with `--densify_strategy steepest --S_estimator inv_cov`. `inv_cov` must be passed explicitly, because the code's default value is the typo `inv_covar`.
- **Rasterizer:** the official CUDA rasterizer is vendored unmodified in `plugin/ss_steepgs/csrc` (Gaussian-Splatting licence).

## What was ported
- **C1, splitting matrix.** Each step, the backward pass's per-Gaussian sum over pixels of dL/dG is multiplied by the pre-step inverse covariance (the `inv_cov` estimator). It is then averaged over the views where the Gaussian was visible.
- **C2, saddle condition.** A Gaussian is densified only if both hold:
  - its mean NDC screen gradient is ≥ 2e-4;
  - the smallest eigenvalue of its averaged splitting matrix is below −1e-6.
- **C3, steepest-descent offsets.**
  - Split offspring go to p ± ‖N(0, diag s)‖·v_min, using the official sign pattern, with scale s/1.6 and copied opacity.
  - For clones, the copy and the original both move to p + lr·v_min.
- **C4, opacity halving.** The paper has it, but the official code leaves it off by default, so it is off here.
- **Rest of the schedule, as in 3DGS:**
  - densify every 100 steps between steps 500 and 15000;
  - prune at opacity < 0.005;
  - world-size prune after step 3000;
  - no screen-size prune, which is a no-op in the official code;
  - opacity reset every 3000 steps.

## Verification
- `ss-verify`: `declared_checks_passed`, 8 probes. The kernel's dL/dG and screen gradient match an independent float64 compositor to 1e-7 (relative).
- 5 unit tests pass; ruff is clean.

| garden, 2000 steps | PSNR | SSIM | LPIPS | Gaussians | train s |
|---|---|---|---|---|---|
| gsplat baseline | 23.02 | 0.643 | 0.419 | 845k | 174 |
| steepgs-adc (same rasterizer, official ADC) | 23.23 | 0.651 | 0.412 | 844k | 308 |
| **steepgs** | 21.83 | 0.575 | 0.468 | 296k (0.35×) | 250 |

The ADC control matches gsplat, so the renderer and the runtime are sound. At 2k steps SteepGS trails because it densifies much less early on. The paper's final result on garden is −0.26 dB with 0.38× the points.

## Benchmark request
Garden and counter on `mipnerf360` (images_4/2, as in the official `full_eval.py`), against `baseline3dgs-mipnerf360-30000`.

| scene | paper SteepGS | paper 3DGS | paper points (SteepGS / 3DGS) | our baseline |
|---|---|---|---|---|
| garden | 27.159 | 27.417 | 2.195M / 5.845M | 26.89, 4.76M |
| counter | 28.737 | 29.006 | 0.592M / 1.196M | 28.85, 1.22M |

Report PSNR and the Gaussian count against our baseline. The paper's shape is a small PSNR loss with roughly half or fewer of the points:
- garden: about −0.26 dB, 0.38× the points;
- counter: about −0.27 dB, 0.49× the points.

## Fidelity gaps
1. **Update order.** The runtime takes the optimizer step before density control; the official code densifies first.
2. **Shared runtime differences.**
   - Views are sampled with replacement.
   - SSIM uses valid convolution.
   - The SH degree steps up one iteration later.
   - LPIPS comes from the `lpips` package.
3. **Background.** Black, as in the official code; the runtime default is white.
4. **Code followed over the paper.**
   - Opacity is copied, not halved.
   - The split sign pattern follows the `repeat` rows.
   - A clone's copy and its original end up at the same position.
5. **Splitting matrix.** The code's `inv_cov` estimator is c·Σ⁻¹, a simplification of the paper's. As a result v_min is always the Gaussian's shortest axis.
6. **Speed.** About 1.4× slower per step than gsplat, because of the vendored rasterizer.

## Review-1 resolution (orchestrator)
- **Review outcome:** approved, with 0 critical and 0 major issues. The benchmark is launched: `steepgs` and the `steepgs-adc` control on garden and counter.
- **The `inv_covar` default crashes; it is not a silently used typo.** The estimator lookup raises a KeyError. The paper's runs must therefore have passed an estimator explicitly, and the README names `inv_cov` as the default.
- **Residual risk:**
  - The paper's Hessian may instead correspond to the `partial` estimator, which has not been ported.
  - If the 30k results land far from the paper's, `partial` will be ported as a labelled variant.
- **What `inv_cov` reduces to:**
  - It drops the paper's σ weighting and the outer-product term, and flips the sign of the dominant term.
  - Selection therefore reduces to "c < 0 and high screen gradient".
  - This is a difference between the paper and the code; the port follows the code.
