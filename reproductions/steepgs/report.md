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

## Benchmark result (steepgs-mipnerf360, 30k, images_4/2) — MATCHED on PSNR; compaction weaker than the paper
| scene | steepgs | paper SteepGS | Δ | steepgs-adc control | paper 3DGS | ΔPSNR vs control ours / paper |
|---|---|---|---|---|---|---|
| garden | 27.11 / 0.856 / 0.143 | 27.16 / 0.851 / 0.142 | −0.05 | 27.36 | 27.42 | −0.26 / −0.26 |
| counter | 28.75 / 0.903 / 0.277 | 28.74 / 0.900 / 0.217 | +0.02 | 28.98 | 29.01 | −0.23 / −0.27 |

| scene | steepgs Gaussians | control Gaussians | ratio ours | paper SteepGS | paper 3DGS | ratio paper |
|---|---|---|---|---|---|---|
| garden | 3.74M | 6.09M | 0.61 | 2.20M | 5.85M | 0.38 |
| counter | 0.77M | 1.24M | 0.62 | 0.59M | 1.20M | 0.49 |

- PSNR and the quality cost of compaction both match the paper closely: −0.26 / −0.23 vs −0.26 / −0.27.
- The control matches the paper's 3DGS within 0.06 dB.
- Our Gaussian count is 1.6× (garden) and 1.3× (counter) the paper's, so the count reduction is weaker: about 39% vs 51–62%.
- Not investigated (no grounded fix identified). Candidates:
  - the split-direction / negative-eigenvalue threshold numerics in our port;
  - gsplat's tighter radii, which affect the screen-space gradient statistics.
- The garden run OOMed twice because other users shared the GPU; the third attempt completed.
