# DOF-GS (plan #60): native port, build and smoke-test report

**Outcome: matched on all 3 benchmark scenes.**
- All-in-focus test PSNR:
  - **cake** 26.73 vs 26.87 (−0.14)
  - **caps** 24.39 vs 24.63 (−0.24)
  - **tools** 27.79 vs 27.73 (+0.06)
- **Gain over plain gsplat** on the same blurry inputs: +5.24 / +8.78 / +3.59 dB. The paper has no 3DGS row; its gains over Mip-Splatting are +5.12 / +3.01 / +3.93.
- **Scope:** this is a 3-scene subset, not the paper's 9-scene real-defocus average.

## Build
- **Methods:** three variants, one per official real-scene config group (`config_real_<scene>.yaml`; only `band_limit` and `lambda_mask_loss` differ between the 9 scenes):

  | Method | band_limit | λ_mk / λ_reg | Scenes |
  |---|---|---|---|
  | `dofgs` | true | 0.002 / 0.0002 | cake, daisy |
  | `dofgs-bl-m001` | true | 0.001 / 0.0001 | cisco, sausage, seal, tools |
  | `dofgs-nobl` | false | 0.001 / 0.0001 | caps, cupcake, cups |

- **Code:** the official DOFGS_implementation @cfd635c. Its DOF CUDA kernel is vendored unmodified (Inria/MPII licence) and JIT-built.
- **Renderer `dof-raster`:** renders training views defocused, using each view's learned focus and aperture. All other cameras are rendered all-in-focus (aperture 0), as in the official evaluation.
- **Per-view bank:** created and stepped by `dofgs-density` on the official schedule. Gradients accumulate before step 100, updates run until step 15000, and the parameters are frozen after that. The bank is saved in `density_state["bank"]`.
- **`dofgs-objective`:** L_rec up to step 5000, then the official code's two composite terms. It replaces the base photometric loss.
- **`dofgs-mask`:** Pearson mask loss and entropy regulariser, weighted λ_mk / 0.1 λ_mk per variant (see table).
- **`dofgs-density`:**
  - 3DGS densification: percent_dense 0.03, prune below opacity 0.005, until step 25k.
  - Compactness densification every 1k steps.
  - Extra points added at step 2k.
  - When band_limit is true: Mip 3D filter in rendering and a filter-aware opacity reset. When false: no 3D filter, no filter computation, and the plain reset min(σ(o), 0.01).
  - xyz LR 5e-4 → 5e-6 over 80k steps.

## Evidence
- **Checks:** `ss-inspect` reports no issues for all three variants, 23 pytest tests pass, ruff is clean, and `ss-verify --device cuda` reports `declared_checks_passed` for `dofgs`, `dofgs-bl-m001` and `dofgs-nobl`.
- **Per-scene config:** the `official-config` probe pins each variant to the official yaml values of its scenes; pytest checks that transcription against all 9 `config_real_<scene>.yaml` files. The `band-limit-switch` probe shows a band_limit=false render matches the unfiltered kernel call (≤1e-6) and resets to min(σ(o), 0.01).
- **Kernel vs an independent float64 renderer** (apertures 0, 40 and 120):
  - RGB error ≤ 7e-8; depth and CoC error ≤ 1e-6.
  - A renderer with the CoC term removed differs by 0.07–0.26, so the check would catch its loss.
- **Aperture-0 limit:** matches a standard render to 1.8e-7. Gaussians on the focal plane are not blurred.
- **Analytic blur:** the fitted variance equals σ'² + 0.1 + R²/(2 ln 4) to 1e-6 relative.
- **Gradients:** focus and aperture gradients are within 5e-6 / 1e-7 relative. The CoC→depth gradient is off, as it is in the official code.
- **Smoke run** (cake, all-in-focus test PSNR):

  | Run | Test PSNR |
  |---|---|
  | `dofgs`, 1950 steps | 25.88 |
  | gsplat, 2000 steps | 23.65 |
  | `dofgs`, 2000 steps | 20.70 |

  The 2000-step drop is a snapshot effect: about 25k new points are added at step 2000 and have not been optimised yet.

## Data
- **Source:** the Deblur-NeRF real defocus set, from the authors' Google Drive via gdown (dataset key `deblurnerf-real`).
- **Ready:** cake and caps.
- **tools:** all images are downloaded, but 4 COLMAP sparse files are still rate-limited or refused by Drive.
- **Scene build:** PIL resize ÷4 to 600×400 with the official per-scene holdout. Validation is a copy of test.
- **Synthetic set:** its Drive folder fails to list (HTTP 500), so it is not used.

## Paper targets
Per-scene numbers come from the supplementary real-data table.

| Scene | DOF-GS | Mip-Splatting |
|---|---|---|
| cake | 26.87 | 21.75 |
| caps | 24.63 | 21.62 |
| tools | 27.73 | 23.80 |

The paper has no 3DGS row.

## Fidelity gaps
1. **Paper vs official code (we follow the code):**
   - The composite loss stays on the defocused branch after warm-up.
   - Mask weights follow the per-scene configs: 0.002 / 0.0002 for cake and daisy, and 0.001 / 0.0001 (the paper's values) for the other 7 scenes.
   - Depth quantiles are 5/95 %.
   - The xyz LR decays over 80k steps.
   - Compactness densification exists only in the code.
2. **Official quirks ported as-is:**
   - The compactness test uses raw log-scales.
   - Only about 540 of the 60k extra-point candidates survive (539 on cake), as in the official code.
   - The aperture never goes below 101.
3. **Runtime differences:**
   - Views are sampled with replacement.
   - The optimizer steps before densification. On every densify/prune, compactness and reset step (~245 of 40k), the official code replaces all Gaussian parameters before `optimizer.step`, so it skips the Adam step for **all** Gaussians. The runtime steps them. Negligible.
   - The SH degree is raised one step later (step 1001, not 1000). Negligible.
   - The last optimizer step is not skipped.
4. **Metrics:**
   - Only PSNR is comparable, because the paper uses skimage SSIM.
   - The official val scripts pick the iteration per scene (val_real.sh: cake 15k, caps 15k, tools 40k), which is test-set selection. The protocol and headline result are the final 40k state. A 15k number may appear only as a labelled diagnostic.
7. **band_limit=false variant:** the official code still computes the 3D filter but never uses it. The port skips that computation, which does not change the result.
5. **Baseline:** no shared gsplat baseline exists for this dataset, so one is run alongside.
6. **Mip filter coefficient:** computed in log space. This is mathematically identical.



## Training stability fix
- **What failed:** the `dofgs` benchmark on cake stopped twice, at steps 26000–26500 and 34500–35000, with `quats: invalid optimizer moment exp_avg_sq`. This is the runtime's `check_state`. The quats themselves stayed finite.
- **Reproduction:** an instrumented 40k run (scratch `nanhunt.py`) failed at step **27874**.
  - The offender is Gaussian 60079: log-scales (8.45, 0.14, 0.34), filtered to 4663 × 1.18 × 1.43 units, at camera depth 271. Its quat norm is 0.79 (not degenerate) and σ(o)·coef = 0.023.
  - Its kernel gradient was dL/dq = 8.3e19 (view 17, defocused render), above the overflow threshold √FLT_MAX = 1.8e19. Scales and means grads were 1e15–5e16.
  - The same thing happened earlier at lower size: from step ~12k, other needles (log-scale ~7, about 1000 units long, depth ~165) gave 1e13–1.3e17.
- **Root cause:** float32 cancellation inside the unmodified official kernel.
  - The needle's 2D covariance entries are ~2.6e7. The kernel computes det = a·c − b² (det_0, det_1 for the Mip 2D filter, and `denom` for the conic gradient).
  - At that size the float32 ulp of a·c is ~6.7e7, but the true det is ~2.3e8. The computed value is therefore rounding noise: 0, negative, or det_0 = det_1.
  - It is then clamped (1e-6, or 1/(denom² + 1e-7)) in the backward pass, which amplifies the gradient.
  - Replaying that single Gaussian with its camera gives |dL/dq| between 3e-4 and 1.2e21 when one log-scale changes by 2e-5.
  - Adam's g² then overflows, so exp_avg_sq = +inf.
- **What the official code does:** it uses the same kernel and plain torch Adam with no state checks. The inf stays forever (β₂·inf = inf), so that element's update is m/inf = 0 for the rest of training: it silently freezes. The parameter does not become NaN unless g itself overflows, which never happened.
  - The `dofgs-nobl` caps run completed. It had no 3D filter, it is a different scene, and it was not instrumented.
- **Fix:** `ss_dofgs/density.py:clamp_overflowed_moments`, called at the top of `Density.after_backward` (after the runtime's Adam step, before `check_state`).
  - Any exp_avg_sq entry above FLT_MAX/2 (only reachable by overflow) is pinned to FLT_MAX on every step, and its exp_avg is set to 0.
  - Each later step moves that entry by about lr·0.1·|g|/1.8e19 (~1e-4·lr at |g| = 1e17). This is the official freeze to float noise. All other entries are bit-identical (tested).
  - NaN moments are left untouched, so a real bug still fails.
  - The number of pinned entry-steps is reported in `density_state["counts"]["moments_clamped"]`.
  - The kernel is unchanged, no hyper-parameters changed, and no Gaussians are pruned or altered.
- **Tests (26 pass):**
  - `test_overflowed_adam_moment_is_frozen_like_official`: unchecked Adam against the port.
  - `test_overflow_freeze_leaves_nan_moments`.
  - `test_needle_gaussian_overflows_adam_and_freeze_keeps_state_valid_cuda`: the real step-27874 Gaussian and camera through the official kernel. |g_q| > 1.9e19, then inf exp_avg_sq, then finite after the fix.
  - `ss-verify dofgs --device cuda`: `declared_checks_passed`.
- **Validation run** (scratch `fixed_run.py`, cake, 40k, same settings as the benchmark): completed.
  - At the end, 21 moment entries were frozen in at most 3 Gaussians (means 4, quats 10, scales 7). `moments_clamped` = 137081 entry-steps.
  - Final count: 307449 Gaussians.
  - Validation PSNR 26.91, SSIM 0.899, LPIPS 0.172. Validation is a copy of test, so this is a sanity check only. The benchmark number must come from `tools/bench.py`.


## Benchmark results
Setup: 40000 steps, Deblur-NeRF real defocus at 600×400, the official per-scene holdout, and all-in-focus test renders (aperture 0). We report the final 40k state.

| scene | method (variant) | PSNR ours / paper (Δ) | SSIM | LPIPS | #G | gsplat baseline (ours) | Mip-Splatting (paper) |
|---|---|---|---|---|---|---|---|
| cake | `dofgs` (band_limit, λ_mk 0.002) | **26.73 / 26.87 (−0.14)** | 0.896 | 0.176 | 318k | 21.49 | 21.75 |
| caps | `dofgs-nobl` (no band_limit, λ_mk 0.001) | **24.39 / 24.63 (−0.24)** | 0.859 | 0.197 | 190k | 15.61 | 21.62 |
| tools | `dofgs-bl-m001` (band_limit, λ_mk 0.001) | **27.79 / 27.73 (+0.06)** | 0.942 | 0.114 | 260k | 24.20 | 23.80 |

- Runs: `runs/dofgs-cake/`, `runs/dofgs-caps/`, `runs/baseline-gsplat-deblurnerf-40000/`.
- SSIM is not comparable with the paper (skimage vs valid-convolution SSIM), and the paper reports no LPIPS.

**cake run history.** Two earlier attempts stopped at steps 26k and 34.5k on the runtime's optimizer-state check; the training-stability fix above explains why. Those attempts are kept as `.stale-*`. The reported run uses the fixed plugin.

**caps.** It ran on the plugin before the fix. The fix only acts when an Adam moment overflows, and caps completed with no such event, so its result is unaffected.

**tools.** The authors' Google Drive refuses `cameras.bin` and `images.bin` for `defocustools/sparse/0` ("Cannot retrieve the public link"). The scene was rebuilt from its LLFF `poses_bounds.npy`, which encodes the same COLMAP poses, plus the official `points3D.bin` (`tools/llff_to_native.py`).
- The conversion is c2w columns in permutation (1,0,2) with signs (1,−1,1), translation unchanged, focal = hwf[2], and a centred principal point.
- Checked on cake and caps, which have the full COLMAP model: rebuilding cake this way reproduces its real conversion to a maximum pose difference of 9e-16, with identical intrinsics.
