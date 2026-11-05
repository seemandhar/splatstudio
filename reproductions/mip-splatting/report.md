# Mip-Splatting — native SplatStudio reproduction report

**Paper**: Mip-Splatting: Alias-free 3D Gaussian Splatting (Yu, Chen, Huang, Sattler, Geiger;
CVPR 2024; [arXiv 2311.16493](https://arxiv.org/abs/2311.16493)). Plan id 26.
**Official code**: <https://github.com/autonomousvision/mip-splatting>, followed at commit
`1ecf7d5`. That is the last commit before the April 2024 "add GOF densification" commits, which
postdate the paper and are not reproduced.
**Method**: `mip-splatting` (plugin `plugin/`, module `ss_mip_splatting`). No custom CUDA is used.
**Outcome**: **matched**. All 3 scenes are within ±0.5 dB of the paper's PSNR: bicycle −0.17,
garden −0.33 (both after the iteration-1 image-protocol fix) and counter −0.19 (iteration 0).
On every scene the gain over our 3DGS baseline is at least the paper's gain over its 3DGS*.
A matched PSNR is not proof of fidelity; all deviations are listed below. Single seed (0).

## 1. What was implemented

| contribution | paper | implementation |
|---|---|---|
| C1 3D smoothing filter | §4.1, Eq. 8–9 | Per Gaussian, `filter_3d = sqrt(0.2)·d_min/f_max`. `d_min` is the smallest camera depth z among training cameras that see the Gaussian (z > 0.2, centre projected inside the image with a 15 % margin); `f_max` is the largest focal_x. Gaussians no camera sees get the largest visible depth. Rendering uses scales `sqrt(s²+f²)` and opacity `σ(o)·sqrt(Πs²/Π(s²+f²))`, with the filter held constant (no gradient). The filter is a per-Gaussian field, so it is saved in the checkpoint and used at test time. |
| C1 schedule | §4.1 "every m iterations" + `train.py` | The filter is recomputed at initialisation (through the runtime hook `set_training_cameras`, training split only), after every densification step (600…14900), and every 100 steps in 15100…29800 (not at 15000 or ≥ 29900). |
| C1 filter-aware opacity reset | code `reset_opacity` | At 3000, 6000, 9000 and 12000 the *filtered* opacity is clamped to 0.01 and Adam moments are zeroed. Exactly one reset per reset step. |
| C2 2D Mip filter | §4.2, Eq. 10 | gsplat `rasterize_mode="antialiased"` with `eps2d=0.1`. This is identical to the official `computeCov2D` (Σ2D + 0.1 I, opacity × sqrt(det ratio), differentiated through), up to 1e-6 guards. The built-in `gsplat-aa` uses gsplat's default 0.3 and has no 3D filter. |
| unchanged | — | gsplat DefaultStrategy densification (grad 2e-4, 500–15000 every 100), L1 + 0.2 D-SSIM, optimiser/LRs, SH schedule, `--recipe 3dgs` initialisation. |

## 2. Protocol
Mip-NeRF 360, single-scale training and testing at the same scale (paper main Table 3; per-scene
numbers in supplementary `tab/normal_setting_360_supp.tex`).
- Splits: llffhold8.
- Resolution: outdoor ¼, indoor ½.
- Training: 30k steps, `--recipe 3dgs`, all SfM points kept.
- Metrics: LPIPS-VGG.
- Scenes: bicycle and garden (outdoor), counter (indoor).
- Paper comparison: the paper's improvement is measured against **3DGS\*** (3DGS retrained by the
  Mip-Splatting authors). Our comparison is against `baseline3dgs-*` (gsplat with the
  opacity-reset fix, §4.2).

## 3. Results

### Iteration 0: dataset `mipnerf360` (pre-downsampled `images_4` / `images_2`)
Runs: `runs/mip-splatting-mipnerf360/mip-splatting/`, baseline `runs/baseline3dgs-mipnerf360-30000/gsplat/`.

| scene | PSNR ours | PSNR paper | Δ paper | SSIM ours / paper | LPIPS ours / paper | Gaussians (M) | train min* |
|---|---:|---:|---:|---|---|---:|---:|
| bicycle | 25.17 | 25.72 | −0.55 | 0.762 / 0.780 | 0.243 / 0.206 | 7.14 | 246 |
| garden | 27.13 | 27.76 | −0.63 | 0.862 / 0.875 | 0.131 / 0.103 | 4.92 | 112 |
| counter | 28.97 | 29.16 | **−0.19** | 0.908 / 0.916 | 0.261 / 0.179 | 1.26 | 75 |

Baseline-relative (PSNR):

| scene | our gsplat baseline3dgs | ours − baseline | paper 3DGS\* | paper gain over 3DGS\* | baseline − 3DGS\* |
|---|---:|---:|---:|---:|---:|
| bicycle | 24.92 (SSIM 0.757, LPIPS 0.243, 6.59 M) | **+0.25** | 25.63 | +0.09 | −0.71 |
| garden | 26.89 (0.860, 0.131, 4.76 M) | **+0.23** | 27.70 | +0.06 | −0.81 |
| counter | 28.85 (0.907, 0.260, 1.22 M) | **+0.12** | 29.08 | +0.08 | −0.23 |

The gains over our baseline match or exceed the paper's. The absolute gaps on the outdoor scenes
come from the baseline already sitting 0.7–0.8 dB below the paper's 3DGS\*. That pointed to a
data-protocol difference rather than the method (gap G2).

\*Wall-clock times are not comparable between runs: the host was heavily shared, and one smoke
run spent about 35 minutes just loading the dataset.

### Iteration 1: protocol-only change, dataset `mipnerf360-pilr`
**Change**: the official scripts run `train.py -r 4` with the default `--images images`. That
loads the full-resolution JPEGs and resizes them with PIL (`Image.resize`, default bicubic) to
`round(W/4)` (banker's rounding, so bicycle is 1236 px wide); it does not use the dataset's
`images_4`. `tools/make_pil_resized.py` rebuilds that input (intrinsics rescaled; poses, splits
and points from the native conversion). The method, its config and the seed are unchanged.
Outdoor scenes only: counter was already within tolerance and was not re-run.
Runs: `runs/mip-splatting-i1-pilr/{mip-splatting,gsplat}/mipnerf360-pilr-{bicycle,garden}/`.

| scene | PSNR ours | PSNR paper | Δ paper | SSIM ours / paper | LPIPS ours / paper | Gaussians (M) |
|---|---:|---:|---:|---|---|---:|
| bicycle | 25.55 | 25.72 | **−0.17** | 0.776 / 0.780 | 0.236 / 0.206 | 6.83 |
| garden | 27.43 | 27.76 | **−0.33** | 0.869 / 0.875 | 0.129 / 0.103 | 4.35 |

| scene | our gsplat (PIL) | ours − baseline | paper gain | baseline − 3DGS\* |
|---|---:|---:|---:|---:|
| bicycle | 25.35 (SSIM 0.770, LPIPS 0.237, 6.26 M) | **+0.20** | +0.09 | −0.28 |
| garden | 27.27 (0.867, 0.129, 4.15 M) | **+0.16** | +0.06 | −0.43 |

The PIL protocol recovers +0.38 / +0.30 dB for Mip-Splatting and +0.43 / +0.37 dB for the
baseline. The remaining offset is that our runtime's 3DGS baseline sits 0.3–0.4 dB below the
paper's 3DGS\*, which is shared by both methods (gap G3).

### Final per-scene status
| scene | final PSNR | Δ paper | within ±0.5 | gain ≥ paper gain |
|---|---:|---:|:---:|:---:|
| bicycle (i1) | 25.55 | −0.17 | yes | yes (+0.20 vs +0.09) |
| garden (i1) | 27.43 | −0.33 | yes | yes (+0.16 vs +0.06) |
| counter (i0) | 28.97 | −0.19 | yes | yes (+0.12 vs +0.08) |

LPIPS is systematically higher than the paper's (+0.03 outdoor, +0.08 counter), for us and for
our baseline alike. This is the known LPIPS implementation offset in
`reproductions/_calibration/`; compare LPIPS only as deltas, which are ≈ 0, as in the paper.
Because the gains are larger than the paper's, the extra gain may partly come from the gsplat
runtime itself (e.g. the 2D filter acting differently on a baseline that sits below 3DGS\*).
With a single seed this was not separated.

## 4. Incidents and fixes

### 4.1 Opacity reset produced NaN (first benchmark crashed in all 3 scenes)
- **Symptom**: `ValueError: opacities: nonfinite Gaussian state` right after the reset at step
  3000 (garden) or 6000 (bicycle, counter).
- **Cause** (reproduced with a minimal fixture): the official coefficient `prod(s²)/prod(s²+f²)`
  underflows to 0 in float32 for Gaussians much smaller than their filter (s = 1e-8, f = 4e-3 →
  coef 0). The reset `logit(min(σ·c, 0.01)/c)` then becomes 0/0 = NaN. A saturated σ = 1.0
  gives +inf. The official code has the same latent problem.
- **Fix** (documented deviation):
  - The coefficient is computed in log space, `Σ log s − ½ Σ logaddexp(2 log s, 2 log f)`, for
    both rendering and the reset.
  - The reset is evaluated as the algebraically identical `logit(clamp(min(σ, 0.01/c), 1e-6, 1−1e-6))`.
  - It agrees with the official formula wherever that is finite.
- **Guards**:
  - `tests/test_reset_finite_in_failure_regime` shows the official formula is non-finite on the
    fixture while ours is finite and equal elsewhere.
  - The `filter-aware-reset` probe fails when the old formula is swapped back in (checked).
  - A garden run through step 3100 (`evidence/reset-3100`) completed with finite opacities.
  - The benchmark jobs were relaunched after the fix; all results above come from the fixed plugin.

### 4.2 gsplat 1.5.3 never resets opacity (found during planning)
`DefaultStrategy.step_post_backward` checks `if step % self.reset_every == 0 & step > 0`.
Because `&` binds tighter than comparisons, this means `(step % reset_every == 0) and (0 > 0)`,
which is always False. The original shared baselines (`baseline-*-30000`) therefore never reset
opacity, which probably contributes to their larger outdoor Gaussian counts.
- **Runtime change (coordinator)**: `splatstudio.backends.gsplat.Density` now performs the 3DGS
  reset itself (`opacity_reset: true` by default), and new baselines `baseline3dgs-*-30000`
  include it. We use those.
- **Plugin side**:
  - The plugin wraps `DefaultStrategy` directly and applies only its own filter-aware reset.
  - It refuses to start on a gsplat whose built-in reset works, to prevent a double reset.
  - A unit test pins the gsplat behaviour.
  - The `single-reset` probe fails on a missing, unfiltered or doubled reset (checked by
    deliberately breaking each case).
- The temporary `mip-splatting-noreset` variant was removed.

### 4.3 Training-camera access
The first version found the training cameras by looking up the runtime's `dataset` in the
calling frame. It now uses the runtime hook `set_training_cameras` (training split only), and
`before_backward` raises if the hook was never called (probe `camera-hook`). The built-in
renderer-gradient probe does not call the hook, so it is replaced by an equivalent probe that does.

## 5. Probes (`ss-verify mip-splatting --device cuda` → `declared_checks_passed`)
| probe | checks | fails if |
|---|---|---|
| filter3d-frequency-bound | the computed filter matches an independent per-point/per-camera loop (max error 1.9e-9), the hand-computed value sqrt(0.2)·5/100, and the fallback for Gaussians no camera sees | filter missing, wrong coefficient, wrong visibility rule |
| filter3d-convolution (CUDA) | the render matches gsplat given the explicit covariance Σ + f²I and the det-based opacity factor (max error 1.2e-7); filtered and unfiltered renders differ | 3D filter not applied or not compensated |
| mip-filter-2d (CUDA) | peak opacity of an on-axis Gaussian equals the analytic σ·sqrt(det/det(+0.1 I)) (0.38938 vs 0.38938) | classic mode, eps2d 0.3, no compensation |
| filter-aware-reset | filtered opacity is min(old, 0.01), moments are zeroed, and tiny Gaussians stay finite | unfiltered reset, NaN regime |
| single-reset (CUDA) | one filter-aware reset at step 3000, none at 2900 | missing, doubled or unfiltered reset |
| camera-hook | the filter comes from `set_training_cameras`; training without the hook raises | silent zero filter |
| filter-schedule | 292 recompute steps (600…29800, excluding 15000) and resets at {3000, 6000, 9000, 12000} | schedule changed |
| state-transitions, photometric-gradient, renderer-gradient (hook-aware) | built-in infrastructure checks, including `filter_3d` through duplicate/split/prune | — |

Unit tests: 9 pass (CPU). ruff is clean. `ss-inspect` reports no issues.

## 6. Fidelity gaps / deviations
- **G2 Image protocol**: iteration 0 used `images_4`/`images_2`; iteration 1 matched the
  official PIL resize for the outdoor scenes. Counter (indoor, `images_2`) still uses the
  pre-downsampled images.
- **G3 Shared runtime differences**: white background (official 360 runs use black), view
  sampling with replacement, densification/filter update/reset after (not before) the optimiser
  step, gsplat near plane 0.01 vs 0.2 culling, alpha clamp 0.999 vs 0.99, valid-convolution
  SSIM. These cause our 3DGS baseline to sit 0.28–0.43 dB below the paper's 3DGS\* under the
  PIL protocol.
- **G5 Principal point**: the visibility projection uses K's cx, cy (official assumes W/2, H/2);
  these are identical for these datasets.
- **G6 Numerical**: log-space coefficient and clamped reset (§4.1); identical wherever the
  official formula is finite.
- **Paper vs code**: the paper writes s/ν̂, while the code uses variance 0.2/ν̂² and a global
  f_max instead of per-camera f_n. We follow the code (identical here, one intrinsics per scene).
- **Not evaluated**: multi-scale protocols (Blender multi-scale training and testing;
  single-scale training with multi-scale testing on Blender and 360), which are the paper's
  headline anti-aliasing results. The single-scale 360 benchmark only shows that the filters do
  no harm and slightly help at the training scale. Also: only 3 of 9 scenes, a single seed,
  and no T&T/DB.

## 7. Review (reviews/review-1.json, approved; minors)
1. Update/reset run after the optimiser step and use gsplat's near plane: shared with the
   baseline, documented (G3).
2. `filter_3d` FieldSpec lr = 1e-8 (the runtime requires lr > 0): inert because the field is
   used detached. A unit test asserts no gradient reaches it.
3. `logit` could reach inf for saturated σ: fixed by the clamp in §4.1, which also covers the
   0/0 underflow the review did not anticipate.

## 8. Reproduce
```bash
T=tools/ssenv
cd reproductions/mip-splatting/plugin && $T pip install -e . && cd -
$T ss-inspect mip-splatting
$T python -m pytest -q reproductions/mip-splatting/plugin/tests
$T ss-verify mip-splatting --device cuda --output reproductions/mip-splatting/evidence/verify-mip-splatting.json
# iteration 0 (images_4 / images_2)
python3 tools/bench.py launch --exp mip-splatting-mipnerf360 --methods mip-splatting \
  --dataset mipnerf360 --scenes bicycle garden counter --steps 30000 --recipe 3dgs
python3 tools/bench.py summarize --exp mip-splatting-mipnerf360 \
  --paper reproductions/mip-splatting/paper_results.json
# iteration 1 (official PIL resize of full-res images)
python3 tools/make_pil_resized.py            # builds ~/datasets/mipnerf360/pilr (see tools/datasets.json)
python3 tools/bench.py launch --exp mip-splatting-i1-pilr --methods mip-splatting gsplat \
  --dataset mipnerf360-pilr --scenes bicycle garden --steps 30000 --recipe 3dgs
python3 tools/bench.py summarize --exp mip-splatting-i1-pilr \
  --paper reproductions/mip-splatting/paper_results.json
```
Evidence: `evidence/verify-mip-splatting.json`, `evidence/test.json`, `evidence/smoke-2`
(garden, 2000 steps, 23.15 dB vs gsplat 22.99), `evidence/reset-3100`.
