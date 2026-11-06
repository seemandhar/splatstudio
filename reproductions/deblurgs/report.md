# DeblurGS (plan #76): native port, build and smoke-test report

**Outcome:** built and verified; the benchmark has not been run yet.
- **Data:** 2 of the 10 scenes are ready; the other 8 are downloading.
- **Target:** Real Motion Blur average (paper Table 1), PSNR / SSIM / LPIPS-Alex 26.28 / 0.862 / 0.086. The paper's 3DGS row is 21.69.
- **Scope:** Table 1 gives averages only, so the comparison needs all 10 scenes.

## Build
- **Official code:** taekkii/deblurgs @e0587475. Its README marks it "under construction". The se(3) maps (pytorch3d code) are vendored verbatim.
- **No custom CUDA.** The official rasterizer is stock Inria, and gsplat renders all sub-frames in one batched call.
- **Methods:**
  - `deblurgs`: ν learning rate 3e-3, the paper's value.
  - `deblurgs-nu0`: ν learning rate 0, the released-code default.
- **Renderer `deblurgs-renderer`:**
  - Training views: renders 21 sub-frame poses on each view's Bézier se(3) trajectory in one batched gsplat call, with one random background per step.
  - All other cameras: renders sharp, as x^(1/2.2) over white.
  - Official activations: opacity clamp (with a straight-through 0.999 cap for gsplat), SH degree ≤ 2, near plane 0.2.
- **Density `deblurgs-density`:**
  - Owns the per-image trajectory bank: control points [n, 10, 6] and ν [n, 19], with its own Adam.
  - Bank schedule, as in the official code: frozen and single sub-frame before step 1000; learning rates halve every 15k; frozen again at 100k; ν learning rate from 30k.
  - Densification: the gradient threshold anneals from 4e-4 to 2e-4 over 25k steps. Statistics come from all sub-frames. Densify every 200 steps until 75k; reset to min(o, 0.1) every 3000 steps; prune by opacity only.
- **Losses:**
  - L1 of the blur against GT^2.2.
  - Smoothness term λ_t·mean|Δsub|, with λ_t going 1e-3 → 1e-5.
  - Opacity hinge, weight 0.1.
  - No D-SSIM.
- **Evaluation (`python -m ss_deblurgs.evaluate`):** the official test.py test-pose post-optimisation, then PSNR, SSIM and LPIPS (Alex and VGG). Writes `<run>/test_tpo.json`.

## Evidence
- **Checks:** `ss-inspect` is clean, 10 pytest tests pass, ruff is clean, and `ss-verify` reports `declared_checks_passed` for both methods.
- **Probes:**
  - Bézier curve matches de Casteljau to 3e-16; the se(3) exp map is exact.
  - The blur equals the mean of 21 independent renders to 1.2e-7.
  - Gamma handling is exact.
  - The annealing schedule is checked.
  - Test-pose post-optimisation recovers a perturbed camera (PSNR 27.6 → 61.7).
- **Smoke run** (ball, 2000 steps):

  | run | PSNR | Gaussians |
  |---|---|---|
  | `deblurgs`, unaligned | 22.13 | 4.5k |
  | `deblurgs`, after test-pose alignment | 22.51 | 4.5k |
  | gsplat | 23.43 | 87k |

  Being behind gsplat at 2k steps is expected: the trajectory curve only starts at step 1000, and densification is deliberately slow.

## Data
- **Source:** Deblur-NeRF `real_camera_motion_blur` (10 scenes), from the authors' Google Drive.
  - Downloader: `tools/download_motionblur.py`, which keeps retrying through rate limits.
  - Scene builder: `tools/prepare_motionblur.py`, dataset key `deblurnerf-motion`.
- **Build:** PIL ÷4 to 600×400 from the full-resolution JPGs. Test views are int(name) % hold == 0, using the release's `hold=N` files.
- **Missing COLMAP files:** if any are refused, rebuild from `poses_bounds.npy` with `reproductions/dofgs/tools/llff_to_native.py`, which was verified exact on Deblur-NeRF.

## Fidelity gaps
1. **Released code vs paper (we follow the code):**
   - Smoothness λ is 1e-3 → 1e-5 on mean-L1; the paper says 0.05 → 0.01 on L2.
   - Gamma is applied to the GT, with linear-space L1.
   - No D-SSIM.
   - Opacity uses a clamp plus hinge.
   - SH degree 2.
   - Densification runs every 200 steps until 75k, with reset to 0.1.
   - The θ annealing values appear only in the code.
2. **ν learning rate:** 0 in the code, 3e-3 in the paper. It is unknown which produced Table 1, so the two methods bracket it.
3. **Runtime differences:** views are sampled with replacement, the Adam step runs before densification, and SH degree rises one step later.
4. **Metrics:** post-optimised metrics follow test.py. LPIPS-Alex comes from the `lpips` package.
5. **Cost:** 150k steps × 21 renders per step is roughly 5–10 GPU-h per scene, about 60–100 GPU-h for all 10.
6. **Baseline:** no shared gsplat baseline exists for this dataset, and ours is indicative only (no gamma, no pose alignment).

*Report saved by the orchestrator; the build subagent could not write it.*

## Review-1 resolution (orchestrator)
**Major, fixed: evaluator metric conventions.** `evaluate.py` now writes an `aligned_official` block, which is the headline. It follows the official `test.py` `evaluate()`:
- no clamping;
- per-channel mean PSNR;
- same-padded SSIM;
- LPIPS-Alex fed [0,1] images directly, matching the official lpipsPyTorch convention.

On the smoke checkpoint (ball, 2k steps), LPIPS moves from 0.470 to 0.388 and PSNR from 22.510 to 22.528. The old clamped, runtime-convention metrics stay as `aligned`.

**Headline method:** `deblurgs` (ν lr 3e-3). The paper states 3e-3, the code's default of 0 carries the comment "# 3e-3", and the paper's ablation shows that learning ν helps. `deblurgs-nu0` is a bracket only, run on at most 2 scenes.

**Fidelity gaps added:**
- **ν schedule:** ν learns only between 30k and 100k (code schedule with the paper's lr). The paper says the lr halves every 15k.
- **Alpha cap:** the Inria forward caps alpha at 0.99; gsplat caps at 0.999.
- **Input images:** the release also ships `images_4/*.png`. We use PIL ÷4 of the full-res JPGs, which is the likely official path.
- **3DGS row:** the paper's 3DGS row (21.69) is post-optimised. Our gsplat baseline is not, so compare absolute numbers only.

**No cheaper target exists.** The only per-scene numbers are on ExBlur-NP, which is unreleased. The Real Motion Blur average needs all 10 scenes at 150k steps, about 60–100 GPU-h.
