# Mini-Splatting2 (plan #29) — reproduction report

**Outcome: matched** (4 of 5 scenes within ±0.5 dB; mean gap −0.21 dB; garden −0.56).
Paper: *Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification*,
pinned to arXiv **2411.12788v1**. arXiv v2 is a later journal version and is not reproduced.
Official code: fatPeter/mini-splatting2 @ 5ec9202 (`msv2/train.py`, `config/fast`, 18000 iterations).
Plugin: `plugin/` (`ss_mini_splatting2`). The benchmark ran `mini-splatting2` (outdoor importance
metric) on bicycle and garden, and `mini-splatting2-indoor` on counter, drjohnson and playroom.
This is the official README's imp_metric assignment. One seed, 18000 steps, 3DGS protocol
(llffhold8, images_4 outdoor / images_2 indoor). Only these 5 reported scenes were run, so
compare them with the paper's per-scene values, not with its dataset averages.

## Results (test split) vs paper (v1 appendix per-scene table)

| scene | method | PSNR ours | PSNR paper | Δ | SSIM ours / paper | LPIPS ours / paper | #G final ours / paper (M) | peak #G ours | train time ours / paper |
|---|---|---:|---:|---:|---|---|---|---:|---|
| bicycle | mini-splatting2 | 24.86 | 25.25 | −0.39 | 0.750 / 0.771 | 0.280 / 0.220 | 0.64 / 0.68 | 4.57M | 1164 s / 171 s |
| garden | mini-splatting2 | 26.24 | 26.80 | **−0.56** | 0.833 / 0.850 | 0.181 / 0.142 | 0.67 / 0.73 | 4.72M | 1864 s / 205 s |
| counter | -indoor | 28.61 | 28.59 | +0.02 | 0.906 / 0.908 | 0.255 / 0.193 | 0.54 / 0.55 | 4.62M | 1318 s / 279 s |
| drjohnson | -indoor | 29.69 | 29.60 | +0.09 | 0.909 / 0.910 | 0.318 / 0.238 | 0.77 / 0.80 | 6.55M | 622 s / 180 s |
| playroom | -indoor | 30.26 | 30.47 | −0.21 | 0.910 / 0.914 | 0.310 / 0.242 | 0.54 / 0.49 | 4.57M | 730 s / 151 s |
| **mean (5)** | | **27.93** | **28.14** | **−0.21** | | | | | |

The paper also reports a peak of 5.1M Gaussians at 3K iterations for bicycle; our peak is 4.57M.
Final Gaussian counts are within 10% of the paper on every scene. `depth_views_fallback` = 0 on
every run: all training views were drawn before step 2000, so every depth point got a
ground-truth colour.

**Comparison against our shared gsplat baseline** (`baseline3dgs-*-30000`, 30k steps, with the
opacity-reset fix). Compare our improvement over this baseline with the paper's improvement over
its own 3DGS row:

| scene | gsplat baseline PSNR | MSv2 − baseline (ours) | MSv2 − 3DGS (paper) | baseline #G / time |
|---|---:|---:|---:|---|
| bicycle | 24.92 | −0.06 | +0.06 | 6.59M / 6796 s |
| garden | 26.89 | −0.65 | −0.59 | 4.76M / 3199 s |
| counter | 28.85 | −0.24 | −0.48 | 1.22M / 4129 s |
| drjohnson | n/a (baseline run not finished) | – | +0.51 | – |
| playroom | n/a (baseline run not finished) | – | +0.49 | – |

On the three scenes with a baseline, the relative effect matches the paper within 0.25 dB. That
includes garden: the paper itself reports MSv2 0.59 dB below 3DGS there.

**Training time.** Time is the paper's headline metric. Our absolute times are 4–9× the paper's.
They were measured on an 8×RTX A6000 host shared with roughly 7–10 other training jobs (every GPU
was at 97–100% utilisation from other users), and use gsplat plus our Python statistics passes
instead of the Taming rasterizer on a dedicated RTX 3090. Only the speed-up over a baseline on the
same host is meaningful:

| scene | baseline/MSv2 time, ours | 3DGS/MSv2 time, paper |
|---|---:|---:|
| bicycle | 5.8× | 12.1× |
| garden | 1.7× | 10.5× |
| counter | 3.1× | 5.1× |

Our speed-up is smaller than the paper's and noisy, because host load varied between runs. The
speed claim is therefore **not verified** quantitatively on this host. Peak memory was 8.2–12.4 GB
(paper: 3.7–4.6 GB on average, using its SH-disable trick; we keep 15 zeroed shN coefficients
allocated).

## About the garden and bicycle offset
Both misses are outdoor scenes. The runtime calibration (`reproductions/_calibration/report.md`)
shows our gsplat runtime 0.3–0.5 dB below the original 3DGS code on outdoor scenes (bicycle −0.46,
garden −0.29) and 0.3 dB above it on indoor scenes (counter +0.32). Adding those offsets explains
most of the bicycle and garden gaps and the ≈0 indoor gaps. Relative to our own baseline, the
garden result (−0.65) agrees with the paper's own garden drop (−0.59). The most likely remaining
cause is outdoor protocol differences: white background, gsplat alpha numerics, and
densification statistics. No paper-grounded fix exists, so no fix iteration was run. The LPIPS
gap (+0.04 to +0.08 on every scene) comes from the `lpips` package: the calibration shows the
same +0.04 offset for gsplat versus 3DGS.

## Mechanism evidence (`evidence/verify-*.json`, `evidence/test.json`)
`ss-verify` returns `declared_checks_passed` for all 4 methods, the 9 unit tests pass and ruff is
clean. Probes:
- **msv2-stats-kernel**: the copied CUDA kernel agrees with a brute-force per-pixel blend. Argmax
  agreement is 1.0, max weight error 1.7e-5, T error 1.2e-7, and alpha matches gsplat exactly.
- **msv2-critical-clone**: the critical set and post-clone population (35 survivors + 33 clones)
  match an independent reference. Opacity and scale match the general 3DGS-MCMC binomial formula
  to about 1e-7. Copies start with zero Adam moments; sources keep theirs.
- **msv2-visibility-culling**: per-view masks equal the 0.99 CDF reference, and
  factor = count_vis/(count_rad+0.1) exactly. The culled render equals the subset render (error 0)
  and differs from the full render by 0.06. Culled Gaussians get no gradient and are not updated.
- **msv2-sparse-adam**: invisible rows and their moments are untouched. Visible rows match torch
  Adam to 3e-7, using gsplat's fused kernel on CUDA and torch on CPU.
- **msv2-depth-reinit**: point count equals N × factor, points lie on the plane (error 3e-5),
  colours come from the half-resolution ground truth, opacity is 0.1, Adam is fresh and culling is
  cleared.
- **msv2-half-res**: training views render at int(W/2)×int(H/2) up to step 3000 and at full
  resolution from 3001. Non-training views always render at full resolution.
- **msv2-schedule**: aggressive clones at 500–2750 every 250 steps (skipping 2000), densification
  at 600–2900, reinit at 2000, simplification at 3000 and 8000, culling off at 13000. The learning
  rate uses the 30000-step horizon, SH degree 0 until 4000, opacity learning rate 0.025.
- **Mutation checks**: a vanilla clone, a renderer that ignores culling, Adam without bias
  correction, and removing the clone events each make the matching probe fail.
- **Benchmark runs**: the event logs show the intended trajectory. For bicycle: 4.57M Gaussians
  at 2750, 0.89M after the 3000 sampling, 0.64M after 8000, culling off at 13000.

## Fidelity gaps
1. The runtime steps the optimizer before density control. The official code skips Adam on
   densification steps.
2. White background (runtime) instead of black (official default).
3. Views are sampled at random with replacement. The official code uses a shuffled stack, which
   also keeps half-resolution views until the stack empties after step 3000; we switch exactly at
   3001.
4. gsplat rasterizer (alpha clamp 0.999, 3σ tile extent) instead of the Taming rasterizer (0.99,
   opacity-aware extent).
5. Runtime SSIM (valid convolution) instead of fused-ssim (same padding). LPIPS-VGG comes from the
   `lpips` package.
6. Half-resolution targets use torch bicubic with antialiasing instead of PIL bicubic.
7. A training view never drawn before step 2000 would get rendered instead of ground-truth depth
   colours. This did not happen in any benchmark run (fallback = 0).
8. Timing is measured with gsplat on a shared, heavily loaded A6000 host instead of Taming on a
   dedicated RTX 3090. Absolute times are not comparable.
9. Peak memory is higher than the paper's because inactive SH coefficients stay allocated (zeroed).
10. CUDA statistics are copied from the Mini-Splatting port and inherit its gsplat numerics.

## Review minor items (`reviews/review-1.json`, approved)
- If the renderer↔density table lookup misses (for example, a training viewmat mismatch), training
  silently becomes stock gsplat. Suggested guard: raise in `before_backward` when a training frame
  has no view index before step 13000. Not applied. The benchmark event logs and the fallback = 0
  counts show the link worked in every run.
- The culling-off step and learning-rate horizon follow `total_steps = 18000` in the config,
  independently of `--steps`. The benchmark used `--steps 18000`. Any other step count needs the
  matching config.
- The half-resolution and sampling differences are already listed as gap 3. Keep reporting
  `depth_views_fallback`.
- The paper's dataset averages (main table) differ slightly from the means of its per-scene
  appendix values (M360 27.33 vs 27.35, DB 30.08 vs 30.04). This report compares per-scene values
  only.
- The half-res probe does not check the intrinsics scaling with a projection test. The code
  review confirmed `scale_intrinsics` preserves the FoV.

## Paths
- Runs:
  - `runs/mini-splatting2-mipnerf360/mini-splatting2/mipnerf360-{bicycle,garden}/`
  - `runs/mini-splatting2-indoor-mipnerf360/mini-splatting2-indoor/mipnerf360-counter/`
  - `runs/mini-splatting2-indoor-db/mini-splatting2-indoor/db-{drjohnson,playroom}/`
- Paper numbers: `paper_results.json`
- Specification and plan: `paper.md`, `plan.md`
- Smoke tests: `evidence/` (2000-step real schedule, and a compressed schedule covering every event)
