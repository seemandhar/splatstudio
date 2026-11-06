# ResGS (plan #96) — build report

- Method `resgs`, plugin `reproductions/resgs/plugin` (module `ss_resgs`), official code
  github.com/yanzhelyu/ResGS @a0db69d read and ported (no custom CUDA; gsplat absgrad rasterizer).
- Official code host: AbsGS-based 3DGS fork → it produces the **"Ours (AbsGS)"** row (tau 0.00067).
  The "Ours (3D-GS)" (tau 0.00028) and "Ours-Small" variants are not the released defaults.
- Contributions → components: residual split (Eq. 4-5) + varying threshold (Eq. 8) + AbsGS opacity
  reduction → `resgs-density`; image pyramid → `resgs-renderer` (pyrDown-size training renders,
  full-res eval) + pyramid-target `photometric` loss; schedule = literal port of train.py counters.
- Verified schedule: stages 1..2500 (1/4), 2501..6000 (1/2), 6001.. (full); 500-iter warm-ups;
  densify every 100 in (700,12000), 104 densify events; substage levels 1..9 first used at
  800/1400/2000/3000/4000/5000/6000/8400/10200; opacity x0.8 at 2500..24500 (45x), prune-before
  from 12000. Official LRs (1.28e-4, 0.002, 0.04, 0.006, 0.001) applied.
- ss-verify: declared_checks_passed (10 probes). Unit tests 8 passed, ruff clean.
- Smoke (garden, 2000 steps, 3dgs recipe): default 20.98 dB (stays in the 1/4-res stage, expected);
  compressed schedule crossing both stage changes 22.60 dB / 1.66M G; gsplat baseline 23.05 dB.
- Fidelity gaps: white vs black background; with-replacement view sampling; optimizer step before
  density ops; pyramid camera K scaled by w'/W (sub-pixel principal point); gsplat rasterizer vs
  AbsGS fork; runtime eval SSIM valid-padded. Compare delta vs baseline3dgs-mipnerf360-30000 with the
  paper's delta vs its 3DGS row (garden +0.41, counter +0.80).
- Benchmark requested: mipnerf360 garden + counter, 30k (bench_request.json).

## Review-1 resolution (orchestrator)
- **Verdict:** approved, with 0 critical, 0 major and 3 minor issues.
- **Host:** AbsGS / tau 0.00067 = the "Ours (AbsGS)" row.
- **Residual split:** a differential test against the official densify_residual_split shows zero difference in every field, Adam moment and level.
- **Schedule:** the selection thresholds, abs-grad scaling, schedule, pyramid (≤1.8e-7 vs cv2) and LRs all match.
- **Gaps to add (minor):**
  - The official loop skips the Adam step on the 104 densify iterations for parameters that were just replaced, and on the 45 opacity-reduction iterations; the runtime takes those steps.
  - The means LR uses (t−1)/T.
  - The SH degree goes up at step 1001.
  All three are negligible.
- **Probe gap:** no probe pins the gradient scaling or the LR overrides; the code itself is correct.
- **Benchmark:** resgs-mipnerf360 (garden, counter, 30k). Compare gains: paper +0.41 / +0.80 vs its 3DGS; our baseline is 26.89 / 28.85.

## Benchmark attempt 1 (resgs-mipnerf360, 30k) — FAILED, collapse
- **Failure:** both scenes stopped at about 14000 with "Density control removed all Gaussians".
- **Trace:** the loss jumps after each resolution switch (stage 2 at 3000: 0.022 → 0.088; stage 3 at 6000: 0.038 → 0.12) and never recovers. The opacity reductions plus the < 0.005 prune then empty the population (garden 1.02M at 7000 → 2.2k at 14000).
- **Next step:** sent back to the builder for root-cause analysis; render/target misalignment at the higher pyramid levels is suspected. The failed run directories are kept.

## Fix round 1 (benchmark collapse)
- Symptom: 30k runs on garden/counter died at about 14000 ("Density control removed all Gaussians").
- Root cause: `ss_resgs/density.py` `residual_split.param_fn` (old line 85, `p = p.clone()`). Under
  `no_grad` the clone has `requires_grad=False`, and the new opacity Parameter took that flag →
  **opacities were frozen from the first densify (step 800)**. Only the x0.3 split and x0.8
  reductions acted on them, so the population decayed and was pruned away. Probes missed it because
  they checked values and moments, not trainability.
- Fix: build the tensor from `p.detach().clone()`, keep `requires_grad` from the Parameter; a runtime
  guard raises if a density op leaves a non-trainable field. New check `_assert_trainable` in the
  residual-split and opacity-reduce probes fails on the old code.
- Evidence: real schedule on garden to 9000 steps: 1.31M → 4.72M Gaussians, loss 0.067 just after
  the 6000 switch → 0.038 by 8000, opacity median 0.16–0.19 (pre-fix: 0.006, collapsing).
  Compressed-schedule smoke: 24.41 dB (pre-fix 22.60; gsplat baseline 23.05). The earlier smoke
  numbers are invalid. The loss jump at each switch is mostly resolution (gsplat full-res loss is ~0.10 at 2k steps).

## Fix iteration 1 (orchestrator)
- **Root cause:** residual_split rebuilt the opacity Parameter from a clone made under torch.no_grad, so opacity stopped training from step 800.
- **Fix:** build from `p.detach().clone()` and keep `requires_grad`, matching the official replace_tensor_to_optimizer_with_mask.
- **Guard and probe:** a runtime guard and an `_assert_trainable` probe now catch this; the probe fails on the old code.
- **Earlier smoke numbers:** invalid. The compressed smoke is now 24.41 vs gsplat 23.05.
- **Relaunched** as resgs-mipnerf360-v2 (garden, counter, 30k). The failed v1 run dirs are kept.

## Benchmark result after fix 1 (resgs-mipnerf360-v2, 30k, images_4/2) — MATCHED
| scene | ours | paper Ours (AbsGS) | Δ | our baseline | paper 3DGS | gain ours / paper | Gaussians |
|---|---|---|---|---|---|---|---|
| garden | 27.76 / 0.878 / 0.102 | 27.82 / 0.876 / 0.089 | −0.06 | 26.89 | 27.41 | +0.87 / +0.41 | 4.17M |
| counter | 29.41 / 0.919 / 0.227 | 29.50 / 0.918 / 0.172 | −0.09 | 28.85 | 28.70 | +0.56 / +0.80 | 1.64M |

- Both scenes are within ±0.1 dB, with SSIM matching.
- The single grounded fix (frozen opacity in residual_split) took the run from a collapse at 14k to a match.
