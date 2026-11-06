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
