# Spec-Gaussian: build report (plan id 90)
**Status:** built, verified and smoke-tested; benchmark pending.

## What was ported
- **Code path:** the official real-scene path, ingra14m/Specular-Gaussians @0b40ce3: `train.py --is_real --asg_degree 12 [--is_indoor]`, τ_g = 0.0005. The README says `run_wo_anchor.sh` produced the paper's "Ours" rows.
- **Methods:** `spec-gaussian` (outdoor), `spec-gaussian-indoor`, and `-c2f` variants of both with the paper-text resolution schedule.

| contribution | implementation | probes |
|---|---|---|
| ASG appearance field | `spec_feat` [12] per Gaussian. The bank holds Θ = Linear(12→32) → 8 ASGs on fixed anisotropic axes and Ψ = MLP(κ, d, PE₂(d)) → c_s. Colour = clamp_min(SH+0.5, 0) + c_s, with c_s from step 3000. The decoder is packed into `spec_blob` for ss-eval. | asg-basis, decoder-architecture, colour-composition, decoder-blob |
| coarse-to-fine, part 1 | gsplat absgrad (same statistic as the official fabsf kernel), grow_grad2d 0.0005 | absgrad-densify |
| coarse-to-fine, part 2 (resolution 1/4→1 over 5k) | `-c2f` variants only | c2f-resolution |
| official schedules | decoder lr decays linearly 0.0025→0.000125; spatial_lr_scale 5 | official-schedules |

No custom CUDA.

## Benchmark
- **Scenes:** mipnerf360-pilr garden (spec-gaussian) and counter (spec-gaussian-indoor), 30k steps.
- **Baseline:** baseline3dgs-mipnerf360-pilr-30000: garden 27.35, counter 29.03.
- **Paper targets:** garden 28.07, counter 30.12.
- **Paper's own 3D-GS row:** 27.73 / 29.12, a stronger 3DGS than the 3DGS paper's 27.41 / 28.70. Spec-Gaussian's gains over that row are +0.34 / +1.00 dB.
- **Primary criterion:** absolute ±0.5 dB, with the gain over our baseline compared alongside.

## Smoke test (garden pilr, 2000 steps)
| run | PSNR | SSIM | LPIPS | #G |
|---|---|---|---|---|
| spec-gaussian (ss-eval test) | 23.55 | 0.708 | 0.352 | 1.72M |
| gsplat baseline | 23.19 | 0.657 | 0.409 | 0.81M |
| diagnostic, c_s from step 500 | 23.73 | 0.720 | 0.347 | 1.71M |

A bug was fixed during testing: ss-eval had switched on the untrained decoder before step 3000. The blob now records the step it was baked at.

## Fidelity gaps
1. **Official code vs paper text** (code followed):
   - 8 ASGs, not 32;
   - f ∈ R¹², not R²⁴;
   - view direction instead of reflection direction, with no normals;
   - decoder 32 wide with 2 or 3 layers, not 3×64.
   The synthetic-scene variant is not ported.
2. **Resolution coarse-to-fine** is absent from the official train.py, so it is off in the benchmarked methods and available in the -c2f variants. The supp. ablation reports w/o c2f 28.01 vs 28.18 with it.
3. **Background:** white (official: black).
4. **Screen-size prune:** the 20-px prune is not used, matching our baseline.
5. **Runtime differences:**
   - Optimizer steps before densification.
   - SH degree increments at step+1.
   - Views are sampled with replacement.
6. **Decoder init:** the last-layer bias starts at zero; the other layers use the torch default, as in the official code.

## Review-1 resolution (orchestrator)
- **Verdict:** approved, with 0 critical, 0 major and 7 minor issues.
- **Exact match:** the official SpecularNetworkReal matches the plugin exactly in float64 (outdoor and indoor). Eval with c_s on matches in-process validation exactly.
- **Code path:** confirmed. The no-anchor row gives garden 28.07; the anchor row gives garden 27.97, so it is not the source.
- **Fixed:** paper_results.json now has a mipnerf360-pilr alias. The scenes are launched as two experiments, specgs-garden and specgs-counter.
- **Smoke caveat:** the 2k smoke (+0.36 dB) tests only densification and the LR changes. c_s is off before step 3000.
- **Resolution schedule:** the supp ablation (w/o c2f 28.01 vs Full 28.18) suggests the paper's Full row used the resolution schedule, which train.py lacks. If 30k falls short, the -c2f variants are the labelled ablation.
- **Gaussian count:** watch it. It is 2.1× baseline at 2k; the paper averages 3.10M.
- **Minor, not fixed:**
  - The official 20-px screen prune and `>=` threshold are not used.
  - The c2f camera ignores the half-pixel principal-point offset.
