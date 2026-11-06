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

## Result: counter (spec-gaussian-indoor, 30k, mipnerf360-pilr) — within tolerance
- **Result:** 29.69 / 0.921 / 0.226, with 1.19M Gaussians.
- **Paper:** 30.12 (Δ −0.43).
- **Gain:** +0.66 over our pilr baseline (29.03), vs the paper's +1.00 over its own 3D-GS row.
- **Still pending:** garden.

## Result: garden (spec-gaussian, 30k, mipnerf360-pilr) — outside tolerance
- **Result:** 27.26 / 0.873 / 0.117, vs paper 28.07 (Δ **−0.81**).
- **Gain:** −0.09 vs our baseline (27.35), where the paper reports +0.34 vs its 3D-GS row.

## Fix iteration 1: paper-text coarse-to-fine resolution schedule
- **Why:** the paper describes resolution coarse-to-fine (1/4 → full over 5k steps) as part of the method. Its supp. ablation lists "w/o c2f" at 28.01 against Full at 28.18, so the paper's headline row used it, but the released train.py lacks it (review-1 minor 4).
- **What:** spec-gaussian-c2f (garden) and spec-gaussian-indoor-c2f (counter) are run at 30k as the paper-text configuration (exps specgs-c2f-garden / specgs-c2f-counter).
- **Kept:** the code-faithful runs.
- **c2f counter (spec-gaussian-indoor-c2f):** 29.90 / 0.922 / 0.227 vs paper 30.12 (Δ −0.22). The code-path run gave 29.69, so c2f adds +0.21.
- **c2f garden (spec-gaussian-c2f):** 27.56 / 0.874 / 0.118, vs paper 28.07 (Δ −0.51). The code-path run gave 27.26, so c2f adds +0.30.

## Final verdict: PARTIAL (counter within ±0.5 dB; garden 0.01 dB outside)
| scene | code path (train.py) | paper-text c2f (fix 1) | paper | Δ (c2f) | our baseline | gain c2f / paper |
|---|---|---|---|---|---|---|
| garden | 27.26 | **27.56** | 28.07 | −0.51 | 27.35 | +0.21 / +0.34 |
| counter | 29.69 | **29.90** | 30.12 | −0.22 | 29.03 | +0.87 / +1.00 |

- The paper's c2f schedule (fix iteration 1, grounded in the paper text and its ablation) improves both scenes by +0.2–0.3 dB.
- **Garden** is −0.51, at the tolerance edge. Part of the residual is the paper's stronger 3D-GS (27.73 vs our 27.35); the gain is +0.21 vs the paper's +0.34.
- **No further iteration:** the batch allows 1–2 grounded fixes and no further grounded change is identified.
