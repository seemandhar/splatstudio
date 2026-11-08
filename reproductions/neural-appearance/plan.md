# Plan — neural-appearance (plan #124), module `ss_neural_appearance`

## Base-pipeline decision
The paper's per-scene "Neural" and "SH" rows come from the **same** pipeline: Faster-GS
rasterizer + Faster-GS FusedAdam + Morton, **MCMC** densification (scene caps), **PPISP** with
controller distillation, black background, 30k (+5k PPISP-controller) iterations.
The reproduction `reproductions/faster-gs` (matched) is the Faster-GS *main* configuration (3DGS ADC,
opacity lr 0.025, no MCMC, no PPISP). So we build on its **kernels** (`ss_faster_gs.raster`,
`ss_faster_gs.optim.FusedAdam`, `ss_faster_gs.morton`; cited) and add the MCMC + PPISP + appearance
pieces from the official FasterGSVDA code. Both appearance models are shipped so the bench measures
the paper's paired gain (Neural − SH on the same base):
- `neural-appearance-<scene>`: C1 neural model.
- `neural-appearance-sh-<scene>`: the paper's SH row (degree-3 SH via the vendored rasterizer).
Scene suffix = MCMC cap (paper_configs): garden 5.0M, room 1.5M (also bicycle 6M, truck 2.5M
registered for optional runs). ss-train has no config overrides, hence per-scene entry points (as BAGS).

## Components
| component | kind | factory | requires / provides | config |
|---|---|---|---|---|
| na-renderer | renderer | `ss_neural_appearance.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb, appearance.neural-ppisp | appearance (neural/sh), F=8, k=1, degrees [0..3], 16x2 MLP, near 0.2, black bg |
| na-mcmc | density | `ss_neural_appearance.density:create_density` | appearance.neural-ppisp / – | cap, MCMC 600..24900/100, noise 5e5, Morton 5000..25000, lrs + schedules, PPISP, train_steps 30000, distill_steps 5000 |
| photometric | loss | runtime | 0.8 L1 + 0.2 D-SSIM (unchanged) |
| na-regularizers | loss | `ss_neural_appearance.losses:create_regularizers` | render.rgb | 0.01 opacity, 0.01 scale, PPISP reg 1.0; off after train_steps |

## State
| field | shape | lr | init | notes |
|---|---|---|---|---|
| sh0 | runtime | Neural: 0.00025/Y0 cosine 7k→37k; SH: 0.0025 const | (rgb−0.5)/Y0 | base colour c0 = Y0·sh0 (Adam scale invariance ⇒ same colour-space step as official) |
| shN | runtime | SH: 0.000125 const; Neural: unused (no grad) | 0 | |
| na_features | (8,) | 0.008 cosine 7k→37k | 0 | Neural only (declared for both, unused for SH) |
| na_blob | (4,) | 0 | 0 | packs MLP + PPISP state at the last step so plain `ss-eval` restores them |
Component-owned (not per-Gaussian): MLP weights (32×16, 16×16, 16×3; Adam β2 0.99, warm-up schedule),
PPISP module (vendored nv-tlabs/ppisp, CUDA JIT) — held by `bank.Bank`, created in
`set_training_cameras`, stepped in `after_backward`, exported to `density.state` and `na_blob`.

## Step mapping (official callback at 0-based iteration i ↔ runtime after_backward(step=i))
after_backward(s): bank step (MLP Adam + PPISP Adam/scheduler) → MCMC noise (lr of step s) →
MCMC if s∈[600,24900], s%100==0 → Morton if s%5000==0, s≤25000 → set lrs for step s+1 →
s==30000: prune opacity<1/255 + degenerate, Morton → s==30000+5000: write blob.
Steps 30001..35000 = PPISP controller distillation: renderer rasterizes under no_grad (scene frozen,
fields get no gradients so Adam skips them), regularizers off, no noise. **ss-train --steps 35000.**
The runtime's means-lr schedule (over --steps) is overwritten every step with the official
LRDecayPolicy over 30000.

## Probes
| probe | measures / fails when |
|---|---|
| na-neural-decoder | MLP input layout, freq-encoding, SH basis, tanh, zero output init, degree gating vs independent formula; fails if residual not gated or encoding differs |
| na-direction-gradient | gradient of colour wrt means is nonzero for Neural (view direction path) |
| na-render-composition | rendered image with sh0 + residual/Y0 equals rasterizing precomputed relu(0.5+c0+A) (via a 1-Gaussian analytic pixel) |
| na-mcmc | relocation formula vs independent binomial evaluation; dead → copied, moments reset, +5% growth capped |
| na-schedules | cosine (7k/37k), MLP warm-up, means lr, noise op_sigmoid values |
| na-ppisp | vendored CUDA PPISP == ppisp torch reference; controller used for novel views; blob round trip restores identical renders |

## Fidelity gaps (expected)
1. MLP in fp32 PyTorch, evaluated for all Gaussians (unfused reference path) vs fused fp16 tcnn — paper shows w/o fusion = same quality.
2. tcnn RNG for hidden init not bit-identical (same distribution).
3. Runtime SSIM (valid conv) vs fused-ssim; sampling with replacement; white-bg protocol flag irrelevant (black bg render, targets have no alpha).
4. MCMC noise RNG (torch randn vs Philox) — same distribution.
5. Single seed vs the paper's 5-run mean (std ≈0.02–0.06 dB).
6. Training images on the runtime's loader (images_4/images_2) vs NeRFICG scale factor.
