# Improving ADC for 3DGS (arXiv 2503.14274): build report

**Outcome: matched on Tanks & Temples and Deep Blending** (both complete 2-scene averages).
- **T&T:** 23.93 vs the paper's 24.18 (−0.25).
- **DB:** 29.56 vs the paper's 29.63 (−0.07).
- **Gain over our gsplat baseline:** +0.57 on T&T and +0.28 on DB. The paper reports +0.44 and +0.13 over its own 3DGS.
- **Mip-NeRF 360: not evaluated.** It was paused at 2 of 9 scenes. The paper reports only the 9-scene average, and a subset mean is not comparable with it.

## Results
Setup: 30k steps with `--max-gaussians 16000000`, against our baselines `baseline3dgs-{tandt,db}-30000`.

| scene | PSNR | our 3DGS | Δ | SSIM | LPIPS | #G |
|---|---|---|---|---|---|---|
| truck | 25.71 | 25.21 | +0.51 | 0.891 | 0.144 | 2.98M |
| train | 22.15 | 21.52 | +0.63 | 0.833 | 0.211 | 2.18M |
| **T&T mean** | **23.93** (paper 24.18) | 23.36 | **+0.57** (paper +0.44) | | | |
| drjohnson | 29.28 | 28.76 | +0.52 | 0.902 | 0.315 | 2.91M |
| playroom | 29.84 | 29.80 | +0.04 | 0.908 | 0.309 | 2.00M |
| **DB mean** | **29.56** (paper 29.63) | 29.28 | **+0.28** (paper +0.13) | | | |

Mip-NeRF 360 scenes already finished: bicycle 25.12 and flowers 21.64. These are kept as data points only, not compared. They are in `runs/improving-adc-mipnerf360/`, and the run can be resumed with the same launch command.

# Build notes

**Official code:** fraunhoferhhi/Improving-ADC-3DGS @ 65901d8, a PixelGS fork. It was read but never run.

## Mapping to the paper
All changes live in one density component. The renderer and the L1 + 0.2 D-SSIM loss are stock.

- **C1, corrected extent** (Sec. 4.1, Eq. 13): e' is the mean distance from the SfM points to the mean training-camera centre. It sets the clone/split cutoff (0.01·e') and the means learning-rate scale (the official `spatial_lr_scale`). On garden, e = 4.92 and e' = 6.49, which matches the commented-out table in the paper.
- **C2, rising threshold** (Sec. 4.2, Eq. 14): the densification threshold is 1e-4·4^(k/145) at call k, over 144 calls from step 600 to 14900.
- **C3, significance pruning** (Sec. 4.3, Eq. 15/16, Alg. 2):
  - σ = Σ α·T per Gaussian since the last densification.
  - A Gaussian is pruned if o < 0.005 **and** σ is below the quantile of σ over the old, non-split Gaussians, at fraction N_prune/N_old.
  - There is no size pruning.
- **Base method, PixelGS gradient:** the gradient is weighted by pixel count and scaled by min(1, (z/(0.37·e_cam))²). The square comes from the code, not the paper.
- **Base method, revised opacity:** 1−√(1−o) is applied to the clone, to its parent (overwritten in place) and to the split children.
- **CUDA kernel** (`csrc/adc_stats.cu`, adapted from the Taming-3DGS port):
  - It is forward-only and returns pixel count and σ.
  - It replaces the official fake-colour second rasterization. A probe shows the two agree to a maximum error of 2e-6.

## Verification
- **Static checks:** `ss-inspect` reports no issues. 6 unit tests pass, and ruff is clean.
- **Probes:** `ss-verify` reports `declared_checks_passed` with 10 probes. One checks background preservation: a low-opacity, high-σ background Gaussian survives while hidden low-σ ones are pruned. Another runs an opacity-only ablation and confirms the background Gaussian is then removed.
- **Smoke run** (garden, 2000 steps):

  | Run | PSNR / SSIM / LPIPS | Gaussians |
  |---|---|---|
  | improving-adc | 23.60 / 0.703 / 0.366 | 1.87M |
  | gsplat | 23.06 / 0.643 / 0.418 | 0.84M |

## Benchmark request
- **Scenes:** all 9 Mip-NeRF 360 scenes, using the official images_4/2 `full_eval.py` setup. This is the primary result. Also T&T ×2 and Deep Blending ×2.
- **Baselines:** `baseline3dgs-*-30000`. Our averages are 27.28 (Mip-NeRF 360), 23.36 (T&T) and 29.28 (DB).
- **Paper targets:**

  | | Mip-NeRF 360 | T&T | DB |
  |---|---|---|---|
  | Paper, Improving ADC | 27.71 | 24.18 | 29.63 |
  | Paper, its own 3DGS row | 27.39 | 23.74 | 29.50 |
  | Paper gain | +0.32 | +0.44 | +0.13 |

## Fidelity gaps
1. **Background:** the runtime uses white; the official code uses black.
2. **Update order:** the runtime steps Adam before densifying; the official code densifies first.
3. **Rasterizer numerics:** gsplat clamps alpha at 0.999, Inria at 0.99. Pixel counts and σ follow gsplat.
4. **Sampling and SSIM:** views are sampled with replacement, and SSIM uses valid convolution.
5. **Quantile fraction:** clamped to at most 1, where the official code would raise.
6. **Known runtime pitfalls:** the screen-size prune and the gsplat gradient drop do not apply here, because the official code disables all size pruning.
7. **Gaussian budget:** early Gaussian growth may hit the 12M limit. This was not checked beyond 2k steps.
8. **3DGS baseline row:** the paper's 3DGS row (27.39) differs from the 3DGS paper's own number (27.21), and the paper does not say where it comes from.

*Report saved by the orchestrator; the build subagent could not write it.*

## Review-1 resolution (orchestrator)
**Review outcome:** approved, with 0 critical and 0 major issues. The benchmark was launched on all 13 scenes with `--max-gaussians 16000000`. That cap stays below the 2^24-element limit of `torch.quantile`; above it, density would crash, just as the official code does.

**Additional documented items:**
- **Early Gaussian growth:** about 1.6× our full baseline at 2k steps, not 2.2×. The 0.84M baseline figure in the smoke table came from a 2k smoke schedule.
- **Unreproducible contribution:** the paper's introduction lists a fourth contribution, "opacity-momentum-guided densification". It has no section, no ablation row and no code, so it is not reproducible.
- **Densify-step optimizer behaviour:** on every densify step the official code skips the whole optimizer step, because every `.grad` is None after the parameters are replaced. The runtime does take that step.
- **Context row:** the paper's Revising Densification row is 27.61 / 23.93 / 29.50. "Ours" exceeds it by +0.10 / +0.25 / +0.13 dB.
- **Table 2 protocol:** Table 2 (truck / garden / drjohnson) uses a different protocol, full resolution capped at 1600 px, and is not targeted.
