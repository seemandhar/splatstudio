# MaskGaussian (plan #95) — build report

Status: **built, awaiting benchmark** (garden + counter, mipnerf360-1600, 30k steps).

- Official code: kaikai23/MaskGaussian @ 2f581d0 (vanilla Inria 3DGS host; per-scene tables =
  Ours-α, λ_m = 0.1 on steps 19000–20000) + kaikai23/mask-diff-gaussian-rasterization @ 540dc11,
  vendored in `plugin/ss_maskgaussian/csrc/maskrast` (Gaussian-Splatting licence), JIT-built with
  torch.utils.cpp_extension.
- Protocol: run_all.py uses `images/` with -r -1, i.e. width 1600 → `mipnerf360-1600`;
  baseline `baseline3dgs-mipnerf360-1600-30000`.

## Contributions → components
| contribution | implementation | probe |
|---|---|---|
| Masked rasterization (Eq. 3–4, Eq. 7) | vendored CUDA forward/backward, `raster._MaskedRasterize` returns dL/dM | `maskgs-raster-exact`, `maskgs-unsampled-gradient` |
| Probabilistic masks (hard Gumbel-Softmax, ST) | renderer `sample_mask` on `mask_scores` (+ offset [10,1] = official init), lr 0.01 | `maskgs-gumbel` |
| Squared mask loss + window | `losses.create_mask_loss` | `maskgs-mask-loss` |
| Mask-sampling pruning | `density.MaskDensity`: never-sampled-in-10 at each densify step, every 1000 from 15000, one final sample at 30000 | `maskgs-prune-schedule` |

## Probe evidence (evidence/verify.json, all passed)
- Kernel vs float64 dense compositor: image max err 5.6e-7; `exact` build dL/dM matches autograd
  of Eq. 3–4 (rel 7e-7); released (`official`) build matches an independent replica of its own
  recursion (rel 6e-7).
- **Finding:** the released backward.cu uses the *current* Gaussian's mask in the `accum_rec`
  update (it should use the mask of the Gaussian behind). So the official kernel's dL/dM differs
  from the exact gradient by up to 32 % of max|dL/dM| when masks are mixed (13/40 masked), and is
  exact when all M = 1. The default build keeps the released kernel because that is what produced
  the paper numbers. `maskgaussian-exactgrad` is an optional variant with the exact gradient.
- Unsampled gradients: all 32/32 visible unsampled Gaussians get nonzero mask-score gradients and
  leave the image unchanged (max diff 0 vs. rendering with them removed). The same check on the
  naive mask×opacity renderer finds 0/32 and fails, as it should.

## Smoke tests (garden 1600 px, 2000 steps, shared GPU)
| run | PSNR | SSIM | LPIPS | #G | s |
|---|---|---|---|---|---|
| gsplat baseline | 23.03 | 0.626 | 0.439 | 779k | 313 |
| maskgaussian (λ window not reached) | 23.14 | 0.632 | 0.431 | 785k | 486 |
| compressed α schedule (not paper) | 23.53 | 0.638 | 0.429 | 461k | 480 |
The compressed run uses densify until 1500, λ = 0.1 on steps 1200–1300, and a final sample at 2000.
The mask loss was active only inside its window, mask_prune and the final-sample keep fired, and the state stayed aligned. The window is only 100 steps, so the reduction is small: Adam moves the scores by about lr per step, and the paper's 1000-step window is needed for the roughly 60 % cut.

## Paper targets
garden 27.34 dB / 2.13M (paper 3DGS 27.38 / 5.81M); counter 29.01 / 0.328M (3DGS 29.09 / 1.16M).
Compare our Δ vs our gsplat 1600 baseline (26.93 / 4.55M; 29.04 / 1.19M) and the count ratio.

## Fidelity gaps
Optimizer step before density op (runtime order); views sampled with replacement; final-step
optimizer step not skipped; different RNG (final model is a random sample, as in the official code); runtime
SSIM/LPIPS; schedules are absolute, so `--steps` must be 30000.

## Review-1 resolution (orchestrator)
- **Verdict:** approved, with 0 critical, 0 major and 5 minor issues.
- **Confirmed against the sources:**
  - Setting: Ours-α (Table 1 caption; the per-scene averages equal the α row).
  - Resolution: -r -1, width 1600, so the data is mipnerf360-1600 and the baseline baseline3dgs-mipnerf360-1600-30000.
  - Mask field: τ = 1, hard ST Gumbel; the mask is used in α-blending only. L_m is computed over all N Gaussians.
  - The pruning schedule is exact.
- **Official kernel bug, independently confirmed:**
  - accum_rec uses the current Gaussian's mask instead of the mask of the Gaussian behind it.
  - Against float64 autograd on a 3-Gaussian stack:
    - dL/dM relative error 0.09–0.52 on mixed masks;
    - dL/dopacity relative error 0.16–1.05.
  - So it also biases the opacity and geometry gradients of sampled Gaussians, not only dL/dM.
  - The default keeps the released kernel because it produced the paper numbers. `maskgaussian-exactgrad` is benchmarked alongside as a diagnostic.
- **Smoke caveat:** the compressed smoke's 461k count comes mostly from densification ending at 1500; masking removed only about 11k.
- **Confounds vs our gsplat baseline:**
  - re-implemented Inria density control vs gsplat DefaultStrategy;
  - black vs white background.
  A mask-off control was not run, so the count ratio includes the density-implementation difference.
- **Steps:** total_steps is not tied to --steps, so the benchmark must use exactly 30000.
- **Benchmark:** maskgaussian-1600 (maskgaussian and maskgaussian-exactgrad, garden and counter, 30k). Targets: garden 27.34 / 2.13M, counter 29.01 / 0.328M.

## Benchmark result (maskgaussian-1600, 30k, mipnerf360-1600) — MATCHED (PSNR and count)
| scene | maskgaussian | Gaussians | paper | Δ PSNR | count ratio vs our baseline / paper | exactgrad |
|---|---|---|---|---|---|---|
| garden | 27.27 / 0.856 / 0.146 | 2.20M | 27.34 / 2.13M | −0.07 | 0.48 / 0.37 | 27.29, 2.20M |
| counter | 29.03 / 0.914 / 0.255 | 0.339M | 29.01 / 0.328M | +0.02 | 0.29 / 0.28 | 29.00, 0.336M |

- Both PSNR and the absolute Gaussian counts match the paper (within 3% and 3%).
- **Ratio caveat:** our baseline has fewer Gaussians than the paper's 3DGS on garden (4.55M vs 5.81M), so the garden ratio looks weaker. The absolute counts agree.
- **Exactgrad:** fixing the official backward-kernel bug changes PSNR by ±0.03 and the count by <1%. The bug has no practical effect at this setting.
