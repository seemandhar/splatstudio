# Revising Densification in Gaussian Splatting (arXiv 2404.06109): build report

**Status:** ready for benchmark. The plugin `revising-densification` (`plugin/`, module `ss_revising_densification`) is built and passes every probe. It has had one 2000-step smoke run and no benchmark yet.

**Official code:** none was released. The method is re-derived from the arXiv LaTeX source of the paper and its supplementary material.

## Mapping to the paper
The renderer and the L1 + 0.2 D-SSIM loss are stock. The paper's changes live in one density component (`revdens-density`) and one extra loss (`revdens-aux`).

- **C1, error-based densification** (Sec. 3.2):
  - The per-pixel error is 𝓔(u) = 1 − SSIM(u), using an 11×11 window with same padding, averaged over RGB.
  - A CUDA kernel (`csrc/err_stats.cu`, adapted from the improving-adc kernel) attributes it to each Gaussian as E^π_k = Σ_u 𝓔(u)·α·T over gsplat's tile lists.
  - The score E_k is the maximum over the views seen since the last run.
  - Gaussians with E_k > 0.1 are candidates.
- **C2, opacity correction on clone** (Sec. 3.3): the source and the copy both get 1−√(1−α). Split children keep their α, as the paper says.
- **C3, growth control** (Sec. 3.4):
  - Each run adds at most ⌊0.05·N⌋ Gaussians, highest score first.
  - The global cap is the final count of our `baseline3dgs-*-30000` run for that scene, looked up from the scene's SfM point count.
- **C4, opacity regularisation** (Sec. 3.5):
  - Every opacity drops by 0.001 after each densification run.
  - There is no opacity reset.
  - A 0.1·mean(1−alpha) transmittance loss is added.
- **C5, longer densification** (Sec. 4.2): runs at steps 600 to 26900, every 100 steps (264 runs).

## Verification
- **Static checks:** `ss-inspect` reports no issues, 6 unit tests pass, and ruff is clean.
- **Probes:** `ss-verify` reports `declared_checks_passed` with 12 probes.
- **Alpha-compositing oracle:** the CUDA E_k matches a brute-force per-pixel loop to 8e-6, and the paper's own auxiliary-loss gradient route to 2e-4.
- **Smoke run** (garden, 2000 steps):

  | Run | PSNR / SSIM / LPIPS | Gaussians | Time |
  |---|---|---|---|
  | revising-densification | 21.57 / 0.593 / 0.445 | 273k | 209 s |
  | gsplat | 23.06 / 0.643 / 0.418 | 842k | 164 s |

  The −1.49 dB gap at 2k steps is most likely the 5 %-per-run growth limit. This is not confirmed. By calculation, the cap is reached around step 7.5k on garden.

## Benchmark request
Garden and room are compared per scene. The T&T truck and train runs are compared as the complete T&T average.

| Target | Paper, ours | Paper, 3DGS | Paper gain | Our gsplat baseline |
|---|---|---|---|---|
| garden | 27.622 | 27.332 | +0.29 | 26.89 |
| room | 31.902 | 31.395 | +0.51 | 31.52 |
| T&T average | 23.93 | 23.44 | +0.49 | 23.36 |

## Fidelity gaps
1. **Error formula:** the paper only says "SSIM". We use 1−SSIM; (1−SSIM)/2 is another reading. Because the budget binds (median E_k ≈ 12.5), the threshold matters little.
2. **Order within a run:** prune, then grow, then the decrement. Each clone or split counts +1 against the budget. This is our reading of the paper.
3. **Cap:** the cap is our gsplat baseline's count, not the authors' baseline.
4. **Pruning:** the 3DGS world-size prune is kept. The screen-size prune is not used, since it is inert in the official code.
5. **Runtime differences:**
   - white background;
   - Adam steps before density;
   - sampling with replacement;
   - gsplat clamps alpha at 0.999 versus Inria's 0.99;
   - valid-convolution SSIM in the loss.
6. **Single run:** we train one run per scene; the paper averages 5 runs (standard deviation 0.03–0.16 dB).
7. **Not targeted:** the Mip-Splatting variant.

## Review-1 resolution (orchestrator)
**Review outcome:** approved, with 0 critical and 0 major issues. The benchmark is launched on garden and room (`mipnerf360`) and on truck and train (`tandt`).

**Notes recorded from the review:**
- **Early growth deficit:** the 2k smoke deficit is what the paper's 5 %-per-run rule predicts (138.8k × 1.05^15 ≈ 289k, against 273k observed). The cap should be reached around step 8–9k.
- **Check after the 30k runs:** confirm in the training log that the Gaussian count reaches the cap and holds it until 27k.
- **Cap source:** the cap is the final count of a single run of our baseline. The paper uses the median of 5 runs.
- **Error scale:** the paper does not specify the scale of the per-pixel error ("e.g. SSIM"). Because the per-run budget binds and ranking does not depend on scale, this has no practical effect.
