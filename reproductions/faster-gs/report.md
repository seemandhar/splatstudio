# Faster-GS (plan #100) — build report (benchmark pending)
- **Official code:** the CUDA rasterizer and Adam from FasterGSCudaBackend (main @ 090e8ad, Apache-2.0), vendored unmodified and JIT-built.
- **Pinned configuration:** the paper's "Ours / Full" row.
  - 3DGS training is otherwise unchanged: densify every 100 over 600..14900, resets 3000..12000, opacity lr 0.025, black background.
  - Faster-GS adds its rasterizer, CUDA Adam (swapped in place for the runtime optimizer) and Morton ordering at steps 0/5k/10k/15k.
- **Excluded (optional rows or borrowed methods):** the fused backward+optimizer branch, skip-invisible updates, MCMC, Mip-Splatting, Speedy-Splat, AbsGS, PPISP.
- **Labelled borrowed and lossless:** the warp-level sub-tile culling in the main blend kernel, which comes from a separate HPG 2026 paper and cannot be switched off.

## Checks
- ss-verify reports declared_checks_passed for all 8 probes:
  - render and all six gradients vs a float64 dense reference: ≤1.4e-6;
  - exact tile counts;
  - densification statistics 7.5e-7;
  - CUDA Adam vs torch Adam 1.1e-5;
  - Morton ordering, densify/prune and optimizer-state alignment.
- Unit tests pass (7) and ruff is clean.

## Smoke test (garden, 2000 steps, shared A6000)
| | PSNR | SSIM | LPIPS | Gaussians | train s | peak GiB |
|---|---|---|---|---|---|---|
| faster-gs | 23.42 | 0.653 | 0.412 | 795k | 65.0 | 1.05 |
| gsplat | 22.99 | 0.643 | 0.419 | 844k | 149.8 | 1.28 |

Per-phase profile (ms; 795k Gaussians):

| path | forward | backward | update | total |
|---|---|---|---|---|
| Faster-GS | 6.7 | 13.3 | 6.0 | 36.7 |
| gsplat + torch Adam | 9.2 | 27.8 | 14.0 | 69.4 |

Faster-GS is 1.37× faster in forward, 2.1× in backward and 2.3× in the update. These timings are noisy because the GPU is shared.

## Benchmark
- **Data:** the paper reports dataset averages only, so the benchmark uses the complete T&T (truck, train) and DB (drjohnson, playroom) averages at 30k.
- **Targets:** T&T 23.75 (paper 3DGS 23.77) and DB 29.78 (29.81), a gain of about 0.
- **Our baselines:** 23.36 / 29.28.
- **Speed:** profile each final checkpoint with `python -m ss_faster_gs.profile --run RUN --views 50`.

## Fidelity gaps
1. Runtime SSIM (valid convolution) instead of fused-ssim.
2. Views sampled with replacement.
3. The means LR runs one step behind.
4. Single seed; the paper averages 5.
5. Our baseline differs: opacity lr 0.05 and a white background.
6. The kernels are from main (16×12 tiles plus warp culling), not the testbed branch; both changes are lossless.
7. Timing is on a shared A6000 with CUDA 12.1; the paper used a 4090 with CUDA 12.8.
8. LPIPS is VGG from the lpips package.

## Review-1 resolution (orchestrator)
- **Verdict:** approved, with 0 critical, 0 major and 8 minor issues.
- **Confirmed against the paper and official code:**
  - The headline row excludes the fused and skip-invisible variants (Table 3).
  - The hyper-parameters match Trainer.py/Model.py.
  - The vendored CUDA is byte-identical.
  - CUDA Adam stays aligned through append, permute, reset and remove (16-step check).
  - The T&T/DB resolutions match.
- **Gap 8 corrected:** the official LPIPS is VGG on [-1,1], the same as ours.
- **Near plane:** 0.2 (from the garden config) is an ambiguity; the official T&T loader uses 0.01.
- **Speed evidence:** the profile tool times per phase against gsplat. It is not the paper's end-to-end timing table and is noisy on shared GPUs.
- **Criterion:**
  - Primary: absolute PSNR within ±0.5 dB of the full-dataset averages (T&T 23.75, DB 29.78).
  - Secondary: our gain over our baseline (23.36 / 29.28, which sit 0.41 / 0.53 below the paper's 3DGS) vs the paper's ≈0.
- **Benchmark:** faster-gs-tandt and faster-gs-db at 30k.

## Result: Deep Blending (faster-gs-db, 30k) — within tolerance
- **Complete DB average:** **29.78** (drjohnson 29.47, playroom 30.09), SSIM 0.907, LPIPS 0.310, vs paper 29.78 (Δ 0.00).
- **Gain:** +0.50 over our baseline (29.28), vs the paper's −0.03 over its 3DGS.
  - Our gsplat baseline sits 0.53 below the paper's 3DGS. The difference is consistent with the training-recipe differences listed in gap 5 (opacity lr, black background).
- **Still pending:** T&T.

## Final verdict: MATCHED (both complete dataset averages within ±0.5 dB)
| dataset | ours | paper | Δ | our baseline | paper 3DGS | gain ours / paper |
|---|---|---|---|---|---|---|
| T&T (truck 25.39, train 21.91) | 23.65 / 0.852 / 0.204 | 23.75 | −0.10 | 23.36 | 23.77 | +0.29 / −0.02 |
| DB (drjohnson 29.47, playroom 30.09) | 29.78 / 0.907 / 0.310 | 29.78 | 0.00 | 29.28 | 29.81 | +0.50 / −0.03 |

- **Quality:** our port of the official kernels reproduces the paper's quality.
- **Gains:** our positive gains come from the training-recipe differences against our gsplat baseline (opacity lr, black background), not from Faster-GS.
- **Speed:** the per-phase speed-up from the 2k smoke (forward 1.37×, backward 2.1×, update 2.3×) was measured on a shared GPU. `ss_faster_gs.profile` was not rerun on the final checkpoints.
