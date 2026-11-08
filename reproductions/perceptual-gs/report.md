# Perceptual-GS — reproduction report

Paper: Zhou & Ni, *Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian
Splatting*, ICML 2025, arXiv 2506.12400. Plan id 119.

Plugin: `perceptual-gs` (`plugin/`, module `ss_perceptual_gs`). It is a port of the official code
(https://github.com/eezkni/Perceptual-GS, Inria 3DGS research licence). The CUDA statistics and
depth kernels are vendored from our Mini-Splatting port, which adapts the `_ms` rasterizer that
Perceptual-GS also ships.

**Outcome: matched.** Both reported scenes are within ±0.5 dB of the paper's per-scene PSNR
(Appendix E, Table 12).

## Results (30k steps, `mipnerf360-pilr`, test split = llffhold8)
Run: `runs/perceptual-gs-mipnerf360-pilr-30000` (summary.json). Baseline:
`baseline3dgs-mipnerf360-pilr-30000` (gsplat).

| scene | ours PSNR | paper PSNR | Δ | ours SSIM / LPIPS | paper SSIM / LPIPS | #G ours / paper | train time |
|---|---|---|---|---|---|---|---|
| bicycle (outdoor, -r 4) | 25.79 | 25.956 | −0.17 | 0.802 / 0.179 | 0.805 / 0.165 | 5.16M / 3.89M | 94 min |
| counter (indoor, -r 2) | 29.21 | 29.452 | −0.24 | 0.921 / 0.216 | 0.922 / 0.157 | 1.65M / 1.49M | 84 min |

Each method's gain over its own 3DGS baseline:

| scene | ours − our gsplat | paper − paper 3DGS* |
|---|---|---|
| bicycle | 25.79 − 25.45 = +0.34 | 25.956 − 25.617 = +0.34 |
| counter | 29.21 − 29.03 = +0.18 | 29.452 − 29.144 = +0.31 |

LPIPS has a systematic offset because LPIPS-VGG comes from the `lpips` package and SSIM uses valid
convolution (runtime-wide).

## Scene-adaptive modes taken (decision at step 600, from the run logs)
| scene | γ (sparse test, τ 0.55) | β (scene sensitivity, τ 0.85) | mode |
|---|---|---|---|
| bicycle | 0.5513 → sparse | 0.9233 → split_only | Split-only HD densification. Depth reinit at 5000 (2.66M → 3.46M) and 10000 (5.44M → 3.46M). No opacity reset. Self-reinit at 15000 (5.16M). |
| counter | 0.4480 → not sparse | 0.7177 → clone mode | HD clones small and splits large Gaussians. MD splits. 3DGS opacity reset every 3000. |

Bicycle's γ is only just above the 0.55 threshold, so its mode is sensitive to small numerical
differences. Bicycle's final count is 5.16M against the paper's 3.89M, which is plausibly a
consequence of this mode.

## Smoke and verification
- `ss-verify`: declared_checks_passed (9 probes).
- 7 unit tests pass and ruff is clean.
- Counter 2k smoke: 24.33 dB against gsplat's 24.90. This is a timing artifact: step 2000 is a
  high-sensitivity densify step whose opacity decline briefly lowers opacity (see review). The
  forced-sparse smoke exercised depth reinit on a real scene.
- Review: `reviews/review-1.md`, approved with 0 critical, 0 major and 5 minor items.

## Deviations from the paper / official code
1. **Code over paper where they disagree:**
   - split-only when β > 0.85 (the paper text says clone-only when β < τ);
   - the medium band is the open interval (0.3, 0.9);
   - BCE is computed on 1 − R[1 − σ(ε)] over black, against the bicubic-resized (non-binary) map.
2. **Background:** the RGB branch renders over white (runtime); the sensitivity branch uses black,
   as in the official code.
3. **Optimizer step order:** the runtime steps the optimizer before density control. On ordinary
   densify steps the official code skips Adam; HD/MD steps match exactly.
4. **Kernel numerics:** the statistics and depth kernels use gsplat's 0.999 alpha cap and its stop
   test (official: 0.99, `test_T < 1e-4` with continue).
5. **View sampling:** views are sampled with replacement (official: shuffled stack). Depth-reinit
   colours come from a cache of rendered training targets, which raises an error if a view was
   never seen.
6. **Small timing and numeric differences:**
   - the γ/β decision is evaluated one Adam step later than official;
   - the SH degree increments one step later;
   - the opacity-decline output is clamped to [1e-7, 1−1e-7] before the logit.
7. **Sensitivity maps:** computed offline (`ss-pgs-prep`) from the full-resolution images, using
   the installed Pillow 12.3 `Image.point` and its resize. They are stored in
   `/bigdata/sejain/perceptual-gs-sens/` and matched to frames by viewmat.
8. **Scope:** only 2 of the 13 reported scenes were benchmarked (batch rule). No dataset averages
   are claimed.
9. **Hyper-parameters:** none were tuned; all values are the paper / official defaults.
