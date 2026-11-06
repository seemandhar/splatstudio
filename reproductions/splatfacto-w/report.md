# Splatfacto-W (arXiv 2407.12306), plan #86 — build report

**Status: built and smoke-tested; benchmark not yet run, so paper fidelity is unverified.**

- Plugin `plugin/` (`ss_splatfactow`), with methods `splatfacto-w`, `splatfacto-w-a` and `splatfacto-w-t`.
- Written fresh from the paper and the official Nerfstudio code (KevinXu02/splatfacto-w, Apache-2.0). Where the two disagree the code was followed; `paper.md` §7 lists each case:
  - colour is SH + 0.5 clamped, not a sigmoid;
  - the background is raw SH;
  - the alpha loss uses a mean, not a sum;
  - 55k steps (code) vs 65k (paper text).
- No custom CUDA.

## Contributions → components (each with a probe)
- **Appearance:** per-Gaussian 72-d features, per-image 48-d embeddings and the colour network in the bank; colour = max(SH + 0.5, 0). Probe: appearance-colour.
- **Robust mask:** the code's mask (0–40 %, top 40 % never masked, 5×5 box filter, from step 6000) in the photometric loss. Probe: robust-mask.
- **Background:** degree-4 SH from the embedding, rgb = clamp(C + (1−α)·bg). Probe: background-model.
- **Alpha loss:** 0.15·mean α where the 3×3 box of |gt−bg| < 0.003 exceeds 0.6. Probe: alpha-loss.
- **Splatfacto base:** densification, initialisation, 12 LR schedules, 1/4 → 1/2 → full resolution schedule, Nerfstudio normalisation. Probes: splatfacto-schedule, splatfacto-refine, ns-normalisation.
- **Protocol:** following the official code, test images are trained on their left half and scored on the right half. Probes: masked-view-padding, nerfw-right-half. Data layout `phototourism-sfw` comes from `tools/prepare_sfw.py`.

## Smoke tests (Brandenburg Gate, phototourism-sfw, 2000 steps, still at 1/4 resolution)
Compare like for like: full-image PSNR against gsplat. The right-half number is the paper protocol and is listed separately.

| run | full-image PSNR / SSIM / LPIPS-VGG | right-half PSNR / SSIM / LPIPS-Alex | Gaussians | time |
|---|---|---|---|---|
| splatfacto-w after review-1 fixes (`evidence/smoke-2`) | **21.09** / 0.835 / 0.395 | 21.15 / 0.833 / 0.288 | 90k | 134 s |
| splatfacto-w before fixes (`evidence/smoke-1`) | 21.63 / 0.841 / 0.396 | 21.47 / 0.839 / 0.288 | 85k | 128 s |
| gsplat, same data and args | **19.79** / 0.863 / 0.316 | – | 213k | 177 s |

- The fixed method is +1.3 dB over gsplat on full-image PSNR. The −0.5 dB against smoke-1 is within run-to-run noise at 2k steps, and densification has changed.
- **Compressed-schedule debug run** (`evidence/smoke-compressed-2`, `tools/smoke_compressed.py`):
  - full-image PSNR 20.99;
  - duplicates rose from 15044 to 18463 with the official shrink-then-duplicate order;
  - every later phase ran without errors.
- **Alpha loss:**
  - It is now logged every step as `sfw-alpha-selected-frac` in training.jsonl. Cumulative totals are in the checkpoint at `density_state.counters.alpha_*`.
  - smoke-2: active on 3 of 2000 steps (49 px). Compressed run: active on 13 steps (146 px).
  - Low activity this early is expected (the reviewer confirmed this). If it stays near 0 through 55k steps, report the C3 alpha loss as inactive.

## Fidelity gaps
1. Views are sampled with replacement; Nerfstudio uses a per-epoch shuffle.
2. Crop views are padded to 2·(W//2), and the alpha loss sees only their left half.
3. Crop views are detected with a principal-point heuristic (cx > 0.75·W).
4. 55k steps (code) vs 65k (paper text).
5. LPIPS-Alex comes from the `lpips` package.
6. No shared Photo Tourism baseline; compare with the paper's 3DGS row (19.99 / 17.57).
7. SH is evaluated in the Nerfstudio frame.
8. On crop views, densification statistics use the full image size (h, 2w), as the code does. Gaussians that appear only in the right half are visible in the code (with zero gradient) and raise vis_counts there, but are not counted in the port. 10 of 773 and 21 of 851 training views are affected.
9. The diagnostic `sfw-alpha-selected-frac` adds a gradient-free value (≤ 1) to the logged total `loss`.
10. Split/duplicate order now follows the code: split scales are divided by 1.6 in place before duplicates are chosen (review-1 major).

## Benchmark request
- splatfacto-w on phototourism-sfw, brandenburg-gate (26.87) and sacre-coeur (22.53), 55k steps.
- Score with `python -m ss_splatfactow.evaluate --run RUN`; the headline is `metrics.nerfw`.

## Review-1 resolution (orchestrator)
- 1 major and 4 minor issues, all fixed by the builder:
  - The split/duplicate order now follows the official code; the new probe case fails on the old order.
  - Crop-view statistics now use the full width.
  - Alpha-selection counters are logged.
- Benchmark: sfw-pt (splatfacto-w, phototourism-sfw, brandenburg-gate + sacre-coeur, 55k steps). Score with ss_splatfactow.evaluate and report the alpha_* counters.

## Benchmark result (sfw-pt, 55k, phototourism-sfw) — MATCHED
NeRF-W protocol, scored on the right half of each test image with its jointly trained embedding:

| scene | ours PSNR / SSIM / LPIPS-Alex | paper | Δ | full image |
|---|---|---|---|---|
| brandenburg-gate | 26.49 / 0.931 / 0.128 | 26.87 / 0.932 / 0.124 | −0.38 | 27.49 |
| sacre-coeur | 22.31 / 0.878 / 0.154 | 22.53 / 0.876 / 0.158 | −0.22 | 23.71 |

- Both scenes are within ±0.5 dB, and SSIM and LPIPS are close.
- The paper's 3DGS row is 19.99 / 17.57, so the method's +6.5 / +4.7 dB gain is reproduced.
- **Alpha loss (C3) was active:**
  - brandenburg-gate: on 35,920 of 55,000 steps (130M selected pixels);
  - sacre-coeur: on 26,813 steps (75M pixels);
  - it was active up to the last step in both.
- **Final Gaussians:** 203k and 241k.
- **Evaluation:** bench.py's generic ss-eval step fails for this method (the renderer needs the bank), so both jobs show "failed" in bench status. The training runs completed, and the headline numbers come from ss_splatfactow.evaluate (test_sfw.json).
