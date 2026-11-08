# OMG: Optimized Minimal 3D Gaussian Splatting (Lee, Ko, Park, NeurIPS 2025). Native SplatStudio reproduction

**Outcome: matched.** We benchmarked OMG-XS (τ 0.96) and evaluated the decoded model. The PSNR is
within ±0.5 dB of the paper on both reported scenes: bicycle −0.27, counter −0.46. Counter is close
to the edge of the tolerance.

A matched PSNR is not proof of fidelity; every known deviation is listed below.

- **Paper:** arXiv 2503.16924. Per-scene numbers come from the supplement, Table tab:perscene360.
- **Official code:** https://github.com/maincold2/OMG. It ships no licence file, so it was read and
  re-implemented rather than copied.
- **Base:** our matched Mini-Splatting port (`reproductions/mini-splatting`, freeze a6428eb7),
  copied into the plugin.
- **Plugin:** `plugin/`, with methods `omg` (outdoor importance metric), `omg-indoor` and `omg-m`.
- **Runs:**
  - `runs/omg-mipnerf360/omg/mipnerf360-bicycle`
  - `runs/omg-indoor-mipnerf360/omg-indoor/mipnerf360-counter`
- **Protocol:**
  - 30k steps with `--recipe 3dgs`, all SfM points, llffhold 8.
  - images_4 for bicycle, images_2 for counter.
  - Test metrics come from `ss-eval` on the decoded state. That state is encoded at step 30000 (fp16
    positions, G-PCC, Huffman-coded SVQ indices, fp16 codebooks and MLPs, LZMA), decoded from the
    bytes, and written back in place of the parameters.

## Results (test split, decoded OMG-XS)

| scene | PSNR ours / paper | Δ | SSIM ours / paper | LPIPS ours / paper | #Gauss ours / paper | size MB ours (official formula) / paper | real container MB | G-PCC xyz bytes: flag-0 stream (stored) / official flags | train time |
|---|---|---|---|---|---|---|---|---|---|
| bicycle (`omg`) | 24.68 / 24.95 | **−0.27** | 0.737 / 0.743 | 0.319 / 0.276 | 420,575 / 480,772 | 4.18 / 4.61 | 4.18 | 1,002,783 / 1,008,662 | 75 min |
| counter (`omg-indoor`) | 27.94 / 28.40 | **−0.46** | 0.888 / 0.899 | 0.291 / 0.206 | 299,466 / 310,056 | 2.94 / 2.95 | 2.94 | 677,693 / 681,673 | 99 min |

- **Units:** MB = 2^20 bytes, the same as the official "Actual storage".
- **Official formula:** `write_storage`, i.e. the official-flag G-PCC stream, Huffman streams, fp16
  codebooks and fp16 MLP weights.
- **Container:** the real LZMA file that was decoded for evaluation. It holds the flag-0 G-PCC
  stream, because TMC13 v14 cannot decode streams made with the official
  `--cabac_bypass_stream_enabled_flag=1`. The flag-0 stream is 0.6 % smaller and also lossless.
- **Training time** was measured on shared, loaded A6000s. The paper reports 18–25 min on an RTX 4090.

### Storage breakdown (official formula, bytes)

| scene | xyz | scale | rotation | appearance | MLPs |
|---|---|---|---|---|---|
| bicycle | 1,008,662 | 903,751 | 925,777 | 1,514,651 | 30,720 |
| counter | 681,673 | 636,585 | 661,806 | 1,073,429 | 30,720 |

Our models have 13 % (bicycle) and 3 % (counter) fewer Gaussians than the paper's. Storage tracks
the Gaussian count: bicycle is −9 %, counter −0.3 %.

### Relative to the baselines

| scene | OMG − our Mini-Splatting | paper OMG-XS − paper Mini-Splatting | OMG − our gsplat 3DGS |
|---|---|---|---|
| bicycle | 24.68 − 24.95 = −0.27 | 24.95 − 25.15 = −0.20 | 24.68 − 24.92 = −0.24 |
| counter | 27.94 − 28.50 = −0.56 | 28.40 − 28.41 = −0.01 | 27.94 − 28.85 = −0.91 |

- **Bicycle:** OMG costs about the same relative to its Mini-Splatting base as in the paper.
- **Counter:** ours loses 0.56 dB against our Mini-Splatting run, while the paper loses 0.01 dB. This
  is the largest unexplained piece of the gap; the deviations below are the candidate causes.
- **LPIPS** is higher on both scenes. Our gsplat and Mini-Splatting runs show the same LPIPS offset
  (lpips package, valid-convolution SSIM; see `_calibration`).

## Verification

- **Contract and tests:**
  - `ss-inspect` reports no issues;
  - 12 unit tests pass and ruff is clean;
  - `ss-verify` gives `declared_checks_passed` for `omg` and `omg-indoor`.
- **Probes:** 6 copied Mini-Splatting probes plus 6 OMG probes:
  - LD prune against an independent bit-loop Morton reference;
  - space-feature MLPs against a float64 sin/cos + matmul reference;
  - live render and gradient flow;
  - net LR against a torch ChainedScheduler replay;
  - SVQ layout, nearest-code indices and frozen indices;
  - codec round trip (Huffman, fp16↔uint16, G-PCC, bake).
- **Smoke run** (garden, compressed 2000-step schedule, every event fired): decoded PSNR 19.22.
  This is a diagnostic only (`evidence/test.json`).
- **Decode drift** in the real runs: decoding with fp16 positions and weights changes the appearance
  by a mean absolute 0.0014 (SH DC) and 0.0017 (opacity).
- **Review:** `reviews/review-1.md` approved with 4 minor items:
  - our decode is fp16 throughout, while the official in-training decode uses fp32 codes
    (`decompress=False`); estimated effect ≤ 0.02 dB;
  - the codec probe;
  - the numbers quoted in the smoke note.

## Deviations from the paper (all present in the reported numbers)

1. **Runtime, inherited from the Mini-Splatting port:** white background (official: black), gsplat
   rasteriser numerics, view sampling with replacement, and SSIM with valid convolution.
2. **MLPs:** fp32 instead of tcnn fp16, including tcnn's `__sinf` on the 2^15·π frequency features.
   The weights are rounded to fp16 only at encode.
3. **K-means:** torch k-means++ seeding plus Lloyd iterations (tolerance 1e-4, at most 1000)
   instead of cuML's scalable-k-means++.
4. **G-PCC:** the container stores the flag-0 stream (see above). The geometry is identical and
   both sizes are reported.
5. **Container format:** our own (canonical Huffman tables + JSON + LZMA) instead of an LZMA pickle
   of dahuffman tables. The official-formula size is reported alongside the real container size.
6. **Decoded model:** fp16 codebooks throughout, as in the official `render.py --decode`. The
   official in-training decode uses fp32 codes (review item, ≤ 0.02 dB).
7. **Update order:** the per-Gaussian Adam steps before the density hook, so positions get one extra
   tiny step at 30k.
8. **LD neighbours:** we follow the released code (the Morton permutation applied twice), not the
   paper text (adjacent Gaussians in Morton order). The paper's Morton→KNN ablation shows
   ±0.01 dB, so the choice is expected to matter little.
9. **Run length:** the step constants are absolute (15k/20k/29k/30k), so only 30k-step runs are
   accepted.

## Reproduce
```bash
cd reproductions/omg/plugin && ../../../tools/ssenv pip install -e .
python3 tools/bench.py launch --exp omg-mipnerf360 --methods omg --dataset mipnerf360 --scenes bicycle
python3 tools/bench.py launch --exp omg-indoor-mipnerf360 --methods omg-indoor --dataset mipnerf360 --scenes counter
tools/ssenv python reproductions/omg/plugin/report_size.py <run>
```
