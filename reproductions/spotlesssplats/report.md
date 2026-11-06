# SpotLessSplats (plan #77) — build report (pre-benchmark)

**Method**: `spotlesssplats` (SLS-mlp + UBP, paper headline); variants `spotlesssplats-noubp`,
`spotlesssplats-agg`. Plugin: `plugin/` (`ss_spotlesssplats`). Port of the official gsplat trainer
(lilygoli/SpotLessSplats `examples/spotless_trainer.py` @ 0caae3c); code values override paper values
where they differ (paper.md §7).

## Contributions → components
| contribution | component | probe |
|---|---|---|
| Masked L1 + RobustNeRF mask (Eq. 1–2), no D-SSIM | `sls-robust` | robust-mask, pipeline-step |
| Discounted residual histogram quantiles (§4.2.2) | `sls-robust` | quantile-histogram |
| SLS-mlp classifier on SD 2.1 features + PE, hinge (Eq. 5–8) | `sls-robust` + `sls-mlp` | mlp-equivalence, mlp-loss-order |
| SLS-agg cluster vote (Eq. 3–4) | `sls-robust` (mask_mode agg) | agg-vote |
| Bernoulli warm-up (Eq. 9) | `sls-robust` | warmup-schedule |
| UBP (Eq. 10), custom CUDA `csrc/ubp.cu` | `sls-density` | ubp-kernel (exact vs per-pixel autograd), density-schedule |
| No opacity reset, SH reset at 8002 | `sls-density` | density-schedule |
| GLO appearance (Eq. 11) | **not implemented** (absent from official code) | – |

## Data
Authors' preprocessed scenes with released SD features (Borealis doi:10.5683/SP3/WOFXFT), android
(RobustNeRF) and patio (On-the-go), 3.0 GB under `/bigdata/sejain/datasets/spotless` (symlink
`~/datasets/spotless`). Converted with `tools/prepare_spotless.py` (official Parser: images_8, cv2
undistortion, world normalisation; train = clutter, test = clean `extra` views, val = copies of test);
registered as `robustnerf` / `onthego` in `tools/datasets.json`. SD 2.1 weights
(`/bigdata/sejain/models/sd21`, sha256 in `SHA256SUMS`) + `tools/extract_sd_features.py`: parity with
released features cos 0.94 (our own run-to-run noise: 0.95) — `evidence/sd_feature_parity.json`.

## Tests
inspect [] ×3, pytest 7 passed, ruff clean, ss-verify `declared_checks_passed` ×3. Smoke (android,
2000 steps): 24.13 dB / 158k Gaussians / 64 s vs gsplat 24.18 dB / 453k / 70 s. Fixed one leak
(hook reference cycle, OOM at 1.7k steps). Patio 700 steps: mlp 20.44, agg 18.29 dB.

## Benchmark request
android + patio, 30k steps, plus `gsplat` on the same scenes. Targets: android 25.15, patio 22.24
(paper, 5-run mean, with GLO); official gsplat README (no GLO): 24.96 / 22.2.

## Fidelity gaps
GLO absent; β1 = 3e-3 (code) vs 3e-4 (paper); UBP threshold 1e-14 (code) vs 1e-8 (paper); view
sampling with replacement; optimizer step before refine; scene scale from train cameras only;
identity initial quats; 1 seed vs 5; LPIPS-VGG (paper) vs AlexNet (official code).

## Review-1 resolution (orchestrator)
Review result: approved, with 0 critical and 0 major issues. The benchmark is launched with `spotlesssplats` and `gsplat` on robustnerf/android and onthego/patio, 30k steps.

**Targets**
- **Primary:** the official README numbers, android 24.96 and patio 22.2. They come from this exact code and config: no GLO, β1 3e-3, UBP 1e-14, a single run.
- **Secondary:** the paper, android 25.15 and patio 22.24. These are 5-seed means, with GLO and β1 3e-4.
- **Tolerance:** about ±0.3 dB, since we run a single seed.

**Comparing gains**
- **Official README:** +1.73 dB on android and +5.43 dB on patio over its 3DGS (23.23 / 16.77).
- **Paper:** +1.82 dB and +3.99 dB over its 3DGS (23.33 / 18.25).
- **Our gsplat baseline:** it uses L1 + 0.2 D-SSIM, a white background and an opacity reset. That matches neither paper's 3DGS row, but the data, splits and resolution are the same.

**Note:** patio runs at images_8 (240×135). That is the SpotLessSplats code default, not the native NeRF On-the-go protocol.

## Benchmark: RobustNeRF android (spotless-robustnerf, 30k) — MATCHED
spotlesssplats 25.07 / 0.840 / 0.184 vs paper SLS-mlp+UBP 25.15 (Δ −0.08; official README 24.96).
gsplat on the same scene: 24.03, so the gain is +1.04 dB. The paper's 3DGS row is 23.33, a gain of +1.82 for SLS.
Our gsplat baseline is already stronger than the paper's 3DGS row. On-the-go patio is still pending.

## Benchmark: On-the-go patio (spotless-onthego, 30k) — MATCHED
spotlesssplats 22.30 / 0.859 / 0.127 vs paper SLS-mlp+UBP 22.24 (Δ +0.06; official README 22.2).
gsplat on the same scene scores 17.24, a gain of +5.06 dB; the paper's 3DGS row is 18.25, a gain of +3.99.

## Final verdict: MATCHED (2/2 reported scenes within ±0.5 dB)
| scene | ours | paper | Δ | gain over gsplat (ours / paper vs 3DGS) |
|---|---|---|---|---|
| android | 25.07 | 25.15 | −0.08 | +1.04 / +1.82 |
| patio | 22.30 | 22.24 | +0.06 | +5.06 / +3.99 |
