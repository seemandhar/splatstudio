# WE-GS gap analysis 1

The 150k runs are in `runs/we-gs-phototourism/we-gs/`. Protocol: NeRF-W split, downscale 2, test
appearance encoded feed-forward from the test image, full-image metrics, and our evaluator
`ss_wegs.evaluate` (`metrics.full`). Diagnostics script: `tools/diag_appearance.py`. Its outputs
are `evidence/diag-{brandenburg,sacre}.json` and `evidence/diag-brandenburg-img/` (columns: GT |
render | render with the mean training embedding).

## Gap table (PSNR)

| scene | ours | paper | abs gap | our gsplat (ds2, 10k) | paper 3DGS | baseline offset | delta gap |
|---|---|---|---|---|---|---|---|
| brandenburg-gate | 25.21 | 27.74 | **−2.53** | 19.63 | 20.72 | −1.09 | **−1.44** |
| sacre-coeur | 22.98 | 23.62 | −0.64 | 16.74 | 17.57 | −0.83 | +0.19 |

**Caveats on the baselines.**
- Our gsplat reference was trained for 10k steps (the brief allowed diagnostics of at most 10k
  steps) with a white background. It is scored on the full image.
- The existing WildGaussians gsplat run (`runs/wg-gsplat-pt`, full resolution, 30k) gives
  18.77 on brandenburg with the runtime metric and 19.38 with the WildGaussians evaluator.
- The paper's 3DGS row matches the GS-W paper's own numbers. That is the authors' protocol, not
  ours, so `baseline_offset` is only indicative.

## Evidence

1. **The model is healthy on the training views.**
   - Gaussian counts are 284k and 272k. That fits the paper's 66.4 MB, which is about 280k at
     59 floats per Gaussian.
   - Densification stops at 15k, as in 3DGS.
   - Every loss term is active: L_regTS touches 0–657 Gaussians per step, and the mask mean
     during training is 0.7–0.9.
   - Training took 5.1 h and 6.9 h on a shared A6000; the paper reports 1.8 h on a 4090.
2. **Per-image appearance is genuinely used.**
   - The embedding std per dimension across 60 training images is 0.75–1.6, with norm ≈5.6.
   - Rendering with the mean training embedding costs about 4 dB on train views (brandenburg
     22.54 → 18.66; sacre 22.04 → 16.98) and about 4.4 dB on test views.
   - The SH residual is small on average: mean ‖Δsh‖ is 0.006–0.008, and |Δdc| is about 0.001.
     It is still decisive.
3. **The brandenburg gap is a test-time appearance (colour) error.**
   - Diagnostic only, never used for model selection: a global per-channel affine colour fit to
     the GT raises test PSNR from 25.21 to 27.83 (+2.62 dB).
   - The same fit gains only +0.36 dB on train views.
   - So the encoder reproduces training appearances well but generalises worse to the 10 unseen
     test images.
   - Two views carry most of the gap: 84113133 (19.19, 23.19 after the fit) and 86628129
     (18.39, 25.52 after the fit). Excluding them, the mean is 26.8.
   - In 86628129 the façade is rendered brighter and whiter than an overcast, darker GT.
   - In 84113133 there is also a geometry/season mismatch: leafy trees behind the columns and a
     blurry ground plane, which no appearance code can fix.
   - Sacre: affine fit 22.98 → 24.24 (+1.25) on test vs +0.46 on train. Same direction, smaller.
4. **The test-time pipeline is the training pipeline.**
   - Same downscale-2 LANCZOS frames and the same encoder input (the image resized only to a
     multiple of 16).
   - Same [0,1] normalisation and no masking in the evaluator.
   - The runtime's live-bank `validation.json` equals `ss_wegs.evaluate`'s numbers (25.21 /
     22.98), so there is no loading or normalisation bug.

## Classification

- **brandenburg: method / generalisation.** The gap is concentrated in feed-forward appearance
  on unseen images: 2.6 dB of it is recoverable by a global colour fit, against 0.36 dB on train
  views.
- **sacre-coeur: protocol.** Our gain over our gsplat is +6.24 dB, against the paper's +6.05
  over its 3DGS (delta gap +0.19). The −0.64 is explained by the −0.83 baseline offset.
- **brandenburg detail.** The −2.53 splits into −1.09 baseline offset and −1.44 method (delta
  gap).

## Ranked hypotheses

1. **Encoder over-fits per-image identity rather than illumination.** Expected effect: large on
   brandenburg.
   - For: the train/test affine-fit asymmetry. It sees the full target image, has 512-channel
     convs, runs for 150k steps on 763 images, and nothing regularises the embedding (the paper
     has no embedding loss).
   - Against: nothing direct. The paper's architecture is followed as written.
   - No grounded fix. The paper specifies no embedding regulariser, augmentation or encoder
     dropout. The released inference model differs in ways that are not tied to generalisation:
     256² resized input, mask fed as a 4th input channel instead of attention, dim 32. Switching
     to it would revert the review-1 M1 decision. It is also the TVCG variant, not the WE-GS
     paper configuration.
2. **Paper-silent loss choices: D-SSIM mask stop-gradient, ε = 0.1, L_regSH mean reduction.**
   Expected effect: under 0.3 dB on the colour error.
   - These change mask strength and residual magnitude, not test-time colour generalisation.
     Test masks are already about 0.92 / 0.85.
   - The literal alternatives collapse (smoke-1, review-1).
3. **TVCG learning rates (1e-4 / 1e-4 / 1e-5).** Unknown effect.
   - The paper gives no learning rates, so there is no grounded alternative.
4. **Densification only to 15k of 150k steps.**
   - "Following similar settings as in 3DGS" (§5.1). The count matches the paper's storage, and
     train-view fidelity is fine. Not the cause.
5. **Base/protocol offset.** See the table. The paper's 3DGS row was taken from GS-W, so it has
   an unknown resolution or eval convention.

## Recommendation

No paper- or released-code-grounded change is likely to close ≥0.5 dB on brandenburg. Report
`not-matched`:
- brandenburg −2.53. This is the encoder's test-time appearance generalisation, with the
  diagnostic affine-fit evidence above.
- sacre −0.64.

Possible follow-ups, outside the paper and so labelled if ever run:
- Evaluate the authors' released WE-GS Brandenburg checkpoint (Google Drive, PT-OVS models) with
  our evaluator. This would separate implementation from method. It is the TVCG configuration,
  so it is only indicative.
- A seed repeat, to size the noise on the 10 brandenburg test views. Two views dominate.
