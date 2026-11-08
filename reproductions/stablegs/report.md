# StableGS (plan #9) — reproduction report

**Outcome: matched** (T&T complete average, 30k steps; not separately reviewed).
Paper arXiv 2503.18458, Table 1 row "StableGS w/o L_prior" (averages only). Official code
MooreThreads/StableGS @397a9c00 (incomplete release). Runs: `runs/stablegs-tandt`.

| scene | PSNR | Gaussians | train time |
|---|---|---|---|
| truck | 25.83 | 2.58 M | 105 min |
| train | 22.44 | 1.09 M | 138 min |
| **T&T mean** | **24.13** | | |
| paper "w/o L_prior" | 24.165 | | |
| difference | **−0.03** | | |

Improvement over 3DGS: we gain +0.77 over our gsplat baseline (23.36), while the paper gains
+0.23 over its retrained 3DGS (23.93). Absolute PSNR matches. Our gain is larger mainly because
our gsplat baseline sits 0.57 dB below the paper's 3DGS(updated) on T&T, so the gain is not
directly comparable.

## Deviations from paper / official code
1. **Reconstructed renderer and arguments.** The release lacks `gaussian_renderer/` and
   `arguments/`. The renderer was rebuilt from the vendored RaDe-GS@ec9199f rasterizer
   (kernel_size 0, black background). Learning rates follow Taming-3DGS (opacity and α' 0.025,
   SH-rest 2.5e-4).
2. **Pair set P from the paper's criteria.** We use COLMAP covisibility > 30, rotation 16–60°
   and homography inlier ratio < 0.8 at 4 px. The official mvs.py uses BoQ retrieval with a
   rotation/translation filter instead, and its weights are not available here. Our P on truck
   has 5372 pairs, probably denser than the official graph.
3. **No DUSt3R prior (C3).** We benchmarked the "w/o L_prior" row. In the paper the full model
   gains only +0.03 dB on T&T.
4. **Reproduced code bug.** View 0's SSIM is counted twice in the colour loss, as in
   train_pair.py:128. Config `ssim_views: "both"` gives the corrected form.
5. **NaN guard (not in the official code).** In the consistency warp, float32 cancellation can
   make the transformed z exactly 0, which turned the backward into NaN. We clamp |z| < 1e-6 to
   1e-6. Those points project far outside the image, so in-view values are unchanged.
6. **Runtime differences.**
   - The runtime steps the optimizer before density control; the official code does it after.
   - The density controller's pair stack drives view sampling and ignores the runtime's random
     frame.
   - The Taming score uses gsplat renders.
   - Evaluation (LPIPS-VGG and SSIM) uses the runtime's implementation.
7. **Code over paper.**
   - Depth consistency uses the official one-sided form. The paper's two-sided form is available
     as `stablegs-paper-consis`.
   - The RaDe-GS depth-normal loss (0.05) is included. It is in run.sh but not in the paper.
   - α is frozen at 15k.

## Evidence
- `evidence/verify.json`: all probes passed.
- `evidence/test.json`: smoke tests and fixes.
- `plan.md`, `paper.md`, `paper_results.json`.
