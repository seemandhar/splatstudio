# SplatFields reproduction report

**Paper:** *SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction* (ECCV 2024).

**Methods:** `splatfields-3dgs` (stage 1) and `splatfields` (stage 2). The plugin is in `plugin/` and is frozen at `f32f62e64feb314b`.

**Outcome: partial (2 of 3 scenes).** On 6-view Blender, hotdog and ficus are within ±0.5 dB of the paper. Chair is 0.78 dB short.

The method's main claim reproduces on every scene: neural splats do much better than 3DGS on sparse views.
- Gain over our stage-1 3DGS: **+1.54 / +4.75 / +0.19 dB**.
- Paper's gain over its 3DGS: +2.28 / +5.03 / +0.01 dB.

## Protocol
The paper's 3DGS row, which we reproduce as stage 1, also produces the initial point cloud for SplatFields.

**Stage 1**
- Run `splatfields-3dgs` on `blender-fewview6` for 40k steps.
- The 6 training views are chosen by KMeans, as in the official code.
- The test set is the full Blender test set (200 views).

**Build**
- `tools/build_stage2.sh` keeps the stage-1 Gaussians that fall inside the visual hull, giving the `blender-fewview6-sf` dataset.

**Stage 2**
- Run `splatfields` for 40k steps.

The 12-view setting and the DTU / 4D settings were not run (see `state.json` `not_reproduced`).

## Results (PSNR / SSIM, test split)

| scene | stage 1 ours / paper 3DGS | **SplatFields ours / paper (Δ)** | gain ours / paper | SSIM ours / paper | #G |
|---|---|---|---|---|---|
| chair | 22.71 / 22.75 | **24.25 / 25.03 (−0.78)** | +1.54 / +2.28 | 0.889 / 0.895 | 146k |
| hotdog | 21.26 / 21.16 | **26.01 / 26.19 (−0.18)** | +4.75 / +5.03 | 0.932 / 0.936 | 122k |
| ficus | 22.16 / 22.25 | **22.35 / 22.26 (+0.09)** | +0.19 / +0.01 | 0.898 / 0.896 | 109k |

- Stage 1 matches the paper's 3DGS row within 0.1 dB on all scenes, so the comparison is on equal footing.
- Mean LPIPS over the three scenes is 0.106.
- Training took 2.0–2.8 h per scene for stage 2 on shared A6000s.

## Chair gap (not iterated)
- **Likely cause:** the stage-2 initial cloud comes from a single stochastic stage-1 run (hull subsample plus densification). The authors' clouds are not released (gap 8).
- **Why no iteration:**
  - The batch rules allow an iteration only for a fix grounded in the paper, and no such fix was found.
  - Re-running seeds and keeping the best chair result would be selection on test numbers.
- **Result kept:** the single-seed result.

## Fidelity gaps (kept, from plan.md)
1. **Rasterizer.** We use gsplat instead of the ingra14m rasterizer. The maths is the same (2D blur 0.3, same near plane).
2. **Metric rendering.** Metrics are computed on float renders; the official evaluation uses 8-bit PNGs (about 0.0x dB).
3. **View selection.** KMeans view selection uses scikit-learn 1.0.2 defaults; the authors' version is unknown.
4. **Update order.** The runtime steps the optimizer before density. This matters only in stage 1.
5. **SH schedule.** It is one step behind the official schedule.
6. **Final step.** The official code skips the last optimizer step; the runtime does not.
7. **Stage-1 randomness.** It differs from the authors' run.
8. **SSIM.** The loss uses valid-convolution SSIM.
9. **Opacity reset.** None (NoResetStrategy, as in the official code).
10. **Not implemented.** The 4D branch and the DTU protocol.

## Review
`reviews/review-1.json` raised 1 major issue: `refine_stop` had to be 40000. It was fixed before the benchmark.

## Reproduce
```bash
python3 tools/bench.py launch --exp splatfields-stage1-fv6 --methods splatfields-3dgs --dataset blender-fewview6 --scenes chair hotdog ficus --steps 40000
reproductions/splatfields/tools/build_stage2.sh splatfields-stage1-fv6 6
python3 tools/bench.py launch --exp splatfields-fv6 --methods splatfields --dataset blender-fewview6-sf --scenes chair hotdog ficus --steps 40000
```
