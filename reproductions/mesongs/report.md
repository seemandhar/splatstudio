# MesonGS reproduction report

**Paper:** *MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation* (ECCV 2024, arXiv 2409.09756).

**Method:** `mesongs-c1-<scene>`, the official released c1 configuration. The pipeline starts from 3DGS trained for 30k steps, then:
1. Importance pruning.
2. Octree positions.
3. RAHT on colour, with block-wise quantisation and a 2048-entry codebook.
4. 8000 fine-tune steps.

The plugin is in `plugin/` and is frozen at `992f4f4016742a48`.

**Outcome: matched.** The decoded-model PSNR is within ±0.22 dB of the official per-scene c1 results on all 3 scenes (−0.16 / −0.02 / −0.22). Sizes are 5–16 % larger, in line with our Gaussian counts.

## Targets
Table 1 of the paper gives only 7-scene averages: 28.61 dB and 27.62 MiB for MesonGS-FT. The per-scene targets therefore come from the official repo's `results/MipNeRF360/*.csv`, c1 rows. The paper's headline result uses the c1 configuration. The pass criterion was fixed in `state.json` before benchmarking.

## Results
Setup:
- 38k steps in total: 30k for 3DGS, then the compression stage with 8000 fine-tune steps.
- Data: `mipnerf360-1600`, the official `-r -1` loader.
- Split: llffhold 8.
- Evaluated on the decoded model.

Sizes come from `plugin/report_size.py`, which writes `bins.zip` with the official `save_npz` members. MB = 1e6 bytes.

| scene | PSNR ours / c1 (Δ) | SSIM ours / c1 | LPIPS ours / c1 | size MB ours / c1 (Δ) | #G ours / c1 (before prune) |
|---|---|---|---|---|---|
| bicycle | 24.40 / 24.56 (**−0.16**) | 0.717 / 0.716 | 0.317 / 0.289 | 56.4 / 48.7 (+16 %) | 4.32M / 3.69M (6.17M) |
| garden | 26.37 / 26.40 (**−0.02**) | 0.829 / 0.829 | 0.192 / 0.167 | 51.2 / 48.5 (+5 %) | 3.75M / 3.53M (4.58M) |
| counter | 28.33 / 28.55 (**−0.22**) | 0.901 / 0.904 | 0.274 / 0.202 | 14.7 / 14.0 (+5 %) | 1.04M / 0.99M (1.18M) |

- Size per Gaussian is close to the reference on every scene: 13.1 / 13.6 / 14.2 bytes, against 13.2 / 13.8 / 14.1.
- The size differences therefore come from how many Gaussians survive pruning, which depends on our gsplat base model.
- Compression against an equivalent 3DGS PLY is 27× / 22× / 20×.

## Comparison with the baseline
Our baseline is `baseline3dgs-mipnerf360-1600-30000`.

| scene | our 3DGS | MesonGS − ours |
|---|---|---|
| bicycle | 24.93 | −0.53 |
| garden | 26.93 | −0.56 |
| counter | 29.04 | −0.71 |

The paper's 7-scene average is MesonGS-FT − 3DGS = 28.61 − 28.98 = −0.37 dB. Our per-scene costs are larger by about 0.2–0.3 dB. The absolute PSNR still matches the official c1 runs, which suggests the authors' 3DGS inputs scored somewhat lower than ours. This cannot be checked per scene, because the repo CSV has no matching 3DGS column.

LPIPS is higher across scenes. This comes from the runtime's metric convention (the `lpips` package with a white background).

## Fidelity gaps (kept, from plan.md)
1. The base model is our gsplat 3DGS, not the authors' 3DGS checkpoints.
2. The fine-tune length of 8000 is taken from the released script; the paper does not state the Mip-NeRF 360 value.
3. The official code keeps the best model by **test** PSNR, checked every 400 fine-tune iterations. We report the final model, which is the honest choice; the official selection is optimistic.
4. Views are sampled with replacement.
5. The gsplat rasteriser is used for both importance and rendering. gsplat also drops gradients at saturated alpha (see `TODO-runtime.md`).
6. The Euler `sqrt` input is clamped at 1e-12 to avoid NaN gradients.
7. The background is white; the released script uses black.
8. The decoded model renders from quaternions recovered from the Euler rotation matrix, a float-level difference.

## Review
`reviews/` raised one major issue, about targets. It was resolved by switching to the c1 config and the repo's per-scene c1 rows, with sizes reported in both MiB and 1e6 bytes.

## Reproduce
```bash
for s in garden bicycle counter; do
  python3 tools/bench.py launch --exp mesongs-c1-$s --methods mesongs-c1-$s --dataset mipnerf360-1600 --scenes $s --steps 38000
done
tools/ssenv python reproductions/mesongs/plugin/report_size.py runs/mesongs-c1-*/*/mipnerf360-1600-*
```
