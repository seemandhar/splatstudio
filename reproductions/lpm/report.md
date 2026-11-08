# LPM — Improving Gaussian Splatting with Localized Points Management (CVPR 2025) — reproduction report

**Outcome: matched.** Both benchmarked scenes are within ±0.5 dB of the paper's per-scene PSNR. Single seed, 30k steps.
The match is in absolute PSNR. The *improvement over 3DGS* is not reproduced consistently: our gain is +0.30 dB on bicycle and −0.17 dB on room, against the paper's +0.23 / +0.05.

Native SplatStudio port of the official code, https://github.com/Surrey-UP-Lab/GS-LPM @7c06026. Plugin: `plugin/` (methods `lpm`, `lpm-indoor`).
Review: `reviews/review-1.md` (approved, 0 critical/major, 4 minor). The plugin's zones equal a reimplementation of the official ray path to 1.7e-3 world units.

## Results (Mip-NeRF 360, `mipnerf360` protocol = 3DGS images_4 outdoor / images_2 indoor, llffhold8)
| scene | method | ours PSNR | paper PSNR | Δ | ours SSIM / LPIPS | paper SSIM / LPIPS | Gaussians | train min |
|---|---|---:|---:|---:|---|---|---:|---:|
| bicycle | `lpm` | 25.22 | 25.40 | −0.18 | 0.774 / 0.226 | 0.776 / 0.203 | 6.50M | 101 |
| room | `lpm-indoor` | 31.35 | 31.58 | −0.23 | 0.919 / 0.289 | 0.923 / 0.209 | 1.73M | 78 |

Gain over 3DGS:
| scene | our gsplat baseline (`baseline3dgs-mipnerf360-30000`) | our gain | paper 3DGS* (authors' retrain) | paper gain |
|---|---:|---:|---:|---:|
| bicycle | 24.92 | +0.30 | 25.166 | +0.23 |
| room | 31.52 | −0.17 | 31.53 | +0.05 |

- Our LPIPS is LPIPS-VGG from the `lpips` package, so absolute LPIPS values are not comparable; compare deltas only.
- Runs: `runs/lpm-mipnerf360`, `runs/lpm-indoor-mipnerf360` (`summary.json` in each).
- Smoke and diagnostic evidence: `evidence/test.json`.
  - The 2000-step smoke run scored 19.19 dB because step 2000 is a localized-reset step.
  - At 1950 steps: 20.97 vs baseline 21.34.

## Deviations from the paper / official code
1. **Lazy pairing.** The runtime gives density components only the cameras, not the images. So training targets are cached as they are rendered (uint8, lossless). SuperPoint+LightGlue pairs (3 random neighbours within `angle`) are built when a view first needs LPM, using only neighbours already seen, and topped up later. The official code matches all views offline. If none of a view's neighbours has been seen yet, that LPM step does nothing. This is rare and only happens early (0 occurrences in the smoke run).
2. **Dropped 20 px screen-size prune.** The official `lpm_densify_and_prune` also prunes `max_radii2D > 20` after step 3000. In the official code `max_radii2D` is reset only when zones are found, so this prune can fire. The port omits it, which matches our shared gsplat baseline (only the world-size prune is kept).
3. **White background.** The runtime renders on white; the official code uses black for these datasets. This affects both the training loss and LPM's error maps. No visible effect is expected on the 360 scenes, which have no empty background.
4. **LightGlue version.** cvg/LightGlue @5a650c4 with the released `superpoint_v1` / `superpoint_lightglue` weights. The official code uses its vendored `LightGlue/` copy, so match sets may differ slightly. SuperPoint weights carry a non-commercial licence.
5. **Indoor setting.** The README suggests reset_interval 200 and angle 90° for indoor scenes; we use that for room (`lpm-indoor`). The paper's reset-interval ablation (tab:lpm_reset_ini) instead favours 100 indoors (31.03 vs 29.85). The paper does not say which setting produced its per-scene tables. Room is our weakest result, so this ambiguity is the first suspect. I did not test the alternative, to avoid tuning on test images.
6. **"Add points at the zone centre."** The paper mentions it, but it is absent from the official code, so it is not implemented.
7. **Neighbour angle.** Computed between centre-ray directions; the official code uses the 6-D [o, d] vector. They disagree on 0.1% of view pairs.
8. **Rays.** Built from our K (keeps cx, cy) rather than 3DGS's symmetric projection; sub-pixel difference.
9. **Runtime (shared by all native ports).** The optimizer steps before density control (the official code densifies first); views are sampled at random with replacement; gsplat rasterizer.

## Observations
- Official zones are axis-aligned boxes around the closest points of index-paired perimeter rays. They are often very large (official median extent 2.5, p90 58 world units; scene scale ≈ 5).
- As a result, the localized reset hits up to ~17k Gaussians in a single call. The early-training PSNR is depressed until resets stop at 15k.
- This behaviour is reproduced as-is because it is the official algorithm.
