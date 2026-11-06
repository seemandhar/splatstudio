# WildGaussians (plan #84) — build report (pre-benchmark, after review-1)
Paper arXiv 2407.08447; official code jkulhanek/wild-gaussians (MIT) ported; no custom CUDA.
Methods:
- **wildgaussians:** On-the-go with appearance on (the paper's "Ours").
- **wildgaussians-noapp:** the released nerfonthego.yml; a required second row.
- **wildgaussians-pt:** Photo Tourism, 200k steps.

| contribution | component(s) | probe(s) |
|---|---|---|
| appearance MLP + embeddings (Eq.4-5) | wg-renderer (2 passes), bank, field wg_embedding (Fourier init) | appearance-toning, render-two-pass |
| D-SSIM raw / L1 toned (Eq.6) | wg-color | split-objective |
| DINO uncertainty (Eq.8-9) | wg-uncertainty, bank head, own DINOv2 (== timm) | dino-backbone, uncertainty-loss |
| binary mask + warm-up/protected (Eq.10-11) | wg-uncertainty -> outputs.wg_mask -> wg-color | uncertainty-loss, split-objective |
| test-time appearance (NeRF-W) | ss_wildgaussians.evaluate | nerfw-protocol |
| sky init, Mip filters, abs-grad densify (never-reset stats), official reset | wg-density, geometry | sky-init, densify-official, filter-3d |

**Checks:** ss-verify declared_checks_passed for all 3 variants; 12 unit tests pass; ruff is clean.

**Data:**
- On-the-go: ~/datasets/onthego-undist/native/{patio,patio-high}, from the HF undistorted set. The SpotLessSplats patio is the distorted x8 version, so it was not reused.
- Photo Tourism: ~/datasets/phototourism/native/brandenburg-gate (IMC dense model, NeRF-W split, full res).
- DINOv2 weights: /bigdata/sejain/models/dinov2 (sha256 f433177089a6...641fb).

**Smoke test** (patio-high, 3500 steps, same official metrics for both): WildGaussians 21.13 / 0.671 / 0.280 vs gsplat 20.61 / 0.656 / 0.295.
- Before the fixes, at 2000 steps: 20.60 vs 20.79.
- The 3000-step runs ended exactly on the opacity-reset step and are invalid.

**Benchmark request** (all runs scored with `python -m ss_wildgaussians.evaluate`):
- wildgaussians and wildgaussians-noapp on patio-high (paper 22.23) and patio (21.44), 30k steps.
- wildgaussians-pt on brandenburg-gate (27.77), 200k steps.
- gsplat control on all 3 scenes.

## Review resolution (review-1: 1 major, 5 minor; all fixed)
- **Opacity reset** (major): now ports the official reset with the doubled 3D filter, so sub-filter Gaussians end below 0.01 and are pruned at the next densify. The filter-3d probe checks it against an independent re-derivation; the old formula fails the probe.
- **Abs-grad statistic:** now the official L1 scalar sum_px(|gx|+|gy|).
- **gsplat control:** scored with the same official metrics.
- **wildgaussians-noapp:** now a required row.
- **Deviation list:** extended (below).

## Deviations
- **Paper vs code** (code followed):
  - The colour gain has no +1.
  - The Photo Tourism opacity reset is 15000, not 3000 as in the paper.
  - On-the-go appearance is on in the paper but off in the released config.
- **Training:**
  - Views are sampled with replacement.
  - Gaussian Adam steps before density, while the official code effectively skips the Gaussian Adam step on densify/reset iterations.
  - Exact KNN instead of distCUDA2.
- **gsplat rasterizer:**
  - Culls Gaussians with opacity < 1/255 and uses an opacity-aware extent, which changes denom for faint and sub-filter Gaussians.
  - Alpha cap is 0.999 (straight-through) vs the official 0.99.
- **Other:**
  - Sky logit 20 vs +inf.
  - Single seed.
  - ss-eval reports raw untoned renders (no bank).
