# AuGS — Augmented Radiance Field (plan #110, ICLR 2026, arXiv 2602.19916)

**Outcome: matched.** Both scenes are within ±0.5 dB of the paper's per-scene PSNR. The gain over the base model is reproduced (ours is slightly larger than the paper's on both scenes).

Variant: "Ours (3DGS, sh=3)", from the supplementary tables per_scene_m360 and diff_ps_mipnerf. Protocol `mipnerf360-pilr`: the official loader's `-r 4/2`, i.e. PIL-resized full-resolution images, llffhold8. Method `augs` (plugin `plugin/`, module `ss_augs`). We read the official code (github.com/XiaoXinyyx/AuGS_Code @57cca00) for its numbers and never ran it.

## Results (test PSNR)
| scene | our base (in-run, step 30k) | ours final | gain | paper base (final − Δ) | paper final | paper gain | final vs paper |
|---|---|---|---|---|---|---|---|
| bicycle | 25.42 (6.24M G) | **25.60** (6.85M G) | +0.18 | 25.74 | 25.85 | +0.11 | −0.25 |
| room | 31.62 (1.68M G) | **32.43** (1.84M G) | +0.81 | 31.67 | 32.39 | +0.72 | +0.04 |

The shared pilr gsplat baseline gives bicycle 25.45 and room 31.69, consistent with the in-run bases. The bicycle shortfall in absolute PSNR comes from the base: ours is 0.32 dB below the paper's implied base, so the deficit is not in the enhancement.

## Augmented primitives
| scene | base Gaussians | augmented (10 %) | views | skipped (<3 pts → zfar) | cluster retries | scale-fit failures | mean T init |
|---|---|---|---|---|---|---|---|
| bicycle | 6,236,727 | 617,629 | 169 | 1,207 | 63 | 0 | 0.2 |
| room | 1,676,481 | 167,198 | 272 | 0 | 5 | 0 | 0.1 |

Wall time is 650 min for bicycle and 907 min for room. The enhancement step alone took 8.4 h on bicycle and 13.7 h on room, almost all of it inverse-splatting projection (7.9 h and 12.7 h). Rendering and 2D fitting took 0.6 h and 1.0 h. The paper reports about 15 min on a 4090. Speeding up projection is an open engineering item; it does not affect fidelity.

## Implementation
- **Base:** 30k steps of the built-in gsplat 3DGS method.
- **Depth maps:** the Gaussian where transmittance crosses the threshold (custom CUDA `csrc/contrib_stats.cu`, adapted from gsplat 1.5.3), then the max-response depth along the ray. The threshold is 0.7, from the code; the paper says 0.5.
- **2D fitting:** error-driven 2D Gaussians drawn with gsplat `rasterize_to_pixels` over the rendered image, using vendored fused-ssim (MIT; the SSIM the official code uses).
- **Inverse splatting** (GPU-batched): single-linkage clustering in float64, weighted PCA, the Eq. 6 scale fit, and lobe initialisation T = clip(7θ̄/π, 0.01, 1).
- **Joint optimisation:** 30 × |train| steps. Sparse Adam without bias correction: augmented rows fully trained, originals opacity-only (used Gaussians only). The view-dependent opacity lobe (Eq. 1) lives in the renderer.
- **Checks:** 8 probes pass in `evidence/verify.json`, 12 unit tests pass, and `evidence/test.json` has the smoke run. Reviewer review-1 approved with 5 minors; the float64 clustering fix was applied before launch.

## Deviations from paper / official code
1. White background and with-replacement view sampling. These are runtime-wide and apply to both base and method.
2. The photometric loss uses valid-convolution SSIM, where the official code uses fused-ssim with same padding. This is runtime-wide; the 2D stage uses the official fused-ssim.
3. The base is gsplat 3DGS (opacity lr 0.05, etc.), not the official train_3dgs.py: −0.32 dB vs the paper's implied base on bicycle, −0.05 dB on room.
4. 2D rasterization uses gsplat (alpha clamp 0.999, exclusive termination) instead of the Inria 2D rasterizer (0.99, inclusive). Pixel-coverage counts use the Inria rules.
5. "Used" original Gaussians are detected as those with a nonzero opacity gradient. The official code uses a per-pixel `is_used` flag from its rasterizer.
6. Cluster ties (when the pivot's cluster has fewer than 3 points) are broken by the smallest point index, not by sklearn's label order.
7. Edge cases are guarded where the official code would hang or abort:
   - each round adds at least 1 kernel (official code loops forever for per-view targets ≤ 2);
   - a scale-fit assertion failure gives that kernel a tiny scale instead of dropping the whole view (none occurred).
8. Back-projection uses the full intrinsics K; the official code assumes a centred principal point. The two are identical for these cameras.
9. RNG streams differ, and both scenes ran a single seed (0).
