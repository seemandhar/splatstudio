# Vol3DGS — reproduction report

Paper: *Volumetrically Consistent 3D Gaussian Rasterization* (Talegaonkar et al., CVPR 2025, arXiv 2412.03378). Plugin `plugin/` (`splatstudio-method-vol3dgs`), method **`vol3dgs-black`**. Ported from the official code (Vol3DGS, MIT; slang-gaussian-rasterization @5eca118, Apache-2.0). Custom CUDA kernel `csrc/vol_raster.cu` (JIT; adapted from gsplat 1.5.3 RasterizeToPixels3DGS).

**Outcome: matched** (both reported scenes within ±0.5 dB PSNR).

## Results — Mip-NeRF 360 at width 1600 (official `-r -1` default), 30k steps, test split
| scene | PSNR | paper | Δ | SSIM (paper) | LPIPS (paper) | Gaussians (paper) | time | gsplat baseline PSNR* | gain |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | 25.00 | 25.04 | −0.04 | 0.760 (0.759) | 0.242 (0.216) | 4.11M (3.7M) | 313 min | 24.93 | +0.07 |
| room | 31.73 | 32.00 | −0.27 | 0.931 (0.936) | 0.267 (0.185) | 1.65M (1.6M) | 231 min | 31.52 | +0.21 |

Run: `runs/vol3dgs-mipnerf360-1600-30000` (summary.json). Paper: Supp. E per-scene table.

*The baselines `baseline3dgs-mipnerf360-1600-30000` (bicycle) and `-1600b-30000` (room) use a **white** background. The gains above therefore mix the method's effect with the black-vs-white background difference. The paper gives its 3DGS-Slang baseline only as an average (Vol3DGS is −0.22 dB below it on Mip-NeRF 360), so we cannot compare per-scene improvements.

LPIPS is higher than the paper on both scenes, most clearly on room (0.267 vs 0.185). This matches the runtime-wide LPIPS gap of our gsplat baselines (LPIPS-VGG from the `lpips` package, see `_calibration/`) and is not specific to the method.

## Verification
- `ss-verify vol3dgs-black`: declared_checks_passed.
- The kernel forward agrees with a float64 dense volumetric reference to about 3e-7.
- Kernel gradients and end-to-end parameter gradients agree with the reference to ≤1e-5 relative.
- Analytic line-integral checks pass:
  - the centre alpha is 0.99θ;
  - the off-axis profile follows the closed form;
  - stretching a Gaussian along the ray raises its alpha.
- Densification rules, learning rates and the SH schedule each have their own probe.
- 2k-step smoke run on bicycle-1600: 20.22 dB, against 21.15 dB for gsplat.

## Deviations from the paper / official code
1. **Mirrored code quirk (intentional):**
   - The official `density2alpha` (general_utils.py:56) evaluates `1 - exp(-x)/0.99` because of operator precedence, so it is not the inverse of `alpha2density`.
   - We reproduce it as written, because this code produced the paper's numbers.
   - Consequences: halved θ for clones and splits is about 0.0101 below a true halving, and the "split if θ ≥ 0.99" rule never fires.
   - The mathematically intended inverse is available as `density2alpha: inverse`; it has not been benchmarked.
2. Where the paper and code disagree, we follow the code:
   - κ is divided by 2.506; paper Eq. 20 omits this.
   - The split threshold is 1e-7; the paper says 1e-8.
   - Only the cloned copy has its density halved; the original keeps its θ.
3. A clamped θ = 0 maps to logit −30 instead of −inf, because the runtime requires finite parameters. Such Gaussians render with κ≈0 and are pruned at the next densification step.
4. Tile boxes come from gsplat's EWA projection, using the same 3σ + 0.3 formula as the slang vertex shader. Gaussians on the box boundary may be assigned to tiles differently.
5. The ray principal point is taken from K, not W/2, H/2. The two are identical for centred intrinsics.
6. The kernel is built with `--use_fast_math`. The NaN-gradient skip also drops inf gradients.
7. Runtime-wide differences:
   - training views are sampled with replacement;
   - SSIM uses valid convolution (the official code uses fused SSIM);
   - the optimizer steps before densification (Inria densifies first);
   - LPIPS comes from the `lpips` package.
8. The official screen-size prune never fires in the official code, so it is omitted. Opacity reset is disabled, as in the official script.

## Review
`reviews/review-1.md` raised 2 majors and 4 minors.
- Major 1 (density2alpha precedence) is fixed by mirroring the official expression.
- Major 2 (black background) is fixed: `vol3dgs-black` is the benchmarked method.
- The minors (fast-math/NaN skip, optimizer order, EWA tile boxes, docs) are documented in the plugin and listed above.
