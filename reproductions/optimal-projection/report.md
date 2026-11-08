# Optimal Gaussian Splatting (plan 123) — reproduction report

Paper: "On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy"
(Huang et al., ECCV 2024, arXiv 2402.00752). Method `optimal-gs`, control `optimal-gs-ewa`.
Plugin `plugin/` is a port of the official op43dgs pinhole rasterizer (Inria licence, cited):
- tangent-plane projection with the official closed-form covariance gradient;
- unit-sphere rasterizer, as a JIT CUDA extension on gsplat 1.5.3 tile lists.

**Outcome: matched.** Both reported scenes are within ±0.5 dB. The projection gain we
measure is small, though (see below).

## Results (Mip-NeRF 360, 30k steps, llffhold8, images_2 room / images_4 treehill)

Runs: `runs/optimal-gs-mipnerf360`. Baseline: `runs/baseline3dgs-mipnerf360-30000`.

| scene | paper Ours | optimal-gs | Δ vs paper | control optimal-gs-ewa | gsplat baseline | paper 3DGS row |
|---|---|---|---|---|---|---|
| room | 31.58 | **31.41** | −0.17 | 31.39 | 31.52 | 30.63 |
| treehill | 22.77 | **22.39** | −0.38 | 22.19 | 22.28 | 22.49 |

SSIM and LPIPS:
- room: optimal-gs 0.916 / 0.298, control 0.916 / 0.298, gsplat 0.916 / 0.298.
- treehill: optimal-gs 0.628 / 0.364, control 0.626 / 0.365, gsplat 0.627 / 0.361.

The paper gives per-scene PSNR only (its SSIM/LPIPS tables are commented out in the LaTeX).

Gaussians (M): room 1.68 / 1.69 / 1.73, treehill 4.89 / 4.97 / 5.07 (optimal-gs / control / gsplat).

Training time (h): room 8.1 / 3.7 / 2.2, treehill 7.3 / 6.9 / 1.5. GPUs were shared, so these
times are only indicative.

### Gains (projection effect)

| scene | optimal − control (like-for-like) | optimal − gsplat | paper Ours − paper 3DGS |
|---|---|---|---|
| room | +0.02 | −0.11 | +0.95 |
| treehill | +0.20 | +0.11 | +0.28 |

How to read the gains:
- **The control is the like-for-like comparison.** `optimal-gs-ewa` uses the same kernel and
  the Inria blending rules, with the EWA projection and planar evaluation. That is the paper's
  own comparison: optimal vs EWA on the Inria rasterizer.
- **The paper's 3DGS row is stale.** It equals the published Inria numbers digit for digit; it
  was not retrained.
- **Room:** our gsplat baseline (31.52) already reaches the paper's method number, so the
  paper's +0.95 room gain is mostly baseline offset. We measure +0.02.
- **Treehill** is the cleaner test (baseline offset −0.21). Our +0.20 is consistent with the
  paper's +0.28.
- **Two scenes cannot establish the dataset-level gain.** The paper's M360 average gain is
  +0.27. Counter (cleanest indoor signal: paper gain +0.35, baseline offset +0.15) would be the
  next scene to add.

## Verification

- `ss-inspect`: no issues. 7 unit tests pass; ruff clean.
- `ss-verify`: `declared_checks_passed`, 8/8 probes:
  - the projection Jacobian equals both the official atan2 formula and the paper's Q·J_p;
  - the official coefficients equal autograd except j12_x;
  - CUDA forward and backward match a float64 dense reference (errors < 4e-6 and < 1e-6);
  - off-axis splats under a wide field of view: optimal alpha is 2–6× closer than EWA to the
    exact ray maximum of the 3D Gaussian.
- Smoke (garden, 2k steps): optimal-gs 23.06, control 23.06, gsplat 22.98.
- Review: `reviews/review-1.md` approved (0 critical, 0 major code issues, 4 minor). Its
  protocol requirement (run the control) was met.

## Deviations from the paper and official code

1. **Covariance-gradient typo kept on purpose (default `official_cov_grad: true`).** The
   official `backward.cu` j12_x term has `2*y2*(x2*z2)` where the exact derivative has
   `(x2+z2)`; we keep it to match the code that produced the paper's numbers. Effect: the
   gradient-vector cosine vs the exact gradient is ≥ 0.9994 on room and treehill (review §2).
   `official_cov_grad: false` gives the exact gradient.
2. **The mean-2D gradient is exact, not the official one.** The official
   d(u,v)/d(mean2D) puts the unit pixel ray into a formula derived for the z=1 ray, so it is
   not the exact derivative. Our kernel computes the exact derivative with dual numbers. The
   difference in |dL/dmean2D| is: median ≤ 0.3%, p95 ≤ 1.4% at the image edges, angle ≤ 0.8°;
   it slightly changes densification statistics at the image edges. Documented, not replicated.
   The earlier comment in `sphere_raster.cu` / `plan.md` that called the official version "the
   same derivative" was wrong.
3. **White background is hard-coded** (`renderer.py`, runtime convention); the official code
   uses black. The method, control and baseline share it, so the deltas are unaffected. There
   is no `background` option yet.
4. **Runtime-wide differences:** gsplat DefaultStrategy densification (with the 3DGS opacity
   reset fix), random view sampling with replacement, SSIM with valid convolution,
   LPIPS-VGG from the `lpips` package.
5. **Inria blending rules vs the gsplat baseline.** The method follows the official rules:
   alpha clamp 0.99 with ungated gradient, near plane 0.2, stop when T < 1e-4. The gsplat
   baseline uses a 0.999 gate and near 0.01. This is why the control is the like-for-like
   comparison.
6. **Tile rectangles** follow the gsplat isect rule (floor/ceil), not Inria `getRect`. The
   radius formula is the official one (naive-EWA mid with the optimal det, 3σ).
7. **Smaller differences:**
   - the principal point is taken from K (official code assumes W/2; identical here);
   - the tangent basis is computed algebraically instead of with atan2 (checked by a probe);
   - the f' = max(512, f) floor is kept (a no-op here: fx = 1586 room, 1054 treehill);
   - an expected-depth 4th channel is rendered (not in the official code; unused by any loss,
     about 25% extra raster bandwidth).
8. **Speed:** about 2.4–3.7× slower than gsplat (torch projection plus per-pixel tangent
   evaluation). Not optimised; the paper itself notes its CUDA implementation is slow.
