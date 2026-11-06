# 3DGUT (plan #81) — build report

**Status:** ready for benchmark. Method `3dgut`, plugin `reproductions/3dgut/plugin` (`ss_3dgut`).
Paper arXiv 2412.12507; official code nv-tlabs/3dgrut @ bb203b1 (read, not run).

## Implementation
- **Renderer**: gsplat 1.5.3 3DGUT kernels (UT projection `fully_fused_projection_with_ut` +
  3D max-response `rasterize_to_pixels_eval3d`, contributed to gsplat by the 3DGUT authors), driven
  directly so the paper's UT parameters (α=1, β=2, κ=0, any-sigma-point-valid) are used —
  `gsplat.rasterization(with_ut=True)` silently uses α=0.1 / require-all-valid. Distortion
  (OpenCV radial/tangential/thin-prism, fisheye, F-theta) and rolling shutter are supported through
  optional camera-dict keys; `ss_3dgut.cameras` loads distorted transforms.json plugin-side.
  No custom CUDA build.
- **Density**: port of official `GSStrategy` — statistic ‖∇μ·dist‖/2, densify every 300 (500–15000),
  clone/split threshold 2e-4, opacity prune every 100, reset every 3000, no screen/world prune;
  official COLMAP init (scale 0.01×nearest-camera distance, random rotations).
- **Loss**: 0.8 L1 + 0.2 D-SSIM (official code; the paper text says L2 + 0.2 SSIM).

## Verification
ss-verify: 9/9 probes passed (`evidence/verify.json`): UT moments == independent float64 Eq. 5–9
(cov rel err 1e-6; α=0.1 differs by 1%); UT closer to Monte-Carlo than EWA under fisheye
(KL 0.0010 vs 0.0092); eval3d alpha == ray-space Eq. 10 (4e-7), differs from 2D EWA by 0.21;
kernel gradients == float64 autograd reference (rel err <1e-6); fisheye render peak at analytic
projection (6e-4 px); densification statistic/routing/cadence; official init.

Smoke (garden, 2000 steps, recipe 3dgs): **22.81 dB** / SSIM 0.605 / LPIPS 0.443, 380k Gaussians,
204 s, vs gsplat baseline 23.07 dB, 843k Gaussians (fewer densification rounds at 300-step
cadence and the smaller official init scales, as the official code does).

## Benchmark request
Mip-NeRF 360 garden (paper 26.90; official README re-run 27.18) + counter (paper 29.03; re-run 29.14),
30k steps, vs `baseline3dgs-mipnerf360-30000`. Paper Δ vs 3DGS-paper per scene: garden −0.51,
counter +0.33.
Distorted-camera results (ScanNet++ fisheye, Waymo rolling shutter) not benchmarked: licensed
data, and the SplatStudio core loader rejects distorted cameras (needs a core camera-schema
extension; renderer side is ready and probed).

## Fidelity gaps
1. gsplat eval3d: max α 0.999 vs official 0.99; no min-kernel-response cut (0.0113 ≈ 3σ).
2. gsplat 3.33σ opacity-aware tile bounding vs official StopThePop culling (mostly speed).
3. White background (runtime) vs official black.
4. Distorted/rolling-shutter training not reachable through ss-train (core loader).
5. Sorted (k-buffer MLAB) variant and secondary-ray hybrid rendering not implemented.
6. Random view sampling with replacement; 1-based steps; scene extent from train cameras only.
7. Metric implementations (valid-conv SSIM, LPIPS-VGG) differ from the official evaluation.

## Review-1 resolution (orchestrator)
**Verdict:** approved, with 0 critical and 0 major issues. The benchmark is launched on garden and counter (`mipnerf360`).

**Targets:**
- **Paper (supp. Tab. 7):** garden 26.90, counter 29.03.
- **Official README re-run (secondary):** garden 27.18, counter 29.14.

**Notes from the review:**
- **UT-parameter fix.** It matches the official config, but on undistorted pinhole data it barely affects quality: RGB differs by at most 1.6e-3, because UT only drives tile bounding. No PSNR difference is attributed to it.
- **Distorted-camera claim.** It is validated only at renderer level, through the UT-vs-Monte-Carlo and fisheye probes. There is no training-level evidence.
- **Remaining gaps:**
  - init quaternions are normalised, while the official code stores the raw draw;
  - alpha clamps at 0.999 with no 0.0113 minimum response;
  - observer points and the scene extent use training cameras only;
  - the background is white.

## Benchmark result (3dgut-mipnerf360, 30k, images_4/2) — MATCHED
| scene | ours | paper (Ours) | Δ | gsplat baseline |
|---|---|---|---|---|
| garden | 26.92 / 0.850 / 0.148 | 26.90 / 0.851 / 0.121 | +0.02 | 26.89 |
| counter | 28.97 / 0.910 / 0.255 | 29.03 / 0.908 / 0.197 | −0.06 | 28.85 |
- Both scenes are within ±0.5 dB, and SSIM matches.
- LPIPS is higher because the runtime uses LPIPS-VGG and the paper a different net.
- As in the paper, the unscented transform costs nothing in quality vs EWA splatting on pinhole images (gain over our baseline +0.03 / +0.13).
- The distorted-camera and secondary-ray claims were not benchmarked.
