# splatstudio-method-optimal-gs

"On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy"
(Huang et al., ECCV 2024, https://arxiv.org/abs/2402.00752). Native SplatStudio port of the
official code https://github.com/LetianHuang/op43dgs (pinhole rasterizer; Inria
Gaussian-Splatting licence, non-commercial research use). Paper fidelity unverified.

## Mechanism
- `projection.py`: J = diag(f', f') [e_u; e_v] / |t| (Q J_p, Eq. 14/15/18; official
  `computeCov2D`), f' = max(512, f); Σ_2D = J Σ_cam Jᵀ + 0.3 I. By default the gradient of J
  w.r.t. the camera-space mean uses the official closed form (`computeCov2DCUDA`), including its
  `j12_x` typo (`official_cov_grad: false` gives the exact gradient). Tile radius as in the
  official preprocess (naive-EWA mid, optimal det, 3 sigma). Near plane 0.2.
- `csrc/sphere_raster.cu` (JIT): per-pixel tangent-plane coordinates (Eq. 16-17, official
  `renderCUDA`), Inria blending (alpha clamp 0.99, gradient through the clamp, 1/255,
  T < 1e-4 stop), exact d(u,v)/d(mean2d) via dual numbers. Tiling adapted from
  reproductions/linear-splatting (gsplat 1.5.3 RasterizeToPixels3DGS).
- Training: unmodified base gsplat density (3DGS schedule + opacity reset) and L1 + 0.2 D-SSIM.

`optimal-gs-ewa` is a control: same kernel/blending with the EWA projection.

## Config (renderer)
projection "optimal"|"ewa", focal_floor 512, eps2d 0.3, near 0.2, alpha_max 0.99,
official_cov_grad true, tile_size 16 — all from the official code.

## Fidelity gaps
White background and gsplat DefaultStrategy densification / sampling (runtime-wide);
Inria blending rules differ from the gsplat baseline (use the control); gsplat tile-rect rule;
principal point from K; algebraic instead of atan2 basis (probe-checked).

## Commands
    tools/ssenv pip install -e reproductions/optimal-projection/plugin
    tools/ssenv ss-verify optimal-gs --device cuda
    tools/ssenv python -m pytest -q reproductions/optimal-projection/plugin/tests
