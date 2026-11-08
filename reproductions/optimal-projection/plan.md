# Plan — optimal-gs (plan id 123)

Method `optimal-gs`, module `ss_optimal_gs`, package `splatstudio-method-optimal-gs`.
Control `optimal-gs-ewa`: same CUDA rasterizer/semantics with the standard EWA projection
(planar pixel-space evaluation) — isolates the projection change from rasterizer semantics.

## Components
| kind | name | factory | requires / provides | config |
|---|---|---|---|---|
| renderer | optimal-renderer | `ss_optimal_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb, render.depth.expected_z, gradient.screen.signed | projection "optimal", focal_floor 512, eps2d 0.3, near 0.2, alpha_max 0.99, official_cov_grad true, tile 16 |
| density | adaptive-density (unchanged gsplat base, 3DGS opacity reset) | base | gradient.screen.signed | base (grow_grad2d 2e-4) |
| loss | photometric (unchanged) | base | render.rgb | — |

No new fields. Schedules: none method-specific.

## Implementation (equations → code)
- `projection.py` (torch, autograd): t = R x + T; near cull z <= 0.2; means2d = K-projection
  (pixel, gsplat +0.5 convention = Inria ndc2Pix); Σ_cam = R Σ R^T (gsplat quat_scale_to_covar);
  J = diag(f'x,f'y)[e_u; e_v]/|t| (Eq. 14/15 premultiplied by Q Eq. 18, code computeCov2D);
  `official_cov_grad`: custom autograd.Function whose backward uses the 18 official
  coefficients verbatim (incl. the j12_x typo); else exact autograd. Σ_2D = JΣJ^T + 0.3I, conic.
  Radius: naive EWA mid + optimal det (code preprocess). radii=0 when rect has no tile.
- `csrc/sphere_raster.cu` (adapted from reproductions/linear-splatting csrc/linear_raster.cu,
  itself gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu; per-pixel math from official
  forward.cu/backward.cu renderCUDA): pixel ray t̂, Gaussian ray μ̂ from mean2d, tangent coords
  (u,v) (Eq. 16), Inria α rules (0.99 clamp with straight-through gradient, 1/255, stop
  T(1-α)<1e-4), back-to-front backward; d(u,v)/d(mean2d) via forward-mode dual numbers.
  `projection="ewa"` evaluates d = mean - pixel instead.
- SH: gsplat spherical_harmonics, clamp_min(c+0.5, 0). Expected depth as a 4th channel.

## Probes
| probe | measures | fails if |
|---|---|---|
| tangent-jacobian | J vs official trig formula (verbatim) and vs paper Q·J_p (Eq. 14,18); differs from EWA J off-axis | J replaced by EWA / wrong basis |
| cov-gradient | custom backward vs autograd: 17 coefficients exact, j12_x matches the official typo | backward wrong |
| raster-forward-vs-dense | CUDA RGB/alpha/depth vs float64 dense reference written from official trig formulas | per-pixel tangent evaluation missing/wrong |
| raster-backward-vs-dense | CUDA grads (means2d, conics, colours, opacities) vs autograd of the dense reference | backward wrong |
| projection-error | off-axis wide-FOV splats: optimal render closer to exact ray-max 3D Gaussian response than EWA (control) | mechanism removed |
Base probes (state-transitions, photometric-gradient, renderer-gradient) kept.

## Fidelity gaps
1. Runtime: white background (official black), gsplat DefaultStrategy densification, sampling
   with replacement, SSIM valid conv — shared with the baseline; compare deltas.
2. Baseline gsplat uses α clamp 0.999 / gated gradient and near 0.01; this method uses the
   official Inria 0.99 / ungated / 0.2. The `optimal-gs-ewa` control isolates this.
3. Tile assignment uses gsplat isect rule (floor/ceil) instead of Inria getRect (pixel centre
   offset by 0.5); bounding heuristic only, the radius formula itself is the official one.
4. Pixel/mean rays use per-camera cx, cy from K (official assumes centred principal point).
5. Basis computed algebraically instead of atan2/sin/cos (equal up to float rounding; probe).
