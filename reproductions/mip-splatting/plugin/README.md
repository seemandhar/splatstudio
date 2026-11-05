# splatstudio-method-mip-splatting

Native SplatStudio port of **Mip-Splatting: Alias-free 3D Gaussian Splatting** (Yu et al.,
CVPR 2024, [arXiv 2311.16493](https://arxiv.org/abs/2311.16493)). Numerics follow the official
code at commit `1ecf7d5` (paper era, before the GOF densification update). Paper fidelity is
unverified until the benchmark is run.

## Mechanism
- **3D smoothing filter** (Eq. 8–9): per Gaussian, `filter_3d = sqrt(0.2) * d_min / f_max`,
  where `d_min` is the smallest camera-z depth over training cameras whose frustum (projected
  centre within ±15 % of the image, z > 0.2) contains the centre and `f_max` the largest
  focal_x; never-visible Gaussians use the largest visible depth. Rendering uses
  scales `sqrt(s² + f²)` and opacity `σ(o)·sqrt(Πs² / Π(s² + f²))`; `filter_3d` is a constant
  per-Gaussian field (saved in the checkpoint, used at test time).
  Recomputed at initialisation, after each densification step (600…14900) and every 100 steps
  in 15100…29800. Opacity reset (every 3000 steps until 15000) clamps the *filtered* opacity.
- **2D Mip filter** (Eq. 10): gsplat `rasterize_mode="antialiased"` with `eps2d=0.1` — exactly
  the official `computeCov2D` (Σ2D + 0.1 I, opacity × sqrt(det ratio), differentiated). The
  built-in `gsplat-aa` uses eps2d 0.3 and has no 3D filter.
- Everything else is the gsplat 3DGS baseline (DefaultStrategy, L1 + 0.2 D-SSIM).
No custom CUDA is needed.

## Opacity reset
Exactly one reset per reset step (3000, 6000, 9000, 12000): the controller wraps gsplat's
`DefaultStrategy` directly (not `splatstudio.backends.gsplat.Density`, whose unfiltered reset
would double it), gsplat 1.5.x's own reset never fires, and the controller refuses to start on a
gsplat whose reset works. Probe `single-reset` fails on a missing, unfiltered or doubled reset.

## Config (paper/official values)
renderer `kernel_size_2d 0.1`; density `filter_coef 0.2, near 0.2, frustum_margin 0.15,
filter_interval 100, densify_until 15000, total_steps 30000, opacity_reset_every 3000,
opacity_reset_value 0.01` + DefaultStrategy `grow_grad2d 2e-4`.

## Fidelity gaps
- **Numerical deviation (reset/coef)**: the official det-ratio coefficient prod(s²)/prod(s²+f²)
  underflows to 0 in float32 for Gaussians much smaller than their filter, and its reset
  logit(min(σ·c, 0.01)/c) then gives 0/0 = NaN (and +inf for saturated σ = 1). This crashed the
  first benchmark at the 3000/6000 reset. We compute the coefficient in log space and evaluate
  the reset as logit(clamp(min(σ, 0.01/c), 1e-6, 1-1e-6)), which is algebraically identical wherever
  the official formula is finite (unit test `test_reset_finite_in_failure_regime`).
- Training cameras arrive through the runtime hook `set_training_cameras` (train split only);
  the filter is computed there, and `before_backward` raises if the hook was never called.
- Paper images are PIL-resized from full resolution (`-r 4/-r 2`); our data are `images_4/_2`.
- Shared runtime differences: white background, random-with-replacement sampling,
  densification after the optimiser step, alpha clamp 0.999.
- Visibility projection uses K's principal point (official assumes W/2, H/2; identical here).

## Commands
```bash
pip install -e .
ss-inspect mip-splatting && ss-verify mip-splatting --device cuda
ss-train mip-splatting --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs --num-gaussians 10000000
pytest -q tests
```
