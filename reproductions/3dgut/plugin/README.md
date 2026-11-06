# splatstudio-method-3dgut

Native SplatStudio port of **3DGUT** (Wu et al., CVPR 2025, arXiv 2412.12507), unsorted "Ours"
variant. Official code: github.com/nv-tlabs/3dgrut @ bb203b1 (Apache-2.0), read for numerics.

| contribution | code |
|---|---|
| UT projection (Eq. 5-9), alpha=1 beta=2 kappa=0, any-sigma-point-valid | `renderer.py` → gsplat `fully_fused_projection_with_ut(ut_params=...)` |
| 3D max-response evaluation (Eq. 10), no gradient through projection | `renderer.py` → gsplat `rasterize_to_pixels_eval3d` |
| distorted / fisheye / rolling-shutter cameras | optional camera-dict keys (`renderer.camera_kwargs`), `cameras.py` loader |
| densification on ‖∇μ·dist‖/2, every 300 its, official COLMAP init | `density.py` (port of `threedgrut/strategy/gs.py`) |
| loss 0.8 L1 + 0.2 D-SSIM (code; paper text says L2) | SplatStudio `photometric` |

Why not `gsplat.rasterization(with_ut=True, with_eval3d=True)`: it does not forward `ut_params`
and runs gsplat's defaults (alpha=0.1, require_all_sigma_points_valid=True).

Config sources: `configs/render/3dgut.yaml`, `configs/strategy/gs.yaml`, `configs/base_gs.yaml`,
`configs/initialization/colmap.yaml`.

Fidelity gaps: max alpha 0.999 vs 0.99 and no 0.0113 min-response cut (gsplat kernels); gsplat
tile bounding vs StopThePop culling; white background (official black); `ss-train` can only feed
pinhole cameras (core loader) so distorted training needs a core camera-schema extension;
sorted k-buffer variant and secondary rays not implemented.

```bash
tools/ssenv pip install -e reproductions/3dgut/plugin
tools/ssenv ss-verify 3dgut --device cuda
tools/ssenv ss-train 3dgut --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs \
    --num-gaussians 10000000 --max-gaussians 12000000
```
