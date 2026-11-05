# splatstudio-method-improving-densification

Native SplatStudio port of **Improving Densification in 3D Gaussian Splatting for High-Fidelity
Rendering** (ImprovedGS, arXiv 2508.12313), from the official code
https://github.com/XiaoBin2001/Improved-GS (20b5db3). Paper fidelity unverified.
Non-commercial research use only (ImprovedGS + Inria 3DGS licences, see `csrc/VENDORED.md`).

Method `improving-densification` = official rasterizer (vendored unmodified, JIT-built on first
use, ~1 min) + ImprovedGS density + base photometric loss.

| mechanism | code | values (source) |
|---|---|---|
| EAS (Eq. 5-7) | `density._eas_scores`, `core.edge_map` | Laplacian edge maps, 10 views/densify (all at 3400/6400), per-view min-max, abs-grad τ = 0.0003 (App. A.2) |
| LAS (Eq. 8) | `core.long_axis_children` | d = 0.45, long ×(1−d), short ×√(1−d²), opacity ×0.6 |
| GC (Eq. 9) | `core.budget` | √ curve 500→14500; N_max 3M (bicycle/garden/stump), 1.5M (flowers/treehill/drjohnson/truck), 1M others; config `budget` overrides |
| RAP | `density._rap` | prune op<0.02 @300; reset to 0.05 @3k..12k; prune lowest 20 % @3300/6300 |
| MU | `optim.GatedAdam`, `core.mu_update` | summed-grad steps every 5 from 20000, every 20 from 24000 (code; paper text 15000/22500) |
| lrs | `density.xyz_lr` | position 4e-5→2e-6 ×extent, opacity 0.025 |

The density reloads the training images from `ss-train --data` (or config `data`) for the
edge maps. Use `--recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000`.

Fidelity gaps: runtime view sampling with replacement; runtime SSIM (valid conv); torch Adam
instead of the official CUDA Adam (dense, same β/eps); single seed vs paper's best-of-3;
data-resolution ambiguity (images_4/images_2 used). See ../report.md.

```bash
tools/ssenv ss-train improving-densification --data ~/datasets/mipnerf360/native/garden \
  --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
