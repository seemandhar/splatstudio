# splatstudio-method-gof — Gaussian Opacity Fields (native port)

Yu, Sattler, Geiger, *Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in
Unbounded Scenes*, SIGGRAPH Asia 2024, arXiv 2404.10772. Official code:
github.com/autonomousvision/gaussian-opacity-fields @ 5245b20.

## Mechanism → code
| paper | code |
|---|---|
| Ray–Gaussian intersection volume rendering (§3.1) | `ss_gof/csrc` = official `diff-gaussian-rasterization` of GOF, unmodified, JIT-built (`raster.py`) |
| Opacity field O(x) (§3.2) | `raster.integrate`, `raster.opacity_field` (official `integrate` kernel, min over views) |
| Depth distortion, detached weights (§3.3) | kernel channel 8; `losses.create_regularizer`, λ=100 from step 15000 |
| Intersection-plane normal + depth-normal consistency (§3.3) | kernel channels 3-5, `raster.depth_to_normal`; λ=0.05 from 15000 |
| Gradient-norm densification metric (§3.3) | kernel `means2D.grad[:, 2]`; `density.Density.densify_and_prune` (quantile Q) |
| Mip-Splatting 3D filter, filter-aware reset, clone resampling (code/supp.) | `filters.py`, `density.py` |

Config defaults = official `arguments/__init__.py` and `train.py` (min_opacity 0.05, black
background, kernel_size 0). The official screen-size prune never fires (max_radii2D zeroed in
`densification_postfix`), so it is not applied.

## Fidelity gaps
- Runtime samples views with replacement; optimizer step precedes densification.
- Tetrahedral mesh extraction (CGAL Delaunay + binary-search marching tetrahedra) not ported.
- Decoupled appearance model (TnT only) not ported.
- 3D filter coefficient in log space (same value, NaN-safe).
- Official backward of the distortion drops the 1/(1-T)^2 normalisation (kept: official CUDA).

## Commands
```
tools/ssenv pip install -e reproductions/gof/plugin
tools/ssenv ss-verify gof --device cuda
tools/ssenv ss-train gof --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs --num-gaussians 10000000
```
Licence: vendored CUDA is under the Inria/MPII Gaussian-Splatting licence (non-commercial research), `ss_gof/csrc/LICENSE-GOF.md`.
