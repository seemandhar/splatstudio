# splatstudio-method-gaussianpro

Native SplatStudio port of **GaussianPro: 3D Gaussian Splatting with Progressive Propagation**
(Cheng et al., ICML 2024, arXiv 2402.14650), following the official code
(github.com/kcheng1021/GaussianPro @ 86bb048). Paper fidelity unverified.

Method `gaussianpro` (Mip-NeRF 360 settings, dataset mode `360`).

## Mechanism → code
| paper | code |
|---|---|
| §4.1 depth/normal maps (Eq. 3-4) | `renderer.py` geometry pass: gsplat with colours [z, camera-facing shortest axis in camera frame], black bg, normal `F.normalize` |
| §4.2 propagation | `density.py` `propagate`: vendored kernel `csrc/propagation` (unmodified), geometric filter `geometry.check_geometric_consistency`, selection abs-rel > σ, `insert` (stride 8, distCUDA2 scales from vendored `csrc/simple_knn`) |
| §4.3 Eq. 8 / 9 | `losses.normal_loss` (β = 0.001 + 0.001), `losses.flatten_loss` (γ = 100) |
| §4.4 schedule / Eq. 10 | propagation in (1000, 12000), every 50 steps, each view once; opacity-masked photometric loss (`losses.masked_photometric`) |

Standard clone/split/prune/opacity reset is `splatstudio.backends.gsplat.Density` (identical
to our gsplat baseline). Neighbour views: 4 training views with most shared COLMAP points
(paper App. A), read from `images.bin` (found next to `--data` or in the converter's source
scene, e.g. `~/datasets/mipnerf360/360_v2/<scene>/sparse/0`; config `sparse_dir`). The density
reloads the training images from `--data` (runtime exposes cameras only).

## Config (defaults = paper/official)
Density: `propagation_interval 50`, `propagated_iteration_begin 1000`, `_after 12000`
(code default; demo.sh 6000), `depth_error_max/min_threshold 1.0/0.8`, `patch_size 20`,
`depth_min/max 0.1/20`, `num_neighbours 4`, `geo_pixel_threshold 2`, `geo_depth_threshold 0.01`,
`geo_min_views 2`, `min_new_pixels 100`, `backproject_stride 8`, `new_opacity_logit 20`
(sigmoid = 1.0 exactly: finite stand-in for the official +inf), `color_from_previous_view
true` (official variable reuse), `add_geometry_gradient true`. Loss: `lambda_l1_normal 0.001`,
`lambda_cos_normal 0.001`, `lambda_flatten 100`, opacity mask 0.999 before step 1000.

## Replicated official quirks (code > paper)
Kernel runs 6 propagation iterations (paper u = 3); the kernel ignores the rendered normal
(random initial normals at the rendered depth); new Gaussians fully opaque until the next
opacity reset; their colour comes from the previous step's GT image; neighbour depth caches are
never refreshed, a missing neighbour cache is propagated with the reference view's neighbour
list, and the reference cache is overwritten with 300 outside the inserted pixels; every
insertion zeroes the densification statistics.

## Fidelity gaps
gsplat rasterizer (alpha cap 0.999, straight-through opacity cap) instead of Inria's; white
background, sampling with replacement, valid-conv SSIM (runtime conventions shared with our
baseline); propagation at the end of step t instead of its start; neighbour graph regenerated
from shared SfM points (authors' pair files unreleased); the paper's Mip-NeRF 360 numbers use the
authors' re-run COLMAP (unreleased).

## Commands
```bash
pip install -e .
ss-verify gaussianpro --device cuda        # CUDA probes need the GPU exposed as device 0
ss-train gaussianpro --data ~/datasets/mipnerf360/native/room --output runs/x --recipe 3dgs \
  --num-gaussians 10000000 --max-gaussians 12000000
```
The first run JIT-compiles the two vendored extensions (~2.5 min, cached in
`~/.cache/torch_extensions`). Licences: `csrc/propagation/LICENSE-GaussianPro.md` (MIT file;
README: commercial use → contact the authors), `csrc/simple_knn/LICENSE-3DGS.md` (Inria).
