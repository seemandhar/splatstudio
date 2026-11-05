# Vendored CUDA (unmodified)

Source: https://github.com/turandai/gaussian_surfels commit 01d1249 ("dual visible").

* `cuda_rasterizer/`, `rasterize_points.{cu,h}`, `ext.cpp`, `third_party/glm/glm` —
  `submodules/diff-gaussian-rasterization` (Gaussian Surfels rasterizer: zero-thickness surfels,
  per-pixel tangent-plane depth, rendered normals/opacity, x10 normal gradient in backward.cu:727).
* `utils/cuda_utils.{cu,h}`, `utils/ext.cpp` — `utils/` of the same repo (occupancy grid for the
  volumetric cutting, `gaussian2occgrid`).

Every file carries the Inria GRAPHDECO header: "free for non-commercial, research and evaluation
use under the terms of the LICENSE.md file" — the Gaussian-Splatting License (copy in
`LICENSE-GaussianSplatting.md`; the gaussian_surfels repo ships no separate licence file).
GLM: MIT/Happy Bunny licence (`third_party/glm/copying.txt`).
Files are byte-identical to the upstream commit (checksums in `SHA256SUMS`).
