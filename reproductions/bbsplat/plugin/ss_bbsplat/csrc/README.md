# Vendored CUDA — official BBSplat rasterizer

Source: https://github.com/david-svitov/diff-bbsplat-rasterization @ 2dbc336db1591b19819451cc1b3f94b8dacfb7d8
(the `submodules/diff-bbsplat-rasterization` of https://github.com/david-svitov/BBSplat @ cff8d6a).
Copied **unmodified** (setup.py sources + headers): `cuda_rasterizer/{auxiliary.h, backward.{cu,h},
config.h, forward.{cu,h}, grid_sample.h, rasterizer.h, rasterizer_impl.{cu,h}}`,
`cuda_rasterizer/stopthepop_2DGS/*`, `rasterize_points.{cu,h}`, `ext.cpp`; `SHA256SUMS.upstream`
lists their checksums (verify with `sha256sum -c SHA256SUMS.upstream` from this directory).
Build flags are the upstream defaults (TILE_SORTING 0, PIXEL_RESORTING 0, FAST_INFERENCE 0).
`third_party/glm/` — GLM headers from the same repository (MIT / Happy Bunny, `glm-copying.txt`).

Licences: BSD-3-Clause for the BBSplat additions (`LICENSE-BBSplat.txt`) on top of the Inria
Gaussian-Splatting licence (non-commercial research and evaluation use, `LICENSE-GaussianSplatting.md`).
Built at first use with `torch.utils.cpp_extension.load` (see `ss_bbsplat/raster.py`).
