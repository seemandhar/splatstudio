# Vendored rasterizer

`cuda_rasterizer/`, `rasterize_points.cu`, `rasterize_points.h`, `ext.cpp`, `third_party/glm/` are
copied **unmodified** from `submodules/diff-gaussian-rasterization/` of
https://github.com/XiaoBin2001/Improved-GS at commit 20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a
(2026-06-05). The Python package `diff_gaussian_rasterization/` is not vendored; its
`_RasterizeGaussians` autograd wrapper is re-implemented in `../raster.py` with the same argument
order. `cuda_rasterizer/adam.cu` and `utils.cu` are compiled (ext.cpp binds them) but unused.

Licences: the rasterizer is Inria/MPII Gaussian-Splatting code (`LICENSE-3DGS-rasterizer.md`,
non-commercial research and evaluation use only) with ImprovedGS/TamingGS modifications; the
ImprovedGS repository licence is `LICENSE-ImprovedGS.md` (non-commercial research).
