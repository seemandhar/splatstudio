# Vendored CUDA — official HDGS rasterizers

Source: https://github.com/TimSong412/HDGS
* `stage1/`  — branch `main` @ b13a359, `submodules/diff-surfel-rasterization`
  (`cuda_rasterizer/{auxiliary.h,backward.{cu,h},config.h,forward.{cu,h},rasterizer.h,rasterizer_impl.{cu,h}}`,
  `rasterize_points.{cu,h}`, `ext.cpp`): 2DGS surfels + per-ray k-buffer sort + per-tile
  centre-depth keys + 5-sample frustum average; SH colour at the surfel centre.
* `texture/` — branch `texture` @ 01c6fd4, same files + `tex.h`: bilinear per-surfel texture DC,
  SH at the intersection direction.
* `third_party/glm/` — GLM headers shipped in the same submodule (MIT / Happy Bunny,
  `third_party/glm-copying.txt`).

Only the setup.py-compiled sources are copied (the `*_origin.cu`, `*_nocone.cu`, ... variants
are not built upstream). **One modification**: in both `cuda_rasterizer/auxiliary.h`,
`#define SORT_WINDOW 24` is wrapped in `#ifndef SORT_WINDOW ... #endif` so the k-buffer size is
the build flag `-DSORT_WINDOW=k` (upstream README: edit the macro and recompile; 16 for
Mip-NeRF 360). `SHA256SUMS.upstream` lists the byte-identical upstream checksums (before the
guard was added).

Licence: Inria Gaussian-Splatting licence ("free for non-commercial, research and evaluation
use"), copy in `LICENSE-GaussianSplatting.md`.
