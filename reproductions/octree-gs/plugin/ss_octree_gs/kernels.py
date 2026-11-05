"""JIT-built simple-knn (`distCUDA2`): mean squared distance to the 3 nearest neighbours, used by Octree-GS's
create_from_pcd for the initial anchor scaling.

Vendored unmodified from the official Octree-GS repository (github.com/city-super/Octree-GS @ 9db64fe,
submodules/simple-knn: simple_knn.cu, simple_knn.h, spatial.cu, spatial.h, ext.cpp; Copyright (C) 2023 Inria
GRAPHDECO, Gaussian-Splatting licence, csrc/simple_knn/LICENSE.md, non-commercial research use).
`-include cfloat` supplies FLT_MAX, which newer toolkits no longer pull in transitively (the source is untouched).
Built on first use by torch.utils.cpp_extension and cached in ~/.cache/torch_extensions.
"""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc" / "simple_knn"


@functools.cache
def simple_knn():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_octree_gs_simple_knn",
        sources=[str(CSRC / "spatial.cu"), str(CSRC / "simple_knn.cu"), str(CSRC / "ext.cpp")],
        extra_include_paths=[str(CSRC)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "-include", "cfloat"],
        verbose=False,
    )


def dist_cuda2(points):
    """simple_knn._C.distCUDA2(points) -> [N] mean squared 3-NN distance (float32, CUDA)."""
    return simple_knn().distCUDA2(points.float().contiguous())
