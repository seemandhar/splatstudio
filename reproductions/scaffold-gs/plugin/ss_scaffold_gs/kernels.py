"""JIT-built vendored simple-knn `distCUDA2` (Inria 3DGS licence, csrc/LICENSE-3DGS.md), used by Scaffold-GS's
anchor initialisation (scene/gaussian_model.py:create_from_pcd). Copied from the HGS native plugin
(reproductions/hard-gs/plugin/ss_hard_gs/csrc/simple_knn.cu)."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_scaffold_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "simple_knn.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def dist_cuda2(points):
    return ext().dist_cuda2(points.detach().float().contiguous())
