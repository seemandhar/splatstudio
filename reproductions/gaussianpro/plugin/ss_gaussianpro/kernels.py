"""JIT builds of the vendored, unmodified official CUDA sources (see csrc/README.md).

* ``propagation()``: GaussianPro ``submodules/Propagation`` (``propagate``: ACMH checkerboard
  PatchMatch of plane hypotheses seeded by a rendered depth; returns [4,H,W] = depth + world
  normal of the reference view). Built like the official setup.py (``-O3``), sm_86 via
  ``TORCH_CUDA_ARCH_LIST`` (tools/ssenv).
* ``simple_knn()``: Inria simple-knn ``distCUDA2`` (mean squared distance to the 3 nearest
  neighbours, Morton-box approximation), used by the official code for new-Gaussian scales.
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def propagation():
    from torch.utils.cpp_extension import load

    src = CSRC / "propagation"
    return load(
        name="ss_gaussianpro_propagation",
        sources=[str(src / f) for f in ("PatchMatch.cpp", "Propagation.cu", "pro.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[str(src)],
        verbose=False,
    )


@functools.cache
def simple_knn():
    from torch.utils.cpp_extension import load

    src = CSRC / "simple_knn"
    return load(
        name="ss_gaussianpro_simple_knn",
        sources=[str(src / f) for f in ("spatial.cu", "simple_knn.cu", "ext.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[str(src)],
        verbose=False,
    )


def dist_cuda2(points):
    """simple-knn distCUDA2 on [N,3] float32 CUDA points -> [N] mean squared 3-NN distance."""
    return simple_knn().distCUDA2(points.float().contiguous())


def propagate(images, intrinsics, poses, depth, depth_intervals, patch_size):
    """Official ``propagate`` call (utils/graphics_utils.depth_propagation).

    images [V,H,W,3] uint8 (reference first), intrinsics [V,3,3], poses [V,4,4] world-to-camera,
    depth [H,W] rendered depth of the reference view, depth_intervals [V,4]. Images and depth
    are passed as host tensors: the kernel copies them with cudaMemcpy*HostToDevice.
    The kernel calls cudaSetDevice(0), so the current CUDA device must be 0.
    Returns (depth [H,W], world normal [H,W,3]) on the CPU.
    """
    if torch.cuda.current_device() != 0:
        raise RuntimeError("The official propagation kernel runs on CUDA device 0 "
                           "(cudaSetDevice(0)); expose the training GPU as device 0")
    images = images.detach().to("cpu", torch.uint8).contiguous()
    depth = depth.detach().to("cpu", torch.float32).reshape(*depth.shape[:2], 1).contiguous()
    normal = torch.zeros_like(depth)
    out = propagation().propagate(images, intrinsics.detach().float().cpu(),
                                  poses.detach().float().cpu(), depth, normal,
                                  depth_intervals.detach().float().cpu(), int(patch_size))
    return out[0], out[1:4].permute(1, 2, 0)
