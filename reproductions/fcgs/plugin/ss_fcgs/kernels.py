"""JIT-built CUDA extensions used by the FCGS codec (all vendored, see csrc/README.md).

- csrc/gridcreater: FCGS's grid creator (Eq. 4: trilinear-weighted average of decoded features
  into dense 3-D / 2-D grids; float-atomic and deterministic integer-atomic variants).
- csrc/gridencoder: FCGS's dense multi-resolution grid interpolation (Eq. 5).
- csrc/freqencoder: sin-cos positional embedding `emb` (Eq. 5).
- csrc/arithmetic: chunked CUDA arithmetic coder (torchac-derived), byte-identical between
  FCGS and HAC; copied from reproductions/hac/plugin/ss_hac/csrc/arithmetic.

Built on first use by torch.utils.cpp_extension and cached in ~/.cache/torch_extensions.
"""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"
_NVCC = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
         "-U__CUDA_NO_HALF2_OPERATORS__"]


def _load(name, folder, sources, include=False, nvcc=None):
    from torch.utils.cpp_extension import load

    src = CSRC / folder
    return load(
        name=f"ss_fcgs_{name}",
        sources=[str(src / s) for s in sources],
        extra_include_paths=[str(src)] if include else None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=nvcc or _NVCC,
        verbose=False,
    )


@functools.cache
def gridcreater():
    return _load("gridcreater", "gridcreater", ["gridcreater.cu", "bindings.cpp"])


@functools.cache
def gridencoder():
    return _load("gridencoder", "gridencoder", ["gridencoder.cu", "bindings.cpp"])


@functools.cache
def freqencoder():
    return _load("freqencoder", "freqencoder", ["freqencoder.cu", "bindings.cpp"])


@functools.cache
def arithmetic():
    return _load("arithmetic", "arithmetic", ["arithmetic.cpp", "arithmetic_kernel.cu"],
                 include=True, nvcc=["-O2"])
