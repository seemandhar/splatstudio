"""JIT-built CUDA extensions vendored from the official HAC repository (submodules/).

- csrc/gridencoder: HAC's multiresolution hash-grid encoder (torch-ngp derived, binary
  embeddings), used for the hash-grid context (paper §3.2, Eq. 2).
- csrc/arithmetic: HAC's chunked CUDA arithmetic coder (calculate_cdf / arithmetic_encode /
  arithmetic_decode), used to produce the real bitstream at the end of training.

Both are built on first use by torch.utils.cpp_extension and cached in
~/.cache/torch_extensions (outside the package, so the source manifest is stable).
"""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def grid():
    from torch.utils.cpp_extension import load

    src = CSRC / "gridencoder"
    flags = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__",
             "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
    return load(
        name="ss_hac_gridencoder",
        sources=[str(src / "gridencoder.cu"), str(src / "bindings.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=flags,
        verbose=False,
    )


@functools.cache
def arithmetic():
    from torch.utils.cpp_extension import load

    src = CSRC / "arithmetic"
    return load(
        name="ss_hac_arithmetic",
        sources=[str(src / "arithmetic.cpp"), str(src / "arithmetic_kernel.cu")],
        extra_include_paths=[str(src)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
