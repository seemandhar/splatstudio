"""JIT build of the vendored Faster-GS CUDA backend (``csrc/``).

``csrc/`` = ``FasterGSCudaBackend/FasterGSCudaBackend/{rasterization,adam,utils}`` of
github.com/nerficg-project/faster-gaussian-splatting @ 090e8ad (Apache-2.0, ``csrc/LICENSE``),
unmodified except ``bindings.cpp`` (filter3d / MCMC modules not bound). Compiler flags follow the
official ``setup.py`` (``ENABLE_FASTMATH = True``: ``-O3 -use_fast_math``, C++17).
"""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"
MODULES = ("rasterization", "adam")


def sources():
    files = [CSRC / "bindings.cpp"]
    for module in MODULES:
        files += sorted((CSRC / module / "src").rglob("*.cu"))
        files += sorted((CSRC / module / "src").rglob("*.cpp"))
    return [str(f) for f in files]


def include_dirs():
    return [str(CSRC / "utils")] + [str(CSRC / m / "include") for m in MODULES]


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_faster_gs_backend",
        sources=sources(),
        extra_include_paths=include_dirs(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-use_fast_math", "-std=c++17"],
        verbose=False,
    )
