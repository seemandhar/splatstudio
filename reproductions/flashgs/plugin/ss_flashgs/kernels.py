"""JIT builds of the vendored CUDA sources (compiled once, cached by torch.utils.cpp_extension).

* ``flashgs()``  -> the official FlashGS extension (InternLandMark/FlashGS @ cdfc4e40, MIT),
  sources in ``csrc/`` byte-identical to upstream, same compiler flags as its setup.py.
* ``ref3dgs()``  -> the original 3DGS rasterizer (graphdeco-inria/diff-gaussian-rasterization
  @ 59f5f77e, Gaussian-Splatting licence, research/evaluation use), sources in
  ``ref3dgs_csrc/`` unmodified. Used only as the paper's speed/quality reference (forward only).

Run through ``tools/ssenv`` so CUDA_HOME / gcc-11 / TORCH_CUDA_ARCH_LIST are set.
"""

import os
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLASHGS_SRC = HERE / "csrc"
REF3DGS_SRC = HERE / "ref3dgs_csrc"


def _build_dir(name):
    root = Path(os.environ.get("SS_FLASHGS_BUILD", Path.home() / ".cache" / "ss_flashgs"))
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@cache
def flashgs():
    from torch.utils.cpp_extension import load

    sources = sorted(str(p) for p in (FLASHGS_SRC / "cuda_rasterizer").glob("*.cu"))
    sources.append(str(FLASHGS_SRC / "pybind.cpp"))
    # Upstream setup.py: CXX ["-g", "-O1", "-std=c++17"], NVCC ["-O1", '-Xptxas="-O1"'].
    return load(
        name="flash_gaussian_splatting",
        sources=sources,
        extra_cflags=["-O1", "-std=c++17"],
        extra_cuda_cflags=["-O1", "-Xptxas=-O1"],
        build_directory=_build_dir("flashgs"),
        verbose=False,
    )


@cache
def ref3dgs():
    from torch.utils.cpp_extension import load

    cr = REF3DGS_SRC / "cuda_rasterizer"
    sources = [str(cr / f) for f in ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [str(REF3DGS_SRC / "rasterize_points.cu"), str(REF3DGS_SRC / "ext.cpp")]
    # Upstream setup.py: nvcc -I third_party/glm/ (default optimisation otherwise).
    return load(
        name="ss_flashgs_ref3dgs",
        sources=sources,
        extra_cuda_cflags=["-I" + str(REF3DGS_SRC / "third_party" / "glm")],
        build_directory=_build_dir("ref3dgs"),
        verbose=False,
    )
