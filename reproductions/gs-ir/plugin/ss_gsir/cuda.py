"""JIT build of ``csrc/gsir_kernels.cu`` (torch.utils.cpp_extension; run through tools/ssenv)."""

from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"
_EXT = {}


def ext():
    if "k" not in _EXT:
        from torch.utils.cpp_extension import load

        _EXT["k"] = load(
            name="ss_gsir_kernels",
            sources=[str(CSRC / "gsir_kernels.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT["k"]
