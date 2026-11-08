"""JIT build of the vendored PPISP CUDA extension (github.com/nv-tlabs/ppisp @ a247368, Apache-2.0).

Sources and nvcc flags follow the official ``setup.py`` (``-O3 --use_fast_math``; arch from
``TORCH_CUDA_ARCH_LIST``, set by tools/ssenv). The only change to ``__init__.py`` is that ``_C``
resolves to this JIT module instead of the ``ppisp_cuda`` wheel.
"""

import functools
from pathlib import Path

HERE = Path(__file__).resolve().parent


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_neural_appearance_ppisp",
        sources=[str(HERE / "src" / "ppisp_impl.cu"), str(HERE / "ext.cpp")],
        extra_include_paths=[str(HERE)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
