"""JIT-built extension (csrc/): CUDA hash grid (tcnn conventions) and a CPU canonical-Huffman decoder.

Built on first use by torch.utils.cpp_extension and cached in ~/.cache/torch_extensions (outside the
package, so the source manifest stays stable).
"""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_compact_3dgs_ext",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "hashgrid.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
