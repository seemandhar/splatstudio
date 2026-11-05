"""JIT-built CUDA arithmetic coder (csrc/arithmetic: calculate_cdf / arithmetic_encode / arithmetic_decode).

Vendored unmodified from the official HAC repository (github.com/YihangChen-ee/HAC, commit 110dcf0a,
submodules/arithmetic.zip), via the HAC native port (reproductions/hac/plugin/ss_hac/csrc/arithmetic). The official
ContextGS codes the same Gaussian-conditional / Bernoulli models with torchac (CPU) and the hyperprior with
compressai's rANS coder; this coder takes the same float CDFs (fidelity gap, plan.md §6). Built on first use by
torch.utils.cpp_extension and cached in ~/.cache/torch_extensions.
"""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def arithmetic():
    from torch.utils.cpp_extension import load

    src = CSRC / "arithmetic"
    return load(
        name="ss_contextgs_arithmetic",
        sources=[str(src / "arithmetic.cpp"), str(src / "arithmetic_kernel.cu")],
        extra_include_paths=[str(src)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
