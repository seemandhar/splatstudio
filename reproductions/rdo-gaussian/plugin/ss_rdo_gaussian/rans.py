"""rANS entropy coding of codebook indices, as the official `utils/entropy_model.py` +
`utils/encode_utils.py` (DiscreteUnconditionalEntropyModel with a Softmax prior).

The coder itself is the vendored CompressAI rANS C++ code (csrc/, see csrc/README.md), built on
first use as a CPU torch cpp_extension. Table construction follows the official
`_build_tables`: pmf = softmax(logits); an overflow (escape) bin of mass max(0, 1 - sum pmf) is
appended; `pmf_to_quantized_cdf(·, precision=16)`; cdf length = K + 2, offset 0; all symbols
use cdf index 0 (one unconditional distribution per attribute).
"""

import functools
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_rdo_gaussian_rans",
        sources=[str(CSRC / "bindings.cpp"), str(CSRC / "rans_interface.cpp"),
                 str(CSRC / "ops.cpp")],
        extra_include_paths=[str(CSRC)],
        extra_cflags=["-O3"],
        verbose=False,
    )


def quantized_cdf(logits, precision=16):
    """Official `_build_tables` for a [1, K] (or [K]) logits tensor -> (cdf list, cdf length)."""
    logits = torch.as_tensor(logits, dtype=torch.float32).reshape(1, -1).cpu()
    pmf = F.softmax(logits, dim=-1)[0]
    overflow = (1.0 - pmf.sum(dim=0, keepdim=True)).clamp_min(0)
    p = torch.cat([pmf, overflow], dim=0)
    cdf = ext().pmf_to_quantized_cdf(p.tolist(), int(precision))
    return [int(c) for c in cdf], len(pmf) + 2


def encode(symbols, logits, precision=16):
    """Integer symbols in [0, K) -> rANS bytes (official `entropy_coding`).

    An empty attribute (e.g. no Gaussian keeps SH degree 3) is the empty string: the CompressAI
    encoder asserts on an empty buffer, so the official code cannot encode it at all.
    """
    sym = torch.as_tensor(symbols).reshape(-1).int().tolist()
    if not sym:
        return b""
    cdf, length = quantized_cdf(logits, precision)
    return ext().RansEncoder().encode_with_indexes(sym, [0] * len(sym), [cdf], [length], [0])


def decode(string, count, logits, precision=16):
    """Inverse of `encode` (official `entropy_decoding`) -> int64 tensor [count]."""
    if int(count) == 0:
        return torch.zeros(0, dtype=torch.int64)
    cdf, length = quantized_cdf(logits, precision)
    values = ext().RansDecoder().decode_with_indexes(string, [0] * int(count), [cdf], [length],
                                                     [0])
    return torch.tensor(values, dtype=torch.int64)


# --- official encode_utils.py byte packing ------------------------------------------------------
def pack_strings(strings):
    out = b""
    for s in strings:
        out += struct.pack(">I", len(s)) + struct.pack(f">{len(s)}s", s)
    return out


def unpack_strings(stream, n):
    strings = []
    for _ in range(n):
        (length,) = struct.unpack(">I", stream[:4])
        strings.append(bytes(stream[4 : 4 + length]))
        stream = stream[4 + length :]
    return strings, stream


def pack_header(boundaries, step_size, min_opacity):
    """header.bin: boundaries as big-endian uint32, then step size and min opacity as float32."""
    b = [int(v) for v in boundaries]
    return struct.pack(f">{len(b)}I", *b) + struct.pack(">2f", float(step_size),
                                                         float(min_opacity))


def unpack_header(data):
    """Official decode_header: 1 boundary if the header is 12 bytes, else 8."""
    n = 1 if len(data) == 12 else 8
    boundaries = struct.unpack(f">{n}I", data[: 4 * n])
    step_size, min_opacity = struct.unpack(">2f", data[4 * n : 4 * n + 8])
    return list(boundaries), step_size, min_opacity
