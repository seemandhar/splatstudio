"""RDO-Gaussian final parameter encoding (paper §3.3) and its decoder.

Port of the official `GaussianModel.apply_and_save_vq` / `GaussianModel.decode` and
`utils/encode_utils.py` (https://github.com/USTC-IMCL/RDO-Gaussian, commit 710ebab8). The
bitstream is the official `bitstream/` directory, built byte for byte in memory:

  index_bitstream.bin  length-prefixed (>I) rANS strings of scale, rot, dc, sh1, sh2, sh3, opa
  header.bin           8 SH-mask cluster boundaries (>I) + opacity step size, min (>f)
  codebook.npz         np.savez of the shrunk codebooks, float32 [1, K', d]
  logits.npz           np.savez of the shrunk logits float32 [1, K'] + 'opa' log-histogram
                       (float64 [1, 256])
  position.npz         np.savez(position=float16 [N, 3])

The paper's rate (MB) is the total size of these files / 1e6 (official results csv
"Size [Bytes]"; Table 3 components header / indexes / codebooks / logits / positions).
"""

import io

import numpy as np
import torch

from . import rans
from .bank import KEYS, SH_KEYS

FILES = ("index_bitstream.bin", "header.bin", "codebook.npz", "logits.npz", "position.npz")
INDEX_KEYS = KEYS + ("opa",)


def shmask_sort(sh_bits):
    """Official shmask_sort: sort Gaussians by the 3-bit SH mask (sh1 = MSB) and return
    (order, 8 boundaries), boundaries[j] = first sorted index whose mask integer is > j for
    j < 7 and boundaries[7] = N. A stable sort is used (official np.argsort quicksort; the
    order inside a cluster does not change the decoded model)."""
    value = np.zeros_like(sh_bits["sh1"], dtype=np.int64)
    for key in SH_KEYS:
        value = (value << 1) | sh_bits[key].astype(np.int64)
    order = np.argsort(value, kind="stable")
    n = value.shape[0]
    padded = np.append(np.insert(value[order], 0, 0), 7)
    diff = np.diff(padded)
    boundaries = []
    for i in np.flatnonzero(diff):
        boundaries.extend([int(i)] * int(diff[i]))
    boundaries = np.array(boundaries, dtype=np.int64)
    if boundaries[-1] != padded.shape[0]:
        boundaries = np.append(boundaries, n)
    return order, boundaries


def sh_bits_from_boundaries(boundaries):
    """Official get_sh_bitmask (sorted order)."""
    n = int(boundaries[-1])
    value = np.zeros(n, dtype=np.int64)
    for i in range(7, -1, -1):
        value[: boundaries[i]] = i
    return {"sh1": (value & 4).astype(bool), "sh2": (value & 2).astype(bool),
            "sh3": (value & 1).astype(bool)}


def index_counts(boundaries):
    """Official get_index_shapes: number of coded indices per attribute."""
    b = [int(v) for v in boundaries]
    n = b[-1]
    if len(b) == 1:
        return {k: n for k in INDEX_KEYS}
    return {"scale": n, "rot": n, "dc": n, "sh1": b[7] - b[3],
            "sh2": b[7] - b[5] + b[3] - b[1],
            "sh3": b[7] - b[6] + b[5] - b[4] + b[3] - b[2] + b[1] - b[0], "opa": n}


def shrink(codebook, logits, index):
    """Official shrink_codebook: keep used codewords, re-index by searchsorted."""
    used = np.unique(index)
    return codebook[:, used], logits[:, used], np.searchsorted(used, index).astype(np.int64)


def opacity_quant(opacities, bits=8):
    """Official opacity_quant on logit opacities [N, 1] float32."""
    precision = 2**bits
    lo, hi = np.min(opacities), np.max(opacities)
    step = (hi - lo) / (precision - 1)
    if not step > 0:
        raise ValueError("opacity quantization needs at least two distinct opacities")
    q = np.round((opacities - lo) / step).astype(np.uint8 if bits <= 8 else np.int64)
    hist, _ = np.histogram(q, bins=precision, density=True)
    log_prob = np.expand_dims(np.log(hist + 1e-10), 0)
    return q, log_prob, step, lo


def savez_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def encode(attrs, precision=16, opacity_bits=8):
    """attrs (numpy, already GS-pruned): xyz [N,3] f32, opacity [N,1] f32 logits, sh_bits
    {sh1,sh2,sh3: bool [N]}, index {key: int [M_key]} (rows of `sh_bits` for SH keys, all rows
    otherwise), codebook {key: f32 [1,K,d]}, logits {key: f32 [1,K]}.

    Returns (files {name: bytes}, info) where info holds the sort order and shrunk indices.
    """
    order, boundaries = shmask_sort(attrs["sh_bits"])
    n = len(order)
    index, codebooks, logits = {}, {}, {}
    for key in KEYS:
        if key in SH_KEYS:
            full = -np.ones(n, dtype=np.int64)
            full[attrs["sh_bits"][key]] = attrs["index"][key]
            sorted_idx = full[order]
            sorted_idx = sorted_idx[attrs["sh_bits"][key][order]]
        else:
            sorted_idx = attrs["index"][key][order]
        codebooks[key], logits[key], index[key] = shrink(
            attrs["codebook"][key], attrs["logits"][key], sorted_idx)
    xyz = attrs["xyz"].astype(np.float16)[order]
    opacity = attrs["opacity"][order]
    q_opacity, opa_log_prob, step, lo = opacity_quant(opacity, opacity_bits)
    logits["opa"] = opa_log_prob

    strings, index_bytes = [], {}
    for key in KEYS:
        s = rans.encode(torch.Tensor(index[key]), torch.Tensor(logits[key]), precision)
        strings.append(s)
        index_bytes[key] = len(s) + 4
    s = rans.encode(torch.Tensor(q_opacity.astype(np.float32)), torch.Tensor(opa_log_prob),
                    precision)
    strings.append(s)
    index_bytes["opa"] = len(s) + 4
    files = {
        "index_bitstream.bin": rans.pack_strings(strings),
        "header.bin": rans.pack_header(boundaries.tolist(), step, lo),
        "codebook.npz": savez_bytes(**codebooks),
        "logits.npz": savez_bytes(**logits),
        "position.npz": savez_bytes(position=xyz),
    }
    info = {"order": order, "boundaries": boundaries, "index": index, "q_opacity": q_opacity,
            "opacity_sorted": opacity, "xyz_sorted": xyz, "index_bytes": index_bytes,
            "used_codewords": {k: int(codebooks[k].shape[1]) for k in KEYS}}
    return files, info


def decode(files, precision=16):
    """Official GaussianModel.decode on the bytes of `encode` -> numpy attributes (sorted
    order): xyz f32 (from f16), opacity logits, scale / rot / dc codewords, shN [N,15,3] with
    zeros on pruned SH degrees, sh_bits, and the decoded indices."""
    xyz = np.load(io.BytesIO(files["position.npz"]))["position"]
    codebooks = np.load(io.BytesIO(files["codebook.npz"]))
    logits = np.load(io.BytesIO(files["logits.npz"]))
    boundaries, step, lo = rans.unpack_header(files["header.bin"])
    counts = index_counts(boundaries)
    strings, rest = rans.unpack_strings(files["index_bitstream.bin"], len(INDEX_KEYS))
    if rest:
        raise ValueError("trailing bytes in index_bitstream.bin")
    index = {}
    for key, s in zip(INDEX_KEYS, strings):
        index[key] = rans.decode(s, counts[key], torch.Tensor(logits[key]), precision).numpy()
    n = int(boundaries[-1])
    sh_bits = sh_bits_from_boundaries(boundaries)
    opacity = (index["opa"].astype(np.float32) * np.float32(step) + np.float32(lo))
    attr = {k: codebooks[k][0, index[k], :] for k in KEYS}
    shn = np.zeros((n, 15, 3), dtype=np.float32)
    shn[sh_bits["sh1"], 0:3, :] = attr["sh1"].reshape(-1, 3, 3)
    shn[sh_bits["sh2"], 3:8, :] = attr["sh2"].reshape(-1, 5, 3)
    shn[sh_bits["sh3"], 8:15, :] = attr["sh3"].reshape(-1, 7, 3)
    return {"xyz": xyz.astype(np.float32), "opacity": opacity, "scale": attr["scale"],
            "rot": attr["rot"], "dc": attr["dc"].reshape(-1, 1, 3), "shN": shn,
            "sh_bits": sh_bits, "index": index, "boundaries": boundaries, "n": n}


def size_report(files, index_bytes):
    sizes = {name: len(files[name]) for name in FILES}
    total = sum(sizes.values())
    return {
        "total_bytes": total,
        "size_MB": total / 1e6,
        "files_bytes": sizes,
        "components_MB": {  # paper Table 3
            "header": sizes["header.bin"] / 1e6,
            "indexes": sizes["index_bitstream.bin"] / 1e6,
            "codebooks": sizes["codebook.npz"] / 1e6,
            "logits": sizes["logits.npz"] / 1e6,
            "positions": sizes["position.npz"] / 1e6,
        },
        "index_MB": {k: v / 1e6 for k, v in index_bytes.items()},  # paper Table A.9
    }
