"""OMG storage (C4): fp16 -> uint16 positions coded by G-PCC, Huffman-coded SVQ indices, fp16
codebooks and fp16 MLPs, one LZMA-compressed container.

Official: gaussian_model.encode/decode, utils/compress_utils.py (dahuffman, LZMA pickle,
write_storage), utils/gpcc_utils.py (tmc3 command lines; sourced there from HAC++).

- Huffman code lengths: port of dahuffman 0.4.1 `HuffmanCodec.from_frequencies` (MIT licence,
  Copyright Stefaan Lippens), including its EOF symbol of frequency 1. Copied from our
  Compact-3DGS port (reproductions/compact-3dgs/plugin/ss_compact_3dgs/codec.py). The container
  stores canonical codes of exactly those lengths. The official byte count of a dahuffman stream
  (`len(codec.encode(data))`) is ceil((payload bits + EOF bits) / 8), and it is reported as such.
- G-PCC: MPEG TMC13 `tmc3` ($OMG_TMC3, `tmc3` on PATH, or ~/models/tmc13 build). The official
  encoder flags are run verbatim for the reported size. Our TMC13 v14.0 cannot decode streams made
  with `--cabac_bypass_stream_enabled_flag=1` ("bypass buffer exceeded"), so the container holds
  the same command with that flag set to 0. That stream decodes losslessly, and both sizes are
  reported.
"""

import io
import json
import lzma
import os
import shutil
import struct
import subprocess
import tempfile
from heapq import heapify, heappop, heappush
from pathlib import Path

import numpy as np
import torch

MiB = 2**20  # official "Actual storage" divides by 2**20

DEFAULT_TMC3 = Path.home() / "models/tmc13/src/build/tmc3/tmc3"


# ------------------------------------------------------------------------------ Huffman
class _EOF:
    """dahuffman _EndOfFileSymbol: smaller than every symbol."""

    def __repr__(self):
        return "_EOF"

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return other.__class__ == self.__class__

    def __hash__(self):
        return hash(self.__class__)


EOF = _EOF()


def dahuffman_table(frequencies):
    """Port of dahuffman 0.4.1 HuffmanCodec.from_frequencies: symbol -> (bitsize, value)."""
    heap = [(f, [(s, (0, 0))]) for s, f in frequencies.items()]
    if EOF not in frequencies:
        heap.append((1, [(EOF, (0, 0))]))
    heapify(heap)
    while len(heap) > 1:
        a = heappop(heap)
        b = heappop(heap)
        merged = (
            a[0] + b[0],
            [(s, (n + 1, v)) for (s, (n, v)) in a[1]]
            + [(s, (n + 1, (1 << n) + v)) for (s, (n, v)) in b[1]],
        )
        heappush(heap, merged)
    return dict(heappop(heap)[1])


def canonical_codes(lengths):
    order = sorted(lengths, key=lambda s: (lengths[s], s))
    codes, code, prev = {}, 0, lengths[order[0]]
    for s in order:
        code <<= lengths[s] - prev
        codes[s] = code
        code += 1
        prev = lengths[s]
    return codes, order


def pack_bits(symbols, lengths, codes, chunk=1 << 20):
    """MSB-first bitstream of the concatenated codes (vectorised)."""
    symbols = np.asarray(symbols).reshape(-1).astype(np.int64)
    keys = np.array(sorted(lengths), dtype=np.int64)
    lut_len = np.array([lengths[int(k)] for k in keys], dtype=np.int64)
    lut_code = np.array([codes[int(k)] for k in keys], dtype=np.uint64)
    pos = np.searchsorted(keys, symbols)
    parts = []
    for s in range(0, len(symbols), chunk):
        p = pos[s:s + chunk]
        ln, cd = lut_len[p], lut_code[p]
        total = int(ln.sum())
        idx = np.repeat(np.arange(len(p)), ln)
        starts = np.cumsum(ln) - ln
        k = np.arange(total, dtype=np.int64) - np.repeat(starts, ln)
        shift = (ln[idx] - 1 - k).astype(np.uint64)
        parts.append(((cd[idx] >> shift) & np.uint64(1)).astype(np.uint8))
    bits = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
    return np.packbits(bits).tobytes(), int(len(bits))


def huffman_encode(symbols):
    """Canonical Huffman stream with dahuffman's code lengths, plus the official byte count."""
    symbols = np.asarray(symbols).reshape(-1).astype(np.int64)
    values, counts = np.unique(symbols, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    table = dahuffman_table(freq)
    lengths = {s: table[s][0] for s in freq}
    bits = sum(freq[s] * lengths[s] for s in freq)
    codes, order = canonical_codes(lengths)
    payload, nbits = pack_bits(symbols, lengths, codes)
    if nbits != bits:
        raise AssertionError("canonical payload differs from the Huffman cost")
    official_bytes = (bits + table[EOF][0] + 7) // 8  # dahuffman encode(): data + EOF, padded
    maxlen = max(lengths.values())
    count = [0] * (maxlen + 1)
    for s in order:
        count[lengths[s]] += 1
    return {"payload": payload, "n": int(symbols.size), "count": count,
            "symbols": [int(s) for s in order], "bits": int(bits),
            "official_bytes": int(official_bytes)}


def huffman_decode(stream):
    """Canonical decoding with a full lookup table (max code length <= 24 here)."""
    count, symbols, n = stream["count"], stream["symbols"], int(stream["n"])
    maxlen = len(count) - 1
    if maxlen > 24:
        raise ValueError("Huffman code too long for the lookup decoder")
    if maxlen == 0:
        return np.zeros(n, dtype=np.int64) + (symbols[0] if symbols else 0)
    lut_sym = np.zeros(1 << maxlen, dtype=np.int64)
    lut_len = np.zeros(1 << maxlen, dtype=np.int64)
    code, k = 0, 0
    for length in range(1, maxlen + 1):
        for _ in range(count[length]):
            lo = code << (maxlen - length)
            hi = (code + 1) << (maxlen - length)
            lut_sym[lo:hi] = symbols[k]
            lut_len[lo:hi] = length
            code += 1
            k += 1
        code <<= 1
    bits = np.unpackbits(np.frombuffer(stream["payload"], np.uint8))
    bits = np.concatenate([bits, np.zeros(maxlen, np.uint8)])
    # window value at every bit offset (MSB first)
    weights = (1 << np.arange(maxlen - 1, -1, -1)).astype(np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(bits, maxlen)
    out = np.empty(n, dtype=np.int64)
    pos = 0
    lsym, llen = lut_sym.tolist(), lut_len.tolist()
    for i in range(n):
        v = int(windows[pos] @ weights)
        out[i] = lsym[v]
        pos += llen[v]
    return out


# -------------------------------------------------------------------------------- G-PCC
def tmc3_path():
    for cand in (os.environ.get("OMG_TMC3"), shutil.which("tmc3"), str(DEFAULT_TMC3)):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def _encode_cmd(tmc3, ply, out, bypass):
    # gpcc_utils.gpcc_encode verbatim (including the missing space after
    # --positionQuantisationEnabled=1); `bypass` toggles --cabac_bypass_stream_enabled_flag.
    return (f"{tmc3} "
            "--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 "
            "--neighbourAvailBoundaryLog2=8 --positionQuantisationEnabled=1"
            "--intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 "
            "--inferredDirectCodingMode=3 --maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 "
            "--planarEnabled=0 --planarModeIdcmUse=0 "
            f"--cabac_bypass_stream_enabled_flag={int(bypass)} "
            f"--uncompressedDataPath={ply} --compressedStreamPath={out}")


def _run(cmd):
    res = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"tmc3 failed ({res.returncode}): {res.stderr[-300:]!r}")


def gpcc_encode(u16):
    """uint16 [N,3] -> (decodable stream, official-flag stream length)."""
    tmc3 = tmc3_path()
    if tmc3 is None:
        raise RuntimeError("G-PCC needs the MPEG tmc3 binary ($OMG_TMC3, PATH or ~/models/tmc13)")
    pts = np.asarray(u16, dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        ply = os.path.join(tmp, "point_cloud.ply")
        with open(ply, "w") as f:  # gpcc_utils.write_ply_geo_ascii
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\n"
                    "property float y\nproperty float z\nend_header\n")
            np.savetxt(f, pts, fmt="%d")
        official = os.path.join(tmp, "official.bin")
        stream = os.path.join(tmp, "stream.bin")
        _run(_encode_cmd(tmc3, ply, official, bypass=True))
        _run(_encode_cmd(tmc3, ply, stream, bypass=False))
        return Path(stream).read_bytes(), Path(official).stat().st_size


def gpcc_decode(data):
    """stream -> int64 [N,3] (unordered, as tmc3 reconstructs it)."""
    from plyfile import PlyData

    tmc3 = tmc3_path()
    if tmc3 is None:
        raise RuntimeError("G-PCC needs the MPEG tmc3 binary")
    with tempfile.TemporaryDirectory() as tmp:
        bin_path, ply = os.path.join(tmp, "s.bin"), os.path.join(tmp, "r.ply")
        Path(bin_path).write_bytes(data)
        _run(f"{tmc3} --mode=1 --outputBinaryPly=1 --compressedStreamPath={bin_path} "
             f"--reconstructedDataPath={ply}")
        el = PlyData.read(ply).elements[0]
        return np.stack([np.asarray(el.data[k]) for k in "xyz"], 1).astype(np.int64)


# ---------------------------------------------------------------------------- container
def write_container(sections, meta):
    names = list(sections)
    header = json.dumps({"meta": meta, "sections": [[k, len(sections[k])] for k in names]},
                        sort_keys=True).encode()
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(header)))
    buf.write(header)
    for k in names:
        buf.write(sections[k])
    return lzma.compress(buf.getvalue())


def read_container(blob):
    data = lzma.decompress(blob)
    (hlen,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + hlen].decode())
    out, pos = {}, 4 + hlen
    for k, length in header["sections"]:
        out[k] = data[pos:pos + length]
        pos += length
    if pos != len(data):
        raise ValueError("container length mismatch")
    return header["meta"], out


def f16_bytes(t):
    return t.detach().cpu().contiguous().to(torch.float16).numpy().tobytes()


def f16_tensor(b, shape):
    return torch.from_numpy(np.frombuffer(b, np.float16).copy()).float().reshape(shape)
