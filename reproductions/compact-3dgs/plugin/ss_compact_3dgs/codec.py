"""Compact-3DGS storage: official size formulas, post-processing (PP), Huffman, byte container.

Official scene/gaussian_model.py: final_prune (storage formulas, "Ours" and "After PP"),
post_quant (8-bit min-max, hash pruning |x| <= 0.1 with min/max taken *before* pruning),
huffman_encode (dahuffman HuffmanCodec.from_data; total bits over the real symbols only),
save_npz / save_npz_pp (what is stored: fp16 xyz, R-VQ indices, codebooks, fp16 or Huffman-coded
opacity/hash, fp16 MLP). MB = 1e6 bytes.

The Huffman code *lengths* are computed by a port of dahuffman 0.4.1
`HuffmanCodec.from_frequencies` (MIT licence, Copyright Stefaan Lippens), including its EOF symbol
of frequency 1 that takes part in the tree. The actual container uses canonical codes with exactly
those lengths (same payload bits; a canonical table is cheaper to store than dahuffman's pickle).
"""

import io
import json
import struct
from heapq import heapify, heappop, heappush

import numpy as np
import torch

from . import kernels

MB = 1e6


# ------------------------------------------------------------------------------ quantization
@torch.no_grad()
def post_quant(param, prune=False, prune_threshold=0.1):
    """Official post_quant: returns (decoded float, quant uint8, (min, max) float32)."""
    max_val = torch.amax(param)
    min_val = torch.amin(param)
    if prune:
        param = param * (torch.abs(param) > prune_threshold)
    param = (param - min_val) / (max_val - min_val)
    quant = torch.round(param * 255.0)
    out = (max_val - min_val) * quant / 255.0 + min_val
    return out, quant.to(torch.uint8), torch.stack([min_val, max_val]).float()


def dequant(quant, minmax):
    min_val, max_val = minmax[0], minmax[1]
    return (max_val - min_val) * quant.float() / 255.0 + min_val


# ----------------------------------------------------------------------------------- Huffman
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


def symbol_counts(symbols):
    values, counts = np.unique(np.asarray(symbols).reshape(-1), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def huffman_lengths(symbols):
    """(lengths {symbol: bits}, counts, official total bits = sum freq * len, excl. EOF)."""
    counts = symbol_counts(symbols)
    table = dahuffman_table(counts)
    lengths = {s: table[s][0] for s in counts}
    total = sum(counts[s] * lengths[s] for s in counts)
    return lengths, counts, total


def canonical_codes(lengths):
    """Canonical prefix codes for the given lengths: sorted by (length, symbol)."""
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
    if len(symbols) and (pos.max() >= len(keys) or not np.array_equal(keys[pos], symbols)):
        raise ValueError("symbol without a code")
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
    lengths, counts, total = huffman_lengths(symbols)
    codes, order = canonical_codes(lengths)
    payload, nbits = pack_bits(symbols, lengths, codes)
    if nbits != total:
        raise AssertionError("canonical payload differs from the Huffman cost")
    maxlen = max(lengths.values())
    count = [0] * (maxlen + 1)
    for s in order:
        count[lengths[s]] += 1
    return {"payload": payload, "n": int(np.asarray(symbols).size), "count": count,
            "symbols": [int(s) for s in order], "bits": total}


def huffman_decode(stream):
    out = kernels.ext().huffman_decode(
        torch.frombuffer(bytearray(stream["payload"]), dtype=torch.uint8)
        if stream["payload"] else torch.zeros(0, dtype=torch.uint8),
        int(stream["n"]), torch.tensor(stream["count"], dtype=torch.int64),
        torch.tensor(stream["symbols"], dtype=torch.int64))
    return out


def pack_indices(idx, bits):
    """Official save_npz: little-endian `bits`-bit fields, packbits over the flat stream."""
    a = np.asarray(idx).reshape(-1).astype(np.uint8)[:, None]
    return np.packbits(np.unpackbits(a, axis=-1, count=bits, bitorder="little").reshape(-1)).tobytes()


def unpack_indices(data, n, bits):
    b = np.unpackbits(np.frombuffer(data, np.uint8))[: n * bits].reshape(-1, bits)
    return np.packbits(b, axis=-1, bitorder="little")[:, 0].astype(np.int64)


# --------------------------------------------------------------------------- size formulas
def paper_sizes(n, n_hash, n_mlp, rvq_bits=6, rvq_num=6, huffman_bits=None):
    """final_prune storage strings, in MB. `huffman_bits` (PP) maps scale/rotation/opacity/hash ->
    total Huffman bits; None gives the "Ours" (no PP) formula."""
    book_s = 2**rvq_bits * rvq_num * 3 * 32 / 8 / MB
    book_r = 2**rvq_bits * rvq_num * 4 * 32 / 8 / MB
    sizes = {"position": n * 3 * 16 / 8 / MB, "mlp": n_mlp * 16 / 8 / MB}
    if huffman_bits is None:
        sizes.update(scale=n * rvq_bits * rvq_num / 8 / MB + book_s,
                     rotation=n * rvq_bits * rvq_num / 8 / MB + book_r,
                     opacity=n * 16 / 8 / MB, hash=n_hash * 16 / 8 / MB)
    else:
        sizes.update(scale=huffman_bits["scale"] / 8 / MB + book_s,
                     rotation=huffman_bits["rotation"] / 8 / MB + book_r,
                     opacity=huffman_bits["opacity"] / 8 / MB,
                     hash=huffman_bits["hash"] / 8 / MB)
    sizes["total"] = sum(sizes[k] for k in ("position", "scale", "rotation", "opacity", "hash",
                                             "mlp"))
    return sizes


# ------------------------------------------------------------------------------- container
def to_bytes(t, dtype):
    return t.detach().cpu().contiguous().to(dtype).numpy().tobytes()


def write_container(sections, meta):
    """[u32 header length][json header][section bytes...]; header lists names and lengths."""
    names = list(sections)
    header = json.dumps({"meta": meta, "sections": [[k, len(sections[k])] for k in names]},
                        sort_keys=True).encode()
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(header)))
    buf.write(header)
    for k in names:
        buf.write(sections[k])
    return buf.getvalue()


def read_container(data):
    (hlen,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + hlen].decode())
    out, pos = {}, 4 + hlen
    for k, length in header["sections"]:
        out[k] = data[pos:pos + length]
        pos += length
    if pos != len(data):
        raise ValueError("container length mismatch")
    return header["meta"], out


@torch.no_grad()
def encode(attrs, pp=True, hash_prune=0.1, rvq_bits=6):
    """attrs: means [N,3], scale_idx/rot_idx [N,L] int64, scale_books [L,C,3], rot_books [L,C,4],
    opacity [N] (activated), hash [P], mlp list of weights. Returns (bytes, report, decoded-values
    for the official in-memory model: post_quant outputs)."""
    n, L = attrs["scale_idx"].shape
    mlp_flat = torch.cat([w.detach().reshape(-1) for w in attrs["mlp"]])
    sections = {
        "xyz": to_bytes(attrs["means"], torch.float16),
        "codebook_scale": to_bytes(attrs["scale_books"], torch.float32),
        "codebook_rotation": to_bytes(attrs["rot_books"], torch.float32),
        "mlp": to_bytes(mlp_flat, torch.float16),
    }
    meta = {"n": n, "rvq_num": L, "rvq_bits": rvq_bits, "pp": bool(pp),
            "n_hash": int(attrs["hash"].numel()), "mlp_shapes": [list(w.shape) for w in attrs["mlp"]],
            "codebook_size": int(attrs["scale_books"].shape[1])}
    bits = None
    if pp:
        _, q_opa, mm_opa = post_quant(attrs["opacity"])
        _, q_hash, mm_hash = post_quant(attrs["hash"], prune=True, prune_threshold=hash_prune)
        streams = {"scale": attrs["scale_idx"], "rotation": attrs["rot_idx"],
                   "opacity": q_opa, "hash": q_hash}
        bits = {}
        for name, sym in streams.items():
            enc = huffman_encode(sym.detach().cpu().numpy())
            bits[name] = enc["bits"]
            sections[f"huf_{name}"] = enc["payload"]
            meta[f"huf_{name}"] = {"n": enc["n"], "count": enc["count"], "symbols": enc["symbols"]}
        sections["minmax_opacity"] = to_bytes(mm_opa, torch.float32)
        sections["minmax_hash"] = to_bytes(mm_hash, torch.float32)
    else:
        sections["scale"] = pack_indices(attrs["scale_idx"].cpu().numpy(), rvq_bits)
        sections["rotation"] = pack_indices(attrs["rot_idx"].cpu().numpy(), rvq_bits)
        sections["opacity"] = to_bytes(attrs["opacity"], torch.float16)
        sections["hash"] = to_bytes(attrs["hash"], torch.float16)
    data = write_container(sections, meta)
    sizes = {"ours": paper_sizes(n, attrs["hash"].numel(), mlp_flat.numel(), rvq_bits, L)}
    if pp:
        sizes["ours_pp"] = paper_sizes(n, attrs["hash"].numel(), mlp_flat.numel(), rvq_bits, L,
                                       bits)
    report = {"paper_size_mb": sizes, "huffman_bits": bits,
              "actual_bytes": {"total": len(data), **{k: len(v) for k, v in sections.items()}},
              "actual_mb": len(data) / MB}
    return data, report


def _f16(b, shape):
    return torch.from_numpy(np.frombuffer(b, np.float16).copy()).float().reshape(shape)


def _f32(b, shape):
    return torch.from_numpy(np.frombuffer(b, np.float32).copy()).reshape(shape)


@torch.no_grad()
def decode(data, device):
    """Container -> decoded attributes (means, scale/rotation via codebooks, opacity, hash, mlp)."""
    meta, s = read_container(data)
    n, L, bits, C = meta["n"], meta["rvq_num"], meta["rvq_bits"], meta["codebook_size"]
    out = {"means": _f16(s["xyz"], (n, 3)).to(device)}
    books_s = _f32(s["codebook_scale"], (L, C, 3)).to(device)
    books_r = _f32(s["codebook_rotation"], (L, C, 4)).to(device)
    if meta["pp"]:
        def huf(name):
            stream = dict(meta[f"huf_{name}"], payload=s[f"huf_{name}"])
            return huffman_decode(stream)

        sidx = huf("scale").reshape(n, L)
        ridx = huf("rotation").reshape(n, L)
        out["opacity"] = dequant(huf("opacity").to(device),
                                 _f32(s["minmax_opacity"], (2,)).to(device))
        out["hash"] = dequant(huf("hash").to(device), _f32(s["minmax_hash"], (2,)).to(device))
    else:
        sidx = torch.from_numpy(unpack_indices(s["scale"], n * L, bits)).reshape(n, L)
        ridx = torch.from_numpy(unpack_indices(s["rotation"], n * L, bits)).reshape(n, L)
        out["opacity"] = _f16(s["opacity"], (n,)).to(device)
        out["hash"] = _f16(s["hash"], (meta["n_hash"],)).to(device)
    sidx, ridx = sidx.to(device), ridx.to(device)
    out["scale_idx"], out["rot_idx"] = sidx, ridx
    out["scales"] = sum(books_s[k][sidx[:, k]] for k in range(L))
    out["quats"] = sum(books_r[k][ridx[:, k]] for k in range(L))
    flat = _f16(s["mlp"], (-1,)).to(device)
    weights, i = [], 0
    for a, b in meta["mlp_shapes"]:
        weights.append(flat[i:i + a * b].view(a, b).clone())
        i += a * b
    out["mlp"] = weights
    return out
