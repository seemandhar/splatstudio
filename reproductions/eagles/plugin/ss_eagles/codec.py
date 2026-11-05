"""EAGLES encoded size and an actual entropy-coded bitstream of the quantized latents.

* `paper_size_bits` is the official `GaussianModelSQ.size()` (what the paper logs and reports as
  "Storage Mem", bits / 8 / 1e6 MB): per latent dimension, sum over values of
  count * clamp(-log2(count/N + 1e-10), 0, 1000); plus 32 bits per decoder parameter (including
  the fixed DFT matrix and the `div` vector, as `parameters()` counts them) and 32 bits per
  unquantized value (position, SH DC, scale: 9 per Gaussian).
* `encode` / `decode` produce and invert a real bitstream: one range coder (constriction) per
  latent attribute, one static categorical model per latent dimension built from that
  dimension's empirical value counts (the model `size()` assumes). The official pkl instead
  uses torchac with one flattened per-attribute CDF; `actual_bytes` reports this coder's bytes.
"""

import numpy as np
import torch

UNQUANTIZED_DIMS = 9  # xyz 3 + f_dc 3 + scaling 3, stored as float32 (DecoderIdentity)


def column_entropy_bits(column):
    """Official size() term for one latent dimension of integer values."""
    _, counts = torch.unique(column, return_counts=True)
    probs = counts / torch.sum(counts)
    information_bits = torch.clamp(-1.0 * torch.log(probs + 1e-10) / np.log(2.0), 0, 1000)
    return float(torch.sum(information_bits * counts).item())


def paper_size_bits(latents_int, bank, n):
    latent_bits = {f: sum(column_entropy_bits(q[:, d]) for d in range(q.shape[1]))
                   for f, q in latents_int.items()}
    decoder_bits = sum(dec.size_bits() for dec in bank.decoders.values())
    unquantized_bits = 32 * n * UNQUANTIZED_DIMS
    total = sum(latent_bits.values()) + decoder_bits + unquantized_bits
    return {"total_bits": total, "latent_bits": latent_bits, "decoder_bits": decoder_bits,
            "unquantized_bits": unquantized_bits}


def _model(counts):
    import constriction

    probs = counts.astype(np.float64) / counts.sum()
    return constriction.stream.model.Categorical(probs, perfect=False)


def encode(latents_int):
    """{field: [N,d] integer tensor} -> {field: {values, counts, words, shape}} (CPU tensors)."""
    import constriction

    out = {}
    for field, q in latents_int.items():
        q = q.detach().cpu().long().numpy()
        enc = constriction.stream.queue.RangeEncoder()
        values, counts = [], []
        for d in range(q.shape[1]):
            vals, idx, cnt = np.unique(q[:, d], return_inverse=True, return_counts=True)
            if len(vals) > 1:
                enc.encode(idx.astype(np.int32), _model(cnt))
            values.append(torch.from_numpy(vals.astype(np.int32)))
            counts.append(torch.from_numpy(cnt.astype(np.int64)))
        words = enc.get_compressed().astype(np.uint32).view(np.int32)
        out[field] = {"values": values, "counts": counts,
                      "words": torch.from_numpy(words.copy()), "shape": list(q.shape)}
    return out


def decode(stream):
    import constriction

    out = {}
    for field, s in stream.items():
        n, dims = s["shape"]
        words = s["words"].numpy().view(np.uint32)
        dec = constriction.stream.queue.RangeDecoder(words)
        cols = []
        for d in range(dims):
            vals, cnt = s["values"][d].numpy(), s["counts"][d].numpy()
            if len(vals) > 1:
                idx = dec.decode(_model(cnt), n)
            else:
                idx = np.zeros(n, dtype=np.int64)
            cols.append(vals[idx])
        out[field] = torch.from_numpy(np.stack(cols, 1).astype(np.float32))
    return out


def actual_bytes(stream, bank, n):
    """Real stored bytes: bitstream words + per-dimension (value int32, count uint32) tables +
    learned decoder parameters (scale, shift; float32) + unquantized float32 attributes."""
    latent = {f: 4 * len(s["words"]) + sum(8 * len(v) for v in s["values"])
              for f, s in stream.items()}
    decoder = sum(4 * (d.scale.numel() + d.shift.numel()) for d in bank.decoders.values())
    unquantized = 4 * n * UNQUANTIZED_DIMS
    return {"total_bytes": sum(latent.values()) + decoder + unquantized,
            "latent_bytes": latent, "decoder_bytes": decoder, "unquantized_bytes": unquantized}


def entropy_lower_bound_bits(column_counts):
    n = column_counts.sum()
    p = column_counts / n
    return float(-(column_counts * np.log2(p)).sum()) if n else 0.0

