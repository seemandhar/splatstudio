"""Arithmetic coding for the ContextGS bitstream.

`encode_gaussian` / `decode_gaussian` / `encode_binary` / `decode_binary` are copied from the HAC native port
(reproductions/hac/plugin/ss_hac/codec.py, an in-memory port of official HAC utils/encodings_cuda.py) and drive the
vendored HAC CUDA coder. Bit counts: (len(stream) + len(cnt)) * 8 + 32 * 3 per Gaussian call, + 32 * 2 per binary
call. The official ContextGS codes the same models with torchac (Gaussian, masks) and compressai rANS (hyperprior);
`encode_factorized` codes the hyperprior symbols with the factorized-prior CDF on the same CUDA coder.
"""

import numpy as np
import torch

from . import kernels

CHUNK_SIZE_CUDA = 10000
CHUNK = 10_000_000  # encoder_gaussian_chunk default chunk_size


def _q_tensor(q, like):
    if isinstance(q, torch.Tensor):
        return q.reshape(-1).contiguous().float()
    return torch.full_like(like, float(q))


def encode_gaussian(x, mean, scale, q):
    """encoder_gaussian_chunk: returns (payload list, bits)."""
    x, mean, scale = (t.reshape(-1).contiguous().float() for t in (x, mean, scale))
    q = _q_tensor(q, mean)
    payloads, bits = [], 0
    for s in range(0, max(x.numel(), 1), CHUNK):
        xs, ms, ss, qs = (t[s : s + CHUNK].contiguous() for t in (x, mean, scale, q))
        if xs.numel() == 0:
            break
        x_int = torch.round(xs / qs)
        lo, hi = int(x_int.min().item()), int(x_int.max().item())
        if hi - lo + 2 > 32767:
            raise ValueError("symbol range exceeds the int16 coder alphabet")
        cdf = kernels.arithmetic().calculate_cdf(ms, ss, qs, lo, hi)
        sym = (x_int - lo).to(torch.int16).contiguous()
        stream, cnt = kernels.arithmetic().arithmetic_encode(sym, cdf, CHUNK_SIZE_CUDA,
                                                              int(cdf.shape[0]),
                                                              int(cdf.shape[1]))
        cnt_b = cnt.cpu().numpy().tobytes()
        stream_b = stream.cpu().numpy().tobytes()
        payloads.append((lo, hi, cnt_b, stream_b))
        bits += (len(stream_b) + len(cnt_b)) * 8 + 32 * 3
    return payloads, bits


def decode_gaussian(payloads, mean, scale, q):
    mean, scale = (t.reshape(-1).contiguous().float() for t in (mean, scale))
    q = _q_tensor(q, mean)
    out = []
    for c, (lo, hi, cnt_b, stream_b) in enumerate(payloads):
        s = c * CHUNK
        ms, ss, qs = (t[s : s + CHUNK].contiguous() for t in (mean, scale, q))
        cdf = kernels.arithmetic().calculate_cdf(ms, ss, qs, lo, hi)
        cnt = torch.tensor(np.frombuffer(cnt_b, dtype=np.int32).copy(), device=mean.device)
        stream = torch.tensor(np.frombuffer(stream_b, dtype=np.uint8).copy(), device=mean.device)
        sym = kernels.arithmetic().arithmetic_decode(cdf, stream, cnt, CHUNK_SIZE_CUDA,
                                                     int(cdf.shape[0]), int(cdf.shape[1]))
        out.append((sym.to(torch.float32) + lo) * qs)
    if not out:
        return mean.new_zeros(0)
    return torch.cat(out)


def _binary_cdf(p1, n, device):
    p = torch.full((n, 1), 1 - p1, dtype=torch.float32, device=device)
    return torch.cat([torch.zeros_like(p), p, torch.ones_like(p)], dim=-1).contiguous()


def encode_binary(x):
    """encoder(): {0,1} symbols under one global Bernoulli probability."""
    x = x.detach().reshape(-1).float()
    p1 = float((x.sum() / x.numel()).item())
    cdf = _binary_cdf(p1, x.numel(), x.device)
    stream, cnt = kernels.arithmetic().arithmetic_encode(
        torch.floor(x).to(torch.int16).contiguous(), cdf, CHUNK_SIZE_CUDA, int(cdf.shape[0]),
        int(cdf.shape[1]))
    cnt_b, stream_b = cnt.cpu().numpy().tobytes(), stream.cpu().numpy().tobytes()
    p1 = float(np.float32(p1))
    return (p1, cnt_b, stream_b), (len(stream_b) + len(cnt_b)) * 8 + 32 * 2


def decode_binary(payload, n, device):
    p1, cnt_b, stream_b = payload
    cdf = _binary_cdf(p1, n, device)
    cnt = torch.tensor(np.frombuffer(cnt_b, dtype=np.int32).copy(), device=device)
    stream = torch.tensor(np.frombuffer(stream_b, dtype=np.uint8).copy(), device=device)
    return kernels.arithmetic().arithmetic_decode(cdf, stream, cnt, CHUNK_SIZE_CUDA,
                                                  int(cdf.shape[0]), int(cdf.shape[1]))


def encode_factorized(sym, prior):
    """Hyperprior symbols [N, C] (integers relative to the median) under the per-channel factorized CDF of
    `prior` (model.FactorizedPrior); one stream per channel with its own symbol range. Returns (payload, bits)."""
    payloads, bits = [], 0
    for ch in range(sym.shape[1]):
        s = sym[:, ch].to(torch.int64)
        if s.numel() == 0:
            payloads.append(None)
            continue
        lo, hi = int(s.min().item()), int(s.max().item())
        if hi - lo + 2 > 32767:
            raise ValueError("hyperprior symbol range exceeds the int16 coder alphabet")
        row = prior.cdf_table(lo, hi)[ch]
        cdf = row[None].expand(s.numel(), -1).contiguous()
        stream, cnt = kernels.arithmetic().arithmetic_encode((s - lo).to(torch.int16).contiguous(),
                                                              cdf, CHUNK_SIZE_CUDA, int(cdf.shape[0]),
                                                              int(cdf.shape[1]))
        cnt_b, stream_b = cnt.cpu().numpy().tobytes(), stream.cpu().numpy().tobytes()
        payloads.append((lo, hi, cnt_b, stream_b))
        bits += (len(stream_b) + len(cnt_b)) * 8 + 32 * 2
    return payloads, bits


def decode_factorized(payloads, n, prior, device):
    cols = []
    for ch, p in enumerate(payloads):
        if p is None:
            cols.append(torch.zeros(0, dtype=torch.int32, device=device))
            continue
        lo, hi, cnt_b, stream_b = p
        row = prior.cdf_table(lo, hi)[ch]
        cdf = row[None].expand(n, -1).contiguous()
        cnt = torch.tensor(np.frombuffer(cnt_b, dtype=np.int32).copy(), device=device)
        stream = torch.tensor(np.frombuffer(stream_b, dtype=np.uint8).copy(), device=device)
        s = kernels.arithmetic().arithmetic_decode(cdf, stream, cnt, CHUNK_SIZE_CUDA,
                                                   int(cdf.shape[0]), int(cdf.shape[1]))
        cols.append(s.to(torch.int32) + lo)
    return torch.stack(cols, dim=1)
