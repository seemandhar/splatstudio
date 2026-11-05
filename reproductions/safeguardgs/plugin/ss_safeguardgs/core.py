"""Pure-torch pieces of SafeguardGS (CPU-testable) and an independent score reference.

Official code: https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS (commit 539ca22e),
`cuda_rasterizer/forward.cu:compute_score` and `rasterizer_impl.cu:markingTopKMasks`.
"""

import torch

SG_36 = 0x24  # CS1 * alpha * T (paper Score_SG_36, official default `safeguard_gs_score_function`)
SG_4 = 0x04  # alpha * T (= EfficientGS score)
QSCALE = 65535.0


def valid_func_id(func_id):
    return 0 <= int(func_id) <= 0x5F and (int(func_id) & 0xF0) <= 0x50


def score(func_id, opacity, alpha, T, dx, dy, gt, color, p_coef=1.0, c_coef=1.0):
    """Official compute_score (forward.cu) for broadcastable tensors; gt/color [..., 3].

    Score_SG_36 (paper Tab. 1): CS1(c_gt, c_i) * alpha_i * T_i with
    CS1 = 1 - ||c_gt - c_i||_1 / C (paper Eq. 2), T_i the transmittance before Gaussian i.
    """
    if not valid_func_id(func_id):
        raise ValueError(f"invalid score function id {func_id:#x}")
    lo, hi = int(func_id) & 0x0F, int(func_id) & 0xF0
    pd = torch.exp(-(p_coef * torch.sqrt(dx * dx + dy * dy)))  # PC(X_r, X_i)
    one = torch.ones_like(alpha)
    base = {
        0x00: one, 0x01: opacity * one, 0x02: alpha, 0x03: opacity * T, 0x04: alpha * T,
        0x05: pd * one, 0x06: pd * opacity, 0x07: pd * alpha, 0x08: pd * opacity * T,
        0x09: pd * alpha * T, 0x0A: opacity + T, 0x0B: alpha + T, 0x0C: pd + opacity,
        0x0D: pd + alpha, 0x0E: pd + opacity + T, 0x0F: pd + alpha + T,
    }[lo]
    if hi == 0x00:
        return base
    l1 = (gt - color).abs().sum(dim=-1)
    cs1 = 1.0 - l1 / 3.0
    cs2 = torch.exp(-c_coef * l1 / 3.0)
    if hi == 0x10:
        cs3 = (gt * color).sum(-1) / (gt.norm(dim=-1) * color.norm(dim=-1))
        return base * cs3
    return {0x20: base * cs1, 0x30: base * cs2, 0x40: base + cs1, 0x50: base + cs2}[hi]


def quantize(s):
    """Official key: __float2uint_rd((1 - score) * 65535) (saturates negatives to 0)."""
    s = torch.as_tensor(s, dtype=torch.float32)
    return torch.floor((1.0 - s) * QSCALE).clamp_min(0).to(torch.int64)


def tile_lengths(tile_offsets_flat, n_isects):
    """Number of intersections per tile from gsplat's isect_offsets (flattened)."""
    off = tile_offsets_flat.to(torch.int64)
    end = torch.cat((off[1:], torch.tensor([n_isects], dtype=torch.int64, device=off.device)))
    return end - off


def padded_offsets(tile_offsets_flat, n_isects, block=256):
    """Tile-private slot ranges of ceil(len / block) * block slots (one per pixel and batch)."""
    slots = (tile_lengths(tile_offsets_flat, n_isects) + block - 1) // block * block
    pad = torch.cumsum(slots, 0) - slots
    return pad, slots, int(slots.sum())


def tile_topk_keep(slot_q, slot_id, slots_per_tile, top_k, n):
    """markingTopKMasks after the stable radix sort of (tile << 32) | q: the first top_k
    entries of every tile (ascending 1-score, ties by slot order) are kept. Returns bool [n]."""
    device = slot_q.device
    tile = torch.repeat_interleave(
        torch.arange(len(slots_per_tile), device=device), slots_per_tile.to(device)
    )
    valid = slot_q >= 0
    key = (tile[valid] << 32) | slot_q[valid].to(torch.int64)
    ids = slot_id[valid].to(torch.int64)
    keep = torch.zeros(n, dtype=torch.bool, device=device)
    if key.numel() == 0:
        return keep
    key_sorted, order = torch.sort(key, stable=True)
    tiles_sorted = key_sorted >> 32
    _, counts = torch.unique_consecutive(tiles_sorted, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(key_sorted), device=device) - torch.repeat_interleave(starts, counts)
    keep[ids[order[rank < int(top_k)]]] = True
    return keep
