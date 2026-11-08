"""Pure tensor functions of OMG (CPU-testable). Re-implemented from the official code
(github.com/maincold2/OMG: scene/gaussian_model.py, utils/compress_utils.py, utils/gpcc_utils.py,
gaussian_renderer/__init__.py). That repository ships no licence file, so nothing is copied
verbatim; numerics follow it line by line (file:function cited per function).
"""

import math

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


# ------------------------------------------------------------------------- Morton order (C1)
def _split_by_3(a):
    """compress_utils.splitBy3: spread the low 21 bits of `a` to every third bit (int64)."""
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def morton_encode(q):
    """compress_utils.mortonEncode on int64 [N,3] in [0, 2^21)."""
    x, y, z = q.long().unbind(-1)
    return _split_by_3(x) | (_split_by_3(y) << 1) | (_split_by_3(z) << 2)


def morton_order(means):
    """gaussian_model.sort_morton: quantise to 21 bits per axis over the bounding box, sort."""
    lo, hi = means.min(0).values, means.max(0).values
    q = ((2**21 - 1) * (means - lo) / (hi - lo)).long()
    return morton_encode(q).sort().indices


def ld_residual(features, order, neighbors="official"):
    """Local distinctiveness term (1/D) sum_{j in N_i} |T_i - T_j|_1 with K = 2 neighbours.

    official (gaussian_model.ld_scoring): order_l = clamp(order - 1, 0), order_r =
      clamp(order + 1, max(order)); neighbours of i are T[order][order_l][i] = T[order[order[i]-1]]
      and T[order[order[i]+1]] (the permutation applied twice; this is what the released code
      computes and what produced the paper's numbers).
    morton (paper text): the Gaussians adjacent to i in Morton order, order[rank(i) -/+ 1].
    The value is mean(|T_l - T_i| + |T_r - T_i|) over the D feature channels.
    """
    n = len(order)
    if neighbors == "official":
        left = order[torch.clamp_min(order - 1, 0)]
        right = order[torch.clamp_max(order + 1, int(order.max()))]
    elif neighbors == "morton":
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n, device=order.device)
        left = order[torch.clamp_min(rank - 1, 0)]
        right = order[torch.clamp_max(rank + 1, n - 1)]
    else:
        raise ValueError("ld_neighbors must be 'official' or 'morton'")
    t = features
    return torch.mean((t[left] - t).abs() + (t[right] - t).abs(), dim=-1)


def ld_importance(importance, features, means, lambda_ld=2.0, neighbors="official"):
    """Paper Eq. (LD): I_i = Ibar_i * residual_i ** lambda."""
    res = ld_residual(features, morton_order(means), neighbors)
    return importance * res**lambda_ld


# --------------------------------------------------------------------- space feature (C2)
def contract_to_unisphere(x):
    """gaussian_model.contract_to_unisphere with aabb [-1,1]^3 (identity normalisation):
    x if |x| <= 1 else (2 - 1/|x|) x/|x|, then x/4 + 0.5."""
    mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    outside = mag > 1
    safe = torch.where(outside, mag, torch.ones_like(mag))
    x = torch.where(outside, (2 - 1 / safe) * (x / safe), x)
    return x / 4 + 0.5


def frequency_encoding(x, n_frequencies=16):
    """tiny-cuda-nn "Frequency" encoding (encodings/frequency.h): output j of an input with
    n_dims features is sin(x[j // (2F)] * 2^((j // 2) % F) * pi + (j % 2) * pi / 2)."""
    n, d = x.shape
    j = torch.arange(d * 2 * n_frequencies, device=x.device)
    feat = j // (2 * n_frequencies)
    freq = (2.0 ** ((j // 2) % n_frequencies)).to(x.dtype)
    phase = (j % 2).to(x.dtype) * (math.pi / 2)
    return torch.sin(x[:, feat] * freq * math.pi + phase)


def padded(n, multiple=16):
    return (n + multiple - 1) // multiple * multiple


def mlp_shapes(n_in, n_out, n_neurons=64, n_hidden_layers=1):
    """tcnn FullyFusedMLP weight matrices (no biases; input/output padded to multiples of 16)."""
    shapes = [(n_neurons, padded(n_in))]
    shapes += [(n_neurons, n_neurons)] * (n_hidden_layers - 1)
    shapes.append((padded(n_out), n_neurons))
    return shapes


def mlp_init(shapes, generator):
    """tcnn initialize_xavier_uniform per weight matrix."""
    out = []
    for fan_out, fan_in in shapes:
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        out.append((torch.rand(fan_out, fan_in, generator=generator) * 2 - 1) * bound)
    return out


def mlp_forward(weights, inp, n_out, activation="relu"):
    """tcnn FullyFusedMLP forward: hidden activation, output activation None, padded input
    columns are ones (tcnn pads with 1), padded output rows are dropped."""
    pad = weights[0].shape[1] - inp.shape[1]
    h = inp
    if pad:
        h = torch.cat([inp, torch.ones(inp.shape[0], pad, device=inp.device, dtype=inp.dtype)], 1)
    for w in weights[:-1]:
        h = h @ w.T
        h = F.relu(h) if activation == "relu" else F.leaky_relu(h, 0.01)
    return (h @ weights[-1].T)[:, :n_out]


def net_lr(base_lr, epoch, warmup_iters=100, warmup_start=0.01, milestones=(1000, 3500, 6000),
           gamma=0.33):
    """Closed form of ChainedScheduler([LinearLR(start_factor, total_iters),
    MultiStepLR(milestones, gamma)]) after `epoch` scheduler steps (construct_net)."""
    linear = warmup_start + (1 - warmup_start) * min(epoch, warmup_iters) / warmup_iters
    return base_lr * linear * gamma ** sum(epoch >= m for m in milestones)


def sh_degree(step, restart_step=15000, max_degree=3):
    """train.py: oneupSHdegree when iteration % 1000 == 0 and iteration > simp_iteration1."""
    ups = step // 1000 - restart_step // 1000
    return int(min(max(ups, 0), max_degree))


# ---------------------------------------------------------------------------- SVQ (C3)
def nearest_code(x, codes, chunk=1 << 15):
    """argmin_j |x - C[j]|^2 (Eq. SVQ index selection), from explicit differences (exact for the
    short sub-vectors of SVQ)."""
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        d2 = (xs[:, None, :] - codes[None]).square().sum(-1)
        out[s:s + chunk] = d2.argmin(dim=-1)
    return out


def _nearest_fast(x, codes, chunk=1 << 18):
    """Expanded-form |x|^2 + |c|^2 - 2 x.c argmin (as cuML/sklearn Lloyd steps)."""
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    c2 = codes.square().sum(-1)
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        out[s:s + chunk] = (c2[None] - 2 * xs @ codes.T).argmin(dim=-1)
    return out


@torch.no_grad()
def kmeans(x, k, max_iter=1000, tol=1e-4, generator=None, init_sample=1 << 16):
    """K-means for one SVQ partition (stands in for cuML KMeans(n_clusters=k, max_iter=1000,
    n_init=1)): k-means++ seeding on a random subsample, then Lloyd iterations until the summed
    squared centre shift is below tol * mean per-dimension variance (sklearn/cuML-style
    tolerance) or max_iter. Empty clusters keep their centre. Returns (centres [k,L], labels)."""
    x = x.float()
    n = x.shape[0]
    dev = x.device
    if n <= k:
        idx = torch.arange(k, device=dev) % n
        centres = x[idx].clone()
        return centres, nearest_code(x, centres)
    perm = torch.randperm(n, device="cpu", generator=generator)
    sample = x[perm[: min(n, init_sample)].to(dev)].double().cpu()  # seeding on the CPU
    first = int(torch.randint(len(sample), (1,), generator=generator))
    picks = [first]
    d2 = (sample - sample[first]).square().sum(-1)
    u = torch.rand(k, generator=generator, dtype=torch.float64)
    for i in range(1, k):
        cdf = torch.cumsum(d2, 0)
        total = float(cdf[-1])
        if total <= 0:
            j = int(torch.randint(len(sample), (1,), generator=generator))
        else:
            j = min(int(torch.searchsorted(cdf, u[i:i + 1] * total)), len(sample) - 1)
        picks.append(j)
        d2 = torch.minimum(d2, (sample - sample[j]).square().sum(-1))
    centres = sample[picks].float().to(dev)
    thresh = tol * float(x.var(dim=0).mean())
    for _ in range(int(max_iter)):
        labels = _nearest_fast(x, centres)
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros_like(centres).index_add_(0, labels, x)
        new = torch.where((counts > 0)[:, None], sums / counts.clamp_min(1)[:, None], centres)
        shift = float((new - centres).square().sum())
        centres = new
        if shift <= thresh:
            break
    return centres, nearest_code(x, centres)


def svq_decode(codes, indices):
    """Eq. SVQ: concatenation of C^(m)[i_m] over the M partitions."""
    return torch.cat([c[i] for c, i in zip(codes, indices)], dim=-1)


# ----------------------------------------------------------------- position coding (C4)
def float16_to_uint16(t):
    """gpcc_utils.float16_to_uint16: fp16 bit pattern XOR the sign bit (lossless)."""
    u = t.half().view(torch.int16).to(torch.int32) & 0xFFFF
    return u ^ 0x8000


def uint16_to_float16(u):
    """gpcc_utils.uint16_to_float16 (inverse of float16_to_uint16)."""
    v = (u.to(torch.int32) ^ 0x8000) & 0xFFFF
    v = torch.where(v >= 0x8000, v - 0x10000, v)
    return v.to(torch.int16).view(torch.float16)


def position_order(u16):
    """gpcc_utils.calculate_morton_order: (x - min) @ (max + 1)^[0,1,2], argsort (stable here)."""
    x = u16.long() - u16.long().min(dim=0, keepdim=True).values
    base = int(x.max()) + 1
    key = x[:, 0] + x[:, 1] * base + x[:, 2] * base * base
    return torch.sort(key, stable=True).indices
