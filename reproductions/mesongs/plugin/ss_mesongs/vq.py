"""C6 vector quantisation of the 1+-degree SH coefficients (MesonGS `vq.py`, "borrowed from
c3dgs"): copied from the c3dgs plugin (reproductions/c3dgs/plugin/ss_c3dgs/vq.py), whose
`vq_features` / `update` follow the official c3dgs `compression/vq.py` that MesonGS vendors
unchanged (same `weighted_distance` kernel, identical file; csrc/ is a copy of the c3dgs plugin's).

Nearest-centroid search uses the official `weighted_distance` CUDA kernel (csrc/, JIT-built;
plain squared L2 with a first-minimum loop: the per-vector weight S(x) of the paper's distance
does not change the argmin). `torch_scatter.scatter(..., reduce="sum")` is replaced by
`index_add_` (same sums). MesonGS calls it with scale_normalize=False.
"""

from pathlib import Path

import torch

CSRC = Path(__file__).parent / "csrc"
_EXT = None
EPS = 1e-5  # official VectorQuantize.eps


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="ss_mesongs_weighted_distance",
            sources=[str(CSRC / "weighted_distance.cu"), str(CSRC / "ext.cpp")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def nearest(x, codebook):
    """(min squared distance, index) per row of x — official weightedDistance kernel."""
    if not x.is_cuda:
        raise ValueError("the official nearest-centroid kernel needs CUDA tensors")
    return extension().weightedDistance(x.float().contiguous(), codebook.float().contiguous())


def nearest_reference(x, codebook, chunk=4096):
    """Brute-force float64 reference (probes / CPU tests): first minimum on ties."""
    dists, idx = [], []
    c = codebook.double()
    for s in range(0, len(x), chunk):
        d = (x[s : s + chunk].double()[:, None, :] - c[None]).square().sum(-1)
        m, i = d.min(dim=1)
        dists.append(m.float())
        idx.append(i)
    return torch.cat(dists), torch.cat(idx)


def ema_inplace(moving_avg, new, decay):
    moving_avg.mul_(decay).add_(new, alpha=(1 - decay))


@torch.no_grad()
def update(codebook, x, importance, decay, assign=nearest):
    """Official VectorQuantize.update (codebook in place); returns the batch min distances.

    new_k = sum_{x in A(k)} S(x) x / (sum_{x in A(k)} S(x) + eps)  (paper centroid equation)
    c_k <- decay c_k + (1 - decay) new_k  (mini-batch EMA; empty clusters decay towards 0)
    """
    min_dists, idx = assign(x, codebook)
    k = codebook.shape[0]
    acc = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, idx, importance)
    sums = torch.zeros_like(codebook).index_add_(0, idx, x * importance[:, None])
    ema_inplace(codebook, sums / (acc[:, None] + EPS), decay)
    return min_dists


@torch.no_grad()
def vq_features(features, importance, codebook_size, batch_size, steps, decay=0.8,
                scale_normalize=False, generator=None, assign=nearest):
    """Official vq_features: uniform init in [min, max] of all values, `steps` mini-batch
    updates of `batch_size` random vectors (with replacement), optional trace re-normalisation
    of covariance codebooks (entries 0, 3, 5) after every step, final assignment."""
    importance_n = importance / importance.max()
    lo, hi = features.aminmax()
    codebook = torch.rand(codebook_size, features.shape[-1], device=features.device,
                          generator=generator) * (hi - lo) + lo
    for _ in range(steps):
        batch = torch.randint(0, features.shape[0], (batch_size,), device=features.device,
                              generator=generator)
        update(codebook, features[batch], importance_n[batch], decay, assign)
        if scale_normalize:
            tr = codebook[:, [0, 3, 5]].sum(-1)
            codebook /= tr[:, None]
    _, indices = assign(features, codebook)
    return codebook, indices
