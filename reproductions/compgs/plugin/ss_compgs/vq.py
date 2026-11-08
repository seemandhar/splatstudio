"""K-means vector quantiser of CompGS, ported from the official `scene/kmeans_quantize.py`
(UCDvision/compact3d, MIT licence, commit dccc07e).

Numerics follow `Quantize_kMeans`:
* `assign` = `cluster_assign`: initial centres are a random subset of the vectors the first time
  (warm start from the current centres afterwards); `num_iters` Lloyd iterations (nearest centre
  by Euclidean `cdist` in chunks of 10000, then centre = sum / (count + eps), empty cluster -> 0);
  then one final nearest-centre assignment with the final centres (centres are not re-averaged
  after it, as in the official code).
* `update` = `update_centers`: centre = mean of the current (detached) vectors under the cached
  assignment, `sum / (len + 1e-6)`. The official gathers into a padded [K, max_cnt] table; this
  uses `index_add_` (same sums, different float summation order).
* `quantize` = `forward_*`: straight-through `x - x.detach() + C[idx]`.
"""

import torch


class KMeansQuantizer:
    def __init__(self, num_clusters, num_iters, chunk=10000):
        if num_clusters < 1 or num_iters < 0 or chunk < 1:
            raise ValueError("invalid k-means configuration")
        self.k = int(num_clusters)
        self.iters = int(num_iters)
        self.chunk = int(chunk)
        self.centers = None  # [K, d]
        self.idx = None  # [N] long
        self.assignments = 0

    @property
    def n(self):
        return 0 if self.idx is None else len(self.idx)

    @staticmethod
    def flat(x):
        return x.detach().reshape(len(x), -1)

    @torch.no_grad()
    def nearest(self, x, centers):
        out = []
        for i in range(0, len(x), self.chunk):
            out.append(torch.cdist(x[i:i + self.chunk], centers).argmin(dim=-1))
        return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=x.device)

    @torch.no_grad()
    def means(self, x, idx, eps=1e-6):
        sums = torch.zeros(self.k, x.shape[1], device=x.device, dtype=x.dtype)
        sums.index_add_(0, idx, x)
        counts = torch.bincount(idx, minlength=self.k).to(x.dtype)
        return sums / (counts[:, None] + eps)

    @torch.no_grad()
    def assign(self, x, generator=None):
        x = self.flat(x)
        if self.centers is None or self.centers.shape[1] != x.shape[1]:
            if len(x) >= self.k:
                pick = torch.randperm(len(x), generator=generator, device=x.device)[: self.k]
            else:  # tiny scenes only (probes): sample with replacement to fill K centres
                pick = torch.randint(len(x), (self.k,), generator=generator, device=x.device)
            self.centers = x[pick].clone()
        centers = self.centers
        for _ in range(self.iters):
            centers = self.means(x, self.nearest(x, centers))
        self.centers = centers
        self.idx = self.nearest(x, centers)
        self.assignments += 1

    @torch.no_grad()
    def update(self, x):
        x = self.flat(x)
        if self.idx is None or len(self.idx) != len(x):
            raise RuntimeError("k-means centroid update with stale assignments")
        self.centers = self.means(x, self.idx)

    def decoded(self, shape):
        return self.centers[self.idx].reshape(shape)

    def quantize(self, x):
        if self.idx is None or len(self.idx) != len(x):
            raise RuntimeError("k-means quantisation with stale assignments")
        return x - x.detach() + self.decoded(x.shape)

    def keep(self, mask):
        """Drop pruned rows from the cached assignment (keep = ~pruned)."""
        if self.idx is not None:
            self.idx = self.idx[mask]


FIELDS = {"dc": "sh0", "sh": "shN", "scale": "scales", "rot": "quats"}


class VQState:
    """The four CompGS codebooks (DC, SH rest, log-scale, raw quaternion), one per field."""

    def __init__(self, sizes, iters, chunk=10000):
        self.quantizers = {FIELDS[k]: KMeansQuantizer(s, iters, chunk) for k, s in sizes.items()}
        self.ready = False
        self.decoded = False

    @property
    def n(self):
        return next(iter(self.quantizers.values())).n

    def assign(self, params, generator=None):
        for f, q in self.quantizers.items():
            q.assign(params[f], generator)
        self.ready = True

    def update(self, params):
        for f, q in self.quantizers.items():
            q.update(params[f])

    def quantized(self, params):
        out = dict(params)
        for f, q in self.quantizers.items():
            out[f] = q.quantize(params[f])
        return out

    def keep(self, mask):
        for q in self.quantizers.values():
            q.keep(mask)

    def to_state(self):
        return {f: {"codebook": q.centers.detach().cpu(), "indices": q.idx.to(torch.int32).cpu()}
                for f, q in self.quantizers.items()}


ACTIVE = {"vq": None}
