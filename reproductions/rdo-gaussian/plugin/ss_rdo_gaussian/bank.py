"""RDO-Gaussian global learned state: six ECVQ codebooks + learned entropy models.

Ported from the official `vq/ecvq.py`, `vq/kmeans.py` and `utils/entropy_model.py`
(https://github.com/USTC-IMCL/RDO-Gaussian, commit 710ebab8). The native runtime cannot hold
global learned tensors, so the density component owns an `RDOBank` (with its own Adam, as the
official `optimizer_vq`) and the renderer reads it through `ACTIVE` (pattern of the EAGLES /
Compact-3DGS ports).

Per key, the codebook is stored as [K, d] and the logits as [K] (official: [1, K, d] / [1, K],
one sub-codebook); files written by the codec use the official [1, ...] shapes.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

KEYS = ("scale", "rot", "dc", "sh1", "sh2", "sh3")
DIMS = {"scale": 3, "rot": 4, "dc": 3, "sh1": 9, "sh2": 15, "sh3": 21}
VQ_DIM = sum(DIMS.values())  # official renderer `vq_dim` = 55
SH_SLICES = {"sh1": (0, 3), "sh2": (3, 8), "sh3": (8, 15)}  # shN coefficient ranges
SH_KEYS = ("sh1", "sh2", "sh3")
LOG_CLAMP = math.log(1e-9)  # official Softmax.log_pmf clamp (only while it requires grad)

ACTIVE = {"bank": None}


def log2_pmf(logits):
    """Official `Softmax(logits).log_pmf() / (-log 2)`: bits per codeword.

    The clamp at log(1e-9) applies only when the result requires grad (training renders), as
    in the official code; under no_grad (final encode, reactivation) it is unclamped.
    """
    lp = F.log_softmax(logits, dim=-1)
    if lp.requires_grad:
        lp = torch.clamp_min(lp, LOG_CLAMP)
    return lp / (-math.log(2))


@torch.no_grad()
def assign(x, codebook, bits=None, lmbda=None, patch_size=65536):
    """Official ECVQ.quant index: argmin_m ||x - c_m||_2 (+ bits_m / lambda if rate on).

    `torch.cdist(p=2)` (Euclidean, not squared) on [1, b, d] x [1, K, d] patches, first index on
    ties (torch argmin).
    """
    out = []
    for b in x.detach().split(patch_size, dim=0):
        dist = torch.cdist(b[None], codebook.detach()[None], p=2)[0]
        if bits is not None:
            dist = dist + bits.detach()[None, :] / lmbda
        out.append(dist.argmin(dim=-1))
    if not out:
        return torch.zeros(0, dtype=torch.long, device=x.device)
    return torch.cat(out)


@torch.no_grad()
def kmeans(samples, num_clusters, num_iters=10, patch_size=65536, generator=None):
    """Official vq/kmeans.py (Euclidean, patched): returns (means [K, d], bins [K]).

    Initial means are `randperm(M)[:K]` samples (with replacement `randint` if M < K); each
    iteration assigns by argmax(-cdist), keeps the old mean of empty clusters; the returned bins
    are the counts of the last assignment.
    """
    m, d = samples.shape
    device = samples.device
    if m >= num_clusters:
        idx = torch.randperm(m, device=device, generator=generator)[:num_clusters]
    else:
        idx = torch.randint(0, m, (num_clusters,), device=device, generator=generator)
    means = samples[idx].clone()
    bins = torch.zeros(num_clusters, dtype=torch.long, device=device)
    for _ in range(num_iters):
        buckets = []
        for b in samples.split(patch_size, dim=0):
            buckets.append((-torch.cdist(b[None], means[None], p=2)[0]).argmax(dim=-1))
        buckets = torch.cat(buckets)
        bins = torch.zeros(num_clusters, dtype=torch.long, device=device)
        bins.scatter_add_(0, buckets, torch.ones_like(buckets))
        zero = bins == 0
        new = torch.zeros(num_clusters, d, dtype=samples.dtype, device=device)
        new.scatter_add_(0, buckets[:, None].expand(-1, d), samples)
        new = new / bins.masked_fill(zero, 1)[:, None].to(samples.dtype)
        means = torch.where(zero[:, None], means, new)
    return means, bins


class RDOBank(nn.Module):
    """Codebooks, logits, their Adam, the index cache and the activation flags."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.codebooks = nn.ParameterDict()
        self.logits = nn.ParameterDict()
        for key in KEYS:
            k, d = int(cfg["codebook_size"][key]), DIMS[key]
            # official init normal(0, 1/sqrt(d)) and zero logits; replaced by k-means at start
            self.codebooks[key] = nn.Parameter(torch.randn(k, d) / math.sqrt(d))
            self.logits[key] = nn.Parameter(torch.zeros(k))
        self.optimizer = None
        self.gs_active = False
        self.sh_active = False
        self.vq_active = False
        self.rate_constrain = False
        self.baked = False
        self.vq_steps = 0  # optimizer_vq / scheduler_vq steps taken
        self.render_step = 1  # step of the next training render (index cache schedule)
        self.index_cache = {}

    @property
    def any_active(self):
        return self.gs_active or self.sh_active or self.vq_active

    def build_optimizer(self):
        # official training_setup: Adam([codebooks] lr cb_lr, [logits] lr logits_lr, eps 1e-15)
        self.optimizer = torch.optim.Adam(
            [{"params": [self.codebooks[k] for k in KEYS], "lr": self.cfg["codebook_lr"]},
             {"params": [self.logits[k] for k in KEYS], "lr": self.cfg["logits_lr"]}],
            lr=0.0, eps=1e-15)

    def lr_factor(self):
        """MultiStepLR([milestone], gamma) after `vq_steps` scheduler steps."""
        return self.cfg["lr_gamma"] if self.vq_steps >= self.cfg["lr_milestone"] else 1.0

    def step_optimizer(self):
        factor = self.lr_factor()
        self.optimizer.param_groups[0]["lr"] = self.cfg["codebook_lr"] * factor
        self.optimizer.param_groups[1]["lr"] = self.cfg["logits_lr"] * factor
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.vq_steps += 1

    def update_index(self):
        """Official `update_index=(iteration - 1) % 10 == 0`."""
        return (self.render_step - 1) % int(self.cfg["index_update_every"]) == 0

    def bits(self, key):
        return log2_pmf(self.logits[key])

    def quantize(self, key, x, index=None):
        """Official ECVQ.forward: (x_hat, bits sum, index); index=None recomputes it."""
        bits = self.bits(key)
        if index is None:
            lmbda = float(self.cfg["lmbda"][key]) if self.rate_constrain else None
            index = assign(x, self.codebooks[key], bits if self.rate_constrain else None,
                           lmbda, int(self.cfg["patch_size"]))
        x_hat = self.codebooks[key][index]
        return x_hat, bits[index].sum(), index

    @torch.no_grad()
    def kmeans_initialize(self, key, samples):
        """Official ECVQ.kmeans_initialize: codebook = k-means means, logits = log(normalize(bins))."""
        means, bins = kmeans(samples.detach().float(), self.codebooks[key].shape[0],
                             int(self.cfg["kmeans_iters"]), int(self.cfg["patch_size"]))
        self.codebooks[key].data.copy_(means)
        self.logits[key].data.copy_(torch.log(F.normalize(bins.to(torch.float32), dim=-1)))

    @torch.no_grad()
    def live_mask(self, key, prob_threshold):
        """Official codebook_info: live = log_pmf > log(threshold)."""
        return F.log_softmax(self.logits[key], dim=-1) > math.log(prob_threshold)

    @torch.no_grad()
    def reactivate(self, key, prob_threshold):
        """Official ECVQ.reactivate_codeword: dead codewords copy a live one sampled by p (+N(0,1e-4))
        and its logit."""
        log_pmf = F.log_softmax(self.logits[key], dim=-1)
        live = log_pmf > math.log(prob_threshold)
        n_dead = int((~live).sum())
        if n_dead == 0 or not bool(live.any()):
            return 0
        cb, lg = self.codebooks[key].data, self.logits[key].data
        pmf = log_pmf[live].exp()
        idx = pmf.multinomial(n_dead, replacement=True)
        disturb = torch.normal(0, 1e-4, size=(n_dead, cb.shape[1])).to(cb.device)
        cb[~live] = cb[live][idx] + disturb
        lg[~live] = lg[live][idx]
        return n_dead

    def cache(self, key, n, device):
        """Global index cache [N] (official index_cache_global_dict, zeros at creation)."""
        c = self.index_cache.get(key)
        if c is None or len(c) != n:
            c = torch.zeros(n, dtype=torch.long, device=device)
            self.index_cache[key] = c
        return c
