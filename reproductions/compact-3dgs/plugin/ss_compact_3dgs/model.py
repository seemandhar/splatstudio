"""Compact-3DGS global learned state: neural colour field (hash grid + tiny MLP) and R-VQ codebooks.

Official sources: scene/gaussian_model.py (GaussianModel.__init__, training_setup,
contract_to_unisphere), gaussian_renderer/__init__.py (render), and the pinned libraries
vector-quantize-pytorch 1.8.1 (ResidualVQ / VectorQuantize / EuclideanCodebook, MIT) and
tiny-cuda-nn (SphericalHarmonics, FullyFusedMLP, BSD-3) whose semantics are ported here.

The runtime has no parameter type for global learned modules, so `compact-density` owns a
`CompactBank` (with its own Adam), registers it in `ACTIVE`, steps it, and records the next step in
`bank.step`; `compact-renderer` reads it (pattern copied from the EAGLES port's DecoderBank,
reproductions/eagles/plugin/ss_eagles/decoders.py; not imported).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .hashgrid import HashGrid

# Process-local registry: set by CompactDensity.initialize, read by CompactRenderer.render.
ACTIVE = {"bank": None}

SH_C0 = 0.28209479177387814

NET_DEFAULTS = {
    "n_levels": 16,
    "n_features": 2,
    "log2_hashmap_size": 19,  # max_hashmap
    "base_resolution": 16,
    "per_level_scale": 1.447,
    "sh_encoding_degree": 3,  # tcnn "degree" 3 -> 9 outputs
    "n_neurons": 64,
    "n_hidden_layers": 2,
    "tcnn_direction_quirk": True,  # official feeds [-1,1] dirs to tcnn's [0,1] SH input
    "seed": 1337,
}


# ----------------------------------------------------------------------------- colour field
def contract_to_unisphere(x):
    """Official contract_to_unisphere with aabb [-1,1]^3 (identity normalisation):
    paper Eq. 8, x if ||x|| <= 1 else (2 - 1/||x||) x/||x||, then mapped to [0,1] by x/4 + 0.5."""
    mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    outside = mag > 1
    safe = torch.where(outside, mag, torch.ones_like(mag))
    x = torch.where(outside, (2 - 1 / safe) * (x / safe), x)
    return x / 4 + 0.5


def sh_encoding_tcnn(d, degree=3, quirk=True):
    """tcnn SphericalHarmonics (common_device.h sh_enc) for degree <= 3, degree^2 outputs.

    tcnn evaluates the polynomials at 2*input - 1 (it expects directions mapped to [0,1]). The
    official code passes unit directions in [-1,1] directly, so with `quirk` the polynomials are
    evaluated at 2d - 1 exactly as in the official runs.
    """
    if degree not in (1, 2, 3):
        raise ValueError("sh_encoding_degree must be 1..3 (official 3)")
    v = 2 * d - 1 if quirk else d
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    out = [torch.full_like(x, 0.28209479177387814)]
    if degree >= 2:
        out += [-0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x]
    if degree >= 3:
        xy, yz, xz, x2, y2, z2 = x * y, y * z, x * z, x * x, y * y, z * z
        out += [1.0925484305920792 * xy, -1.0925484305920792 * yz,
                0.94617469575755997 * z2 - 0.31539156525251999, -1.0925484305920792 * xz,
                0.54627421529603959 * x2 - 0.54627421529603959 * y2]
    return torch.stack(out, dim=-1)


def padded(n, multiple=16):
    return (n + multiple - 1) // multiple * multiple


def mlp_shapes(n_in, n_neurons=64, n_hidden_layers=2, n_out=3):
    """tcnn FullyFusedMLP weight shapes (no biases; input and output padded to multiples of 16)."""
    shapes = [(n_neurons, padded(n_in))]
    shapes += [(n_neurons, n_neurons)] * (n_hidden_layers - 1)
    shapes.append((padded(n_out), n_neurons))
    return shapes


def mlp_forward(weights, inp, n_out=3):
    """tcnn Network = Identity encoding (pads the input with ones) + FullyFusedMLP (ReLU, no bias,
    output activation None); the padded output rows are discarded."""
    pad = weights[0].shape[1] - inp.shape[1]
    h = torch.cat([inp, torch.ones(inp.shape[0], pad, device=inp.device, dtype=inp.dtype)], 1)
    for w in weights[:-1]:
        h = F.relu(h @ w.T)
    return (h @ weights[-1].T)[:, :n_out]


def view_dirs(means, center):
    d = means - center[None, :]
    return d / d.norm(dim=1, keepdim=True)


def camera_center(viewmat):
    return -viewmat[:3, :3].T @ viewmat[:3, 3]


def color_sh0(weights, features, means, center, degree=3, quirk=True):
    """Official render(): shs = mlp_head(cat[recolor(xyz), direction_encoding(dir_pp)])."""
    enc = sh_encoding_tcnn(view_dirs(means, center), degree, quirk)
    return mlp_forward(weights, torch.cat([features, enc], dim=-1))


# ------------------------------------------------------------------------------------ R-VQ
def nearest_code(x, embed, chunk=1 << 20):
    """argmin_k ||x - Z[k]||^2 (vector-quantize-pytorch cdist + argmax(-dist), Eq. 5)."""
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    y2 = embed.square().sum(-1)
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        d2 = xs.square().sum(-1, keepdim=True) + y2[None] - 2 * xs @ embed.T
        out[s:s + chunk] = d2.argmin(dim=-1)
    return out


@torch.no_grad()
def kmeans(samples, num_clusters, num_iters=1, generator=None):
    """vector-quantize-pytorch 1.8.1 kmeans (batched_sample_vectors init by randperm, Euclidean,
    empty clusters keep their previous mean)."""
    n = samples.shape[0]
    if n >= num_clusters:
        pick = torch.randperm(n, device=samples.device, generator=generator)[:num_clusters]
    else:
        pick = torch.randint(0, n, (num_clusters,), device=samples.device, generator=generator)
    means = samples[pick].clone()
    for _ in range(num_iters):
        buckets = nearest_code(samples, means)
        bins = torch.bincount(buckets, minlength=num_clusters)
        new = torch.zeros_like(means).index_add_(0, buckets, samples)
        new = new / bins.clamp_min(1)[:, None].to(samples.dtype)
        means = torch.where((bins == 0)[:, None], means, new)
    return means


class VQStage(nn.Module):
    """One VectorQuantize layer as configured by the official code: Euclidean codebook,
    kmeans_init (kmeans_iters), learnable_codebook, ema_update False, commitment_weight 0,
    in_place_codebook_optimizer Adam(lr). Training forward: kmeans init on first use, nearest code,
    one Adam step on mse(Z[i], x.detach()) (paper L_r / L_s, Eq. 6), re-quantise, straight-through."""

    def __init__(self, dim, codebook_size, kmeans_iters=1, lr=1e-4):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(codebook_size, dim))
        self.register_buffer("initted", torch.tensor(False))
        self.kmeans_iters = int(kmeans_iters)
        self.lr = float(lr)
        self.optimizer = None

    def init_codebook(self, x, generator=None):
        with torch.no_grad():
            self.embed.copy_(kmeans(x.detach().float(), self.embed.shape[0], self.kmeans_iters,
                                    generator))
            self.initted.fill_(True)
        self.optimizer = torch.optim.Adam([self.embed], lr=self.lr)

    def forward(self, x, train):
        if not bool(self.initted):
            self.init_codebook(x)
        xd = x.detach()
        ind = nearest_code(xd, self.embed.detach())
        if train:
            with torch.enable_grad():
                loss = F.mse_loss(self.embed[ind], xd)
                (grad,) = torch.autograd.grad(loss, self.embed)
            self.embed.grad = grad
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            ind = nearest_code(xd, self.embed.detach())
            quantize = self.embed.detach()[ind]
            return x + (quantize - x).detach(), ind  # straight-through
        return self.embed.detach()[ind], ind


class ResidualVQ(nn.Module):
    """vector-quantize-pytorch ResidualVQ: residual -= quantized.detach(); out += quantized.
    With the per-stage straight-through estimator the input receives L x the output gradient."""

    def __init__(self, dim, codebook_size=64, num_quantizers=6, kmeans_iters=1, lr=1e-4):
        super().__init__()
        self.layers = nn.ModuleList(VQStage(dim, codebook_size, kmeans_iters, lr)
                                    for _ in range(num_quantizers))

    def forward(self, x, train=True):
        residual, out, inds = x, 0.0, []
        for layer in self.layers:
            q, ind = layer(residual, train)
            residual = residual - q.detach()
            out = out + q
            inds.append(ind)
        return out, torch.stack(inds, dim=-1)

    def decode(self, indices):
        return sum(layer.embed.detach()[indices[:, k]] for k, layer in enumerate(self.layers))

    @property
    def initted(self):
        return all(bool(layer.initted) for layer in self.layers)


# ----------------------------------------------------------------------------------- bank
def net_lr(base_lr, epoch, warmup_iters=100, warmup_start=0.01, milestones=(5000, 15000, 25000),
           gamma=0.33):
    """Closed form of ChainedScheduler([LinearLR(start_factor, total_iters), MultiStepLR]) after
    `epoch` scheduler steps (official training_setup; stepped once per iteration)."""
    linear = warmup_start + (1 - warmup_start) * min(epoch, warmup_iters) / warmup_iters
    return base_lr * linear * gamma ** sum(epoch >= m for m in milestones)


class CompactBank(nn.Module):
    def __init__(self, net=None, rvq_size=64, rvq_num=6, kmeans_iters=1, codebook_lr=1e-4,
                 rvq_iter=29000):
        super().__init__()
        self.rvq_iter = int(rvq_iter)  # R-VQ renders for steps > rvq_iter (official itr > 29000)
        c = {**NET_DEFAULTS, **(net or {})}
        unknown = set(c) - set(NET_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown net options {sorted(unknown)}")
        self.net = c
        self.hash = HashGrid(c["n_levels"], c["n_features"], c["log2_hashmap_size"],
                             c["base_resolution"], c["per_level_scale"], c["seed"])
        n_in = self.hash.n_output_dims + int(c["sh_encoding_degree"]) ** 2
        g = torch.Generator().manual_seed(int(c["seed"]) + 1)
        weights = []
        for fan_out, fan_in in mlp_shapes(n_in, c["n_neurons"], c["n_hidden_layers"]):
            bound = math.sqrt(6.0 / (fan_in + fan_out))  # tcnn initialize_xavier_uniform
            weights.append(nn.Parameter((torch.rand(fan_out, fan_in, generator=g) * 2 - 1) * bound))
        self.mlp = nn.ParameterList(weights)
        self.n_in = n_in
        self.vq_scale = ResidualVQ(3, rvq_size, rvq_num, kmeans_iters, codebook_lr)
        self.vq_rot = ResidualVQ(4, rvq_size, rvq_num, kmeans_iters, codebook_lr)
        self.rvq_bits = int(round(math.log2(rvq_size)))
        self.rvq_num = int(rvq_num)
        self.optimizer = None
        self.step = 1  # step number of the next training render
        self.baked = False

    def build_optimizer(self, lr):
        self.optimizer = torch.optim.Adam([self.hash.params, *self.mlp], lr=lr, eps=1e-15)

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def features(self, means, hash_params=None):
        return self.hash(contract_to_unisphere(means.detach()), hash_params)

    def color(self, means, center, features=None):
        feats = self.features(means) if features is None else features
        return color_sh0(list(self.mlp), feats, means, center, self.net["sh_encoding_degree"],
                         self.net["tcnn_direction_quirk"])

    def mlp_param_count(self):
        """tcnn mlp_head.params.shape[0] (padded shapes): 8192 for the official config."""
        return sum(w.numel() for w in self.mlp)
