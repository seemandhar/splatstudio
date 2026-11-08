"""OMG global learned state: space-feature field and attribute MLPs (C2), SVQ codebooks (C3).

Official: scene/gaussian_model.py construct_net / apply_svq / kmeans / get_svq_*,
gaussian_renderer/__init__.py render_imp (net branch).
- mlp_cont = MLP_s: tcnn NetworkWithInputEncoding(3 -> Frequency(16) = 96 -> 64 ReLU -> 13).
- mlp_dc = MLP_t, mlp_opacity = MLP_o: Network(16 = [F(13), T(3)] -> 64 LeakyReLU -> 3 / 1).
- mlp_view = MLP_v: Network(16 = [F(13), V(3)] -> 64 LeakyReLU -> 45).
tcnn trains these in fp16; here they are fp32 with the same layouts and init, rounded to fp16
when encoded.

The runtime has no global learned parameters, so the density component owns an `OMGBank` (with
its own Adam optimisers), registers it in `ACTIVE`, and steps it. The renderer reads it. Pattern
from the Compact-3DGS / EAGLES ports (reproductions/compact-3dgs/plugin/ss_compact_3dgs/model.py).
After the final encode -> decode, the decoded model is a plain 3DGS state, so a fresh process
(ss-eval) renders it without the bank.
"""

import torch
from torch import nn

from . import core

# Process-local registry: set by OMGDensity.initialize, read by OMGRenderer.render.
ACTIVE = {"bank": None}

NET_DEFAULTS = {
    "n_frequencies": 16,
    "space_dim": 13,
    "feature_dim": 3,
    "n_neurons": 64,
    "n_hidden_layers": 1,
    "sh_rest": 15,
    "seed": 1337,
}


class OMGBank(nn.Module):
    def __init__(self, net=None, svq_itr=29000, net_itr=15000):
        super().__init__()
        c = {**NET_DEFAULTS, **(net or {})}
        unknown = set(c) - set(NET_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown net options {sorted(unknown)}")
        self.net = c
        g = torch.Generator().manual_seed(int(c["seed"]))
        nn_, hl = c["n_neurons"], c["n_hidden_layers"]
        n_enc = 3 * 2 * c["n_frequencies"]
        n_app = c["space_dim"] + c["feature_dim"]
        self.out_dims = {"cont": c["space_dim"], "view": 3 * c["sh_rest"], "dc": 3, "opacity": 1}
        self.mlps = nn.ModuleDict()
        for name, n_in in (("cont", n_enc), ("view", n_app), ("dc", n_app), ("opacity", n_app)):
            weights = core.mlp_init(core.mlp_shapes(n_in, self.out_dims[name], nn_, hl), g)
            self.mlps[name] = nn.ParameterList([nn.Parameter(w) for w in weights])
        self.activation = {"cont": "relu", "view": "leaky_relu", "dc": "leaky_relu",
                           "opacity": "leaky_relu"}
        self.svq_itr = int(svq_itr)
        self.net_itr = int(net_itr)
        self.net_enabled = False
        self.vq_enabled = False
        self.baked = False
        self.step = 1  # step number of the next training render
        self.codes = {}  # attr -> list of codebook Parameters
        self.indices = {}  # attr -> list of long tensors
        self.optimizer = None
        self.code_optimizer = None

    # ------------------------------------------------------------------ optimisers
    def build_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(list(self.mlps.parameters()), lr=lr, eps=1e-15)

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------ appearance (Eq. 3-4)
    def run(self, name, x, weights=None):
        w = list(self.mlps[name]) if weights is None else weights[name]
        return core.mlp_forward(w, x, self.out_dims[name], self.activation[name])

    def space_feature(self, means, weights=None):
        """F_n = MLP_s(gamma(contract(p_n.detach())))."""
        enc = core.frequency_encoding(core.contract_to_unisphere(means.detach()),
                                      self.net["n_frequencies"])
        return self.run("cont", enc, weights)

    def appearance(self, means, static, view, weights=None):
        """-> sh0 [N,1,3], shN [N,15,3], opacity logits [N] (official render_imp / decode)."""
        f = self.space_feature(means, weights)
        space = torch.cat([f, static], dim=-1)
        viewf = torch.cat([f, view], dim=-1)
        shn = self.run("view", viewf, weights).reshape(-1, self.net["sh_rest"], 3)
        dc = self.run("dc", space, weights).reshape(-1, 1, 3)
        opacity = self.run("opacity", space, weights)[:, 0]
        return dc, shn, opacity

    def weights_list(self):
        return {k: [w for w in v] for k, v in self.mlps.items()}

    def param_count(self):
        return sum(p.numel() for p in self.mlps.parameters())

    # ------------------------------------------------------------------ SVQ (C3)
    def svq_values(self, attr):
        return core.svq_decode(self.codes[attr], self.indices[attr])

    @torch.no_grad()
    def apply_svq(self, data, slices, clusters, lr, max_iter, tol, seed):
        """Official apply_svq: K-means once per sub-vector, indices frozen, codebooks learnable
        (Adam lr, eps 1e-15)."""
        g = torch.Generator().manual_seed(int(seed))
        params = []
        for attr, x in data.items():
            length, k = int(slices[attr]), int(clusters[attr])
            if x.shape[1] % length:
                raise ValueError(f"invalid sub-vector length for {attr}")
            self.codes[attr], self.indices[attr] = [], []
            for m in range(x.shape[1] // length):
                centres, labels = core.kmeans(x[:, m * length:(m + 1) * length], k, max_iter,
                                              tol, g)
                p = nn.Parameter(centres.contiguous())
                self.codes[attr].append(p)
                self.indices[attr].append(labels)
                params.append(p)
        self.code_optimizer = torch.optim.Adam(params, lr=lr, eps=1e-15)
        self.vq_enabled = True
