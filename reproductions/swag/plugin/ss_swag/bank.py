"""Run-wide learned state of SWAG that the native runtime cannot hold (paper Fig. 2): per-image
appearance embeddings l_I, the hash grid emb(x) on Gaussian centres and the MLP F_theta.

Created by the density in ``set_training_cameras``, published as ``ACTIVE`` for the renderer,
stepped by its own Adam after every runtime step, exported to ``density.state["bank"]``.
``ss_swag.evaluate`` rebuilds it from the checkpoint.
"""

import torch
from torch import nn

from .hashgrid import HashGrid

ACTIVE = None

BANK_DEFAULTS = {
    "embedding_dim": 24,          # §5.1 n = 24
    "embedding_init_std": 1.0,    # A5 (nn.Embedding default, NeRF-W)
    "hash_levels": 12,            # §5.1 L
    "hash_features": 2,           # 51 = 3 + 24 + 12*2
    "hash_log2_table": 19,        # T = 2^19
    "hash_n_min": 16,
    "hash_n_max": 2048,
    "mlp_hidden": 64,             # §5.1
    "mlp_layers": 3,              # A1: hidden layers
    "lr": 0.0025,                 # supp. Tab. 7 (embeddings, hash grid, MLP)
    "aabb_quantile": 0.005,       # A2
    "aabb_margin": 0.05,
    "seed": 0,
}


class SwagMLP(nn.Module):
    """F_theta: (c, emb(x), l_I) -> (c^I logits [3], delta alpha^I [1]) (Eq. 7/8)."""

    def __init__(self, in_dim, hidden, layers, seed=0):
        super().__init__()
        torch.manual_seed(int(seed))
        mods, d = [], in_dim
        for _ in range(int(layers)):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, 4))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def robust_aabb(points, q, margin):
    lo = torch.quantile(points.float(), q, dim=0)
    hi = torch.quantile(points.float(), 1 - q, dim=0)
    pad = (hi - lo) * margin
    return torch.stack([lo - pad, hi + pad])


class SwagBank(nn.Module):
    def __init__(self, viewmats, points, cfg, device, export=None):
        super().__init__()
        self.cfg = {**BANK_DEFAULTS, **cfg}
        c = self.cfg
        g = torch.Generator().manual_seed(int(c["seed"]))
        self.register_buffer("viewmats", torch.stack([v.float() for v in viewmats]).to(device))
        if export is not None:
            aabb = export["aabb"]
        else:
            aabb = robust_aabb(points.detach().cpu(), float(c["aabb_quantile"]),
                               float(c["aabb_margin"]))
        self.register_buffer("aabb", aabb.to(device).float())
        n, d = len(viewmats), int(c["embedding_dim"])
        emb = torch.randn(n, d, generator=g) * float(c["embedding_init_std"])
        self.image_embeddings = nn.Parameter(emb.to(device))
        self.grid = HashGrid(c["hash_levels"], c["hash_features"], c["hash_log2_table"],
                             c["hash_n_min"], c["hash_n_max"], seed=c["seed"]).to(device)
        self.mlp = SwagMLP(3 + self.grid.out_dim + d, int(c["mlp_hidden"]),
                           int(c["mlp_layers"]), seed=c["seed"]).to(device)
        self.optimizer = torch.optim.Adam(
            [self.image_embeddings, *self.grid.parameters(), *self.mlp.parameters()],
            lr=float(c["lr"]), eps=1e-15)
        self.step = 0
        self._eval = {"override": None}
        if export is not None:
            self.load_export(export)

    # -- queries ---------------------------------------------------------------------------
    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.to(self.viewmats.device).float()).abs().flatten(1).sum(1)
        i = int(torch.argmin(d))
        return i if float(d[i]) < 1e-4 else None

    def set_override(self, value):
        self._eval["override"] = value

    @property
    def override_embedding(self):
        return self._eval["override"]

    def mean_embedding(self):
        return self.image_embeddings.detach().mean(0)

    def embedding_for(self, idx):
        """Override (evaluation) > training view's embedding > mean training embedding (A7)."""
        if self.override_embedding is not None:
            return self.override_embedding
        if idx is not None:
            return self.image_embeddings[idx]
        return self.mean_embedding()

    def normalise(self, means):
        lo, hi = self.aabb[0], self.aabb[1]
        return ((means - lo) / (hi - lo)).clamp(0.0, 1.0)

    def mlp_outputs(self, colors, means, embedding):
        """Eq. 8 for every Gaussian: [N, 4] = F(c, emb(x), l_I)."""
        h = self.grid(self.normalise(means))
        e = embedding[None].expand(len(means), -1)
        return self.mlp(torch.cat([colors, h, e], dim=-1))

    # -- optimisation ----------------------------------------------------------------------
    def after_step(self, step):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step = step

    # -- persistence -----------------------------------------------------------------------
    def export(self):
        return {"viewmats": self.viewmats.detach().cpu(), "aabb": self.aabb.detach().cpu(),
                "step": int(self.step), "cfg": dict(self.cfg),
                "image_embeddings": self.image_embeddings.detach().cpu(),
                "grid": {k: v.detach().cpu() for k, v in self.grid.state_dict().items()},
                "mlp": {k: v.detach().cpu() for k, v in self.mlp.state_dict().items()}}

    @torch.no_grad()
    def load_export(self, export):
        dev = self.viewmats.device
        self.image_embeddings.copy_(export["image_embeddings"].to(dev))
        self.grid.load_state_dict({k: v.to(dev) for k, v in export["grid"].items()})
        self.mlp.load_state_dict({k: v.to(dev) for k, v in export["mlp"].items()})
        self.step = int(export["step"])

    @classmethod
    def from_export(cls, export, device):
        bank = cls(list(export["viewmats"].to(device)), None, dict(export["cfg"]), device,
                   export=export)
        bank.eval()
        return bank
