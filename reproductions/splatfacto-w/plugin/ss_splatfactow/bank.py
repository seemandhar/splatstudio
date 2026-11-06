"""Run-wide learned state of Splatfacto-W that the native runtime cannot hold: per-image appearance
embeddings l_j, the appearance MLP (C1, ``SplatfactoWField``) and the background MLP (C3,
``BGField``), plus the Nerfstudio scene normalisation the official code trains in.

Built by the density in ``set_training_cameras`` and published as ``ACTIVE`` for the renderer; its
Adam (the 7 Nerfstudio parameter groups and schedules) is stepped by the density after every
runtime step; exported to ``density.state["bank"]``; ``ss_splatfactow.evaluate`` rebuilds it.
"""

import torch
from torch import nn

from .schedule import lr_at, nerfstudio_normalisation

ACTIVE = None

BANK_DEFAULTS = {
    "appearance_embed_dim": 48,     # paper §5.1 / code
    "appearance_features_dim": 72,  # paper §5.1 / code
    "app_num_layers": 3,            # nerfstudio MLP(num_layers - 1) + heads
    "app_layer_width": 256,
    "bg_num_layers": 3,
    "bg_layer_width": 128,
    "sh_degree": 3,
    "bg_sh_degree": 4,
    "bg_model": True,
    "scale_factor": 3.0,            # NerfWDataParserConfig.scale_factor
    "masked_cx_ratio": 0.75,        # tools/prepare_sfw.py left-half crops
    "seed": 0,
}

GROUPS = ("appearance_embed", "appearance_model_encoder", "appearance_model_base",
          "appearance_model_rest", "field_background_encoder", "field_background_base",
          "field_background_rest")


def ns_mlp(in_dim, num_layers, width):
    """nerfstudio MLP(in_dim, num_layers - 1, width, out_dim=width, ReLU, out ReLU), torch impl."""
    layers, d = [], in_dim
    n = num_layers - 1
    if n == 1:
        return nn.Sequential(nn.Linear(in_dim, width), nn.ReLU())
    for i in range(n - 1):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*layers)


class SHHead(nn.Module):
    """encoder -> (sh_base [3], sh_rest [(K-1)*3] zero-initialised) -> [.., K, 3]."""

    def __init__(self, in_dim, num_layers, width, degree):
        super().__init__()
        self.k = (degree + 1) ** 2
        self.encoder = ns_mlp(in_dim, num_layers, width)
        self.sh_base_head = nn.Linear(width, 3)
        self.sh_rest_head = nn.Linear(width, (self.k - 1) * 3)
        nn.init.zeros_(self.sh_rest_head.weight)
        nn.init.zeros_(self.sh_rest_head.bias)

    def forward(self, x):
        h = self.encoder(x).float()
        out = torch.cat([self.sh_base_head(h), self.sh_rest_head(h)], dim=-1)
        return out.view(*x.shape[:-1], self.k, 3)


class SfwBank(nn.Module):
    def __init__(self, viewmats, widths, Ks, cfg, device, export=None):
        super().__init__()
        self.cfg = {**BANK_DEFAULTS, **cfg}
        c = self.cfg
        torch.manual_seed(int(c["seed"]))
        vm = torch.stack([v.float() for v in viewmats]).to(device)
        self.register_buffer("viewmats", vm)
        if export is not None:
            R, centre, s = export["ns_R"], export["ns_centre"], float(export["ns_scale"])
            masked = export["masked"]
        else:
            R, centre, s = nerfstudio_normalisation(list(vm), float(c["scale_factor"]))
            masked = torch.tensor([float(K[0, 2]) > float(c["masked_cx_ratio"]) * float(w)
                                   for K, w in zip(Ks, widths)])
        self.register_buffer("ns_R", R.to(device))
        self.register_buffer("ns_centre", centre.to(device))
        self.ns_scale = float(s)
        self.register_buffer("masked", masked.bool().to(device))
        n, e = len(viewmats), int(c["appearance_embed_dim"])
        self.appearance_embeds = nn.Embedding(n, e).to(device)  # N(0, 1) init (nn.Embedding)
        self.color_nn = SHHead(e + int(c["appearance_features_dim"]), int(c["app_num_layers"]),
                               int(c["app_layer_width"]), int(c["sh_degree"])).to(device)
        self.bg_model = (SHHead(e, int(c["bg_num_layers"]), int(c["bg_layer_width"]),
                                int(c["bg_sh_degree"])).to(device) if c["bg_model"] else None)
        groups = {
            "appearance_embed": list(self.appearance_embeds.parameters()),
            "appearance_model_encoder": list(self.color_nn.encoder.parameters()),
            "appearance_model_base": list(self.color_nn.sh_base_head.parameters()),
            "appearance_model_rest": list(self.color_nn.sh_rest_head.parameters()),
        }
        if self.bg_model is not None:
            groups.update({
                "field_background_encoder": list(self.bg_model.encoder.parameters()),
                "field_background_base": list(self.bg_model.sh_base_head.parameters()),
                "field_background_rest": list(self.bg_model.sh_rest_head.parameters()),
            })
        self.optimizer = torch.optim.Adam(
            [{"params": p, "lr": lr_at(g, 0), "name": g} for g, p in groups.items()], eps=1e-15)
        self.step = 0  # completed steps = 0-based Nerfstudio step of the next render
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

    def embedding_for(self, idx):
        """Override > view's own embedding > embedding 0 (code: use_avg_appearance=False)."""
        if self.override_embedding is not None:
            return self.override_embedding
        i = 0 if idx is None else idx
        return self.appearance_embeds.weight[i]

    def gaussian_sh(self, embedding, features):
        """C1: b_i = MLP(l_j, f_i) -> [N, 16, 3]."""
        e = embedding[None].expand(len(features), -1)
        return self.color_nn(torch.cat([e, features], dim=-1))

    def background_sh(self, embedding):
        """C3: b = MLP(l_j) -> [25, 3]."""
        return self.bg_model(embedding[None])[0]

    def to_ns_dirs(self, dirs):
        return dirs @ self.ns_R.T

    # -- optimisation ----------------------------------------------------------------------
    def after_step(self, step):
        """Adam step, then the Nerfstudio LR for the next (0-based) step ``step``."""
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step = int(step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr_at(g["name"], self.step)

    # -- persistence -----------------------------------------------------------------------
    def export(self):
        return {"viewmats": self.viewmats.detach().cpu(), "step": int(self.step),
                "cfg": dict(self.cfg), "ns_R": self.ns_R.cpu(), "ns_centre": self.ns_centre.cpu(),
                "ns_scale": self.ns_scale, "masked": self.masked.cpu(),
                "state": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @torch.no_grad()
    def load_export(self, export):
        dev = self.viewmats.device
        self.load_state_dict({k: v.to(dev) for k, v in export["state"].items()})
        self.step = int(export["step"])

    @classmethod
    def from_export(cls, export, device):
        vms = list(export["viewmats"].to(device))
        bank = cls(vms, None, None, dict(export["cfg"]), device, export=export)
        bank.eval()
        return bank
