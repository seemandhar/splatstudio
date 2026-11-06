"""Run-wide learned state of WildGaussians that the native runtime cannot hold: per-image
appearance embeddings, the shared appearance MLP and the uncertainty predictor (plus the frozen
DINOv2 backbone and its GT-feature cache).

The density component creates the bank from the training cameras (`set_training_cameras`),
publishes it as ``ACTIVE`` for the renderer and the losses, steps its Adam after every runtime
step, and exports it to ``density.state["bank"]`` (saved in the checkpoint).
`ss_wildgaussians.evaluate` rebuilds it from the checkpoint. ``ss-eval`` has no bank.
"""

import torch
from torch import nn

from .appearance import AppearanceMLP
from .dino import DEFAULT_WEIGHTS, load_backbone
from .uncertainty import DinoFeatures, UncertaintyHead

ACTIVE = None  # live bank of the current process (training, or evaluate.py)

BANK_DEFAULTS = {
    "appearance_enabled": True,
    "uncertainty_enabled": True,
    "appearance_embedding_dim": 32,
    "gaussian_embedding_dim": 24,      # 6 * appearance_n_fourier_freqs
    "appearance_embedding_lr": 1e-3,
    "appearance_mlp_lr": 5e-4,
    "uncertainty_lr": 1e-3,
    "uncertainty_dropout": 0.5,
    "uncertainty_clip_min": 0.1,
    "dino_weights": DEFAULT_WEIGHTS,
    "seed": 0,
}


class WildBank(nn.Module):
    def __init__(self, viewmats, cfg, device, export=None):
        super().__init__()
        self.cfg = {**BANK_DEFAULTS, **cfg}
        c = self.cfg
        g = torch.Generator(device="cpu").manual_seed(int(c["seed"]))
        n = len(viewmats)
        self.register_buffer("viewmats", torch.stack([v.float() for v in viewmats]).to(device))
        self.appearance = bool(c["appearance_enabled"])
        self.uncertainty = bool(c["uncertainty_enabled"])
        groups = []
        if self.appearance:
            # set_num_training_images: normal_(0, 0.01)
            emb = torch.randn(n, int(c["appearance_embedding_dim"]), generator=g) * 0.01
            self.image_embeddings = nn.Parameter(emb.to(device))
            torch.manual_seed(int(c["seed"]))
            self.mlp = AppearanceMLP(int(c["appearance_embedding_dim"]),
                                     int(c["gaussian_embedding_dim"])).to(device)
            groups += [{"params": [self.image_embeddings], "lr": c["appearance_embedding_lr"]},
                       {"params": list(self.mlp.parameters()), "lr": c["appearance_mlp_lr"]}]
        if self.uncertainty:
            self.head = UncertaintyHead(384, c["uncertainty_dropout"], c["uncertainty_clip_min"],
                                        generator=g).to(device)
            self.features = DinoFeatures(load_backbone(c["dino_weights"], device))
            groups.append({"params": list(self.head.parameters()), "lr": c["uncertainty_lr"]})
        self.optimizer = torch.optim.Adam(groups, lr=1.0, eps=1e-15) if groups else None
        self.step = 0  # last completed runtime step
        self._eval = {"override": None}  # plain dict: never registered as a module parameter
        if export is not None:
            self.load_export(export)

    @property
    def override_embedding(self):
        """Evaluation: embedding used for every render (None = training lookup / zeros)."""
        return self._eval["override"]

    def set_override(self, value):
        """nn.Module would register a Parameter assigned as an attribute; keep it in a dict."""
        self._eval["override"] = value

    # -- queries ---------------------------------------------------------------------------
    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.to(self.viewmats.device).float()).abs().flatten(1).sum(1)
        i = int(torch.argmin(d))
        return i if float(d[i]) < 1e-4 else None

    def embedding_for(self, idx):
        """Training view -> its embedding; otherwise the override (evaluation) or zeros
        (official get_embedding(None))."""
        if self.override_embedding is not None:
            return self.override_embedding
        if idx is not None:
            return self.image_embeddings[idx]
        return torch.zeros_like(self.image_embeddings[0]).detach()

    # -- optimisation ----------------------------------------------------------------------
    def after_step(self, step):
        if self.optimizer is not None:
            self.optimizer.step()  # params with grad None (e.g. protected window) are skipped
            self.optimizer.zero_grad(set_to_none=True)
        self.step = step

    # -- persistence -----------------------------------------------------------------------
    def export(self):
        out = {"viewmats": self.viewmats.detach().cpu(), "step": int(self.step),
               "cfg": {k: v for k, v in self.cfg.items()}}
        if self.appearance:
            out["image_embeddings"] = self.image_embeddings.detach().cpu()
            out["mlp"] = {k: v.detach().cpu() for k, v in self.mlp.state_dict().items()}
        if self.uncertainty:
            out["head"] = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
        return out

    @torch.no_grad()
    def load_export(self, export):
        dev = self.viewmats.device
        if self.appearance:
            self.image_embeddings.copy_(export["image_embeddings"].to(dev))
            self.mlp.load_state_dict({k: v.to(dev) for k, v in export["mlp"].items()})
        if self.uncertainty:
            self.head.load_state_dict({k: v.to(dev) for k, v in export["head"].items()})
        self.step = int(export["step"])

    @classmethod
    def from_export(cls, export, device):
        vms = list(export["viewmats"].to(device))
        bank = cls(vms, dict(export["cfg"]), device, export=export)
        bank.eval()
        return bank
