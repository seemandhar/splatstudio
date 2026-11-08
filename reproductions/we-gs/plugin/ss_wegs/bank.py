"""Run-wide learned state of WE-GS that the native runtime cannot hold: the spatial-attention
encoder (Enc^1, CAM, U-Net, Enc^2) and the residual SH transfer MLP G_phi.

Same ownership pattern as reproductions/swag (``ss_swag.bank``): created by the density in
``initialize``, published as ``ACTIVE`` for the renderer, stepped by its own Adam after every
runtime step, exported to ``density.state["bank"]`` and rebuilt by ``ss_wegs.evaluate``.
"""

import torch
from torch import nn

from .networks import SpatialAttentionEncoder, TransNet

ACTIVE = None

BANK_DEFAULTS = {
    "appearance_dim": 16,      # §5.1
    "encoder_mid": 128,        # released ae_cnn.Encoder mid_dim
    "cam_reduction": 16,       # assumption (CBAM default)
    "unet_channels": 128,      # assumption
    "input_size": 0,           # 0 = training image itself (paper W' = W/4); 256 = released TVCG resize
    "sh_coeffs": 16,           # paper L = 4 (3DGS degree 3)
    "transnet_width": 256,     # released TransNet
    "transnet_layers": 4,      # §4.1 four-layer MLP
    "pe_sh": 0,                # released app_pe_freq = 0
    "pe_xyz": 4,               # released pe_freq_xyz = 4
    "lr_transnet": 1e-4,       # released transnet_lr
    "lr_encoder": 1e-4,        # released encodenet_lr
    "lr_mask": 1e-5,           # released mask_net_lr
    "seed": 0,
}


class WEGSBank(nn.Module):
    def __init__(self, cfg, device, export=None):
        super().__init__()
        self.cfg = {**BANK_DEFAULTS, **(cfg or {})}
        c = self.cfg
        # Deterministic init without clobbering the runtime's --seed RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(c["seed"]))
            self.encoder = SpatialAttentionEncoder(c["appearance_dim"], c["encoder_mid"],
                                                   c["cam_reduction"], c["unet_channels"],
                                                   c["input_size"]).to(device)
            self.transnet = TransNet(c["sh_coeffs"], c["appearance_dim"], c["transnet_width"],
                                     c["transnet_layers"], c["pe_sh"], c["pe_xyz"]).to(device)
        self.optimizer = torch.optim.Adam([
            {"params": list(self.transnet.parameters()), "lr": float(c["lr_transnet"]),
             "name": "transnet"},
            {"params": list(self.encoder.feature_parameters()), "lr": float(c["lr_encoder"]),
             "name": "encoder"},
            {"params": list(self.encoder.unet_parameters()), "lr": float(c["lr_mask"]),
             "name": "unet"},
        ], eps=1e-15)
        self.step = 0
        if export is not None:
            self.load_export(export)

    def encode(self, image_hwc):
        """(l_k [dim], M_k [1, 1, H', W'])."""
        lk, m, _ = self.encoder(image_hwc)
        return lk, m

    def residual(self, sh, means, lk):
        return self.transnet(sh, means, lk)

    def after_step(self, step):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step = int(step)

    def export(self):
        return {"cfg": dict(self.cfg), "step": int(self.step),
                "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
                "transnet": {k: v.detach().cpu() for k, v in self.transnet.state_dict().items()}}

    @torch.no_grad()
    def load_export(self, export):
        dev = next(self.parameters()).device
        self.encoder.load_state_dict({k: v.to(dev) for k, v in export["encoder"].items()})
        self.transnet.load_state_dict({k: v.to(dev) for k, v in export["transnet"].items()})
        self.step = int(export["step"])

    @classmethod
    def from_export(cls, export, device):
        bank = cls(dict(export["cfg"]), device, export=export)
        bank.eval()
        for p in bank.parameters():
            p.requires_grad_(False)
        return bank
