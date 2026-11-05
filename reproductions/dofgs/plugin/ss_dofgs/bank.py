"""Per-view DOF camera state (focal distance f_m, aperture Q_m) and the In-Focus Localization
Network (ILN), held in a process-local bank.

The density component creates the bank from the training cameras (``set_training_cameras``),
publishes it as ``ACTIVE`` for the renderer / loss, and steps its optimizers after every
runtime step. Ports of src/model_dofgs/gaussian_model.py (setup_defocus_params,
join_optim_defocus_params) and neural_module.py (MaskNet) and training.py (get_2d_emb) of
github.com/rongduo/DOFGS_implementation @cfd635c.
"""

import numpy as np
import torch
from torch import nn

ACTIVE = None  # the live bank of the current training process (None in ss-eval)


def get_emb(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def get_2d_emb(batch_size, x, y, out_ch, device):
    """training.py get_2d_emb: [B, x, y, out_ch] sinusoidal pixel encoding (x = H, y = W)."""
    out_ch = int(np.ceil(out_ch / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch))
    pos_x = torch.arange(x).type(inv_freq.type()) * 2 * np.pi / x
    pos_y = torch.arange(y).type(inv_freq.type()) * 2 * np.pi / y
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1)
    emb_y = get_emb(sin_inp_y)
    emb = torch.zeros((x, y, out_ch * 2))
    emb[:, :, :out_ch] = emb_x
    emb[:, :, out_ch: 2 * out_ch] = emb_y
    return emb[None].repeat(batch_size, 1, 1, 1).to(device)


class MaskNet(nn.Module):
    """neural_module.MaskNet (act 'sigmoid'): out-of-focus mask m in [0,1] (paper M* = 1 - m)."""

    def __init__(self, num_img, img_embed=32):
        super().__init__()
        self.num_img = num_img
        self.embedding_camera = nn.Embedding(num_img, img_embed)
        self.mlp_base_mlp = nn.Sequential(nn.Conv2d(48 + 16, 20, 1, bias=False), nn.ReLU())
        self.mlp_mask1 = nn.Conv2d(20, 1, 1, bias=False)
        # Official: InstanceNorm2d(64) after a 48-channel conv (affine=False, so the declared
        # channel count is unused).
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(5, 48, 3, padding=1), nn.ReLU(), nn.InstanceNorm2d(64),
            nn.Conv2d(48, 16, 3, padding=1))

    def forward(self, img_idx, pos_enc, img, coc_alpha):
        """img [4,H,W] (rgb + normalised depth), coc_alpha [1,H,W], pos_enc [1,H,W,16]."""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if coc_alpha.dim() == 3:
            coc_alpha = coc_alpha.unsqueeze(0)
        feat_rgb = self.conv_rgb(torch.cat([img, coc_alpha], dim=1))
        idx = torch.tensor([int(img_idx)], device=img.device)
        img_embed = self.embedding_camera(idx)[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0], pos_enc.shape[1], pos_enc.shape[2],
                                     img_embed.shape[-1])
        inp = torch.cat([img_embed, pos_enc], -1).permute(0, 3, 1, 2)
        feat = self.mlp_base_mlp(torch.cat([inp, feat_rgb], 1))
        return torch.sigmoid(self.mlp_mask1(feat)).squeeze(0)  # [1,H,W]


def initial_camera_params(z_world, translate_z, coc_max=15.0):
    """setup_defocus_params: returns (Fs0, Ks0) in the raw reparameterised units.

    f_m init: 1/f = median(1/z) (supp. B.1), official on world z + nerf++ translate_z.
    Q_m init: Q |1/z_min - 1/z_max| = tau with 5%/95% diopter quantiles (code; paper: 10/90).
    """
    inv = 1.0 / z_world
    z_median = 1.0 / torch.quantile(inv, 0.5, interpolation="nearest").item() + translate_z
    z_max = 1.0 / torch.quantile(inv, 0.05, interpolation="nearest").item()
    z_min = 1.0 / torch.quantile(inv, 0.95, interpolation="nearest").item()
    k_value = coc_max / (abs(1.0 / z_min - 1.0 / z_max) + 1e-6)
    return z_median / 10.0, 10.0 * (k_value - 100.0) / 1000.0


def focal_from_raw(fs):
    return torch.clamp(fs, 0.01, 1000) * 10.0  # renderer/__init__.py


def aperture_from_raw(ks):
    return torch.clamp(ks, 0.01, 200) * 100.0 + 100.0


class DofBank(nn.Module):
    def __init__(self, cameras, z_world, translate_z, cfg, device):
        super().__init__()
        n = len(cameras)
        fs0, ks0 = initial_camera_params(z_world, translate_z, cfg["coc_max"])
        self.Fs = nn.Parameter(torch.full((n, 1), fs0, device=device))
        self.Ks = nn.Parameter(torch.full((n, 1), ks0, device=device))
        self.mask_net = MaskNet(n).to(device)
        self.register_buffer("viewmats", torch.stack([c["viewmat"].float() for c in cameras])
                             .to(device))
        H, W = int(cameras[0]["height"]), int(cameras[0]["width"])
        self.register_buffer("pos_enc", get_2d_emb(1, H, W, 16, device))
        self.cfg = cfg
        self.step = 0  # last completed runtime step
        self.opt_cam = torch.optim.Adam(
            [{"params": [self.Fs], "lr": cfg["fdist_lr"]},
             {"params": [self.Ks], "lr": cfg["kapert_lr"]}], lr=0.0, eps=1e-15)
        self.opt_mask = torch.optim.Adam(self.mask_net.parameters(), lr=cfg["mask_nn_lr"],
                                         eps=1e-15)
        self.counts = {"cam_steps": 0, "mask_steps": 0}

    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.to(self.viewmats.device).float()).abs().flatten(1).sum(1)
        i = int(torch.argmin(d))
        return i if float(d[i]) < 1e-4 else None

    def focal(self):
        return focal_from_raw(self.Fs)

    def aperture(self):
        return aperture_from_raw(self.Ks)

    @torch.no_grad()
    def after_step(self, step):
        """Official schedule: F/K join Adam at iter_defocus_start (grads accumulate before),
        are updated through iter_stop_cam_gd and frozen afterwards; ILN updated every step."""
        c = self.cfg
        if c["iter_defocus_start"] <= step and self.Fs.requires_grad:
            if self.Fs.grad is not None or self.Ks.grad is not None:
                self.opt_cam.step()
                self.counts["cam_steps"] += 1
            self.opt_cam.zero_grad(set_to_none=True)
            if step >= c["iter_stop_cam_gd"]:
                self.Fs.requires_grad_(False)
                self.Ks.requires_grad_(False)
        if any(p.grad is not None for p in self.mask_net.parameters()):
            self.opt_mask.step()
            self.counts["mask_steps"] += 1
        self.opt_mask.zero_grad(set_to_none=True)
        self.step = step

    def export(self):
        return {"focal_distance": self.focal().detach().cpu(),
                "aperture": self.aperture().detach().cpu(),
                "Fs": self.Fs.detach().cpu(), "Ks": self.Ks.detach().cpu(),
                "mask_net": {k: v.detach().cpu() for k, v in self.mask_net.state_dict().items()},
                "step": self.step}
