"""Blur Proposal Network (BPN) state of BAGS, held in a process-local bank.

Ports of github.com/snldmt/BAGS @ed04144 scene/kernelnet_multires.py (multi-scale BPN),
scene/kernelnet_single_res.py (``use_another_mlp`` single-scale BPN) and train.py get_2d_emb.
The density component creates the bank from the training cameras (``set_training_cameras``),
publishes it as ``ACTIVE`` for the renderer / losses and steps its Adam after every runtime
step (official: the BPN parameters are extra groups of the Gaussian Adam, lr KERLR = 5e-4,
eps 1e-15; Adam is per-parameter, so a separate optimizer is the same update).
"""

import numpy as np
import torch
from torch import nn

from .schedule import kernel_level, uses_single_scale_net

ACTIVE = None  # the live bank of the current training process (None in ss-eval)


def get_emb(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def get_2d_emb(batch_size, x, y, out_ch, device):
    """train.py get_2d_emb: [B, x, y, out_ch] sinusoidal pixel encoding (x = H, y = W)."""
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


def rgbd_cnn():
    """F_feat: three convs (64, 64, 32 channels; kernels 5, 5, 3), ReLU + InstanceNorm."""
    return nn.Sequential(
        nn.Conv2d(4, 64, 5, padding=2), nn.ReLU(), nn.InstanceNorm2d(64),
        nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(), nn.InstanceNorm2d(64),
        nn.Conv2d(64, 32, 3, padding=1))


class _Embed(nn.Module):
    def _mmf(self, img_idx, pos_enc, img):
        """Multi-Modal Feature: l(i) (+) p(x) (+) f_RGBD(x, i)   (Eq. 2 / Eq. 4)."""
        idx = torch.tensor([int(img_idx)], device=pos_enc.device)
        img_embed = self.embedding_camera(idx)[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0], pos_enc.shape[1], pos_enc.shape[2],
                                     img_embed.shape[-1])
        if self.not_use_pe:
            inp = img_embed.permute(0, 3, 1, 2)
        else:
            inp = torch.cat([img_embed, pos_enc], -1).permute(0, 3, 1, 2)
        if self.not_use_rgbd:
            return inp
        return torch.cat([inp, self.conv_rgbd(img)], 1)


class BlurKernelMS(_Embed):
    """kernelnet_multires.BlurKernel: base1 -> {head1, mask1} (K3=5); base1, base2 ->
    {head2, mask2} (K2=9); base1, base2, base3 -> {head3, mask3} (K1=17). 1x1 convs, no bias."""

    def __init__(self, num_img, img_embed=32, ks1=5, ks2=9, ks3=17, not_use_rgbd=False,
                 not_use_pe=False):
        super().__init__()
        self.num_img = num_img
        self.ks = (ks1, ks2, ks3)
        self.embedding_camera = nn.Embedding(num_img, img_embed)
        self.not_use_rgbd, self.not_use_pe = not_use_rgbd, not_use_pe
        cin = img_embed + (0 if not_use_pe else 16) + (0 if not_use_rgbd else 32)
        self.mlp_base1 = nn.Sequential(nn.Conv2d(cin, 64, 1, bias=False), nn.ReLU())
        self.mlp_head1 = nn.Conv2d(64, ks1**2, 1, bias=False)
        self.mlp_mask1 = nn.Conv2d(64, 1, 1, bias=False)
        self.mlp_base2 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.ReLU())
        self.mlp_mask2 = nn.Conv2d(64, 1, 1, bias=False)
        self.mlp_head2 = nn.Conv2d(64, ks2**2, 1, bias=False)
        self.mlp_base3 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.ReLU())
        self.mlp_head3 = nn.Conv2d(64, ks3**2, 1, bias=False)
        self.mlp_mask3 = nn.Conv2d(64, 1, 1, bias=False)
        self.conv_rgbd = rgbd_cnn()

    def forward(self, img_idx, pos_enc, img, level):
        """level 1/2/3 = scale 3/2/1. Returns (softmax kernel [1,K^2,H,W], sigmoid mask
        [1,1,H,W])   (Eq. 4)."""
        feature = self.mlp_base1(self._mmf(img_idx, pos_enc, img))
        if level == 1:
            weight, mask = self.mlp_head1(feature), self.mlp_mask1(feature)
        elif level == 2:
            feature = self.mlp_base2(feature)
            weight, mask = self.mlp_head2(feature), self.mlp_mask2(feature)
        elif level == 3:
            feature = self.mlp_base3(self.mlp_base2(feature))
            weight, mask = self.mlp_head3(feature), self.mlp_mask3(feature)
        else:
            raise ValueError(f"kernel level must be 1, 2 or 3, got {level}")
        return torch.softmax(weight, dim=1), torch.sigmoid(mask)


class BlurKernelSS(_Embed):
    """kernelnet_single_res.BlurKernel (``use_another_mlp`` after ms_steps)."""

    def __init__(self, num_img, img_embed=32, ks=17, not_use_rgbd=False, not_use_pe=False):
        super().__init__()
        self.num_img = num_img
        self.ks = ks
        self.embedding_camera = nn.Embedding(num_img, img_embed)
        self.not_use_rgbd, self.not_use_pe = not_use_rgbd, not_use_pe
        cin = img_embed + (0 if not_use_pe else 16) + (0 if not_use_rgbd else 32)
        self.mlp_base_mlp = nn.Sequential(
            nn.Conv2d(cin, 64, 1, bias=False), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.ReLU())
        self.mlp_head1 = nn.Conv2d(64, ks**2, 1, bias=False)
        self.mlp_mask1 = nn.Conv2d(64, 1, 1, bias=False)
        self.conv_rgbd = rgbd_cnn()

    def forward(self, img_idx, pos_enc, img, level=None):
        feat = self.mlp_base_mlp(self._mmf(img_idx, pos_enc, img))
        return torch.softmax(self.mlp_head1(feat), dim=1), torch.sigmoid(self.mlp_mask1(feat))


class BlurBank(nn.Module):
    """Per-training-view BPN state + schedule config shared by renderer, losses and density."""

    def __init__(self, cameras, cfg, device):
        super().__init__()
        n = len(cameras)
        self.cfg = dict(cfg)
        ks = tuple(int(k) for k in cfg["kernel_sizes"])
        self.ms = BlurKernelMS(n, ks1=ks[0], ks2=ks[1], ks3=ks[2],
                               not_use_rgbd=cfg["not_use_rgbd"],
                               not_use_pe=cfg["not_use_pe"]).to(device)
        self.ss = (BlurKernelSS(n, ks=int(cfg["kernel_size_ss"]), not_use_rgbd=cfg["not_use_rgbd"],
                                not_use_pe=cfg["not_use_pe"]).to(device)
                   if cfg["use_another_mlp"] else None)
        self.register_buffer("viewmats", torch.stack([c["viewmat"].float() for c in cameras])
                             .to(device))
        self.size = (int(cameras[0]["height"]), int(cameras[0]["width"]))
        self.opt = torch.optim.Adam(self.parameters(), lr=float(cfg["bpn_lr"]), eps=1e-15)
        self.step = 0  # last completed runtime step
        self.updates = 0
        self._pe = {}

    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.to(self.viewmats.device).float()).abs().flatten(1).sum(1)
        i = int(torch.argmin(d))
        return i if float(d[i]) < 1e-4 else None

    def pos_enc(self, H, W):
        key = (int(H), int(W))
        if key not in self._pe:
            self._pe[key] = get_2d_emb(1, key[0], key[1], 16, self.viewmats.device)
        return self._pe[key]

    def net_for(self, step):
        """(network, level, kernel size) used at global step ``step``."""
        c = self.cfg
        if uses_single_scale_net(step, c["ms_steps"], c["use_another_mlp"]):
            return self.ss, None, self.ss.ks
        level = kernel_level(step, c["upsample_steps"])
        return self.ms, level, self.ms.ks[level - 1]

    @torch.no_grad()
    def after_step(self, step):
        if any(p.grad is not None for p in self.parameters()):
            self.opt.step()
            self.updates += 1
        self.opt.zero_grad(set_to_none=True)
        self.step = step

    def export(self):
        return {"ms": {k: v.detach().cpu() for k, v in self.ms.state_dict().items()},
                "ss": ({k: v.detach().cpu() for k, v in self.ss.state_dict().items()}
                       if self.ss is not None else {}),
                "step": self.step, "updates": self.updates}
