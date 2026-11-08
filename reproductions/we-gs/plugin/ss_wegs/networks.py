"""WE-GS networks (arXiv 2406.02407 §4.1-4.2, §5.1).

* ``SpatialAttentionEncoder`` -- Eq. 14-18: Enc^1 (3 convs, 1/4 res) -> channel attention (CAM,
  Eq. 15-16) -> U-Net spatial attention / static mask M_k (Eq. 17) -> F'' = M (x) F' ->
  l_k = Enc^2(F'' + F) (Eq. 18).
  Layer shapes of Enc^1/Enc^2 are the six convolutions of the authors' released WE-GS appearance
  encoder (takinglangsplatw ``in_the_wild_renderer/appearance_encoder/ae_cnn.py`` ``Encoder``,
  mid_dim 128), split 3 + 3 as in the paper; CAM and U-Net widths are assumptions (plan.md).
* ``TransNet`` -- Eq. 9 G_phi: 4-layer ReLU MLP (paper) of width 256 with tanh output (released
  ``TransNet``), input [gamma_0(sh) (identity, code app_pe_freq=0), l_k, gamma_4(mu)].
"""

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """NeRF gamma: (x, sin(f x), cos(f x), ...) with f = 2^linspace(0, n-1, n) (code utils)."""

    def __init__(self, n_freqs):
        super().__init__()
        n = int(n_freqs)
        freqs = 2.0 ** torch.linspace(0, n - 1, n) if n > 0 else torch.zeros(0)
        self.register_buffer("freqs", freqs, persistent=False)

    def out_dim(self, d):
        return d * (1 + 2 * len(self.freqs))

    def forward(self, x):
        out = [x]
        for f in self.freqs:
            out += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(out, -1)


class SpatialAttentionEncoder(nn.Module):
    def __init__(self, dim=16, mid=128, cam_reduction=16, unet_channels=128, input_size=0):
        super().__init__()
        self.input_size = int(input_size)
        c = 4 * mid
        self.enc1 = nn.Sequential(  # Eq. 14, 3 convs -> 1/4 resolution
            nn.ReflectionPad2d(3), nn.Conv2d(3, mid, 7, 1), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(mid, 2 * mid, 4, 2), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(2 * mid, c, 4, 2), nn.ReLU())
        hidden = max(c // int(cam_reduction), 1)
        self.cam = nn.Sequential(nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c))  # Eq. 15
        u = int(unet_channels)  # Eq. 17: 2 convs + 2 transposed convs, U-Net skip
        self.down1 = nn.Conv2d(c, u, 3, 2, 1)
        self.down2 = nn.Conv2d(u, u, 3, 2, 1)
        self.up1 = nn.ConvTranspose2d(u, u, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(u, 1, 4, 2, 1)
        self.enc2 = nn.Sequential(  # Eq. 18, 3 convs
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 4, 2), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 4, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, int(dim), 1, 1, 0))

    def unet_parameters(self):
        for m in (self.down1, self.down2, self.up1, self.up2):
            yield from m.parameters()

    def feature_parameters(self):
        for m in (self.enc1, self.cam, self.enc2):
            yield from m.parameters()

    def prepare(self, image_hwc):
        """input_size 0 (default, paper §4.2/§5.1): the image itself, resized only to the nearest
        multiple of 16 (so W' = W/4 survives the two stride-2 U-Net convs); s > 0: s x s resize
        (released TVCG model, ablation only)."""
        x = image_hwc.permute(2, 0, 1)[None]
        h, w = x.shape[2:]
        s = self.input_size
        size = (s, s) if s > 0 else (max(16, int(h / 16 + 0.5) * 16),
                                     max(16, int(w / 16 + 0.5) * 16))
        if tuple(size) == (h, w):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)

    def unet(self, f):
        d1 = F.relu(self.down1(f))
        d2 = F.relu(self.down2(d1))
        u1 = F.relu(self.up1(d2)) + d1
        return torch.sigmoid(self.up2(u1))

    def forward(self, image_hwc, mask_override=None):
        """Returns (l_k [dim], M_k [1, 1, H', W'], intermediates)."""
        x = self.prepare(image_hwc)
        f = self.enc1(x)                                                         # Eq. 14
        att = torch.sigmoid(self.cam(f.mean((2, 3))) + self.cam(f.amax((2, 3))))  # Eq. 15
        f1 = att[:, :, None, None] * f                                           # Eq. 16
        m = self.unet(f1) if mask_override is None else mask_override             # Eq. 17
        f2 = m * f1                                                              # F''
        lk = self.enc2(f2 + f).flatten()                                         # Eq. 18
        return lk, m, {"F": f, "Fc": att, "F1": f1, "F2": f2}


class TransNet(nn.Module):
    """Eq. 9: Delta sh_jk = G_phi(gamma(sh_j), gamma(mu_j), l_k); 4 layers, tanh output."""

    def __init__(self, sh_coeffs=16, dim=16, width=256, layers=4, pe_sh=0, pe_xyz=4):
        super().__init__()
        self.sh_dim = 3 * int(sh_coeffs)
        self.pe_sh, self.pe_xyz = PositionalEncoding(pe_sh), PositionalEncoding(pe_xyz)
        d = self.pe_sh.out_dim(self.sh_dim) + int(dim) + self.pe_xyz.out_dim(3)
        mods = []
        for _ in range(int(layers) - 1):
            mods += [nn.Linear(d, int(width)), nn.ReLU()]
            d = int(width)
        mods.append(nn.Linear(d, self.sh_dim))
        self.net = nn.Sequential(*mods)

    def features(self, sh, means, lk):
        n = len(means)
        return torch.cat([self.pe_sh(sh.reshape(n, -1)), lk[None].expand(n, -1),
                          self.pe_xyz(means)], -1)

    def forward(self, sh, means, lk):
        """sh [N, K, 3] (K = 16), means [N, 3], lk [dim] -> Delta sh [N, K, 3]."""
        return torch.tanh(self.net(self.features(sh, means, lk))).view(sh.shape)
