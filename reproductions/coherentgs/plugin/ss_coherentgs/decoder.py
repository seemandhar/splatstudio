"""Implicit convolutional decoder (X-Fields style), positional encoding and backward warping.

Ported from the official CoherentGS code (github.com/avinashpaliwal/CoherentGS):
scene/decoder.py (ImplicitDecoder, up), scene/posenc.py (get_embedder(10)), backwarp.py.
Layer widths, kernel sizes, padding modes, initialisation (N(0, 0.02) weights, zero bias),
bilinear align_corners=True upsampling and the coordconv concatenation are unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Up(nn.Module):
    """Official `up`: bilinear upsample -> pad -> conv -> LeakyReLU(0.2)."""

    def __init__(self, in_ch, out_ch, kernel_size, padding, padding_mode, up_x, up_y):
        super().__init__()
        self.up_x, self.up_y = float(up_x), float(up_y)
        self.padding, self.padding_mode = tuple(int(p) for p in padding), padding_mode
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.02)
        self.conv.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(self.up_y, self.up_x), mode="bilinear",
                          align_corners=True)
        x = F.pad(x, self.padding, mode=self.padding_mode)
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


def upsampling_factor_padding(h_res, w_res):
    """Official ImplicitDecoder.upsampling_factor_padding (binary expansion of H and W)."""
    def bits(res):
        out = [res % 2]
        while res != 1:
            res = res // 2
            out.append(res % 2)
        del out[-1]
        return np.flip(out)

    py, px = bits(h_res), bits(w_res)
    lx, ly = len(px), len(py)
    up_x, up_y = 2 * np.ones(lx), 2 * np.ones(ly)
    if lx > ly:
        py = np.append(py, [0] * (lx - ly))
        up_y = np.append(up_y, [1] * (lx - ly))
    if ly > lx:
        px = np.append(px, [0] * (ly - lx))
        up_x = np.append(up_x, [1] * (ly - lx))
    return px, py, up_x, up_y


def layer_specs(ngf, num_layers):
    specs = [ngf * 16, ngf * 16, ngf * 16, ngf * 8, ngf * 8, ngf * 8, ngf * 4]
    specs.extend([ngf * 4] * (num_layers - len(specs)))
    return specs


class ImplicitDecoder(nn.Module):
    """Maps a [B, in_ch, 1, 1] code to a [B, out_ch, H, W] map (official architecture)."""

    def __init__(self, h_res, w_res, ngf=4, in_ch=1, out_ch=1, min_=-1, max_=1):
        super().__init__()
        self.register_buffer("coordconv", torch.tensor(
            [[[[min_, min_], [max_, min_]], [[min_, max_], [max_, max_]]]], dtype=torch.float32))
        padx, pady, up_x, up_y = upsampling_factor_padding(h_res, w_res)
        if len(padx) < 7:  # the official 7-entry layer table needs >= 7 upsampling levels
            raise ValueError(f"ImplicitDecoder needs max(H, W) >= 128, got {h_res}x{w_res}")
        specs = layer_specs(ngf, len(padx))
        self.initial_padding = (0, int(padx[0]), 0, int(pady[0]))
        self.upconv1 = Up(in_ch, specs[0], 1, self.initial_padding, "reflect", up_x[0], up_y[0])
        model, size = [], specs[0] + 2  # + coordconv
        for num, filters in enumerate(specs[1:]):
            padding = (1, 1 + padx[num + 1], 1, 1 + pady[num + 1])
            model.append(Up(size, filters, 3, padding, "reflect", up_x[num + 1], up_y[num + 1]))
            size = filters
        self.model = nn.Sequential(*model)
        self.last_layer = nn.Conv2d(size, out_ch, 3, stride=1)
        nn.init.normal_(self.last_layer.weight, mean=0, std=0.02)
        self.last_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.upconv1(x)
        cc = self.coordconv.tile((x.shape[0], 1, 1, 1))
        cc = F.pad(cc, self.initial_padding, mode="reflect")
        x = torch.cat((x, cc), dim=1)
        x = self.model(x)
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return self.last_layer(x)


def posenc(n_views, multires=10):
    """Official get_embedder(10) applied to linspace(0, 1, N): [N, 20, 1, 1].

    Per frequency 2^k (k = 0..multires-1, log sampling) the channels are sin, cos; no identity.
    """
    x = np.linspace(0, 1, num=n_views, dtype=np.float32)
    feats = []
    for freq in 2.0 ** np.linspace(0.0, multires - 1, multires):
        feats += [np.sin(x * freq), np.cos(x * freq)]
    return torch.tensor(np.stack(feats, -1), dtype=torch.float32)[..., None, None]


def backwarp(img, flow):
    """Official backWarp.forward: I0 <- grid_sample(I1, grid + F_0->1), align_corners=False."""
    _, _, h, w = img.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device),
                            indexing="ij")
    x = gx[None].float() + flow[:, 0]
    y = gy[None].float() + flow[:, 1]
    x = 2 * (x / w - 0.5)
    y = 2 * (y / h - 0.5)
    grid = torch.stack((x, y), dim=3)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
