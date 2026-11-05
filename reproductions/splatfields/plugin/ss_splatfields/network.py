"""SplatFields networks (static branch) and the process-local slot the renderer reads.

Adapted from the official SplatFields code (github.com/markomih/SplatFields, MIT licence,
(c) Marko Mihajlovic et al.): `utils/time_utils.py` (get_embedder, Embedder, GeneralMLP,
SplatFields.forward with n_frames = 0), `scene/tripFields.py` (Tensorial2D, VarTriPlaneEncoder)
and `scene/time_decoders.py` (TimeVAEDecoder with strategy 'none'). The diffusers 0.21.4 blocks
it builds on (ResnetBlock2D, Attention with `_from_deprecated_attn_block`, Upsample2D with
use_conv) and mmcv's kaiming_init/constant_init are re-implemented here without those packages.

The SplatStudio runtime has no parameter type for global learned modules, so the density owns
the network (and its Adam), registers it in `ACTIVE`, and the renderer predicts the splat
attributes through it while training. At the final step the predicted Gaussians are baked into
the base fields (`SplatFieldsNet.baked`), after which renderers render those fields — the paper
also discards the networks after training (§4.1, "the generator g_theta is discarded").
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# Process-local registry: set by SplatFieldsDensity.initialize, read by the renderer.
ACTIVE = {"net": None}

# Official SplatFields(**hyper) defaults for the static Blender run (run_blender.sh passes
# --encoder_type VarTriPlaneEncoder; D/W are not read by SplatFields; composition_rank 0,
# load_time_step 0 -> n_frames 0; use_view_dep_rgb False).
NET_DEFAULTS = {
    "use_triplane": True,
    "triplane": {"in_ch": 8, "out_ch": 16, "noise_res": 20, "block_out_channels": [32, 32, 32, 32],
                 "layers_per_block": 1, "norm_num_groups": 32},
    "deform_weight": 1.0,
    "deform": {"hidden": 128, "layers": 6, "skips": [3], "multires": 6},
    "rgb": {"hidden": 128, "layers": 6, "skips": [3], "multires": 6},
    "scale": {"hidden": 64, "layers": 4, "skips": [2], "multires": 4},
    "opacity": {"hidden": 64, "layers": 4, "skips": [2], "multires": 3},
    "rotation": {"hidden": 64, "layers": 3, "skips": [20], "multires": 3},
}


# ── positional encoding / MLP fields (official utils/time_utils.py) ─────────────────────────

def embed(x, multires):
    """NeRF positional encoding: [x, sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(...)]."""
    if multires <= 0:
        return x
    freqs = 2.0 ** torch.linspace(0.0, multires - 1, multires)
    out = [x]
    for f in freqs.tolist():
        out += [torch.sin(x * f), torch.cos(x * f)]
    return torch.cat(out, dim=-1)


def embed_dim(multires, d=3):
    return d if multires <= 0 else d * (1 + 2 * multires)


OUT_ACT = {
    "none": lambda x: x,
    "sigmoid": torch.sigmoid,
    "normalize": F.normalize,  # dim=1 on [N, 4]
}


class GeneralMLP(nn.Module):
    """Official GeneralMLP (composition_rank 0 -> plain Linear layers, act = leaky_relu).

    net = [Lin(in, h)] + [Lin(h (+in after a skip), h) x layers] + [Lin(h, out)];
    leaky_relu after EVERY layer (including the last), then the output activation; the input
    [PE(x), feat] is concatenated after each layer index in `skips` (except the last).
    """

    def __init__(self, feat_dim, out_features, hidden, layers, skips, multires, out_activation):
        super().__init__()
        self.multires = int(multires)
        self.skips = list(skips)
        in_features = embed_dim(self.multires) + feat_dim
        lins = [nn.Linear(in_features, hidden)]
        for layer_id in range(layers):
            lins.append(nn.Linear(hidden + in_features if layer_id in self.skips else hidden,
                                  hidden))
        lins.append(nn.Linear(hidden, out_features))
        self.net = nn.ModuleList(lins)
        self.out_act = OUT_ACT[out_activation]

    def forward(self, xyz, feat=None):
        h_in = embed(xyz, self.multires)
        if feat is not None:
            h_in = torch.cat([h_in, feat], dim=-1)
        h = h_in
        last = len(self.net) - 1
        for i, layer in enumerate(self.net):
            h = F.leaky_relu(layer(h))
            if i in self.skips and i != last:
                h = torch.cat([h_in, h], -1)
        return self.out_act(h)


# ── tri-plane CNN generator (official tripFields.py / time_decoders.py, diffusers blocks) ───

class ResnetBlock2D(nn.Module):
    """diffusers ResnetBlock2D (temb None, 'default'/'group' norm, swish, scale factor 1)."""

    def __init__(self, cin, cout, groups, eps=1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, cout, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1, 1, 0) if cin != cout else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))  # dropout 0
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """diffusers Attention(heads=1, dim_head=C, norm_num_groups, residual_connection=True,
    bias=True, _from_deprecated_attn_block=True) with AttnProcessor2_0."""

    def __init__(self, channels, groups, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True)])

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        t = self.group_norm(x.view(b, c, h * w)).transpose(1, 2)  # [b, hw, c]
        q, k, v = self.to_q(t), self.to_k(t), self.to_v(t)
        out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        out = self.to_out[0](out)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return out + residual


class Upsample2D(nn.Module):
    """diffusers Upsample2D(use_conv=True): nearest x2 then 3x3 conv."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class PlaneDecoder(nn.Module):
    """Official TimeVAEDecoder(in 8, out 16, 4 x TimeUpDecoderBlock2D, (32,32,32,32),
    layers_per_block 1): conv_in -> mid (resnet, attention, resnet) -> 4 up blocks of
    layers_per_block + 1 resnets, nearest-x2 upsample + conv on all but the last ->
    GroupNorm -> SiLU -> conv_out."""

    def __init__(self, cin, cout, block_out_channels, layers_per_block, groups):
        super().__init__()
        chans = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(cin, block_out_channels[-1], 3, 1, 1)
        top = block_out_channels[-1]
        self.mid_resnets = nn.ModuleList([ResnetBlock2D(top, top, groups),
                                          ResnetBlock2D(top, top, groups)])
        self.mid_attention = AttentionBlock(top, groups)
        self.up_blocks = nn.ModuleList()
        prev = chans[0]
        for i, ch in enumerate(chans):
            resnets = nn.ModuleList([ResnetBlock2D(prev if j == 0 else ch, ch, groups)
                                     for j in range(layers_per_block + 1)])
            up = Upsample2D(ch) if i != len(chans) - 1 else None
            self.up_blocks.append(nn.ModuleDict({"resnets": resnets,
                                                 **({"up": up} if up is not None else {})}))
            prev = ch
        self.conv_norm_out = nn.GroupNorm(groups, block_out_channels[0], eps=1e-6)
        self.conv_out = nn.Conv2d(block_out_channels[0], cout, 3, padding=1)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """TimeVAEDecoder.init_weights: mmcv kaiming_init (normal, fan_out, relu, bias 0) on
        every Conv2d, constant 1 on GroupNorm, zero_init_residual (resnet conv2, attention
        to_out[0] = 0)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, ResnetBlock2D):
                nn.init.constant_(m.conv2.weight, 0)
                nn.init.constant_(m.conv2.bias, 0)
            elif isinstance(m, AttentionBlock):
                nn.init.constant_(m.to_out[0].weight, 0)
                nn.init.constant_(m.to_out[0].bias, 0)

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid_resnets[0](x)
        x = self.mid_attention(x)
        x = self.mid_resnets[1](x)
        for block in self.up_blocks:
            for r in block["resnets"]:
                x = r(x)
            if "up" in block:
                x = block["up"](x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class TriPlane(nn.Module):
    """Official VarTriPlaneEncoder: three Tensorial2D decoders (own fixed noise each), planes
    xy, yz, zx, `F.grid_sample(planes, coord)` (bilinear, zeros, align_corners=False), cat."""

    AXES = ((0, 1), (1, 2), (2, 0))

    def __init__(self, cfg, generator):
        super().__init__()
        self.decoders = nn.ModuleList([
            PlaneDecoder(cfg["in_ch"], cfg["out_ch"], cfg["block_out_channels"],
                         cfg["layers_per_block"], cfg["norm_num_groups"]) for _ in self.AXES])
        r = int(cfg["noise_res"])
        self.register_buffer("noise", torch.randn(3, 1, cfg["in_ch"], r, r, generator=generator))
        self.out_dim = 3 * int(cfg["out_ch"])

    def planes(self):
        return torch.cat([d(self.noise[i]) for i, d in enumerate(self.decoders)], dim=0)

    def forward(self, xyz):
        planes = self.planes()  # [3, C, H, W]
        coord = torch.stack([xyz[None][..., list(ax)] for ax in self.AXES])  # [3, 1, N, 2]
        feat = F.grid_sample(planes, coord, mode="bilinear", padding_mode="zeros",
                             align_corners=False)  # [3, C, 1, N]
        return feat.permute(2, 3, 0, 1).reshape(xyz.shape[0], -1)  # [N, 3C]


class SplatFieldsNet(nn.Module):
    """Official SplatFields (static, n_frames = 0) forward: features -> deform -> attributes."""

    def __init__(self, cfg=None, seed=0):
        super().__init__()
        cfg = {**NET_DEFAULTS, **(cfg or {})}
        self.cfg = cfg
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))  # nn.Linear default init draws from the global RNG
        try:
            if cfg["use_triplane"]:
                self.encoder = TriPlane(cfg["triplane"], g)
                fd = self.encoder.out_dim
                self.refine = nn.Sequential(nn.Linear(fd, fd), nn.ReLU(), nn.Linear(fd, fd))
            else:
                self.encoder, self.refine, fd = None, None, 0
            self.feat_dim = fd

            def mlp(key, out, act):
                c = cfg[key]
                return GeneralMLP(fd, out, c["hidden"], c["layers"], c["skips"], c["multires"],
                                  act)

            self.mlp_deform = mlp("deform", 3, "none")
            self.mlp_rgb = mlp("rgb", 3, "sigmoid")
            self.mlp_scale = mlp("scale", 3, "none")
            self.mlp_opacity = mlp("opacity", 1, "sigmoid")
            self.mlp_rotation = mlp("rotation", 4, "normalize")
        finally:
            torch.random.set_rng_state(cpu_state)
        self.deform_weight = float(cfg["deform_weight"])
        self.optimizer = None
        self.baked = False

    def features(self, xyz):
        if self.encoder is None:
            return None
        return self.refine(self.encoder(xyz))

    def forward(self, xyz, base_scale):
        """xyz: anchors [N,3] (detached), base_scale: exp(log base scale) [N,3] (detached).

        Returns the rendered splats: means (Eq. 7 + residual), scales = MLP + base scale
        (official train.py `ret['scales'] + scaling`), opacities [N] in (0,1), unit quats,
        view-independent RGB in (0,1)."""
        f = self.features(xyz)
        if self.deform_weight > 0:
            means = xyz + self.deform_weight * self.mlp_deform(xyz, f)
        else:
            means = xyz
        return {
            "means": means,
            "scales": self.mlp_scale(means, f) + base_scale,
            "opacities": self.mlp_opacity(means, f)[:, 0],
            "quats": self.mlp_rotation(means, f),
            "rgb": self.mlp_rgb(means, f),
        }


def exp_lr(step, lr_init, lr_final, max_steps):
    """Official get_expon_lr_func (lr_delay_steps 0): log-linear lr_init -> lr_final."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
