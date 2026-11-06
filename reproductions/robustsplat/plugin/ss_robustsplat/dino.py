"""DINOv2 ViT-S/14 with 4 register tokens (``torch.hub`` ``dinov2_vits14_reg``), the frozen
feature extractor of RobustSplat's mask MLP.

Architecture re-implemented after Meta's DINOv2 (Apache-2.0, facebookresearch/dinov2), copied
from the WildGaussians native port (reproductions/wildgaussians/plugin/ss_wildgaussians/dino.py):
embed 384, depth 12, 6 heads, MLP ratio 4, LayerScale, LayerNorm eps 1e-6, positional
embedding interpolation bicubic + antialias, offset 0. RobustSplat uses
``forward_features(x)["x_norm_patchtokens"]`` (all 12 blocks, final LayerNorm, patch tokens).

Official ``utils/mask_utils.DINOFeatureExtractor.forward``: the [0,1] image (no ImageNet
normalisation) is bilinearly resized (align_corners=False) to (14 s, 14 s) and the patch tokens
are returned as [384, s, s].
"""

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_WEIGHTS = "~/models/dinov2/dinov2_vits14_reg4_pretrain.pth"
WEIGHTS_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV2Reg(nn.Module):
    """ViT-S/14 + 4 registers; img_size 518 -> 37x37 positional grid."""

    def __init__(self, dim=384, depth=12, heads=6, patch=14, registers=4, img_size=518):
        super().__init__()
        self.embed_dim, self.num_heads, self.patch_size = dim, heads, patch
        self.num_register_tokens = registers
        self.patch_embed = _PatchEmbed(patch, dim)
        grid = img_size // patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def interpolate_pos_encoding(self, npatch, h, w):
        """DinoVisionTransformer.interpolate_pos_encoding (antialias bicubic, offset 0)."""
        n = self.pos_embed.shape[1] - 1
        if npatch == n and h == w:
            return self.pos_embed
        pos = self.pos_embed.float()
        m = int(math.sqrt(n))
        patch_pos = F.interpolate(
            pos[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2),
            size=(h // self.patch_size, w // self.patch_size), mode="bicubic", antialias=True)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
        return torch.cat((pos[:, :1], patch_pos), dim=1).to(self.pos_embed.dtype)

    def prepare_tokens(self, x):
        b, _, h, w = x.shape
        t = self.patch_embed(x)
        t = torch.cat((self.cls_token.expand(b, -1, -1), t), dim=1)
        t = t + self.interpolate_pos_encoding(t.shape[1] - 1, h, w)
        return torch.cat((t[:, :1], self.register_tokens.expand(b, -1, -1), t[:, 1:]), dim=1)

    def intermediate(self, x, layer=None):
        """get_intermediate_layers(x, n=[layer], reshape=True, norm=True)[-1]; default layer is
        num_heads - 1 (= 5), as in UncertaintyModel._get_dino_cached."""
        layer = self.num_heads - 1 if layer is None else int(layer)
        b, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("DINO input must be a multiple of the patch size")
        t = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i == layer:
                break
        t = self.norm(t)[:, 1 + self.num_register_tokens:]
        return t.reshape(b, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2) \
            .contiguous()


    def forward_features(self, x):
        """``DinoVisionTransformer.forward_features(x)["x_norm_patchtokens"]`` as [B,384,h,w]."""
        b, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("DINO input must be a multiple of the patch size")
        t = self.prepare_tokens(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)[:, 1 + self.num_register_tokens:]
        return t.reshape(b, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2) \
            .contiguous()


def file_sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


_CACHE = {}


def load_backbone(path=DEFAULT_WEIGHTS, device="cpu", check_sha=True):
    """Frozen, eval-mode backbone (one instance per (path, device))."""
    path = str(Path(path).expanduser())
    key = (path, str(device))
    if key not in _CACHE:
        if not Path(path).exists():
            raise FileNotFoundError(f"DINOv2 weights not found: {path}")
        if check_sha and file_sha256(path) != WEIGHTS_SHA256:
            raise ValueError(f"{path}: sha256 differs from the official dinov2_vits14_reg4 weights")
        model = DinoV2Reg()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        _CACHE[key] = model.eval().to(device)
    return _CACHE[key]


@torch.no_grad()
def extract(model, image_chw, grid):
    """Official ``DINOFeatureExtractor.forward(image, feature_size=grid)`` -> [384, grid, grid]."""
    size = int(grid) * model.patch_size
    x = F.interpolate(image_chw[None].float(), size=(size, size), mode="bilinear",
                      align_corners=False)
    return model.forward_features(x)[0]
