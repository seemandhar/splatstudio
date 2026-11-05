"""DNGaussian Neural Color Renderer (paper §3.4 / supp. A.2; official scene/neural_renderer.py,
gaussian_renderer/__init__.py:render_neural, encoding.py, gridencoder/, shencoder/).

The CUDA hash-grid and SH encoders are the official torch-ngp kernels (MIT, ashawkey),
vendored unchanged in csrc/ and JIT-built with torch.utils.cpp_extension (run through
tools/ssenv so nvcc / gcc-11 are configured). The Python wrappers below follow the official
gridencoder/grid.py and shencoder/sphere_harmonics.py (fp32 only; the official code never
enables autocast).

Global learned state: the runtime has no parameter type for global modules, so (as in the
eagles / SplatFields ports) the density component owns the GridRenderer and its Adam and
registers it in ``ACTIVE``; the renderer reads it while training. For evaluation in a fresh
process the paper's inference path is baked into Gaussian fields: ``dng_geo`` [N, 64] holds
the cached geometry feature (the paper: "we store the intermediate result and only calculate
the last MLP layers") and ``dng_mlp`` [N, 8] carries the 5312 colour-MLP weights packed
row-major behind a small header (requires N >= 665 Gaussians; checked, never silently).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ACTIVE = {"net": None, "baked": False}

_EXT = {}
_CSRC = Path(__file__).resolve().parent / "csrc"
_NVCC = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
         "-U__CUDA_NO_HALF2_OPERATORS__"]


def backend(name):
    """JIT-compile (once per machine, cached by torch) the vendored torch-ngp extension."""
    if name not in _EXT:
        from torch.utils.cpp_extension import load

        src = _CSRC / name
        _EXT[name] = load(
            name=f"_dng_{name}",
            sources=[str(src / f"{name}.cu"), str(src / "bindings.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=_NVCC,
            verbose=False,
        )
    return _EXT[name]


# ------------------------------------------------------------------ hash grid (grid.py)
class _GridEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution,
                calc_grad_inputs, gridtype, align_corners, interpolation):
        inputs = inputs.contiguous()
        B, D = inputs.shape
        L = offsets.shape[0] - 1
        C = embeddings.shape[1]
        S = np.log2(per_level_scale)
        H = base_resolution
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
        dy_dx = (torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
                 if calc_grad_inputs else None)
        backend("gridencoder").grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C,
                                                   L, S, H, dy_dx, gridtype, align_corners,
                                                   interpolation)
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners
        return outputs

    @staticmethod
    def backward(ctx, grad):
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)
        grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype) if dy_dx is not None else None
        backend("gridencoder").grid_encode_backward(grad, inputs, embeddings, offsets,
                                                    grad_embeddings, B, D, C, L, S, H, dy_dx,
                                                    grad_inputs, gridtype, ctx.align_corners,
                                                    interpolation)
        if grad_inputs is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)
        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None


class GridEncoder(nn.Module):
    """Official GridEncoder(gridtype='hash', align_corners=False, interpolation='linear')."""

    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16,
                 log2_hashmap_size=19, desired_resolution=None):
        super().__init__()
        per_level_scale = 2.0
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        self.input_dim, self.num_levels, self.level_dim = input_dim, num_levels, level_dim
        self.per_level_scale = float(per_level_scale)
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        offsets, offset = [], 0
        max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            n = min(max_params, (resolution + 1) ** input_dim)
            n = int(np.ceil(n / 8) * 8)
            offsets.append(offset)
            offset += n
        offsets.append(offset)
        self.register_buffer("offsets", torch.from_numpy(np.array(offsets, dtype=np.int32)))
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))
        self.embeddings.data.uniform_(-1e-4, 1e-4)

    def forward(self, inputs, bound=1):
        inputs = (inputs + bound) / (2 * bound)  # map [-bound, bound] to [0, 1]
        prefix = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        out = _GridEncode.apply(inputs, self.embeddings, self.offsets, self.per_level_scale,
                                self.base_resolution, inputs.requires_grad, 0, False, 0)
        return out.view(prefix + [self.output_dim])


# ------------------------------------------------------------------ SH encoder (degree 4)
class _SHEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, degree, calc_grad_inputs):
        inputs = inputs.contiguous()
        B, input_dim = inputs.shape
        outputs = torch.empty(B, degree ** 2, dtype=inputs.dtype, device=inputs.device)
        dy_dx = (torch.empty(B, input_dim * degree ** 2, dtype=inputs.dtype, device=inputs.device)
                 if calc_grad_inputs else None)
        backend("shencoder").sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)
        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]
        return outputs

    @staticmethod
    def backward(ctx, grad):
        inputs, dy_dx = ctx.saved_tensors
        if dy_dx is None:
            return None, None, None
        B, input_dim, degree = ctx.dims
        grad_inputs = torch.zeros_like(inputs)
        backend("shencoder").sh_encode_backward(grad.contiguous(), inputs, B, input_dim, degree,
                                                dy_dx, grad_inputs)
        return grad_inputs, None, None


def sh_encode(d, degree=4):
    d = d.float().reshape(-1, 3)
    return _SHEncode.apply(d, degree, d.requires_grad)


# ------------------------------------------------------------------ GridRenderer
class MLP(nn.Module):
    """Official MLP: bias-free Linear layers with in-place ReLU between them."""

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.net = nn.ModuleList(
            nn.Linear(dim_in if i == 0 else dim_hidden,
                      dim_out if i == num_layers - 1 else dim_hidden, bias=False)
            for i in range(num_layers)
        )

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i != len(self.net) - 1:
                x = F.relu(x, inplace=True)
        return x


NET_DEFAULTS = {"num_levels": 16, "level_dim": 2, "base_resolution": 16, "log2_hashmap_size": 19,
                "desired_resolution": 512, "hidden": 64, "geo_feat_dim": 64, "sigma_layers": 3,
                "color_layers": 2, "sh_degree": 4, "lr_grid": 5e-3, "lr_net": 5e-4,
                "bound_margin": 1.2}


class GridRenderer(nn.Module):
    """Official GridRenderer: hash grid -> sigma_net (1 + 64) ; [SH4(d), geo] -> color_net -> 3."""

    def __init__(self, bound, coord_center, cfg=None):
        super().__init__()
        cfg = {**NET_DEFAULTS, **(cfg or {})}
        self.cfg = cfg
        self.register_buffer("bound", torch.as_tensor(float(bound), dtype=torch.float32))
        self.register_buffer("coord_center", torch.as_tensor(coord_center, dtype=torch.float32))
        self.encoder_x = GridEncoder(3, cfg["num_levels"], cfg["level_dim"], cfg["base_resolution"],
                                     cfg["log2_hashmap_size"],
                                     cfg["desired_resolution"] * float(bound))
        self.sigma_net = MLP(self.encoder_x.output_dim, 1 + cfg["geo_feat_dim"], cfg["hidden"],
                             cfg["sigma_layers"])
        self.color_net = MLP(cfg["sh_degree"] ** 2 + cfg["geo_feat_dim"], 3, cfg["hidden"],
                             cfg["color_layers"])

    @classmethod
    def from_points(cls, xyz, cfg=None):
        """Official create_from_pcd: bound = max extent / 2 * 1.2, centre = mean (initial cloud)."""
        cfg = {**NET_DEFAULTS, **(cfg or {})}
        xyz = xyz.detach()
        bound = float((xyz.max(0).values - xyz.min(0).values).max()) / 2 * cfg["bound_margin"]
        return cls(bound, xyz.mean(0).cpu(), cfg).to(xyz.device)

    def geo(self, x):
        enc = self.encoder_x(x - self.coord_center, bound=self.bound)
        return self.sigma_net(enc)[..., 1:]  # sigma output unused (combine_opacity)

    def color(self, geo, d):
        h = torch.cat([sh_encode(d, self.cfg["sh_degree"]), geo], dim=-1)
        return color_head(self.color_net, h)

    def forward(self, x, d):
        return self.color(self.geo(x), d)

    def optimizer(self):
        cfg = self.cfg
        return torch.optim.Adam([
            {"params": self.encoder_x.parameters(), "lr": cfg["lr_grid"], "name": "neural_encoder"},
            {"params": self.sigma_net.parameters(), "lr": cfg["lr_net"], "name": "neural_sigma"},
            {"params": self.color_net.parameters(), "lr": cfg["lr_net"], "name": "neural_color"},
        ], lr=0.0, eps=1e-15)


def color_head(color_net, h):
    """sigmoid(color_net(h)) * (1 + 2e-3) - 1e-3 (official GridRenderer.color)."""
    return torch.sigmoid(color_net(h)) * (1 + 2 * 0.001) - 0.001


def view_dirs(means, viewmat):
    """normalize(mu - camera centre) (official render_neural)."""
    center = torch.linalg.inv(viewmat)[:3, 3]
    d = means - center[None]
    return d / d.norm(dim=1, keepdim=True)


# ------------------------------------------------------------------ baking
PACK_MAGIC = 7351.0
PACK_WIDTH = 8
HEADER = 8  # magic, n_weights, n_gaussians, mean|means|, hidden, n_layers, sh_degree, geo_dim


def color_net_weights(net):
    return torch.cat([layer.weight.detach().reshape(-1) for layer in net.color_net.net])


def means_fingerprint(means):
    return float(means.detach().abs().double().mean())


@torch.no_grad()
def bake(params, net):
    """Write the cached geometry feature and the packed colour MLP into the Gaussian fields."""
    n = len(params["means"])
    w = color_net_weights(net)
    need = HEADER + w.numel()
    if n * PACK_WIDTH < need:
        raise RuntimeError(f"cannot bake the colour MLP into {n} Gaussians (need >= "
                           f"{math.ceil(need / PACK_WIDTH)})")
    params["dng_geo"].data.copy_(net.geo(params["means"]))
    cfg = net.cfg
    flat = torch.zeros(n * PACK_WIDTH, device=params["means"].device)
    flat[:HEADER] = torch.tensor([PACK_MAGIC, w.numel(), n, means_fingerprint(params["means"]),
                                  cfg["hidden"], cfg["color_layers"], cfg["sh_degree"],
                                  cfg["geo_feat_dim"]], device=flat.device)
    flat[HEADER:need] = w
    params["dng_mlp"].data.copy_(flat.view(n, PACK_WIDTH))


def unpack_color_net(params):
    """Rebuild the colour MLP from the packed field; raise if absent or stale."""
    flat = params["dng_mlp"].detach().reshape(-1)
    n = len(params["means"])
    if flat.numel() < HEADER or float(flat[0]) != PACK_MAGIC:
        raise RuntimeError("DNGaussian renderer: no baked neural colour state (train to a bake "
                           "step: step % bake_every == 0 or step == total_steps)")
    n_w, n_b, fp, hidden, layers, sh_deg, geo_dim = (float(v) for v in flat[1:HEADER])
    if int(n_b) != n or not math.isclose(fp, means_fingerprint(params["means"]), rel_tol=1e-5):
        raise RuntimeError("DNGaussian renderer: baked colour state is stale (Gaussians changed "
                           "after the last bake)")
    hidden, layers, sh_deg, geo_dim = int(hidden), int(layers), int(sh_deg), int(geo_dim)
    mlp = MLP(sh_deg ** 2 + geo_dim, 3, hidden, layers).to(flat.device)
    w = flat[HEADER:HEADER + int(n_w)]
    i = 0
    for layer in mlp.net:
        k = layer.weight.numel()
        layer.weight.data.copy_(w[i:i + k].view_as(layer.weight))
        i += k
    if i != int(n_w):
        raise RuntimeError("DNGaussian renderer: packed colour MLP has the wrong size")
    return mlp, sh_deg
