"""WSR renderer: gsplat 1.5.3 projection + SH, then the order-independent weighted-sum
rasterizer (csrc/wsr_raster.cu) and the Eq. 7 normalisation in torch."""

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from .kernels import ALPHA_THRESHOLD, rasterize_wsr
from .wsr import compose, depth_weight, view_dependent_opacity

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")
# background: c_B of Eq. 7, a model quantity (not the runtime's white compositing convention);
# black as in the paper's 3DGS evaluation setting and the unofficial code (white_background=False).
# cull_nonpositive=False is UNSAFE: compose() clamps the denominator to >= den_eps, which maps
# negative denominators (possible with signed u) to 1e-4 and blows up the colour; it would need a
# sign-preserving guard before being used as an ablation. Culled Gaussians (u <= 1/255 or w <= 0)
# receive no gradient and are never pruned (same as the unofficial code; fidelity gap).
OPTIONS = {"weight": "lc", "u_offset": 0.1, "cull_nonpositive": True, "den_eps": 1e-4,
           "clamp_output": True, "tile_size": 16, "eps2d": 0.3, "background": 0.0}
GLOBAL_FIELDS = ("wsr_sigma", "wsr_wbg")


def global_value(params, name):
    """Global learnable scalar stored as identical per-Gaussian rows (see density.py).

    Value = row 0 (all rows are identical); the zero-valued straight-through term gives EVERY row
    the full scalar gradient g (not g/N), so Adam on the identical rows equals a scalar Adam step
    independently of the population size N (review-1 M1). Not written as mean*N - mean*(N-1):
    that cancels catastrophically in float32 for N ~ 1e5-1e6 (seen in smoke-2: sigma quantised to
    39.0)."""
    p = params[name]
    return p[0].detach() + (p - p.detach()).sum()


def render_wsr(params, camera, sh_degree=3, weight="lc", u_offset=0.1, cull_nonpositive=True,
               den_eps=1e-4, clamp_output=True, tile_size=16, eps2d=0.3, background=0.0):
    means = params["means"]
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    W, H = int(camera["width"]), int(camera["height"])
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, params["quats"], params["scales"].exp(), viewmats, Ks, W, H,
        eps2d=eps2d, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False)
    campos = torch.inverse(viewmats)[..., :3, 3]
    dirs = means[None] - campos[:, None]  # [1,N,3]
    # Eq. 11 view-dependent, unclamped opacity u_i; Eq. 10 weight w_i
    u = view_dependent_opacity(sh_degree, params["opa_sh0"], params["opa_shN"], dirs[0],
                               u_offset)
    v = params["opacities"].sigmoid()
    sigma = global_value(params, "wsr_sigma")
    w_bg = global_value(params, "wsr_wbg")
    w = depth_weight(weight, depths[0], v, sigma)
    with torch.no_grad():
        if cull_nonpositive:  # unofficial reference: u <= 0, v <= 0 or d >= sigma never render
            live = (u >= ALPHA_THRESHOLD) & (w > 0)
        else:
            live = (u.abs() >= ALPHA_THRESHOLD) & (w != 0)
        radii = torch.where(live[None, :, None], radii, torch.zeros_like(radii))
    shs = torch.cat((params["sh0"], params["shN"]), dim=1)
    colors = spherical_harmonics(sh_degree, dirs, shs[None], masks=(radii > 0).all(dim=-1))
    colors = torch.clamp_min(colors + 0.5, 0.0)
    feats = torch.cat((colors[0], depths[0][:, None]), dim=-1)
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    # Sort-free: tiles are binned by tile id only (constant depth key), no depth ordering.
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, torch.zeros_like(depths), tile_size, tw, th, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
    feat_sum, w_sum = rasterize_wsr(means2d[0], conics[0], feats, u, w, offsets, flatten_ids,
                                    W, H, tile_size)
    bg = torch.full((3,), float(background), device=means.device)
    rgb, alpha = compose(feat_sum[..., :3], w_sum, w_bg, bg, den_eps)  # Eq. 7
    if clamp_output:  # rebuttal: final image clamped as in 3DGS
        rgb = rgb.clamp(0.0, 1.0)
    depth = feat_sum[..., 3:4] / w_sum.clamp_min(1e-10)
    info = {
        "radii": radii, "means2d": means2d, "depths": depths, "conics": conics,
        "width": W, "height": H, "tile_size": tile_size, "tile_width": tw, "tile_height": th,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids, "flatten_ids": flatten_ids,
        "isect_offsets": offsets[None], "n_batches": 1, "n_cameras": 1, "gaussian_ids": None,
        "camera_ids": None,
        "wsr": {"u": u, "w": w, "v": v, "sigma": sigma, "w_bg": w_bg, "w_sum": w_sum,
                "feat_sum": feat_sum},
    }
    return {"rgb": rgb, "depth": depth, "alpha": alpha, "info": info}


class WSRRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown WSR renderer options: {sorted(unknown)}")
        if cfg["weight"] not in ("lc", "dir"):
            raise ValueError("weight must be 'lc' or 'dir'")
        self.cfg = cfg
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        return render_wsr(params, camera, sh_degree, **self.cfg)


def create_renderer(spec):
    return WSRRenderer(spec)
