"""AuGS renderer: the built-in gsplat 1.5.3 classic rasterizer with the view-dependent opacity
lobe (paper Eq. 1) applied to augmented Gaussians, plus the T-threshold "median" depth used by
the 2D stage and inverse splatting (official modified gsplat, rendering.py RGB+MD)."""

import math

import torch
import torch.nn.functional as F
from gsplat import rasterization

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")
NEAR, FAR = 0.01, 1e10  # gsplat.rasterization defaults (official render() passes none)


def camera_center(viewmat):
    return torch.linalg.inv(viewmat)[:3, 3]


def lobe_attenuation(means, lobe_dir, lobe_t, lobe_beta, campos):
    """Eq. 1 factor ((cos(clip(theta/T, 0, pi)) + 1) / 2)^exp(beta), official compute_opacity:
    theta = acos(clip(d.v, -0.9999, 0.9999)), T = clamp(T, max=pi), 0 where base < 1e-5."""
    view = F.normalize(campos[None] - means, dim=-1)
    d = F.normalize(lobe_dir, dim=-1)
    theta = torch.acos((d * view).sum(-1).clip(-0.9999, 0.9999))
    base = 0.5 * (torch.cos(torch.clip(theta / lobe_t.clamp(max=math.pi), 0, math.pi)) + 1)
    valid = base >= 1e-5
    out = torch.zeros_like(base)
    out[valid] = torch.pow(base[valid], torch.exp(lobe_beta[valid]))
    return out


def augmented_opacities(params, campos):
    """sigmoid(opacity), multiplied by the lobe for augmented Gaussians (aug_flag > 0.5)."""
    opac = params["opacities"].sigmoid()
    if "aug_flag" not in params:
        return opac
    idx = torch.nonzero(params["aug_flag"].detach() > 0.5).squeeze(1)
    if idx.numel() == 0:
        return opac
    att = lobe_attenuation(params["means"][idx], params["lobe_dir"][idx], params["lobe_t"][idx],
                           params["lobe_beta"][idx], campos)
    return opac.index_put((idx,), opac[idx] * att)


def rasterize(params, camera, sh_degree, opacities):
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=opacities,
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    return render, alpha, info


class AugRenderer:
    def __init__(self, spec):
        if spec.config:
            raise ValueError("AugRenderer accepts no configuration")
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        opac = augmented_opacities(params, camera_center(camera["viewmat"]))
        render, alpha, info = rasterize(params, camera, sh_degree, opac)
        # Dataset convention (same as splatstudio.backends.gsplat.Renderer): white background.
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
        }


def create_renderer(spec):
    return AugRenderer(spec)


@torch.no_grad()
def median_depth(params, camera, t_threshold=0.7, tile_size=16):
    """Official "MD" depth: the Gaussian at which transmittance falls below `t_threshold`
    (code: 0.7; paper text: 0.5) is found by the CUDA pass; depth = view-z of its maximum
    response point along the pixel ray (ray-Gaussian max response, double precision).
    No hit -> FAR; closer than NEAR -> NEAR. Returns [H,W] float32."""
    from gsplat.cuda._wrapper import fully_fused_projection

    from .kernels import contrib_stats, tile_lists

    viewmat, K = camera["viewmat"], camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    means, quats = params["means"], F.normalize(params["quats"], dim=-1)
    scales = params["scales"].exp()
    opac = augmented_opacities(params, camera_center(viewmat))
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmat[None], K[None], W, H, eps2d=0.3, near_plane=NEAR,
        far_plane=FAR, radius_clip=0.0, packed=False, calc_compensations=False,
        opacities=opac)
    offsets, flatten_ids = tile_lists(means2d[0], radii[0], depths[0], W, H, tile_size)
    idx, _ = contrib_stats(means2d[0], conics[0], opac, offsets, flatten_ids, W, H, tile_size,
                           t_threshold=t_threshold, alpha_max=0.999, inclusive_stop=False)
    idx = idx.flatten().long()
    hit = idx >= 0
    safe = idx.clamp(min=0)
    rows, cols = torch.meshgrid(torch.arange(H, device=means.device),
                                torch.arange(W, device=means.device), indexing="ij")
    ray_cam = torch.stack(
        ((cols.flatten().double() + 0.5 - K[0, 2].double()) / K[0, 0].double(),
         (rows.flatten().double() + 0.5 - K[1, 2].double()) / K[1, 1].double(),
         torch.ones(H * W, dtype=torch.float64, device=means.device)), dim=-1)
    R = viewmat[:3, :3].double()
    ray = F.normalize(ray_cam @ R, dim=-1)  # world direction (R^T x)
    cam = camera_center(viewmat).double()
    from gsplat.utils import normalized_quat_to_rotmat

    rot = normalized_quat_to_rotmat(quats[safe]).double()  # [P,3,3]
    inv_s = 1.0 / scales[safe].double()
    dg = torch.einsum("pji,pj->pi", rot, ray) * inv_s  # S^-1 R^T d
    og = torch.einsum("pji,pj->pi", rot, cam[None] - means[safe].double()) * inv_s
    tau = -(dg * og).sum(-1) / (dg * dg).sum(-1)
    z = tau * (ray * R[2][None]).sum(-1)
    z = torch.where(hit, z, torch.full_like(z, FAR))
    z = z.clamp(min=NEAR)
    return z.reshape(H, W).float()
