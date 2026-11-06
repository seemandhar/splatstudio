"""Splatfacto-W renderer on gsplat (official ``SplatfactoWModel.get_outputs``).

Training renders (grad on, bank in train mode) follow the splatfacto resolution schedule
(1/4 -> 1/2 -> full every 3000 steps). Colours: b_i = MLP(l_j, f_i) (C1), c_i =
max(SH_k(b_i, R(mu_i - o)) + 0.5, 0) — gsplat's SH colour, as the code (the paper writes a
sigmoid); R is the Nerfstudio orientation so the SH frame matches the code. Background (C3):
C_bg(r) = SH_kb(MLP_bg(l_j), R d_ray(r)) (raw SH, as the code), rgb = clamp(C + (1 - alpha) C_bg,
0, 1). Without the background model: a random colour per training render, a fixed colour at eval.
"""

import torch
from gsplat import rasterization, spherical_harmonics

from . import bank as bank_mod
from .schedule import downscale_factor
from .schedule import sh_degree as degree_at

CAPABILITIES = ("gradient.screen.absolute", "render.antialiased_2d", "render.rgb")
DEFAULTS = {
    "bg_model": True,
    "sh_degree": 3,
    "sh_degree_interval": 2000,
    "bg_sh_degree": 4,
    "num_downscales": 2,
    "resolution_schedule": 3000,
    "eval_background": [0.1490, 0.1647, 0.2157],  # viser default colour (code)
    "near_plane": 0.01,
    "far_plane": 1e10,
}


def scaled_camera(camera, d):
    """nerfstudio Cameras.rescale_output_resolution(1/d)."""
    if d == 1:
        return camera
    K = camera["K"].clone()
    K[:2] /= d
    return {**camera, "K": K, "width": int(int(camera["width"]) / d),
            "height": int(int(camera["height"]) / d)}


def ray_directions(camera):
    """World-space ray directions through pixel centres, [H, W, 3] (unnormalised)."""
    K, W, H = camera["K"], int(camera["width"]), int(camera["height"])
    dev = K.device
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                          torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    d = torch.stack([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], torch.ones_like(x)], -1)
    c2w = torch.linalg.inv(camera["viewmat"])
    return d @ c2w[:3, :3].T


def gaussian_colors(params, camera, bank, embedding, degree):
    coeffs = bank.gaussian_sh(embedding, params["appearance_features"])
    campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
    dirs = bank.to_ns_dirs(params["means"] - campos[None])
    return torch.clamp_min(spherical_harmonics(int(degree), dirs, coeffs) + 0.5, 0.0)


def background_image(camera, bank, embedding, degree):
    coeffs = bank.background_sh(embedding)
    dirs = bank.to_ns_dirs(ray_directions(camera))
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    flat = dirs.reshape(-1, 3)
    out = spherical_harmonics(int(degree), flat, coeffs[None].expand(len(flat), -1, -1))
    return out.view(*dirs.shape[:2], 3)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown splatfacto-w renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        """``sh_degree`` from the runtime is ignored: the code's own schedules are used."""
        bank = bank_mod.ACTIVE
        if bank is None:
            raise RuntimeError("splatfacto-w renderer needs the bank (sfw-density / evaluate)")
        c = self.cfg
        training = torch.is_grad_enabled() and bank.training
        k = bank.step
        d = downscale_factor(k, int(c["num_downscales"]), int(c["resolution_schedule"]), training)
        cam = scaled_camera(camera, d)
        idx = bank.view_index(camera["viewmat"])
        if training and idx is None:
            raise RuntimeError("training view not registered in the splatfacto-w bank")
        emb = bank.embedding_for(idx)
        deg = degree_at(k, int(c["sh_degree_interval"]), int(c["sh_degree"]))
        colors = gaussian_colors(params, cam, bank, emb, deg)
        q = params["quats"]
        render, alpha, info = rasterization(
            means=params["means"], quats=q / q.norm(dim=-1, keepdim=True),
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=colors, viewmats=cam["viewmat"][None], Ks=cam["K"][None],
            width=int(cam["width"]), height=int(cam["height"]), sh_degree=None,
            near_plane=float(c["near_plane"]), far_plane=float(c["far_plane"]), packed=False,
            render_mode="RGB", absgrad=True, rasterize_mode="antialiased")
        rgb, alpha = render[0], alpha[0]
        if c["bg_model"]:
            bdeg = degree_at(k, int(c["sh_degree_interval"]) // 2, int(c["bg_sh_degree"]))
            background = background_image(cam, bank, emb, bdeg)
        elif training:
            background = torch.rand(3, device=rgb.device).expand_as(rgb)
        else:
            background = torch.tensor(c["eval_background"], device=rgb.device).expand_as(rgb)
        out = torch.clamp(rgb + (1 - alpha) * background, 0.0, 1.0)
        masked = bool(bank.masked[idx]) if (training and idx is not None) else False
        return {"rgb": out, "alpha": alpha, "background": background, "info": info,
                "view_index": idx, "downscale": d, "masked_view": masked,
                "foreground": rgb}


def create_renderer(spec):
    return Renderer(spec)
