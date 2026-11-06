"""Spec-Gaussian renderer: c = clamp_min(SH(d) + 0.5, 0) + c_s(ASG field), rasterized by gsplat.

gaussian_renderer/__init__.py:render (hybrid=True) of the official code: the SH diffuse colour is
evaluated in Python, the ASG specular colour c_s from the global decoder is added, and the sum is
passed to the rasterizer as precomputed colours. c_s = 0 for iteration <= 3000 (train.py).
The official rasterizer accumulates |dL/dmean2D| per pixel and component for densification
(backward.cu:550, the paper's L1 gradient constraint); gsplat's ``absgrad=True`` is the same
statistic, so the renderer provides ``gradient.screen.absolute``.

Optional paper-text coarse-to-fine resolution (Sec. 3.3, r(i) = min(round(r_s + (r_e - r_s) i /
tau), r_e)): training renders use intrinsics/size scaled by s(i) = min(s0 + (1 - s0) i/tau, 1);
the ``spec-photometric`` loss resizes the target to the rendered size.
"""

import torch
from gsplat import rasterization, spherical_harmonics

from . import bank as bank_mod
from .asg import DEFAULTS as ASG_DEFAULTS
from .asg import baked_step, unpack

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "gradient.screen.absolute"}
DEFAULTS = {
    **ASG_DEFAULTS,
    "spec_start": 3000,  # train.py: mlp_color = 0 unless iteration > 3000
    "near_plane": 0.01,  # gsplat default; Gaussians behind it are culled anyway
    "c2f": False,  # official train.py has no resolution schedule
    "c2f_start_scale": 0.25,  # paper Sec. 4.1: r_s is 4x downsampled
    "c2f_until": 5000,  # paper Sec. 3.3: tau = 5k
}
BLOB = "spec_blob"
BLOB_WIDTH = 4  # floats per Gaussian row of the packed decoder (needs >= 651 Gaussians)
FEAT = "spec_feat"


def camera_center(viewmat):
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    return -R.T @ t


def c2f_scale(step, start_scale, until):
    return min(start_scale + (1.0 - start_scale) * step / until, 1.0)


def scaled_camera(camera, s):
    if s >= 1.0:
        return camera
    w = max(int(round(camera["width"] * s)), 16)
    h = max(int(round(camera["height"] * s)), 16)
    K = camera["K"].clone()
    K[0] = K[0] * (w / camera["width"])
    K[1] = K[1] * (h / camera["height"])
    return {**camera, "K": K, "width": w, "height": h}


def diffuse_colour(params, dirs, sh_degree):
    """c_d: official ``clamp_min(eval_sh(active_degree, shs, d) + 0.5, 0)``."""
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)
    return torch.clamp_min(spherical_harmonics(sh_degree, dirs, coeffs) + 0.5, 0.0)


class SpecRenderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.bank = None  # explicit bank (probes); otherwise bank.ACTIVE
        self._decoded = None
        self._key = None

    def active_bank(self):
        return self.bank if self.bank is not None else bank_mod.ACTIVE

    def decoded(self, params):
        blob = params[BLOB].detach().reshape(-1)
        key = (blob.data_ptr(), tuple(blob.shape), float(blob[2]) if blob.numel() > 2 else 0.0)
        if self._key != key:
            self._decoded = unpack(blob, self.cfg)
            self._key = key
        return self._decoded

    def colours(self, params, camera, sh_degree, network, spec_on):
        means = params["means"]
        dirs = means - camera_center(camera["viewmat"])
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        colour = diffuse_colour(params, dirs, sh_degree)
        if not spec_on:
            return colour, None
        z = means @ camera["viewmat"][2, :3] + camera["viewmat"][2, 3]
        idx = torch.nonzero(z > self.cfg["near_plane"], as_tuple=True)[0]
        c_s = network(params[FEAT][idx], dirs[idx])
        return colour.index_add(0, idx, c_s), c_s

    def render(self, params, camera, sh_degree=3):
        bank = self.active_bank()
        if bank is not None:
            network, step = bank.network, bank.current_step()
        else:  # ss-eval / fresh process: decoder from the checkpointed blob
            network, step = self.decoded(params), None
        # after training, the render state is that of step (last completed step + 1), exactly
        # as the runtime's in-process validation sees it through the live bank
        cur = step if step is not None else baked_step(params[BLOB]) + 1
        spec_on = cur > self.cfg["spec_start"]
        cam = camera
        if self.cfg["c2f"] and step is not None:
            cam = scaled_camera(camera, c2f_scale(step, self.cfg["c2f_start_scale"],
                                                  self.cfg["c2f_until"]))
        colour, c_s = self.colours(params, cam, sh_degree, network, spec_on)
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(), colors=colour,
            viewmats=cam["viewmat"][None], Ks=cam["K"][None],
            width=cam["width"], height=cam["height"], sh_degree=None, packed=False,
            render_mode="RGB+ED", absgrad=True, near_plane=self.cfg["near_plane"])
        info["spec"] = {"spec_on": spec_on, "c_s": c_s, "width": cam["width"],
                        "height": cam["height"]}
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4],
                "alpha": alpha[0], "info": info}


def create_renderer(spec):
    return SpecRenderer(spec)
