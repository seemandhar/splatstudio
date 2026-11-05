"""DN-Splatter renderer (dn_model.py:get_outputs, official commit 97588b4).

* RGB + expected depth in one gsplat pass (``RGB+ED``, classic, absgrad, near 0.01, tile 16).
* Random background while training, the viser grey (0.1490, 0.1647, 0.2157) at evaluation
  (populate_modules with background_color="random"); rgb clamped to [0, 1].
* Depth: pixels with zero accumulation get the detached maximum depth.
* Normals (paper Eq. 4, 7): n_i = R(q_i) onehot(argmin s_i), flipped to face the camera every
  step, rotated into the camera frame and alpha-composited in a second pass that uses the
  *detached* 2D means but differentiable conics/opacities, like the official legacy
  ``rasterize_gaussians(xys.detach(), ...)`` call. The legacy rasterizer composites over a white
  background (gsplat 1.0 cuda_legacy/_wrapper.py:339) *in the OpenGL camera frame*, then the map
  is normalised per pixel and encoded as (n + 1) / 2.
* Optional Splatfacto resolution schedule (``num_downscales`` > 0, vanilla splatfacto.py:653-660,
  737-742): on training render k (0-based, = Nerfstudio ``self.step``) the camera is rendered at
  1 / 2^max(num_downscales - k // resolution_schedule, 0) (fx, fy, cx, cy scaled by 1/d, width /
  height truncated like ``Cameras.rescale_output_resolution``); ``outputs["downscale"]`` tells the
  photometric loss to area-downscale the target (``resize_image``). Evaluation renders are always
  full resolution. DN-Splatter itself uses num_downscales = 0 (dn_model.py:104).

Frame convention: official normals live in the Nerfstudio (OpenGL) camera frame and are compared
with Omnidata pngs converted by D = diag(1, -1, -1). Elementwise |D a - D b| = |a - b| and the TV
term is sign invariant, so rendering in the OpenCV frame (R_w2c n) and comparing with the raw
png values gives exactly the official loss values and gradients, provided the background is also
mapped through D: the official white (1, 1, 1) in the OpenGL frame is (1, -1, -1) in OpenCV.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterize_to_pixels
from gsplat.utils import normalized_quat_to_rotmat

from . import sidecar

CAPABILITIES = {
    "render.rgb",
    "render.depth.expected_z",
    "render.normal.camera",
    "gradient.screen.absolute",
    "prior.sensor_depth",
    "prior.mono_normal",
}
NORMAL_CAPABILITIES = {"render.normal.camera", "prior.sensor_depth", "prior.mono_normal"}


def capabilities(normal_pass=True):
    return set(CAPABILITIES) if normal_pass else set(CAPABILITIES) - NORMAL_CAPABILITIES
DEFAULTS = {
    "random_background": True,
    "eval_background": [0.1490, 0.1647, 0.2157],
    "near_plane": 0.01,
    "normal_eps": 1e-12,
    "normal_pass": True,
    "num_downscales": 0,
    "resolution_schedule": 3000,
}
# official legacy-rasterizer white background (OpenGL camera frame) expressed in the OpenCV frame
NORMAL_BACKGROUND_CV = (1.0, -1.0, -1.0)


def downscale_factor(train_step, num_downscales, resolution_schedule):
    """splatfacto.py:_get_downscale_factor for a 0-based training step."""
    return 2 ** max(int(num_downscales) - int(train_step) // int(resolution_schedule), 0)


def gaussian_normals(params, viewmat):
    """World normals of every Gaussian, flipped towards the camera centre (dn_model.py:476-492)."""
    quats = F.normalize(params["quats"], dim=-1)
    onehot = F.one_hot(torch.argmin(params["scales"], dim=-1), num_classes=3).to(quats.dtype)
    rots = normalized_quat_to_rotmat(quats)
    normals = F.normalize(torch.bmm(rots, onehot[:, :, None]).squeeze(-1), dim=1)
    center = torch.linalg.inv(viewmat.detach())[:3, 3]
    viewdirs = F.normalize(center[None] - params["means"].detach(), dim=-1)
    flip = (normals * viewdirs).sum(-1) < 0
    return torch.where(flip[:, None], -normals, normals)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DN-Splatter renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = capabilities(self.cfg["normal_pass"])
        self._table = False
        self.train_renders = 0  # number of training renders so far (= Nerfstudio self.step)

    def table(self):
        if self._table is False:
            self._table = sidecar.table()
        return self._table

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        training = torch.is_grad_enabled() and params["means"].requires_grad
        viewmat, K = camera["viewmat"], camera["K"]
        W, H = int(camera["width"]), int(camera["height"])
        d = 1
        if training:
            d = downscale_factor(self.train_renders, self.cfg["num_downscales"],
                                 self.cfg["resolution_schedule"])
            self.train_renders += 1
        if d > 1:
            K = K.clone()
            K[:2] = K[:2] / d
            W, H = int(W / d), int(H / d)
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            tile_size=16,
            packed=False,
            near_plane=float(self.cfg["near_plane"]),
            far_plane=1e10,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )
        if training and self.cfg["random_background"]:
            bg = torch.rand(3, device=dev)
        else:
            bg = torch.tensor(self.cfg["eval_background"], dtype=torch.float32, device=dev)
        alpha = alpha[0]
        rgb = torch.clamp(render[0, ..., :3] + (1 - alpha) * bg, 0.0, 1.0)
        depth = render[0, ..., 3:4]
        depth = torch.where(alpha > 0, depth, depth.detach().max())
        out = {"rgb": rgb, "depth": depth, "alpha": alpha, "info": info, "camera": camera,
               "downscale": d}
        if not training or not self.cfg["normal_pass"]:
            return out
        normals = gaussian_normals(params, viewmat)
        normals_cam = normals @ viewmat[:3, :3].T  # R_w2c n (OpenCV camera frame)
        nimg, nalpha = rasterize_to_pixels(
            info["means2d"].detach(),
            info["conics"],
            normals_cam[None],
            info["opacities"],
            W,
            H,
            info["tile_size"],
            info["isect_offsets"],
            info["flatten_ids"],
        )
        # legacy default background = ones in the OpenGL frame = (1, -1, -1) in OpenCV
        bg_n = torch.tensor(NORMAL_BACKGROUND_CV, dtype=nimg.dtype, device=dev)
        nimg = nimg[0] + (1 - nalpha[0]) * bg_n
        nimg = nimg / nimg.norm(dim=-1, keepdim=True).clamp_min(float(self.cfg["normal_eps"]))
        out["normal"] = (nimg + 1) / 2
        tab = self.table()
        idx = tab.index(viewmat) if tab is not None else None
        out["dn_view"] = tab.view(idx, dev) if idx is not None else None
        return out


def create_renderer(spec):
    return Renderer(spec)
