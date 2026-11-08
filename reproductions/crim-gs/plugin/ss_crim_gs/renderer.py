"""CRiM-GS renderer: Mip-Splatting gsplat rendering + continuous camera-motion blur synthesis.

* Every camera: sharp render I^{tau_0} at the input pose with the Mip-Splatting 3D filter
  (scales / opacities, Eq. 9 of Mip-Splatting) and the 2D Mip filter (gsplat antialiased,
  eps2d = kernel_size 0.1), over a black background (CoMoGaussian ``white_background=False``).
  Its gsplat ``info`` feeds the density statistics (CoMo train.py: densification statistics
  come from the sharp render only).
* Training camera (gradients on, live kernel, viewmat of a training view, step > start_warp):
  the N trajectory poses from the kernel are rendered in one batched gsplat call and combined
    - start_warp < step <= start_pixel_weight: mean of the N renders and I^{tau_0}
      (CoMo train.py: ``rendered_images.append(image_ori); torch.mean``),
    - step > start_pixel_weight: I_blur = sum_i P_i I^{tau_i}, P = softmax(F(I)) and
      I_out = (1 - M) I^{tau_0} + M I_blur (Sec. 4.4); F sees the renders detached (CoMo).
  ``rgb`` is the blurred output (compared with the blurry GT); ``sharp`` is I^{tau_0}.
* Any other camera (ss-eval, validation): ``rgb`` = sharp render at the given pose.
"""

import torch
from gsplat import rasterization

from . import kernel as kernel_mod
from .filters import filtered_scales_opacities

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "render.antialiased_2d", "render.filter_3d", "render.motion_blur")
DEFAULTS = {"kernel_size_2d": 0.1, "background": 0.0}


def rasterize(params, viewmats, Ks, width, height, sh_degree, kernel_size_2d):
    scales, opacities = filtered_scales_opacities(params["scales"], params["opacities"],
                                                  params["filter_3d"])
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=Ks, width=int(width), height=int(height),
        sh_degree=sh_degree, packed=False, render_mode="RGB+ED", eps2d=kernel_size_2d,
        rasterize_mode="antialiased",
    )
    return render[..., :3], render[..., 3:4], alpha, info


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown crim-gs renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def _render(self, params, viewmats, camera, sh_degree):
        n = viewmats.shape[0]
        rgb, depth, alpha, info = rasterize(
            params, viewmats, camera["K"][None].expand(n, 3, 3), camera["width"],
            camera["height"], sh_degree, float(self.cfg["kernel_size_2d"]))
        return rgb + (1 - alpha) * float(self.cfg["background"]), depth, alpha, info

    def render(self, params, camera, sh_degree=3):
        sharp, depth, alpha, info = self._render(params, camera["viewmat"][None], camera,
                                                 sh_degree)
        out = {"rgb": sharp[0], "sharp": sharp[0], "depth": depth[0], "alpha": alpha[0],
               "info": info}
        kern = kernel_mod.ACTIVE
        if kern is None or not torch.is_grad_enabled():
            return out
        idx = kern.view_index(camera["viewmat"])
        step = kern.step + 1  # the runtime step being rendered
        if idx is None or not kern.trajectory_active(step):
            return out
        viewmats, R_a = kern.trajectory(idx)
        renders, _, _, _ = self._render(params, viewmats, camera, sh_degree)  # [N,H,W,3]
        if kern.pixel_weight_active(step):
            weights, mask = kern.weights_and_mask(renders.detach())
            blur = (weights * renders).sum(dim=0)
            rgb = (1 - mask) * sharp[0] + mask * blur
        else:
            weights = mask = None
            rgb = torch.cat((renders, sharp), dim=0).mean(dim=0)
        out.update(rgb=rgb, subframes=renders, subframe_viewmats=viewmats, R_a=R_a,
                   pixel_weights=weights, mask=mask, view_index=idx)
        return out


def create_renderer(spec):
    return Renderer(spec)
