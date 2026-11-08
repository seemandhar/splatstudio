"""Turbo-GS dilated rendering (paper Sec. 4.1) on the gsplat 1.5.3 rasterizer.

Training renders the chessboard sub-lattice {(d*j + o_x, d*i + o_y)} of the full image. The
2D Gaussian is the full-resolution footprint: projected with the original focal length,
low-pass filtered by 0.3 px^2 in *high-resolution* pixel units, with 3-sigma radius r/d on the
downsampled grid (Eq. r_ds = r/d). We express this exactly with gsplat by
  - sub-pixel shifted intrinsics: ds pixel j has its centre at high-res coordinate
    d*j + o + 1/2, i.e. u_ds = (u_hr - o - 1/2)/d + 1/2, so fx' = fx/d, cx' = (cx - o - 1/2)/d + 1/2;
  - eps2d = 0.3 / d^2 (the 0.3 high-res px^2 filter expressed in ds px^2).
The ds covariance is then Sigma_hr/d^2 + 0.3/d^2 = (Sigma_hr + 0.3)/d^2, so the conic evaluated
at a ds pixel equals the full-resolution conic evaluated at the corresponding high-res pixel,
and gsplat's radius = ceil(3 sqrt(lambda_max)) is r/d up to rounding. Tile binning runs in ds
space (16x16 ds tiles), as in the paper.

Evaluation (torch.no_grad) renders the full image with the standard eps2d = 0.3.
"""

import random

import torch
from gsplat import rasterization

DILATED = "render.dilated_subset"
COLORS = "render.color_coefficients"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed", DILATED, COLORS)
OFFSETS = tuple((ox, oy) for oy in range(2) for ox in range(2))


def dilated_camera(camera, d, ox, oy):
    """Camera rendering the high-res pixels (d*j + ox, d*i + oy), size floor(W/d) x floor(H/d)."""
    K = camera["K"].clone()
    K[0, 0] = camera["K"][0, 0] / d
    K[1, 1] = camera["K"][1, 1] / d
    K[0, 1] = camera["K"][0, 1] / d
    K[0, 2] = (camera["K"][0, 2] - ox - 0.5) / d + 0.5
    K[1, 2] = (camera["K"][1, 2] - oy - 0.5) / d + 0.5
    return {**camera, "K": K, "width": int(camera["width"]) // d,
            "height": int(camera["height"]) // d}


def dilated_target(image, d, ox, oy, height, width):
    """HWC image -> the supervised pixel subset image[oy::d, ox::d] cropped to the render size."""
    return image[oy::d, ox::d][:height, :width]


def rasterize(params, camera, sh_degree, eps2d):
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED",
        rasterize_mode="classic",
        eps2d=eps2d,
    )
    return render, alpha, info, colors


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        unknown = set(spec.config) - {"dilation", "eps2d", "seed"}
        if unknown:
            raise ValueError(f"Unknown turbo renderer options: {sorted(unknown)}")
        self.d = int(spec.config.get("dilation", 2))
        if self.d < 1:
            raise ValueError("dilation must be >= 1")
        self.eps2d = float(spec.config.get("eps2d", 0.3))
        self.offsets = tuple((ox, oy) for oy in range(self.d) for ox in range(self.d))
        # Private RNG: does not perturb the runtime's view-sampling stream.
        self.rng = random.Random(int(spec.config.get("seed", 0)))

    def render(self, params, camera, sh_degree=3, offset=None):
        training = torch.is_grad_enabled()
        if training and self.d > 1:
            ox, oy = offset if offset is not None else self.rng.choice(self.offsets)
            cam = dilated_camera(camera, self.d, ox, oy)
            eps2d = self.eps2d / (self.d * self.d)
            dilation = (self.d, ox, oy)
        else:
            cam, eps2d, dilation = camera, self.eps2d, (1, 0, 0)
        render, alpha, info, colors = rasterize(params, cam, sh_degree, eps2d)
        out = {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "dilation": dilation,
            # Non-leaf SH tensor; the density retains its gradient for the colour branch.
            "colors": colors,
        }
        return out


def create_renderer(spec):
    return Renderer(spec)
