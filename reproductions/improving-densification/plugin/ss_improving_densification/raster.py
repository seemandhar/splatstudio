"""Official ImprovedGS rasterizer, JIT-built from the vendored, unmodified ``csrc/``.

``csrc/`` = ``submodules/diff-gaussian-rasterization`` of github.com/XiaoBin2001/Improved-GS
(20b5db3; Inria 3DGS rasterizer with TamingGS per-bucket backward, SpeedySplat boxes, absolute
screen gradients and edge-weighted blend-weight accumulation). The autograd wrapper below
re-implements ``diff_gaussian_rasterization._RasterizeGaussians`` with the same argument order;
camera conventions follow ``scene/cameras.py`` (znear 0.01, zfar 100, symmetric FoV).

Kernel facts used by the method:
- ``means2D`` is [N, 4]; its gradient holds (dL/dx, dL/dy) in channels 0:2 and
  (sum_p |dL_p/dx|, sum_p |dL_p/dy|) in channels 2:4 (backward.cu:647-650, Eq. 7).
- with ``pixel_weights`` [H, W], ``accum_weights[i] = sum_p w_p * T_p,i * alpha_p,i``
  (forward.cu:416, Eq. 5).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        glm = CSRC / "third_party" / "glm"
        sources = [CSRC / "cuda_rasterizer" / f for f in
                   ("rasterizer_impl.cu", "forward.cu", "backward.cu", "adam.cu", "utils.cu")]
        sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
        _EXT = load(name="ss_improvedgs_raster", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3"], verbose=False)
    return _EXT


def raster_settings(camera, background, sh_degree, pixel_weights=None, inference_only=False):
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("Inria rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")
    tanfovx, tanfovy = float(W / (2 * K[0, 0])), float(H / (2 * K[1, 1]))
    proj = torch.zeros(4, 4, device=K.device)  # graphics_utils.getProjectionMatrix
    proj[0, 0] = 1.0 / tanfovx
    proj[1, 1] = 1.0 / tanfovy
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    if pixel_weights is None:
        pixel_weights = torch.empty(0, device=K.device)
    return {
        "bg": background, "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": tanfovx, "tanfovy": tanfovy, "width": W, "height": H,
        "sh_degree": int(sh_degree), "pixel_weights": pixel_weights.float().contiguous(),
        "inference_only": bool(inference_only),
    }


def _forward_args(means3D, dc, sh, opacities, scales, rotations, s):
    return (s["bg"], means3D, opacities, scales, rotations, 1.0, s["viewmatrix"],
            s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"], s["width"], dc, sh,
            s["sh_degree"], s["campos"], False, False, s["inference_only"], s["pixel_weights"],
            False)


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, dc, sh, opacities, scales, rotations, settings):
        (num_rendered, num_buckets, color, invdepth, radii, geom, binning, img, sample,
         accum) = extension().rasterize_gaussians(
            *_forward_args(means3D, dc, sh, opacities, scales, rotations, settings))
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(color, invdepth, means3D, scales, rotations, radii, dc, sh,
                              opacities, geom, binning, img, sample)
        ctx.mark_non_differentiable(radii, accum)
        return color, radii, invdepth, accum

    @staticmethod
    def backward(ctx, g_color, _g_radii, g_invdepth, _g_accum):
        s = ctx.settings
        (color, invdepth, means3D, scales, rotations, radii, dc, sh, opacities, geom, binning,
         img, sample) = ctx.saved_tensors
        if g_invdepth is None:
            g_invdepth = torch.zeros_like(invdepth)
        args = (s["bg"], means3D, radii, opacities, scales, rotations, 1.0, s["viewmatrix"],
                s["projmatrix"], s["tanfovx"], s["tanfovy"], color, invdepth,
                g_color.contiguous(), dc, sh, g_invdepth.contiguous(), s["sh_degree"],
                s["campos"], geom, ctx.num_rendered, binning, img, ctx.num_buckets, sample,
                False, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_dc, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_dc, g_sh, g_opac, g_scales, g_rot, None


def activated(params):
    """GaussianModel getters: exp scales, sigmoid opacity [N,1], normalised wxyz quaternions."""
    return (params["scales"].exp(), params["opacities"].sigmoid()[:, None],
            F.normalize(params["quats"], dim=-1))


def rasterize(params, camera, sh_degree, background, pixel_weights=None):
    """gaussian_renderer.render_3dgs. Returns (color CHW, radii, invdepth 1HW, accum, means2D)."""
    track = torch.is_grad_enabled()
    means = params["means"]
    means2D = torch.zeros((len(means), 4), dtype=means.dtype, device=means.device,
                          requires_grad=track)
    if track:
        means2D.retain_grad()
    inference_only = not track and pixel_weights is None
    settings = raster_settings(camera, background, sh_degree, pixel_weights, inference_only)
    scales, opac, rots = activated(params)
    color, radii, invdepth, accum = _Rasterize.apply(
        means.contiguous(), means2D, params["sh0"].contiguous(), params["shN"].contiguous(),
        opac.contiguous(), scales.contiguous(), rots.contiguous(), settings)
    return color, radii, invdepth, accum, means2D


@torch.no_grad()
def blend_weights(params, camera, pixel_weights):
    """Per-Gaussian sum_p w_p T alpha for one view (EAS, Eq. 5) and radii (visibility)."""
    bg = torch.zeros(3, device=params["means"].device)
    _, radii, _, accum, _ = rasterize(params, camera, 0, bg, pixel_weights)
    return accum, radii
