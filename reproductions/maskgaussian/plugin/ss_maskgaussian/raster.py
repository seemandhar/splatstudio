"""Official MaskGaussian masked rasterizer, JIT-built from ``csrc/maskrast``.

``csrc/maskrast`` = github.com/kaikai23/mask-diff-gaussian-rasterization @ 540dc11 (the
``submodules/mask-diff-gaussian-rasterization`` of MaskGaussian @ 2f581d0): the Inria 3D-GS
rasterizer with a per-Gaussian mask M in the alpha blending (forward.cu: C += c M alpha T,
T *= 1 - M alpha; backward.cu: dL/dM). Gaussian-Splatting licence (Inria/MPII, non-commercial
research use, ``csrc/maskrast/LICENSE.md``). The only edit is an opt-in ``#ifdef
MASKGS_EXACT_ACCUM`` block in backward.cu (see README); the default build is the official kernel.
glm = upstream header library. The Python side reproduces
``mask_diff_gaussian_rasterization/__init__.py`` and ``gaussian_renderer.render``.
"""

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py
ACCUM = ("official", "exact")


@functools.cache
def extension(accum="official"):
    if accum not in ACCUM:
        raise ValueError(f"accum must be one of {ACCUM}")
    from torch.utils.cpp_extension import load

    root = CSRC / "maskrast"
    glm = CSRC / "third_party" / "glm"
    sources = [root / "cuda_rasterizer" / f for f in
               ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [root / "rasterize_points.cu", root / "ext.cpp"]
    flags = ["-O3", f"-I{glm}"] + (["-DMASKGS_EXACT_ACCUM"] if accum == "exact" else [])
    return load(name=f"ss_maskgaussian_rasterizer_{accum}", sources=[str(s) for s in sources],
                extra_cuda_cflags=flags, extra_cflags=["-O3", f"-I{glm}"], verbose=False)


def _check_centered(camera):
    # The Inria kernels place the principal point at (W/2, H/2) (FoV camera model).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("Inria rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
    _check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = torch.zeros(4, 4, device=K.device)
    proj[0, 0] = 2 * fx / W
    proj[0, 2] = 2 * cx / W - 1
    proj[1, 1] = 2 * fy / H
    proj[1, 2] = 2 * cy / H - 1
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    return {
        "bg": background, "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * fx)), "tanfovy": float(H / (2 * fy)),
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


class _MaskedRasterize(torch.autograd.Function):
    """mask_diff_gaussian_rasterization._RasterizeGaussians (SH colours, no precomputed cov)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, masks, scales, rotations, s, accum, stats):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, masks, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num_rendered, color, radii, geom, binning, img = \
            extension(accum).rasterize_gaussians(*args)
        stats["num_rendered"] = int(num_rendered)
        ctx.s, ctx.accum, ctx.num_rendered = s, accum, num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        s = ctx.s
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                grad_color.contiguous(), sh, s["sh_degree"], s["campos"], geom,
                ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_mask, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension(ctx.accum).rasterize_gaussians_backward(*args)
        return (g_means3D, g_means2D, g_sh, g_opac, g_mask, g_scales, g_rot, None, None, None)


def rasterize(params, camera, sh_degree, background, masks, accum="official",
              opacity_mask=False):
    """gaussian_renderer.render with explicit masks [N, 1] (float, may carry a graph).

    ``opacity_mask=True`` is the paper's 'Mask x opacity' ablation: the mask multiplies the
    opacity and the kernel blends with M = 1 (unsampled Gaussians then get no gradient).
    Returns CHW colour, the screen-space tensor (grad in [:, :2]), radii, stats.
    """
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad and means2D.requires_grad:
        means2D.retain_grad()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    opac = params["opacities"].sigmoid()[:, None]
    if opacity_mask:
        opac = opac * masks
        masks = torch.ones_like(masks)
    s = raster_settings(camera, background, sh_degree)
    stats = {}
    color, radii = _MaskedRasterize.apply(
        means.contiguous(), means2D, sh.contiguous(), opac.contiguous(),
        masks.float().contiguous(), params["scales"].exp().contiguous(),
        F.normalize(params["quats"], dim=-1).contiguous(), s, accum, stats)
    return color, means2D, radii, stats
