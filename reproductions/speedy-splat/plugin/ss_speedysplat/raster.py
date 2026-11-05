"""Official Speedy-Splat rasterizer and the stock Inria rasterizer, JIT-built from ``csrc/``.

``csrc/speedy`` = github.com/j-alex-hanson/speedy-splat-rasterizer @ b38abb6 (the
``submodules/diff-gaussian-rasterization`` of speedy-splat @ 34c45c6): Inria rasterizer with
SnugBox/AccuTile tile assignment (``auxiliary.h``), the dL_dG2 pruning-score buffer
(``backward.cu``) and a CUDA-event forward timer. ``csrc/stock`` = the same repository at
59f5f77 (= graphdeco-inria/diff-gaussian-rasterization used by 3D-GS), the unculled 3-sigma
square baseline. Both unmodified; Gaussian-Splatting licence (Inria/MPII, non-commercial
research use, ``csrc/*/LICENSE.md``). glm = upstream header library (copied from the SteepGS
port). The Python side reproduces ``diff_gaussian_rasterization/__init__.py`` and
``gaussian_renderer.render`` of speedy-splat @ 34c45c6.
"""

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py
RASTERIZERS = ("speedy", "stock")


@functools.cache
def extension(kind="speedy"):
    if kind not in RASTERIZERS:
        raise ValueError(f"rasterizer must be one of {RASTERIZERS}")
    from torch.utils.cpp_extension import load

    root = CSRC / kind
    glm = CSRC / "third_party" / "glm"
    sources = [root / "cuda_rasterizer" / f for f in
               ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [root / "rasterize_points.cu", root / "ext.cpp"]
    return load(name=f"ss_speedysplat_{kind}_rasterizer", sources=[str(s) for s in sources],
                extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                verbose=False)


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


def _forward_args(s, means3D, sh, opacities, scales, rotations):
    empty = torch.empty(0, device=means3D.device)
    return (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
            s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
            s["width"], sh, s["sh_degree"], s["campos"], False, False)


def _backward_args(s, ctx, grad_color, means3D, scales, rotations, radii, sh, geom, binning,
                   img):
    empty = torch.empty(0, device=means3D.device)
    return (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
            s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
            grad_color.contiguous(), sh, s["sh_degree"], s["campos"], geom,
            ctx.num_rendered, binning, img, False)


class _SpeedyRasterize(torch.autograd.Function):
    """speedy-splat diff_gaussian_rasterization._RasterizeGaussians (scores = 9th grad)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, scores, s, stats):
        args = _forward_args(s, means3D, sh, opacities, scales, rotations)
        (num_rendered, color, radii, kernel_times, geom, binning,
         img) = extension("speedy").rasterize_gaussians(*args)
        stats["num_rendered"] = int(num_rendered)
        stats["kernel_ms"] = float(kernel_times[0])
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        args = _backward_args(ctx.s, ctx, grad_color, means3D, scales, rotations, radii, sh,
                              geom, binning, img)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales, g_rot,
         g_G2) = extension("speedy").rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, g_G2, None, None


class _StockRasterize(torch.autograd.Function):
    """graphdeco diff_gaussian_rasterization @ 59f5f77 (3-sigma square tiles)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, scores, s, stats):
        args = _forward_args(s, means3D, sh, opacities, scales, rotations)
        num_rendered, color, radii, geom, binning, img = \
            extension("stock").rasterize_gaussians(*args)
        stats["num_rendered"] = int(num_rendered)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        args = _backward_args(ctx.s, ctx, grad_color, means3D, scales, rotations, radii, sh,
                              geom, binning, img)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension("stock").rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None, None, None


def render(params, camera, sh_degree, background, rasterizer="speedy", scores=None):
    """Official gaussian_renderer.render. Returns CHW colour, screen-space tensor, radii, stats.

    ``scores``: optional [N, 1] leaf; after backward its grad holds the Speedy-Splat pruning
    score sum_pixels (o * dL/dalpha)^2 of this view (speedy rasterizer only).
    """
    fn = {"speedy": _SpeedyRasterize, "stock": _StockRasterize}[rasterizer]
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad and means2D.requires_grad:
        means2D.retain_grad()
    if scores is None:
        scores = torch.zeros((len(means), 1), dtype=means.dtype, device=means.device)
    elif rasterizer != "speedy":
        raise ValueError("pruning scores need the speedy rasterizer")
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    s = raster_settings(camera, background, sh_degree)
    stats = {}
    color, radii = fn.apply(
        means.contiguous(), means2D, sh.contiguous(),
        params["opacities"].sigmoid()[:, None].contiguous(), params["scales"].exp().contiguous(),
        F.normalize(params["quats"], dim=-1).contiguous(), scores, s, stats)
    return color, means2D, radii, stats


def speedy_scores(params, cameras, sh_degree, background):
    """train.py prune(): scores = sum over cameras of the dL_dG2 buffer for L = image.sum()."""
    detached = {k: v.detach() for k, v in params.items()}
    n = len(detached["means"])
    total = torch.zeros((n, 1), dtype=torch.float32, device=detached["means"].device)
    with torch.enable_grad():
        for cam in cameras:
            img_scores = torch.zeros_like(total, requires_grad=True)
            image, _, _, _ = render(detached, cam, sh_degree, background, "speedy", img_scores)
            image.sum().backward()
            total += img_scores.grad
    return total[:, 0]
