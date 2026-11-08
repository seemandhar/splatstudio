"""Official BBSplat rasterizer (github.com/david-svitov/diff-bbsplat-rasterization @ 2dbc336, the
``submodules/diff-bbsplat-rasterization`` of github.com/david-svitov/BBSplat @ cff8d6a), vendored
unmodified in ``csrc/`` and JIT-built with torch.utils.cpp_extension (BSD-3 + Inria
Gaussian-Splatting licence, see csrc/README.md).

The Python side follows the official ``diff_bbsplat_rasterization/__init__.py`` and
``gaussian_renderer/__init__.py``; camera matrices follow the official ``scene/cameras.py``
(znear 0.01, zfar 100) as in reproductions/hdgs.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0
_EXT = {}


def extension():
    """JIT-build (cached in ~/.cache/torch_extensions) the official rasterizer."""
    if "ext" not in _EXT:
        from torch.utils.cpp_extension import load

        r = CSRC / "cuda_rasterizer"
        _EXT["ext"] = load(
            name="ss_bbsplat_raster",
            sources=[str(s) for s in (r / "rasterizer_impl.cu", r / "forward.cu",
                                      r / "backward.cu", CSRC / "rasterize_points.cu",
                                      CSRC / "ext.cpp")],
            extra_cuda_cflags=["-O3", f"-I{CSRC / 'third_party'}"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT["ext"]


def raster_settings(camera, background, sh_degree):
    """Official Camera matrices. The kernel's transform uses (W/2, H/2) as principal point (like
    the official code); the projection matrix carries cx, cy for culling/AABB consistency."""
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
        "bg": background.float().contiguous(),
        "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * fx)),
        "tanfovy": float(H / (2 * fy)),
        "width": W,
        "height": H,
        "sh_degree": int(sh_degree),
    }


class _Rasterize(torch.autograd.Function):
    """diff_bbsplat_rasterization._RasterizeGaussians (SH colours, no precomputed transforms)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, tex_alpha, tex_color, scales, rotations, s):
        ext = extension()
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, tex_alpha, tex_color, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num, color, others, radii, impact, geom, binning, img = ext.rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num
        ctx.save_for_backward(color, others, means3D, scales, rotations, tex_alpha, tex_color,
                              radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii, impact)
        return color, radii, impact, others

    @staticmethod
    def backward(ctx, g_color, _g_radii, _g_impact, g_others):
        s = ctx.s
        (color, others, means3D, scales, rotations, tex_alpha, tex_color, radii, sh, geom,
         binning, img) = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, color, others, empty, scales, rotations, tex_alpha,
                tex_color, 1.0, empty, s["viewmatrix"], s["projmatrix"], s["tanfovx"],
                s["tanfovy"], g_color.contiguous(), g_others.contiguous(), sh, s["sh_degree"],
                s["campos"], geom, ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_alpha, g_color_tex, g_means3D, _g_trans, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_alpha, g_color_tex, g_scales, g_rot, None


def rasterize(means, sh, tex_alpha, tex_color, scales, rots, settings):
    """Raw kernel call on activated tensors: colour [3,H,W], radii [N], impact [N] (sum of
    alpha*T per billboard, Eq. 7), allmap [8,H,W] (depth, alpha, normal(3), median depth,
    distortion, median weight)."""
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled())
    return _Rasterize.apply(means.contiguous(), means2D, sh.contiguous(),
                            tex_alpha.contiguous(), tex_color.contiguous(), scales.contiguous(),
                            rots.contiguous(), settings)


def depth_to_normal(settings, depth):
    """utils/point_utils.depth_to_normal of the official code (intrinsics fx = W/(2 tanfovx),
    principal point W/2, H/2, pixel corners). depth [1,H,W] -> normals [H,W,3] (zero border)."""
    W, H = settings["width"], settings["height"]
    dev = depth.device
    c2w = torch.linalg.inv(settings["viewmatrix"].T)
    fx = W / (2 * settings["tanfovx"])
    fy = H / (2 * settings["tanfovy"])
    intrins = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
                           dtype=torch.float32, device=dev)
    gx, gy = torch.meshgrid(torch.arange(W, device=dev).float(),
                            torch.arange(H, device=dev).float(), indexing="xy")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    rays_d = pts @ torch.linalg.inv(intrins).T @ c2w[:3, :3].T
    points = (depth.reshape(-1, 1) * rays_d + c2w[:3, 3]).reshape(H, W, 3)
    out = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    out[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return out
