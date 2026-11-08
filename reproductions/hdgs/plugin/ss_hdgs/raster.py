"""Official HDGS rasterizers (github.com/TimSong412/HDGS), vendored in ``csrc/`` and JIT-built
with torch.utils.cpp_extension (Inria Gaussian-Splatting licence, see csrc/README.md):

* ``stage1``  — branch ``main`` @ b13a359, ``submodules/diff-surfel-rasterization``: 2DGS
  ray-splat intersection, per-ray k-buffer sort (SORT_WINDOW), per-tile radix sort keyed by the
  ray-splat depth at the tile centre, 5-sample frustum average of G, SH colour at the surfel
  centre direction.
* ``texture`` — branch ``texture`` @ 01c6fd4: same geometry/sort/frustum, SH DC replaced by a
  bilinear per-surfel texture lookup, SH evaluated at the exact intersection direction.

The only source change is a ``#ifndef SORT_WINDOW`` guard in ``auxiliary.h`` so that the buffer
size k (README: "set smaller ksort buffer size (e.g. 16) for large scenes in nerf360") is a build
flag. The Python side follows the official ``diff_surfel_rasterization/__init__.py`` and
``gaussian_renderer/__init__.py``; ``depth_to_normal`` is ``utils/point_utils.py``.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py
_EXT = {}


def extension(variant, sort_window):
    """JIT-build (cached in ~/.cache/torch_extensions) the ``variant`` rasterizer with k."""
    key = (variant, int(sort_window))
    if key not in _EXT:
        from torch.utils.cpp_extension import load

        root = CSRC / variant
        r = root / "cuda_rasterizer"
        glm = CSRC / "third_party"
        _EXT[key] = load(
            name=f"ss_hdgs_{variant}_k{int(sort_window)}",
            sources=[str(s) for s in (r / "rasterizer_impl.cu", r / "forward.cu",
                                      r / "backward.cu", root / "rasterize_points.cu",
                                      root / "ext.cpp")],
            extra_cuda_cflags=["-O3", f"-I{glm}", f"-DSORT_WINDOW={int(sort_window)}"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT[key]


def raster_settings(camera, background, sh_degree, sort_window):
    """Official Camera matrices (getWorld2View2 / getProjectionMatrix, znear 0.01, zfar 100).

    The principal point enters as P02 = 2cx/W - 1, P12 = 2cy/H - 1, so the kernel's ndc2pix
    ((ndc+1)W-1)/2 maps a camera point to fx x/z + cx - 0.5: the COLMAP pixel-centre convention
    (identical to the official code when cx = W/2, which it assumes)."""
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
        "sort_window": int(sort_window),
    }


class _Stage1(torch.autograd.Function):
    """diff_surfel_rasterization._RasterizeGaussians of branch main (SH, no texture)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, s):
        ext = extension("stage1", s["sort_window"])
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num, color, others, radii, geom, binning, img = ext.rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii, others

    @staticmethod
    def backward(ctx, g_color, _g_radii, g_others):
        s = ctx.s
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty, s["viewmatrix"],
                s["projmatrix"], s["tanfovx"], s["tanfovy"], g_color.contiguous(),
                g_others.contiguous(), sh, s["sh_degree"], s["campos"], geom, ctx.num_rendered,
                binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_trans, g_sh, g_scales,
         g_rot) = extension("stage1", s["sort_window"]).rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


class _Texture(torch.autograd.Function):
    """diff_surfel_rasterization._RasterizeGaussians of branch texture."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, texture, opacities, scales, rotations, index, s):
        ext = extension("texture", s["sort_window"])
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, texture, index, opacities, scales, rotations, 1.0,
                empty, s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num, color, others, radii, geom, binning, img = ext.rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img, index,
                              texture)
        ctx.mark_non_differentiable(radii)
        return color, radii, others

    @staticmethod
    def backward(ctx, g_color, _g_radii, g_others):
        s = ctx.s
        means3D, scales, rotations, radii, sh, geom, binning, img, index, texture = (
            ctx.saved_tensors)
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, texture, index, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                g_color.contiguous(), g_others.contiguous(), sh, s["sh_degree"], s["campos"],
                geom, ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_trans, g_sh, g_scales, g_rot,
         g_tex) = extension("texture", s["sort_window"]).rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_tex, g_opac, g_scales, g_rot, None, None


def activated(params):
    """Official getters: means, SH [N,16,3], sigmoid opacity [N,1], exp of the two surfel
    scales [N,2] (the runtime's third scale column is unused), normalised wxyz quaternions."""
    return (params["means"],
            torch.cat((params["sh0"], params["shN"]), dim=1).contiguous(),
            params["opacities"].sigmoid()[:, None].contiguous(),
            params["scales"][:, :2].exp().contiguous(),
            F.normalize(params["quats"], dim=-1).contiguous())


def rasterize(means, sh, opac, scales, rots, settings, texture=None, index=None):
    """Raw kernel call: returns colour [3,H,W], radii [N], allmap [7,H,W], means2D (grad
    holder, 2DGS densification statistic dL/dT[:,2] * depth * W/2 written by backward)."""
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    if texture is None:
        color, radii, allmap = _Stage1.apply(means.contiguous(), means2D, sh, opac, scales, rots,
                                             settings)
    else:
        color, radii, allmap = _Texture.apply(means.contiguous(), means2D, sh,
                                              texture.contiguous(), opac, scales, rots,
                                              index.contiguous(), settings)
    return color, radii, allmap, means2D


def depth_to_normal(settings, depth):
    """utils/point_utils.depths_to_points + depth_to_normal (official, incl. its ndc2pix
    without the -1/2 offset). depth [1,H,W] -> normals [H,W,3] (zero border)."""
    W, H = settings["width"], settings["height"]
    wvt, full = settings["viewmatrix"], settings["projmatrix"]
    dev = depth.device
    c2w = torch.linalg.inv(wvt.T)
    ndc2pix = torch.tensor([[W / 2, 0, 0, W / 2], [0, H / 2, 0, H / 2], [0, 0, 0, 1]],
                           dtype=torch.float32, device=dev).T
    projection_matrix = c2w.T @ full
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T
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


def render(params, camera, sh_degree, background, sort_window, depth_ratio=0.0, texture=None,
           index=None):
    """gaussian_renderer.render of the official code. CHW outputs like the official code."""
    means, sh, opac, scales, rots = activated(params)
    s = raster_settings(camera, background, sh_degree, sort_window)
    color, radii, allmap, means2D = rasterize(means, sh, opac, scales, rots, s, texture, index)
    alpha = allmap[1:2]
    rend_normal = (allmap[2:5].permute(1, 2, 0) @ (s["viewmatrix"][:3, :3].T)).permute(2, 0, 1)
    median = torch.nan_to_num(allmap[5:6], 0, 0)
    expected = torch.nan_to_num(allmap[0:1] / alpha, 0, 0)
    surf_depth = expected * (1 - depth_ratio) + depth_ratio * median
    surf_normal = depth_to_normal(s, surf_depth).permute(2, 0, 1) * alpha.detach()
    return {"color": color, "radii": radii, "means2d": means2D, "alpha": alpha,
            "rend_normal": rend_normal, "rend_dist": allmap[6:7], "surf_depth": surf_depth,
            "expected_depth": expected, "surf_normal": surf_normal, "settings": s}
