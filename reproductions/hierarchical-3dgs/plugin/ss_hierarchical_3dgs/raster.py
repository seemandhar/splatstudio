"""Python side of the vendored hierarchy rasterizer.

`_Rasterize` reproduces `diff_gaussian_rasterization/__init__.py::_RasterizeGaussians` of
github.com/graphdeco-inria/hierarchy-rasterizer @63fa247 (same argument order and gradient
mapping). Camera conversion: OpenCV world-to-camera `viewmat` + pinhole K -> the Inria
world_view_transform / full_proj_transform (scene/cameras.py, znear 0.01, zfar 100); the Inria
kernels assume a centred principal point, which is checked.

`render_leaves` = official `gaussian_renderer.render` (train_single.py): all Gaussians,
sigmoid opacity, image clamped to [0, 1].
`render_cut`    = official `gaussian_renderer.render_post` with `interp_python=True`
(train_post.py / render_hierarchy.py): attributes of the cut interpolated with their parents
in Python, alpha interpolation (Eq. alphap) inside the kernel through `interpolation_weights`
and `num_node_kids`; opacity activation abs (falloff); image clamped to [0, 1].
"""

import torch
import torch.nn.functional as F

from .ext import rasterizer

ZNEAR, ZFAR = 0.01, 100.0


def check_centered(camera):
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("The Inria hierarchy rasterizer assumes a centred principal point")


def raster_settings(camera, background, sh_degree, do_depth=False):
    check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = K[0, 0], K[1, 1]
    tanfovx, tanfovy = float(W / (2 * fx)), float(H / (2 * fy))
    # utils/graphics_utils.getProjectionMatrix (z_sign = 1) for a symmetric frustum
    proj = torch.zeros(4, 4, device=K.device)
    proj[0, 0] = 1.0 / tanfovx
    proj[1, 1] = 1.0 / tanfovy
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    return {
        "bg": background, "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": tanfovx, "tanfovy": tanfovy, "width": W, "height": H,
        "sh_degree": int(sh_degree), "do_depth": bool(do_depth),
    }


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, s, ts, kids):
        dev = means3D.device
        empty_i = torch.empty(0, dtype=torch.int32, device=dev)
        empty = torch.empty(0, device=dev)
        args = (s["bg"], empty_i, empty_i, ts, kids, means3D, empty, opacities, scales,
                rotations, 1.0, empty, s["viewmatrix"], s["projmatrix"], s["tanfovx"],
                s["tanfovy"], s["height"], s["width"], sh, s["sh_degree"], s["campos"], False,
                False, s["do_depth"])
        num_rendered, color, radii, geom, binning, img, invdepth = rasterizer().rasterize_gaussians(
            *args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, opacities, geom, binning,
                              img, ts, kids)
        ctx.mark_non_differentiable(radii)
        return color, radii, invdepth

    @staticmethod
    def backward(ctx, grad_color, _grad_radii, grad_depth):
        s = ctx.s
        means3D, scales, rotations, radii, sh, opacities, geom, binning, img, ts, kids = (
            ctx.saved_tensors)
        dev = means3D.device
        empty_i = torch.empty(0, dtype=torch.int32, device=dev)
        empty = torch.empty(0, device=dev)
        if grad_depth is None or grad_depth.numel() == 0:
            grad_depth = torch.zeros(1, s["height"], s["width"], device=dev)
        args = (s["bg"], empty_i, empty_i, ts, kids, means3D, radii, empty, opacities, scales,
                rotations, 1.0, empty, s["viewmatrix"], s["projmatrix"], s["tanfovx"],
                s["tanfovy"], grad_color.contiguous(), grad_depth.contiguous(), sh,
                s["sh_degree"], s["campos"], geom, ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = rasterizer().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None, None, None


def _screen(means):
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad:
        means2D.retain_grad()
    return means2D


def render_leaves(params, camera, sh_degree, background):
    """gaussian_renderer.render: returns CHW colour (clamped), means2D [N,3], radii [N]."""
    means = params["means"]
    means2D = _screen(means)
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    s = raster_settings(camera, background, sh_degree)
    dev = means.device
    color, radii, _ = _Rasterize.apply(
        means.contiguous(), means2D, sh.contiguous(),
        params["opacities"].sigmoid()[:, None].contiguous(), params["scales"].exp().contiguous(),
        F.normalize(params["quats"], dim=-1).contiguous(), s,
        torch.empty(0, device=dev), torch.empty(0, dtype=torch.int32, device=dev))
    return color.clamp(0, 1), means2D, radii


def interpolate_cut(params, render_inds, parent_inds, weights):
    """render_post(interp_python=True): per-attribute lerp of each cut node with its parent.

    Opacity uses the hierarchy activation `abs`; rotations are sign-aligned (dots < 0 -> -q)
    before the lerp and normalised by the rasterizer, as in the official code."""
    t = weights[:, None]
    ti = 1 - t
    scales = params["scales"].exp()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    rot = F.normalize(params["quats"], dim=-1)  # get_rotation
    opac = params["opacities"].abs()[:, None]
    r, p = render_inds, parent_inds
    means = t * params["means"][r] + ti * params["means"][p]
    sc = t * scales[r] + ti * scales[p]
    shs = t[:, :, None] * sh[r] + ti[:, :, None] * sh[p]
    parents = rot[p]
    rots = rot[r]
    dots = (rots * parents).sum(dim=-1)
    parents = torch.where((dots < 0)[:, None], -parents, parents)
    rots = t * rots + ti * parents
    op = t * opac[r] + ti * opac[p]
    return means, sc, shs, rots, op


def render_cut(params, camera, sh_degree, background, render_inds, parent_inds, weights, kids):
    """gaussian_renderer.render_post for the selected cut (no skybox)."""
    means, sc, shs, rots, op = interpolate_cut(params, render_inds, parent_inds, weights)
    means2D = _screen(means)
    s = raster_settings(camera, background, sh_degree)
    color, radii, _ = _Rasterize.apply(
        means.contiguous(), means2D, shs.contiguous(), op.contiguous(), sc.contiguous(),
        rots.contiguous(), s, weights.detach().contiguous(), kids.contiguous())
    return color.clamp(0, 1), means2D, radii
