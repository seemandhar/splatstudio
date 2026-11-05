"""Official 2DGS rasterizer (diff-surfel-rasterization, as shipped in the FatesGS repo, commit
a721d318) JIT-built from the vendored ``csrc/`` (Inria/MPII Gaussian-Splatting licence,
non-commercial research use, see ``csrc/LICENSE-2DGS.md``), plus the Python side of the
official ``gaussian_renderer.render`` (surf_depth, surf_normal, world normals).

Using the original kernel keeps the exact ray-splat intersection depth, 2DGS distortion term,
normals and gradients (incl. the opacity/geometry gradients that gsplat's rasterizer drops
where opac*G > 0.999).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official Camera defaults
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        glm = CSRC / "third_party" / "glm"
        sources = [CSRC / "cuda_rasterizer" / f for f in
                   ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
        sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
        _EXT = load(name="ss_fatesgs_surfel", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as diff_surfel_rasterization._RasterizeGaussians."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, settings):
        ext = extension()
        empty = torch.empty(0, device=means3D.device)
        args = (settings["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                settings["viewmatrix"], settings["projmatrix"], settings["tanfovx"],
                settings["tanfovy"], settings["height"], settings["width"], sh,
                settings["sh_degree"], settings["campos"], False, False)
        num_rendered, color, others, radii, geom, binning, img = ext.rasterize_gaussians(*args)
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii, others

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_others):
        s = ctx.settings
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty, s["viewmatrix"],
                s["projmatrix"], s["tanfovx"], s["tanfovy"], grad_color.contiguous(),
                grad_others.contiguous(), sh, s["sh_degree"], s["campos"], geom,
                ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_trans, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def raster_settings(camera, background, sh_degree):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat).

    P00 = 2 fx / W, P02 = 2 cx / W - 1 (same for y); for the official cx = W/2 this is exactly
    getProjectionMatrix(znear=0.01, zfar=100, FoV) of the FatesGS Camera.
    """
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


def depths_to_points(camera, depth):
    """Official utils/point_utils.depths_to_points: integer pixel grid, z-depth rays. [H*W,3]"""
    viewmat = camera["viewmat"].float()
    c2w = torch.linalg.inv(viewmat)
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].float()
    gx, gy = torch.meshgrid(torch.arange(W, device=depth.device).float(),
                            torch.arange(H, device=depth.device).float(), indexing="xy")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    rays_d = pts @ torch.linalg.inv(K).T @ c2w[:3, :3].T
    return depth.reshape(-1, 1) * rays_d + c2w[:3, 3]


def depth_to_normal(camera, depth):
    """Official utils/point_utils.depth_to_normal; depth [H,W] -> world normals [H,W,3]."""
    H, W = depth.shape
    points = depths_to_points(camera, depth).reshape(H, W, 3)
    out = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    out[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return out


def render(params, camera, sh_degree, background):
    """Official gaussian_renderer.render (depth_ratio = 0). Returns HWC tensors."""
    means = params["means"]
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    opac = params["opacities"].sigmoid()[:, None]
    scales = params["scales"][:, :2].exp()  # 2D surfels: the third stored scale is unused
    rots = F.normalize(params["quats"], dim=-1)
    settings = raster_settings(camera, background, sh_degree)
    color, radii, allmap = _Rasterize.apply(means.contiguous(), means2D, sh.contiguous(),
                                            opac.contiguous(), scales.contiguous(),
                                            rots.contiguous(), settings)
    alpha = allmap[1:2]
    normal_cam = allmap[2:5]
    normal = (normal_cam.permute(1, 2, 0) @ settings["viewmatrix"][:3, :3].T)  # world, HWC
    depth_exp = torch.nan_to_num(allmap[0:1] / alpha, 0, 0)
    dist = allmap[6:7]
    surf_depth = depth_exp  # depth_ratio = 0
    surf_normal = depth_to_normal(camera, surf_depth[0]) * alpha.detach().permute(1, 2, 0)
    return {
        "rgb": color.permute(1, 2, 0),
        "depth": surf_depth.permute(1, 2, 0),
        "alpha": alpha.permute(1, 2, 0),
        "normal": normal,
        "surface_normal": surf_normal,
        "distortion": dist.permute(1, 2, 0),
        "median_depth": torch.nan_to_num(allmap[5:6], 0, 0).permute(1, 2, 0),
        "info": {"means2d": means2D, "radii": radii},
    }
