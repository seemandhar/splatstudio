"""Official RaDe-GS rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-gaussian-rasterization`` of github.com/HKUST-SAIL/RaDe-GS at
commit 2d4bc08 (2024-06-21, the paper-v2 formulation), unmodified (Inria/MPII
Gaussian-Splatting licence, non-commercial research use, see ``csrc/VENDORED.md``). The Python
side reproduces ``gaussian_renderer.render`` and the train.py post-processing of the same commit.

Kernel outputs (``rasterize_points.cu``): color [3,H,W] (+ T·bg), depth [1,H,W] = Σ w_i d_i with
the rasterized per-pixel depth d_i = z_c + p·Δ (paper Eq. 3/14), middepth [1,H,W] = d of the last
Gaussian with T > 0.5, alpha [1,H,W] = Σ w_i, normal [3,H,W] = Σ w_i n_i (camera frame,
paper Eq. 21/22), distortion [1,H,W] = Σ_{j<i} w_i w_j (m_i - m_j)^2 on NDC-mapped depth
m = (f d - f n) / ((f - n) d), n = 0.2, f = 100, weights detached in the backward.
``means2D.grad`` is [N, 3]: x, y = 3DGS screen gradient, z = Σ_pixels |g_x| + |g_y| (GOF metric).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        glm = CSRC / "third_party" / "glm"
        sources = [CSRC / "cuda_rasterizer" / f for f in
                   ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
        sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
        _EXT = load(name="ss_radegs_raster", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


def check_centered(camera, tol=0.51):
    """The kernel (ndc2Pix of a symmetric-FoV projection) and depth_double_to_normal place the
    principal point at (W/2, H/2); refuse cameras that violate this."""
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > tol or abs(float(K[1, 2]) - H / 2) > tol:
        raise ValueError("RaDe-GS rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree):
    """Inria conventions (Camera.world_view_transform / full_proj_transform, FoV model)."""
    check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = K[0, 0], K[1, 1]
    tanfovx, tanfovy = float(W / (2 * fx)), float(H / (2 * fy))
    proj = torch.zeros(4, 4, device=K.device)  # getProjectionMatrix (symmetric frustum)
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
        "tanfovx": tanfovx, "tanfovy": tanfovy,
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as the official diff_gaussian_rasterization."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, settings):
        s = settings
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        (num_rendered, color, depth, middepth, alpha, normal, distortion, radii, geom, binning,
         img) = extension().rasterize_gaussians(*args)
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img, alpha)
        ctx.mark_non_differentiable(radii)
        return color, radii, depth, middepth, alpha, normal, distortion

    @staticmethod
    def backward(ctx, g_color, _g_radii, g_depth, g_middepth, g_alpha, g_normal, g_dist):
        s = ctx.settings
        means3D, scales, rotations, radii, sh, geom, binning, img, alpha = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)

        def c(t):
            return t.contiguous()

        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], c(g_color),
                c(g_depth), c(g_middepth), c(g_alpha), c(g_normal), c(g_dist), sh,
                s["sh_degree"], s["campos"], geom, ctx.num_rendered, binning, img, alpha,
                False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def rasterize(means, means2D, sh, opacities, scales, rotations, settings):
    """Raw kernel call. opacities [N,1] (already 3D-filtered), scales activated [N,3]."""
    return _Rasterize.apply(means.contiguous(), means2D, sh.contiguous(),
                            opacities.contiguous(), scales.contiguous(), rotations.contiguous(),
                            settings)


def depths_double_to_points(camera, depth1, depth2):
    """utils/graphics_utils.depths_double_to_points: camera-frame points, W/2,H/2, +0.5."""
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].float()
    fx, fy = float(K[0, 0]), float(K[1, 1])  # = W / (2 tan(FoVx / 2)) of the official Camera
    dev = depth1.device
    gx, gy = torch.meshgrid(torch.arange(W, device=dev).float() + 0.5,
                            torch.arange(H, device=dev).float() + 0.5, indexing="xy")
    rays = torch.stack([(gx - W / 2.0) / fx, (gy - H / 2.0) / fy, torch.ones_like(gx)], dim=-1)
    rays = rays.reshape(-1, 3)
    return depth1.reshape(-1, 1) * rays, depth2.reshape(-1, 1) * rays


def depth_double_to_normal(camera, depth1, depth2):
    """utils/graphics_utils.depth_double_to_normal; depths [H,W] -> normals [2,H,W,3]
    (camera frame, zero border)."""
    H, W = depth1.shape
    p1, p2 = depths_double_to_points(camera, depth1, depth2)
    points = torch.stack([p1, p2], dim=0).reshape(2, H, W, 3)
    out = torch.zeros_like(points)
    dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    out[:, 1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return out


def activated(params):
    """get_scaling_n_opacity_with_3D_filter, rotations, SH features."""
    from .filters import filtered_scales_opacities

    scales, opac = filtered_scales_opacities(params["scales"], params["opacities"],
                                             params["filter_3d"])
    rots = F.normalize(params["quats"], dim=-1)
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return scales, opac, rots, sh


def render(params, camera, sh_degree, background):
    """Official gaussian_renderer.render + train.py post-processing. Returns HWC tensors."""
    means = params["means"]
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    scales, opac, rots, sh = activated(params)
    settings = raster_settings(camera, background, sh_degree)
    color, radii, depth_acc, middepth, alpha, normal, distortion = rasterize(
        means, means2D, sh, opac[:, None], scales, rots, settings)
    # train.py: rendered_depth / rendered_mask, nan_to_num(0) (mask not detached)
    depth = torch.nan_to_num(depth_acc[0] / alpha[0], 0, 0)
    surface = depth_double_to_normal(camera, depth, middepth[0])
    normal_unit = F.normalize(normal, p=2, dim=0).permute(1, 2, 0)
    return {
        "rgb": color.permute(1, 2, 0),
        "depth": depth[..., None],
        "median_depth": middepth[0][..., None],
        "alpha": alpha[0][..., None],
        "normal": normal_unit,
        "surface_normal_expected": surface[0],
        "surface_normal_median": surface[1],
        "distortion": distortion[0][..., None],
        "info": {"means2d": means2D, "radii": radii, "depth_acc": depth_acc[0]},
    }
