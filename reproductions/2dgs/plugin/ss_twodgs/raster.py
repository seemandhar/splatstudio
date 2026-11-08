"""Official 2DGS surfel rasteriser, JIT-built from the vendored ``csrc/``.

``csrc/`` is github.com/hbb1/diff-surfel-rasterization at commit e0ed020 (the submodule of
github.com/hbb1/2d-gaussian-splatting @f3e3b9f), unmodified (Inria/MPII Gaussian-Splatting licence,
non-commercial research use, see ``csrc/LICENSE-diff-surfel-rasterization.md``; checksums in
``csrc/SHA256SUMS``). This module re-implements the thin Python binding
(``diff_surfel_rasterization/__init__.py``) and ``gaussian_renderer.render`` +
``utils/point_utils.depth_to_normal`` of the official repository. The Inria camera conversion
(``raster_settings``) follows reproductions/pgsr/plugin/ss_pgsr/raster.py.

Kernel conventions (forward.cu, auxiliary.h): integer pixel coordinates, principal point
((W-1)/2, (H-1)/2) inside ndc2pix, scales are 2-vectors, ray-splat intersection
rho = min(|u|^2, 2 |x - c|^2) (low-pass sigma = sqrt(2)/2), per-pixel near cut z < 0.2,
alpha = min(0.99, o G), stop at T < 1e-4, normals flipped towards the camera.
``out_others`` channels: 0 sum(w z), 1 alpha, 2:5 sum(w n) (camera frame), 5 median depth
(last z with T > 0.5), 6 distortion sum w_i w_j (m_i - m_j)^2 with m = f/(f-n) (1 - n/z),
n = 0.2, f = 100. ``means2D.grad`` = dL/dT[:, 2] * depth * W/2 (backward.cu:634, the paper's
"project the gradient of the 3D centre onto the screen").
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
        sources = [CSRC / "cuda_rasterizer" / f
                   for f in ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
        sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
        _EXT = load(name="ss_twodgs_surfel", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


def check_centered(camera):
    """The official kernel assumes a centred principal point (FoV camera model)."""
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("2DGS rasteriser assumes a centred principal point (cx = W/2, cy = H/2)")


def raster_settings(camera, background, sh_degree):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
    check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = K[0, 0], K[1, 1]
    tanx, tany = float(W / (2 * fx)), float(H / (2 * fy))
    proj = torch.zeros(4, 4, device=K.device)  # graphics_utils.getProjectionMatrix
    proj[0, 0] = 1.0 / tanx
    proj[1, 1] = 1.0 / tany
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()  # world_view_transform (row-vector convention)
    return {
        "bg": background, "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),  # full_proj_transform
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": tanx, "tanfovy": tany, "height": H, "width": W,
        "sh_degree": int(sh_degree),
    }


class _Rasterize(torch.autograd.Function):
    """diff_surfel_rasterization._RasterizeGaussians (SH colours, scale/rotation path)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num_rendered, color, others, radii, geom, binning, img = extension().rasterize_gaussians(
            *args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii, others

    @staticmethod
    def backward(ctx, g_color, _g_radii, g_others):
        s = ctx.s
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                g_color.contiguous(), g_others.contiguous(), sh, s["sh_degree"], s["campos"],
                geom, ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_trans, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def activated(params):
    """GaussianModel getters: exp(2 scales), sigmoid opacity, normalised wxyz quaternion."""
    scales = params["scales"][:, :2].exp()  # the third stored column is unused (2D surfels)
    opac = params["opacities"].sigmoid()
    rots = F.normalize(params["quats"], dim=-1)
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return scales, opac, rots, sh


def depth_to_normal(camera, depth):
    """utils/point_utils.depths_to_points + depth_to_normal. depth [H,W] -> world normals [H,W,3].

    Official intrinsics come from full_proj_transform with ndc2pix (W/2, H/2): fx, fy and
    principal point (W/2, H/2) on the integer pixel grid; border pixels are zero."""
    K = camera["K"].float()
    viewmat = camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    c2w = torch.linalg.inv(viewmat)
    dev = depth.device
    gx, gy = torch.meshgrid(torch.arange(W, device=dev).float(),
                            torch.arange(H, device=dev).float(), indexing="xy")
    dirs = torch.stack([(gx - W / 2) / K[0, 0], (gy - H / 2) / K[1, 1], torch.ones_like(gx)], -1)
    rays_d = dirs @ c2w[:3, :3].T
    points = depth[..., None] * rays_d + c2w[:3, 3]
    out = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    out[1:-1, 1:-1] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return out


def render(params, camera, sh_degree, background, depth_ratio=0.0):
    """gaussian_renderer.render. Returns HWC maps (world-frame normals)."""
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad:
        means2D.retain_grad()
    scales, opac, rots, sh = activated(params)
    settings = raster_settings(camera, background, sh_degree)
    color, radii, others = _Rasterize.apply(
        means.contiguous(), means2D, sh.contiguous(), opac[:, None].contiguous(),
        scales.contiguous(), rots.contiguous(), settings)
    viewmat = camera["viewmat"].float()
    alpha = others[1:2]
    # camera -> world: render_normal @ world_view_transform[:3,:3].T == R_w2c^T n
    normal = others[2:5].permute(1, 2, 0) @ viewmat[:3, :3]
    median = torch.nan_to_num(others[5:6], 0, 0)
    expected = torch.nan_to_num(others[0:1] / alpha, 0, 0)
    surf_depth = expected * (1 - depth_ratio) + depth_ratio * median
    surf_normal = depth_to_normal(camera, surf_depth[0]) * alpha.detach().permute(1, 2, 0)
    return {
        "rgb": color.permute(1, 2, 0),
        "depth": surf_depth.permute(1, 2, 0),
        "expected_depth": expected.permute(1, 2, 0),
        "median_depth": median.permute(1, 2, 0),
        "alpha": alpha.permute(1, 2, 0),
        "normal": normal,
        "surface_normal": surf_normal,
        "distortion": others[6:7].permute(1, 2, 0),
        "info": {"means2d": means2D, "radii": radii},
    }
