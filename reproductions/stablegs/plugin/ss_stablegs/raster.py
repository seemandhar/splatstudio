"""StableGS's RaDe-GS rasterizer, JIT-built from the vendored ``csrc/`` (see csrc/VENDORED.md).

The official StableGS release ships the CUDA code but not its Python side
(``gaussian_renderer.render_rade`` and the ``diff_gaussian_rasterization_radegs`` package are
missing, github.com/MooreThreads/StableGS issue #2). The kernel is the RaDe-GS rasterizer at
HKUST-SAIL/RaDe-GS commit ec9199f, so the autograd wrapper below follows that commit's
``diff_gaussian_rasterization/__init__.py`` and the render function its
``gaussian_renderer/__init__.py:render`` (Inria FoV camera model, require_coord False), with the
opacity supplied by the caller (StableGS dual opacity, see renderer.py).

Kernel outputs: color [3,H,W] (+ T·bg), expected depth [1,H,W] = (Σ w_i d_i)/(Σ w_i) of the
rasterized per-pixel depths (0 where no contributor), median depth [1,H,W] (depth where T drops
below 0.5), alpha [1,H,W] = Σ w_i, normal [3,H,W] (camera frame, unit length where covered),
radii [N]. Alpha is clamped to 0.99 in the forward pass and gradients always flow (Inria
semantics). ``means2D.grad`` is [N,3]: x, y = 3DGS NDC screen gradient, z = Σ |g_x| + |g_y|
(GOF absolute gradient).
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
                   ("rasterizer_impl.cu", "forward.cu", "backward.cu", "utils.cu")]
        sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
        _EXT = load(name="ss_stablegs_raster", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


def check_centered(camera, tol=0.51):
    """The kernel (symmetric-FoV projection, W/2 pixel offset) and the depth-normal / warp code
    place the principal point at (W/2, H/2); refuse cameras that violate this."""
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > tol or abs(float(K[1, 2]) - H / 2) > tol:
        raise ValueError("StableGS rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree, kernel_size, require_depth):
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
        "tanfovx": tanfovx, "tanfovy": tanfovy, "kernel_size": float(kernel_size),
        "width": W, "height": H, "sh_degree": int(sh_degree),
        "require_depth": bool(require_depth),
    }


class _Rasterize(torch.autograd.Function):
    """RaDe-GS@ec9199f _RasterizeGaussians (colors_precomp / cov3D_precomp empty,
    scale_modifier 1, prefiltered False, require_coord False)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, settings):
        s = settings
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["kernel_size"],
                s["height"], s["width"], sh, s["sh_degree"], s["campos"], False, False,
                s["require_depth"], False)
        (num_rendered, color, _coord, _mcoord, alpha, normal, depth, mdepth, radii, geom,
         binning, img) = extension().rasterize_gaussians(*args)
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, normal, radii, sh, geom, binning, img,
                              alpha)
        ctx.mark_non_differentiable(radii)
        return color, radii, depth, mdepth, alpha, normal

    @staticmethod
    def backward(ctx, g_color, _g_radii, g_depth, g_mdepth, g_alpha, g_normal):
        s = ctx.settings
        means3D, scales, rotations, normal, radii, sh, geom, binning, img, alpha = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        H, W = s["height"], s["width"]
        zeros3 = torch.zeros(3, H, W, device=means3D.device)

        def c(t):
            return t.contiguous()

        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["kernel_size"],
                c(g_color), zeros3, zeros3, c(g_depth), c(g_mdepth), c(g_alpha), c(g_normal),
                normal, sh, s["sh_degree"], s["campos"], geom, ctx.num_rendered, binning, img,
                alpha, False, s["require_depth"], False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def rasterize(means, opacities, scales, rotations, sh, settings):
    """One render. opacities [N,1] activated; scales activated [N,3]; rotations normalized.
    Returns (dict of CHW maps, means2D) — means2D retains its [N,3] gradient."""
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    color, radii, depth, mdepth, alpha, normal = _Rasterize.apply(
        means.contiguous(), means2D, sh.contiguous(), opacities.contiguous(),
        scales.contiguous(), rotations.contiguous(), settings)
    return {"color": color, "radii": radii, "depth": depth, "median_depth": mdepth,
            "alpha": alpha, "normal": normal}, means2D


def depths_double_to_points(camera, depth1, depth2):
    """train_pair.py:depths_double_to_points: camera-frame points on the (+0.5) pixel grid with
    the principal point at (W/2, H/2). depths [H,W] -> [H*W,3] each."""
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].float()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    dev = depth1.device
    gx, gy = torch.meshgrid(torch.arange(W, device=dev).float() + 0.5,
                            torch.arange(H, device=dev).float() + 0.5, indexing="xy")
    rays = torch.stack([(gx - W / 2.0) / fx, (gy - H / 2.0) / fy, torch.ones_like(gx)], dim=-1)
    rays = rays.reshape(-1, 3)
    return depth1.reshape(-1, 1) * rays, depth2.reshape(-1, 1) * rays


def depth_double_to_normal(camera, depth1, depth2):
    """train_pair.py:depth_double_to_normal; depths [H,W] -> normals [2,H,W,3] (camera frame,
    zero border)."""
    H, W = depth1.shape
    p1, p2 = depths_double_to_points(camera, depth1, depth2)
    points = torch.stack([p1, p2], dim=0).reshape(2, H, W, 3)
    out = torch.zeros_like(points)
    dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    out[:, 1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return out
