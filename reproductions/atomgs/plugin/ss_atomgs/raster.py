"""Official AtomGS feature rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-gaussian-rasterization`` of github.com/RongLiu-Leo/AtomicGS at
commit 5b9ce904 (itself the "diff-gauss" Inria variant with depth/alpha outputs), unmodified
except that only the ``glm/glm`` headers and licence of ``third_party/glm`` are kept
(Inria/MPII Gaussian-Splatting licence, non-commercial research use, ``csrc/LICENSE-AtomGS.md``).

The Python side reproduces the official ``gaussian_renderer.render`` and
``utils/image_utils.py`` (``depth_to_normal`` / ``unproject_depth_map``) of the same commit.

Kernel outputs: colour [3,H,W]; mean depth D = sum_i z_i w_i (NOT normalised by alpha);
median depth (z of the Gaussian where T crosses 0.5, no gradient); alpha = 1 - T.
Depth and alpha have full backward passes. ``means2D.grad`` [N,3]: x, y are the Inria
screen-space gradient used by the densification statistics.
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
        _EXT = load(name="ss_atomgs_raster", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


def check_centered(camera):
    # The official FoV camera ignores the principal point (assumes cx = W/2, cy = H/2).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("AtomGS rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def projection(camera):
    """Official getProjectionMatrix(znear, zfar, fovX, fovY) (row-major P, not transposed)."""
    K = camera["K"].float()
    W, H = int(camera["width"]), int(camera["height"])
    P = torch.zeros(4, 4, device=K.device)
    P[0, 0] = 2 * K[0, 0] / W  # = 1 / tan(fovx / 2)
    P[1, 1] = 2 * K[1, 1] / H
    P[3, 2] = 1.0
    P[2, 2] = ZFAR / (ZFAR - ZNEAR)
    P[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    return P


def raster_settings(camera, background, sh_degree):
    """Inria conventions for an OpenCV pinhole camera (K, world-to-camera viewmat)."""
    check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    P = projection(camera)
    world_view = viewmat.T.contiguous()          # Camera.world_view_transform
    return {
        "bg": background, "viewmatrix": world_view,
        "projection_matrix": P.T.contiguous(),   # Camera.projection_matrix
        "projmatrix": (world_view @ P.T).contiguous(),  # Camera.full_proj_transform
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * K[0, 0])), "tanfovy": float(H / (2 * K[1, 1])),
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as the official diff_gaussian_rasterization."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        (num_rendered, color, depth, median, alpha, radii, geom, binning,
         img) = extension().rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img, alpha)
        ctx.mark_non_differentiable(median, radii)
        return color, depth, median, alpha, radii

    @staticmethod
    def backward(ctx, g_color, g_depth, _g_median, g_alpha, _g_radii):
        s = ctx.s
        means3D, scales, rotations, radii, sh, geom, binning, img, alpha = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty, s["viewmatrix"],
                s["projmatrix"], s["tanfovx"], s["tanfovy"], g_color.contiguous(),
                g_depth.contiguous(), g_alpha.contiguous(), sh, s["sh_degree"], s["campos"],
                geom, ctx.num_rendered, binning, img, alpha, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def unproject_depth_map(depth, s):
    """Official unproject_depth_map: NDC grid with (W-1) normalisation -> world points."""
    H, W = depth.shape
    x = torch.linspace(0, W - 1, W, device=depth.device)
    y = torch.linspace(0, H - 1, H, device=depth.device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    d = depth.reshape(-1)
    xn = (X.reshape(-1) / (W - 1)) * 2 - 1
    yn = (Y.reshape(-1) / (H - 1)) * 2 - 1
    f1 = s["projection_matrix"][2, 2]
    f2 = s["projection_matrix"][3, 2]
    sdepth = (f1 * d + f2) / (d + 1e-8)
    pts = torch.stack([xn, yn, sdepth, torch.ones_like(d)], dim=-1).view(H, W, 4)
    pw = torch.matmul(pts, s["projmatrix"].inverse())
    return (pw[:, :, :3] / pw[:, :, 3:]).view(H, W, 3)


def depth_to_normal(depth, s):
    """Official depth_to_normal (world-frame normals [H,W,3]; zero-padded last row/col)."""
    H, W = depth.shape
    pts = torch.zeros((H + 1, W + 1, 3), device=depth.device)
    pts[:H, :W, :] = unproject_depth_map(depth, s)
    p1, p2, p3 = pts[:-1, :-1, :], pts[1:, :-1, :], pts[:-1, 1:, :]
    n = torch.cross(p2 - p1, p3 - p1, dim=-1)
    return n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)


def render(params, camera, sh_degree, background):
    """Official gaussian_renderer.render. Returns HWC tensors."""
    means = params["means"]
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    s = raster_settings(camera, background, sh_degree)
    sh = torch.cat((params["sh0"], params["shN"]), dim=1).contiguous()
    color, depth, median, alpha, radii = _Rasterize.apply(
        means.contiguous(), means2D, sh, params["opacities"].sigmoid()[:, None].contiguous(),
        params["scales"].exp().contiguous(), F.normalize(params["quats"], dim=-1).contiguous(),
        s)
    mean_depth = depth[0]
    a = alpha[0]
    return {
        "rgb": color.permute(1, 2, 0),
        "mean_depth": mean_depth[..., None],
        "depth": (mean_depth / a.clamp_min(1e-8))[..., None],
        "median_depth": median[0][..., None],
        "alpha": a[..., None],
        "surface_normal": depth_to_normal(mean_depth, s),
        "info": {"means2d": means2D, "radii": radii},
    }
