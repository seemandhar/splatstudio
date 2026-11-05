"""Official GOF ray-tracing rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-gaussian-rasterization`` of
github.com/autonomousvision/gaussian-opacity-fields at commit 5245b20, unmodified
(Inria/MPII Gaussian-Splatting licence, non-commercial research use, see
``csrc/LICENSE-GOF.md``). The Python side reproduces ``gaussian_renderer.render`` /
``integrate`` and ``utils/depth_utils.py`` of the same commit.

Kernel output channels (rasterizer ``OUTPUT_CHANNELS`` = 9):
0-2 RGB, 3-5 blended intersection-plane normals (camera frame), 6 median ray depth
(camera z of the last Gaussian with T > 0.5), 7 accumulated alpha, 8 distortion
(2DGS form on NDC depth, normalised by (1-T)^2, weight gradients detached).
``means2D.grad`` is [N, 3]: x, y = 3DGS signed screen gradient, z = sum over pixels of
|g_x| + |g_y| (the GOF densification metric).
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
        _EXT = load(name="ss_gof_raytrace", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


_OFFSETS = {}


def _subpixel(height, width, device):
    """Official render() passes a zero subpixel offset (unused by the kernels)."""
    key = (height, width, device)
    if key not in _OFFSETS:
        _OFFSETS[key] = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
    return _OFFSETS[key]


def _check_centered(camera):
    # The official kernels place the principal point at (W/2, H/2) (FoV camera model).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("GOF rasterizer assumes a centred principal point (cx = W/2, cy = H/2)")


def raster_settings(camera, background, sh_degree, kernel_size=0.0):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat), as in FatesGS."""
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
        "kernel_size": float(kernel_size),
        "subpixel": _subpixel(H, W, K.device),
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


def _forward_args(s, means3D, sh, opacities, scales, rotations):
    empty = torch.empty(0, device=means3D.device)
    return (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty, empty,
            s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["kernel_size"],
            s["subpixel"], s["height"], s["width"], sh, s["sh_degree"], s["campos"], False,
            False)


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as diff_gaussian_rasterization (GOF)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, settings):
        args = _forward_args(settings, means3D, sh, opacities, scales, rotations)
        num_rendered, color, radii, geom, binning, img = extension().rasterize_gaussians(*args)
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        s = ctx.settings
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                s["kernel_size"], s["subpixel"], grad_color.contiguous(), sh,
                s["sh_degree"], s["campos"], geom, ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales, g_rot,
         _g_v2g) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def depths_to_points(camera, depth):
    """Official utils/depth_utils.depths_to_points: pixel centres (+0.5), z-depth rays."""
    viewmat = camera["viewmat"].float()
    c2w = torch.linalg.inv(viewmat)
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].float().clone()
    K[0, 2], K[1, 2] = W / 2.0, H / 2.0  # official intrinsics use W/2, H/2
    gx, gy = torch.meshgrid(torch.arange(W, device=depth.device).float() + 0.5,
                            torch.arange(H, device=depth.device).float() + 0.5, indexing="xy")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    rays_d = pts @ torch.linalg.inv(K).T @ c2w[:3, :3].T
    return depth.reshape(-1, 1) * rays_d + c2w[:3, 3]


def depth_to_normal(camera, depth):
    """Official depth_to_normal; depth [H,W] -> world normals [H,W,3] (border = 0)."""
    H, W = depth.shape
    points = depths_to_points(camera, depth).reshape(H, W, 3)
    out = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    out[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return out


def activated(params):
    """Mip-Splatting 3D filter applied as in get_scaling/opacity_with_3D_filter."""
    from .filters import filtered_scales_opacities

    scales, opac = filtered_scales_opacities(params["scales"], params["opacities"],
                                             params["filter_3d"])
    rots = F.normalize(params["quats"], dim=-1)
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return scales, opac, rots, sh


def render(params, camera, sh_degree, background, kernel_size=0.0):
    """Official gaussian_renderer.render + train.py post-processing. Returns HWC tensors."""
    means = params["means"]
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    scales, opac, rots, sh = activated(params)
    settings = raster_settings(camera, background, sh_degree, kernel_size)
    out, radii = _Rasterize.apply(means.contiguous(), means2D, sh.contiguous(),
                                  opac[:, None].contiguous(), scales.contiguous(),
                                  rots.contiguous(), settings)
    normal_cam = F.normalize(out[3:6], p=2, dim=0)
    c2w_rot = torch.linalg.inv(camera["viewmat"].float())[:3, :3]
    normal_world = normal_cam.permute(1, 2, 0) @ c2w_rot.T
    depth = out[6]
    return {
        "rgb": out[0:3].permute(1, 2, 0),
        "normal": normal_world,
        "surface_normal": depth_to_normal(camera, depth),
        "depth": depth[..., None],
        "alpha": out[7][..., None],
        "distortion": out[8][..., None],
        "info": {"means2d": means2D, "radii": radii},
    }


@torch.no_grad()
def integrate(params, points, camera, sh_degree=0, background=None, kernel_size=0.0):
    """Official ``integrate``: opacity of arbitrary 3D points seen from one camera.

    Returns ``alpha_integrated`` [M]: sum_k alpha_k G_k(min(t*_k, t_point)) T_k along the ray
    through each point's projection; points that project outside the image keep the kernel's
    initial value (official extract_mesh initialises the min over views with 1).
    """
    dev = params["means"].device
    if background is None:
        background = torch.zeros(3, device=dev)
    scales, opac, rots, sh = activated(params)
    s = raster_settings(camera, background, sh_degree, kernel_size)
    empty = torch.empty(0, device=dev)
    args = (s["bg"], points.contiguous().float(), params["means"].contiguous(), empty,
            opac[:, None].contiguous(), scales.contiguous(), rots.contiguous(), 1.0, empty,
            empty, s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
            s["kernel_size"], s["subpixel"], s["height"], s["width"], sh.contiguous(),
            s["sh_degree"], s["campos"], False, False)
    _, _, alpha_integrated, _, _, _, _, _ = extension().integrate_gaussians_to_points(*args)
    return alpha_integrated


@torch.no_grad()
def opacity_field(params, points, cameras, kernel_size=0.0):
    """GOF O(x) = min over training views (official extract_mesh.evaluage_alpha, before its
    ``1 - final_alpha`` sign flip)."""
    out = torch.ones(len(points), device=points.device)
    for cam in cameras:
        out = torch.minimum(out, integrate(params, points, cam, kernel_size=kernel_size))
    return out
