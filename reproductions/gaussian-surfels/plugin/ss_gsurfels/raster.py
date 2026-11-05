"""Official Gaussian Surfels rasterizer (submodules/diff-gaussian-rasterization of
github.com/turandai/gaussian_surfels, commit 01d1249), vendored unmodified in ``csrc/`` and
JIT-built with torch.utils.cpp_extension (Inria Gaussian-Splatting licence, see csrc/README.md),
plus the Python side of the official ``gaussian_renderer.render`` and ``Camera``.

The kernel splats 3DGS-style EWA ellipses with the third scale forced to zero (config.surface),
computes the per-pixel tangent-plane depth (config.perpix_depth, paper Eq. 8), alpha-normalised
depth (config.normalize_depth, Eq. 7), camera-space normals flipped to face the camera
(config.dual_visible) and scales the rendered-normal gradient by 10 (backward.cu:727, §3.5).
Outputs are CHW like the official code: color [3,H,W], normal [3,H,W] (un-normalised sum of
w_i n_i), depth [1,H,W], opac [1,H,W] = 1 - T.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py
CONFIG = (1.0, 1.0, 1.0, 1.0)  # [surface, normalize_depth, perpix_depth, dual_visible] (arguments)
_EXT = {}


def _load(name, sources):
    if name not in _EXT:
        from torch.utils.cpp_extension import load

        glm = CSRC / "third_party" / "glm"
        _EXT[name] = load(name=name, sources=[str(s) for s in sources],
                          extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3"],
                          verbose=False)
    return _EXT[name]


def extension():
    r = CSRC / "cuda_rasterizer"
    return _load("ss_gsurfels_raster", [r / "rasterizer_impl.cu", r / "forward.cu",
                                        r / "backward.cu", CSRC / "rasterize_points.cu",
                                        CSRC / "ext.cpp"])


def utils_extension():
    """utils/cuda_utils.cu + utils/ext.cpp (gaussian2occgrid for the volumetric cutting)."""
    u = CSRC / "utils"
    return _load("ss_gsurfels_utils", [u / "ext.cpp", u / "cuda_utils.cu"])


class _Rasterize(torch.autograd.Function):
    """diff_gaussian_rasterization._RasterizeGaussians (same argument order and grad mapping)."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, settings):
        empty = torch.empty(0, device=means3D.device)
        s = settings
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["prcppoint"], s["patch_bbox"], s["tanfovx"],
                s["tanfovy"], s["height"], s["width"], sh, s["sh_degree"], s["campos"], False,
                False, s["config"])
        num, color, normal, depth, opac, radii, geom, binning, img = extension().rasterize_gaussians(
            *args)
        ctx.settings = settings
        ctx.num_rendered = num
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, normal, depth, opac, radii

    @staticmethod
    def backward(ctx, g_color, g_normal, g_depth, g_opac, _g_radii):
        s = ctx.settings
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty, s["viewmatrix"],
                s["projmatrix"], s["prcppoint"], s["patch_bbox"], s["tanfovx"], s["tanfovy"],
                g_color.contiguous(), g_normal.contiguous(), g_depth.contiguous(),
                g_opac.contiguous(), sh, s["sh_degree"], s["campos"], geom, ctx.num_rendered,
                binning, img, False, s["config"])
        (g_means2D, _gc, g_opacities, g_means3D, _gcov, g_sh, g_scales, g_rot, _gv, _gp,
         _gcam) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opacities, g_scales, g_rot, None


def raster_settings(camera, background, sh_degree):
    """Official Camera: getProjectionMatrix(znear, zfar, fov, W, H, prcppoint) and
    getWorld2View; P02 = 2cx/W - 1, P12 = 2cy/H - 1 (the kernel's ndc2Pix ignores prcppoint)."""
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dev = K.device
    proj = torch.zeros(4, 4, device=dev)
    proj[0, 0] = 2 * fx / W
    proj[0, 2] = 2 * cx / W - 1
    proj[1, 1] = 2 * fy / H
    proj[1, 2] = 2 * cy / H - 1
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    return {
        "bg": background.float().contiguous(), "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "prcppoint": torch.stack([cx / W, cy / H]).float().contiguous(),
        "patch_bbox": torch.tensor([0.0, 0.0, float(H), float(W)], device=dev),  # random_patch(inf)
        "tanfovx": float(W / (2 * fx)), "tanfovy": float(H / (2 * fy)),
        "width": W, "height": H, "sh_degree": int(sh_degree),
        "config": torch.tensor(CONFIG, dtype=torch.float32, device=dev),
    }


def scaled_camera(camera, factor):
    """loadCam at resolution_scale: (round(W/f), round(H/f)), same FoV and principal point."""
    if factor == 1:
        return camera
    W, H = int(camera["width"]), int(camera["height"])
    w, h = round(W / factor), round(H / factor)
    K = camera["K"].clone().float()
    K[0] *= w / W
    K[1] *= h / H
    return {**camera, "K": K, "width": w, "height": h}


def activated(params):
    """Official getters: exp scales (third column ~ exp(-1e10) = 0), normalised quats, sigmoid."""
    return (params["means"].contiguous(),
            torch.cat((params["sh0"], params["shN"]), dim=1).contiguous(),
            params["opacities"].sigmoid()[:, None].contiguous(),
            params["scales"].exp().contiguous(),
            F.normalize(params["quats"], dim=-1).contiguous())


def render(params, camera, sh_degree, background):
    means, sh, opac, scales, rots = activated(params)
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    settings = raster_settings(camera, background, sh_degree)
    color, normal, depth, opac_map, radii = _Rasterize.apply(means, means2D, sh, opac, scales,
                                                             rots, settings)
    return {"color": color, "normal": normal, "depth": depth, "opac": opac_map, "radii": radii,
            "means2d": means2D, "settings": settings}
