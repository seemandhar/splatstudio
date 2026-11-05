"""Official StopThePop rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is github.com/r4dl/StopThePop-Rasterization at commit d8cad09 (the
``submodules/diff-gaussian-rasterization`` of github.com/r4dl/StopThePop @ 859b11b), copied
unmodified: Inria diff-gaussian-rasterization (Gaussian-Splatting licence, non-commercial
research use, ``csrc/LICENSE.md``) + StopThePop headers ``cuda_rasterizer/stopthepop/`` (MIT,
``csrc/cuda_rasterizer/stopthepop/LICENSE.txt``), nlohmann json (MIT), pybind11_json (BSD-3),
glm (MIT). The Python side reproduces ``diff_gaussian_rasterization/__init__.py`` and
``gaussian_renderer.render`` of the same commits (settings passed as the JSON dict the C++
``from_json`` in ``cuda_rasterizer/rasterizer.h`` parses).
"""

import copy
import functools
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py

SORT_MODES = {"GLOBAL": 0, "PPX_FULL": 1, "PPX_KBUFFER": 2, "HIER": 3}
SORT_ORDERS = {"Z_DEPTH": 0, "DISTANCE": 1, "PTD_CENTER": 2, "PTD_MAX": 3}

# configs/hierarchical.json of github.com/r4dl/StopThePop @ 859b11b (the paper's "Ours").
HIERARCHICAL = {
    "sort_settings": {"sort_mode": 3, "sort_order": 3,
                      "queue_sizes": {"tile_4x4": 64, "tile_2x2": 8, "per_pixel": 4}},
    "culling_settings": {"rect_bounding": True, "tight_opacity_bounding": True,
                         "tile_based_culling": False, "hierarchical_4x4_culling": True},
    "load_balancing": True,
    "proper_ewa_scaling": False,
}
# configs/vanilla.json (3DGS global view-space-z sort through the same kernels).
VANILLA = {
    "sort_settings": {"sort_mode": 0, "sort_order": 0,
                      "queue_sizes": {"tile_4x4": 64, "tile_2x2": 8, "per_pixel": 4}},
    "culling_settings": {"rect_bounding": False, "tight_opacity_bounding": False,
                         "tile_based_culling": False, "hierarchical_4x4_culling": False},
    "load_balancing": False,
    "proper_ewa_scaling": False,
}


def make_settings(base=None, sort_mode=None, sort_order=None, **flat):
    """Settings dict; ``flat`` keys follow ExtendedSettings.set_value (queue sizes / culling)."""
    s = copy.deepcopy(HIERARCHICAL if base is None else base)
    if sort_mode is not None:
        s["sort_settings"]["sort_mode"] = SORT_MODES[sort_mode] if isinstance(sort_mode, str) \
            else int(sort_mode)
    if sort_order is not None:
        s["sort_settings"]["sort_order"] = SORT_ORDERS[sort_order] \
            if isinstance(sort_order, str) else int(sort_order)
    for k, v in flat.items():
        if k in s["sort_settings"]["queue_sizes"]:
            s["sort_settings"]["queue_sizes"][k] = int(v)
        elif k in s["culling_settings"]:
            s["culling_settings"][k] = bool(v)
        elif k in ("load_balancing", "proper_ewa_scaling"):
            s[k] = bool(v)
        else:
            raise ValueError(f"Unknown StopThePop splatting setting {k!r}")
    return s


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    tp = CSRC / "third_party"
    sources = [CSRC / "cuda_rasterizer" / f for f in
               ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
    inc = [f"-I{tp}", f"-I{tp / 'glm'}"]
    return load(name="ss_stopthepop_rasterizer", sources=[str(s) for s in sources],
                extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc], verbose=False)


def _check_centered(camera):
    # The Inria kernels place the principal point at (W/2, H/2) (FoV camera model).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("StopThePop rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree, settings):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
    _check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = torch.zeros(4, 4, device=K.device)  # utils/graphics_utils.getProjectionMatrix
    proj[0, 0] = 2 * fx / W
    proj[0, 2] = 2 * cx / W - 1
    proj[1, 1] = 2 * fy / H
    proj[1, 2] = 2 * cy / H - 1
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    full = (world_view @ proj.T).contiguous()
    return {
        "bg": background, "viewmatrix": world_view, "projmatrix": full,
        "inv_viewprojmatrix": torch.linalg.inv(full).contiguous(),  # cameras.py:57
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * fx)), "tanfovy": float(H / (2 * fy)),
        "width": W, "height": H, "sh_degree": int(sh_degree), "settings": settings,
    }


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as StopThePop diff_gaussian_rasterization."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["inv_viewprojmatrix"], s["tanfovx"],
                s["tanfovy"], s["height"], s["width"], sh, s["sh_degree"], s["campos"], False,
                s["settings"], False, False)
        num_rendered, color, radii, geom, binning, img = extension().rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, opacities, scales, rotations, radii, sh, color, geom,
                              binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        s = ctx.s
        means3D, opacities, scales, rotations, radii, sh, color, geom, binning, img = \
            ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, opacities, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["inv_viewprojmatrix"], s["tanfovx"],
                s["tanfovy"], color, grad_color.contiguous(), sh, s["sh_degree"], s["campos"],
                geom, ctx.num_rendered, binning, img, s["settings"], False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def render(params, camera, sh_degree, background, settings):
    """Official gaussian_renderer.render. Returns CHW RGB, the screen-space tensor, radii."""
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad:
        means2D.retain_grad()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    s = raster_settings(camera, background, sh_degree, settings)
    color, radii = _Rasterize.apply(
        means.contiguous(), means2D, sh.contiguous(),
        params["opacities"].sigmoid()[:, None].contiguous(), params["scales"].exp().contiguous(),
        F.normalize(params["quats"], dim=-1).contiguous(), s)
    return color, means2D, radii


@torch.no_grad()
def forward_raw(params, camera, sh_degree, background, settings, render_depth=False):
    """Forward kernel only. Returns (num_rendered = sort entries, CHW image, radii). With
    ``render_depth`` the kernel's Depth debug output is returned instead of colour:
    channel 0 = sum_i depth_i alpha_i T_i (depth = per-ray t_opt for sorted modes, distance to
    the mean for GLOBAL), channel 1 = final T (stopthepop_common.cuh outputDebugVis)."""
    means = params["means"].float()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1).float().contiguous()
    s = raster_settings(camera, background, sh_degree, settings)
    empty = torch.empty(0, device=means.device)
    args = (s["bg"], means.contiguous(), empty,
            params["opacities"].float().sigmoid()[:, None].contiguous(), empty, empty, 1.0,
            empty, s["viewmatrix"], s["projmatrix"], s["inv_viewprojmatrix"], s["tanfovx"],
            s["tanfovy"], s["height"], s["width"], sh, s["sh_degree"], s["campos"], False,
            s["settings"], bool(render_depth), False)
    args = list(args)
    args[4] = params["scales"].float().exp().contiguous()
    args[5] = F.normalize(params["quats"].float(), dim=-1).contiguous()
    num_rendered, color, radii, *_ = extension().rasterize_gaussians(*args)
    return int(num_rendered), color, radii
