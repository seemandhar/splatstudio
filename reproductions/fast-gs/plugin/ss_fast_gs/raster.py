"""Official FastGS CUDA rasterizer, vendored unmodified in ``csrc/`` and JIT-built.

Source: github.com/fastgs/FastGS @ 44e02a5 ``submodules/diff-gaussian-rasterization_fastgs``
(Inria Gaussian-Splatting licence, non-commercial research use, ``csrc/LICENSE.md``): the
Inria rasterizer with Taming-3DGS per-splat backward, Speedy-Splat SnugBox/AccuTile tile
assignment scaled by the Compact Box factor ``mult`` (beta, supp. Eq. 15), a 4-column
screen-space gradient (``[:, :2]`` signed, ``[:, 2:]`` AbsGS absolute sums) and the per-Gaussian
high-error pixel counter (``metric_map``/``get_flag``). glm = header library shipped with it.

The Python side reproduces ``diff_gaussian_rasterization_fastgs/__init__.py`` and
``gaussian_renderer.render_fastgs``. Camera conversion (OpenCV K/viewmat -> Inria matrices)
follows the SplatStudio Speedy-Splat port (reproductions/speedy-splat/plugin/ss_speedysplat).
"""

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    glm = CSRC / "third_party" / "glm"
    sources = [CSRC / "cuda_rasterizer" / f for f in
               ("rasterizer_impl.cu", "forward.cu", "backward.cu", "adam.cu")]
    sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
    return load(name="ss_fast_gs_rasterizer", sources=[str(s) for s in sources],
                extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                verbose=False)


def _check_centered(camera):
    # The Inria kernels place the principal point at (W/2, H/2) (FoV camera model).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("Inria rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree, mult):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
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
        "bg": background.float().contiguous(), "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * fx)), "tanfovy": float(H / (2 * fy)),
        "width": W, "height": H, "sh_degree": int(sh_degree), "mult": float(mult),
    }


class _Rasterize(torch.autograd.Function):
    """diff_gaussian_rasterization_fastgs._RasterizeGaussians (SH path, no precomputed cov)."""

    @staticmethod
    def forward(ctx, means3D, means2D, dc, sh, opacities, scales, rotations, s, metric_map,
                get_flag, stats):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty, metric_map,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], dc, sh, s["sh_degree"], s["campos"], s["mult"], False, False,
                bool(get_flag))
        (num_rendered, num_buckets, color, radii, geom, binning, img, sample,
         counts) = extension().rasterize_gaussians(*args)
        stats["num_rendered"] = int(num_rendered)
        stats["counts"] = counts
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(means3D, scales, rotations, radii, dc, sh, geom, binning, img,
                              sample)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        means3D, scales, rotations, radii, dc, sh, geom, binning, img, sample = ctx.saved_tensors
        s = ctx.s
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty, s["viewmatrix"],
                s["projmatrix"], s["tanfovx"], s["tanfovy"], grad_color.contiguous(), dc, sh,
                s["sh_degree"], s["campos"], geom, ctx.num_rendered, binning, img,
                ctx.num_buckets, sample, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_dc, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return (g_means3D, g_means2D, g_dc, g_sh, g_opac, g_scales, g_rot, None, None, None,
                None)


def render(params, camera, sh_degree, background, mult, metric_map=None):
    """Official ``render_fastgs``. Returns (CHW colour, means2D [N,4] with retained grad,
    radii [N] int, stats) where stats has ``num_rendered`` (tile-Gaussian pairs) and
    ``counts`` ([N] int32 high-error pixel counts when ``metric_map`` is given)."""
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros((len(means), 4), dtype=means.dtype, device=means.device,
                          requires_grad=grad) + 0
    if grad:
        means2D.retain_grad()
    s = raster_settings(camera, background, sh_degree, mult)
    W, H = s["width"], s["height"]
    get_flag = metric_map is not None
    if metric_map is None:
        metric_map = torch.zeros(H * W, dtype=torch.int32, device=means.device)
    else:
        if metric_map.numel() != H * W:
            raise ValueError("metric_map must have H*W entries")
        metric_map = metric_map.reshape(-1).to(torch.int32).contiguous()
    stats = {}
    color, radii = _Rasterize.apply(
        means.contiguous(), means2D, params["sh0"].contiguous(), params["shN"].contiguous(),
        params["opacities"].sigmoid()[:, None].contiguous(),
        params["scales"].exp().contiguous(), F.normalize(params["quats"], dim=-1).contiguous(),
        s, metric_map, get_flag, stats)
    return color, means2D, radii, stats
