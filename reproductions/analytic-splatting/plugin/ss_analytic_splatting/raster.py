"""Official Analytic-Splatting rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-gaussian-rasterization`` of github.com/lzhnb/Analytic-Splatting
at commit a905939, **unmodified** (Inria 3DGS rasterizer whose pixel shading evaluates the
analytic pixel-area integral, paper Eq. 13, with the conditioned-logistic CDF `approxCdf` in
``cuda_rasterizer/auxiliary.h`` and its hand-written backward in ``backward.cu``). Licences:
``csrc/LICENSE.md`` (Inria Gaussian-Splatting licence, non-commercial research use) and
``csrc/LICENSE-Analytic-Splatting.md`` (repository MIT licence). glm is the header library
shipped in the same submodule (``third_party/glm``; test/doc trees dropped).

The Python side reproduces ``diff_gaussian_rasterization/__init__.py`` and
``gaussian_renderer.render`` of the same commit with ``filter3d=False`` (the paper's "Ours").
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
               ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
    return load(name="ss_analytic_splatting_rasterizer", sources=[str(s) for s in sources],
                extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                verbose=False)


def check_centered(camera):
    # The Inria kernels place the principal point at (W/2, H/2) (FoV camera model).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("Analytic-Splatting rasterizer assumes a centred principal point "
                         "(cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
    check_centered(camera)
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


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as the official diff_gaussian_rasterization."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num_rendered, color, radii, geom, binning, img = extension().rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        s = ctx.s
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                grad_color.contiguous(), sh, s["sh_degree"], s["campos"], geom,
                ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, None


def render(params, camera, sh_degree, background):
    """Official gaussian_renderer.render (filter3d=False).

    Returns CHW colour, the screen-space tensor means2D [N,3] (grad[:, :2] = Inria NDC-space
    screen gradient used by densification) and radii [N].
    """
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad:
        means2D.retain_grad()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    s = raster_settings(camera, background, sh_degree)
    color, radii = _Rasterize.apply(
        means.contiguous(), means2D, sh.contiguous(),
        params["opacities"].sigmoid()[:, None].contiguous(), params["scales"].exp().contiguous(),
        F.normalize(params["quats"], dim=-1).contiguous(), s)
    return color, means2D, radii
