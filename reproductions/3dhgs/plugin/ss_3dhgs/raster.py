"""Official 3D-HGS half-Gaussian rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-gaussian-rasterization`` of github.com/lihaolin88/3DHGS at commit
dde1bd5, unmodified (Inria 3DGS rasterizer modified for half-Gaussians: two opacities and a
splitting-plane normal per primitive, Eq. 10 alpha with an erf factor, half-aware tile
rectangles; Gaussian-Splatting licence, non-commercial research use, see ``csrc/LICENSE.md``).
glm is the header library shipped in the same submodule. The Python side reproduces
``diff_gaussian_rasterization/__init__.py`` and ``gaussian_renderer.render`` of that commit.

Kernel conventions that matter for the probes (all from ``forward.cu``):
* opacities are passed as an ``[N, 2]`` tensor (alpha1, alpha2), the normal raw ``[N, 3]``;
* the normal is moved to camera space with ``transformPoint4x3`` (rotation *and* translation);
* pixels with Gaussian power < -4.8 are skipped in the forward pass only;
* ``cov3D_precomp_small`` is only tested for being non-empty; when it is (official test path,
  model loaded from .ply) the half-aware "small" tile rectangles are used.
"""

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py (Inria)


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    glm = CSRC / "third_party" / "glm"
    sources = [CSRC / "cuda_rasterizer" / f for f in
               ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
    return load(name="ss_3dhgs_rasterizer", sources=[str(s) for s in sources],
                extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                verbose=False)


def _check_centered(camera):
    # The Inria kernels place the principal point at (W/2, H/2) (FoV camera model).
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("3D-HGS rasterizer assumes a centred principal point (cx=W/2, cy=H/2)")


def raster_settings(camera, background, sh_degree):
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
        "bg": background, "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * fx)), "tanfovy": float(H / (2 * fy)),
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


class _Rasterize(torch.autograd.Function):
    """Same argument order / gradient mapping as 3DHGS diff_gaussian_rasterization."""

    @staticmethod
    def forward(ctx, means3D, means2D, normal, sh, opacities, scales, rotations, small, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, normal, empty, opacities, scales, rotations, 1.0, empty, small,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num_rendered, color, radii, geom, binning, img = extension().rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, normal, scales, rotations, radii, sh, geom, binning, img)
        ctx.mark_non_differentiable(radii)
        return color, radii

    @staticmethod
    def backward(ctx, grad_color, _grad_radii):
        s = ctx.s
        means3D, normal, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, normal, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                grad_color.contiguous(), sh, s["sh_degree"], s["campos"], geom,
                ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, g_normal, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_normal, g_sh, g_opac, g_scales, g_rot, None, None


def rasterize(means, normal, sh, opacities2, scales, quats, camera, sh_degree, background,
              test_time_cull=False):
    """Raw call with activated inputs: opacities2 [N,2] in (0,1), scales > 0, unit quats (wxyz).

    Returns CHW colour, means2D (retained grad when autograd is on) and radii.
    """
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    if grad and means2D.requires_grad:
        means2D.retain_grad()
    if test_time_cull:
        # Official test path: GaussianModel.load_ply fills `small_gaussian`; the kernel only
        # checks that the tensor is non-empty (forward.cu:424) and recomputes the splitting
        # ellipse itself (calculate_small), so the values are never read.
        small = torch.zeros(len(means), 6, device=means.device)
    else:
        small = torch.empty(0, device=means.device)
    s = raster_settings(camera, background, sh_degree)
    color, radii = _Rasterize.apply(
        means.contiguous(), means2D, normal.contiguous(), sh.contiguous(),
        opacities2.contiguous(), scales.contiguous(), quats.contiguous(), small, s)
    return color, means2D, radii


def activated_opacities(params):
    """[N, 2] = (sigmoid(opacities), sigmoid(opacity2)) — official `get_opacity` of `_opacity`."""
    return torch.stack([params["opacities"], params["opacity2"]], dim=-1).sigmoid()


def render(params, camera, sh_degree, background, test_time_cull=False):
    """Official gaussian_renderer.render (normal passed raw, as `pc.get_normal`)."""
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return rasterize(params["means"], params["hgs_normal"], sh, activated_opacities(params),
                     params["scales"].exp(), F.normalize(params["quats"], dim=-1), camera,
                     sh_degree, background, test_time_cull)
