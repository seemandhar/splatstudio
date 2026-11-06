"""Fisher-information (diagonal Hessian) rendering with the official FisherRF CUDA kernel.

``csrc/`` is ``diff/`` of github.com/JiangWenPL/FisherRF at commit b747328, unmodified: the
Inria diff-gaussian-rasterization (59f5f77) with a fused backward kernel
(``backward.cu:renderCUDAFused``, ``acc_func = atomicAddSquare``) that, for every
(pixel, Gaussian) pair, back-propagates dL/dpixel through the whole chain (2D mean, conic,
opacity, colour -> SH, cov3D -> scale/rotation, 3D mean) and accumulates the *square* of the
per-pixel gradient. Gaussian-Splatting licence (non-commercial research use), see
``csrc/LICENSE.md``; glm is the header library of the Inria submodule.

The Python side reproduces ``gaussian_renderer.modified_render`` + ``H_reg.py``: the kernel is
fed *activated* tensors (sigmoid opacity, exp scale, normalised quaternion) and
``pred_img.backward(gradient=ones)`` is run, so autograd then multiplies the squared sums by
the activation derivative once (opacity logit: sigma'(x); log-scale: exp(x)); means and SH pass
through unchanged. ``capture()[1:7]`` minus ``--filter_out_grad rotation`` = means, sh0, shN,
scales, opacities.
"""

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py (Inria)
# official capture() order xyz, f_dc, f_rest, scaling, rotation, opacity
FIELDS = ("means", "sh0", "shN", "scales", "quats", "opacities")


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    glm = CSRC / "third_party" / "glm"
    sources = [CSRC / "cuda_rasterizer" / f for f in
               ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
    return load(name="ss_fisherrf_rasterizer", sources=[str(s) for s in sources],
                extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                verbose=False)


def raster_settings(camera, background, sh_degree):
    """Inria FoV camera for an OpenCV pinhole camera dict (K, w2c viewmat, width, height)."""
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if abs(float(cx) - W / 2) > 0.51 or abs(float(cy) - H / 2) > 0.51:
        raise ValueError("The Inria/FisherRF kernel assumes a centred principal point")
    proj = torch.zeros(4, 4, device=K.device)
    proj[0, 0] = 2 * fx / W
    proj[1, 1] = 2 * fy / H
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    return {
        "bg": background.float().contiguous(), "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": float(W / (2 * fx)), "tanfovy": float(H / (2 * fy)),
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


class _FisherRasterize(torch.autograd.Function):
    """modified_diff_gaussian_rasterization._RasterizeGaussians (forward + squared backward)."""

    @staticmethod
    def forward(ctx, means3D, sh, opacities, scales, rotations, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, False)
        num_rendered, color, _depth, radii, geom, binning, img, _count = \
            extension().rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, geom, binning, img)
        return color

    @staticmethod
    def backward(ctx, grad_color):
        s = ctx.s
        means3D, scales, rotations, radii, sh, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                grad_color.contiguous(), sh, s["sh_degree"], s["campos"], geom,
                ctx.num_rendered, binning, img, False)
        (_g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales,
         g_rot) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_sh, g_opac, g_scales, g_rot, None


def render_image(params, camera, sh_degree, background):
    """Forward render (CHW) of the modified kernel; differentiable through the Fisher backward."""
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    s = raster_settings(camera, background, sh_degree)
    return _FisherRasterize.apply(
        params["means"].contiguous(), sh.contiguous(),
        params["opacities"].sigmoid()[:, None].contiguous(), params["scales"].exp().contiguous(),
        F.normalize(params["quats"], dim=-1).contiguous(), s)


def diag_hessian(params, camera, sh_degree, background, exclude=("quats",)):
    """Per-field diagonal Fisher information of one view on the raw parameters.

    Official H_reg.nbvs: ``pred_img.backward(gradient=ones_like(pred_img))`` through
    modified_render, then ``p.grad`` for every captured parameter not filtered out.
    """
    leaves = {k: params[k].detach().clone().requires_grad_(True) for k in FIELDS}
    with torch.enable_grad():
        image = render_image(leaves, camera, sh_degree, background)
        names = [k for k in FIELDS if k not in exclude]
        grads = torch.autograd.grad(image, [leaves[k] for k in names],
                                    grad_outputs=torch.ones_like(image), allow_unused=True)
    return {k: (torch.zeros_like(leaves[k]) if g is None else g.detach())
            for k, g in zip(names, grads)}


def flat(h):
    """Concatenate a per-field Hessian dict in the official capture() order."""
    return torch.cat([h[k].reshape(-1) for k in FIELDS if k in h])
