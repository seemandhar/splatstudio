"""Official DOF-GS rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-gaussian-rasterization-dof`` of github.com/rongduo/
DOFGS_implementation at commit cfd635c, unmodified (Inria/MPII Gaussian-Splatting licence,
non-commercial research use, see ``csrc/LICENSE-DOFGS.md``). The Python side reproduces
``diff_dof_gs_rasterizer/__init__.py`` and ``src/model_dofgs/renderer/__init__.py`` of the same
commit.

Per Gaussian the kernel adds ``a = R^2 / (2 ln 4)``, ``R = 0.5 K[v] |1/F[v] - 1/z_view|``, to the
diagonal of the projected covariance (which already contains the Mip 2D kernel), scales the
opacity by the Mip 2D-filter factor sqrt(det(S'+aI)/det(S'+aI+kI)) and outputs
``color [3,H,W]``, ``depth [1,H,W]`` (alpha-blended ray distance |p_view|), ``coc [1,H,W]``
(alpha-blended R) and radii. ``F``/``K`` are [V,1] tensors indexed by ``vIdx``; their gradients are
returned by the backward kernel. The CoC -> z gradient is disabled (coc2z_coeff = 0, as official).
"""

from pathlib import Path

import torch

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
        _EXT = load(name="ss_dofgs_dof_raster", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


_OFFSETS = {}


def _subpixel(height, width, device):
    """Official render() passes a zero subpixel offset."""
    key = (height, width, str(device))
    if key not in _OFFSETS:
        _OFFSETS[key] = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
    return _OFFSETS[key]


def raster_settings(camera, background, sh_degree, kernel_size=0.1):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if abs(float(cx) - W / 2) > 0.51 or abs(float(cy) - H / 2) > 0.51:
        raise ValueError("DOF rasterizer assumes a centred principal point (cx = W/2, cy = H/2)")
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
        "kernel_size": float(kernel_size), "subpixel": _subpixel(H, W, K.device),
        "width": W, "height": H, "sh_degree": int(sh_degree),
    }


class _RasterizeDOF(torch.autograd.Function):
    """Same argument order / gradient mapping as diff_dof_gs_rasterizer._RasterizeGaussians."""

    @staticmethod
    def forward(ctx, means3D, means2D, sh, opacities, scales, rotations, F, K, v_idx, s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], F, K, int(v_idx), s["tanfovx"], s["tanfovy"],
                s["kernel_size"], s["subpixel"], s["height"], s["width"], sh, s["sh_degree"],
                s["campos"], False, False)
        (num_rendered, color, depth, xy, radii, _radii_diff, coc, geom, binning,
         img) = extension().rasterize_gaussians(*args)
        ctx.s, ctx.v_idx, ctx.num_rendered = s, int(v_idx), num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, F, K, geom, binning, img)
        ctx.mark_non_differentiable(radii, coc)
        return color, depth, coc, radii

    @staticmethod
    def backward(ctx, grad_color, grad_depth, _grad_coc, _grad_radii):
        s = ctx.s
        means3D, scales, rotations, radii, sh, F, K, geom, binning, img = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        grad_xy = torch.zeros((2, s["height"], s["width"]), device=means3D.device)
        if grad_depth is None:
            grad_depth = torch.zeros((1, s["height"], s["width"]), device=means3D.device)
        args = (s["bg"], means3D, radii, empty, scales, rotations, 1.0, empty,
                s["viewmatrix"], s["projmatrix"], F, K, 0.0, ctx.v_idx, s["tanfovx"],
                s["tanfovy"], s["kernel_size"], s["subpixel"], grad_color.contiguous(),
                grad_depth.contiguous(), grad_xy, sh, s["sh_degree"], s["campos"], geom,
                ctx.num_rendered, binning, img, False)
        (g_means2D, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales, g_rot, g_K,
         g_F) = extension().rasterize_gaussians_backward(*args)
        return g_means3D, g_means2D, g_sh, g_opac, g_scales, g_rot, g_F, g_K, None, None


def rasterize(means, scales, opacities, quats, sh, camera, F, K, v_idx, sh_degree,
              background, kernel_size=0.1):
    """One official DOF render. ``scales``/``opacities`` are already activated (and filtered);
    ``quats`` normalised wxyz; ``sh`` [N,16,3]; ``F``/``K`` [V,1] in the kernel's units.

    Returns dict with HWC ``rgb``, ``depth`` [H,W,1] (ray distance), ``coc`` [H,W,1], and
    ``means2d`` (retains grad, [N,3]) / ``radii`` for densification.
    """
    means2D = torch.zeros_like(means, requires_grad=torch.is_grad_enabled()) + 0
    if means2D.requires_grad:
        means2D.retain_grad()
    s = raster_settings(camera, background, sh_degree, kernel_size)
    color, depth, coc, radii = _RasterizeDOF.apply(
        means.contiguous(), means2D, sh.contiguous(), opacities.reshape(-1, 1).contiguous(),
        scales.contiguous(), quats.contiguous(), F.reshape(-1, 1).contiguous(),
        K.reshape(-1, 1).contiguous(), int(v_idx), s)
    return {"rgb": color.permute(1, 2, 0), "depth": depth.permute(1, 2, 0),
            "coc": coc.permute(1, 2, 0), "means2d": means2D, "radii": radii}
