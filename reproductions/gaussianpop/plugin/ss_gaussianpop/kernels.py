"""JIT-built GaussianPOP error quantification (csrc/pop_error.cu, adapted from gsplat 1.5.3
RasterizeToPixels3DGSFwd.cu via the AuGS plugin's contrib_stats.cu)."""

import functools
import math
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_gaussianpop_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "pop_error.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def tile_lists(means2d, radii, depths, width, height, tile_size=16):
    """gsplat isect_tiles + isect_offset_encode for one camera (non-packed)."""
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d[None], radii[None], depths[None], tile_size, tw, th, packed=False, n_images=1
    )
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
    return offsets.int().contiguous(), flatten_ids.int().contiguous()


@torch.no_grad()
def project(params, camera, sh_degree=3):
    """Same projection and colours as gsplat `rasterization` (classic, non-packed, eps2d 0.3,
    near 0.01, far 1e10, SH + 0.5 clamped at 0) for one camera."""
    from gsplat.cuda._wrapper import fully_fused_projection, spherical_harmonics

    viewmat, K = camera["viewmat"], camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    means = params["means"].detach()
    quats = F.normalize(params["quats"].detach(), dim=-1)
    scales = params["scales"].detach().exp()
    opac = params["opacities"].detach().sigmoid()
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmat[None], K[None], W, H, eps2d=0.3, near_plane=0.01,
        far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False,
        opacities=opac,
    )
    radii, means2d, depths, conics = radii[0], means2d[0], depths[0], conics[0]
    campos = torch.inverse(viewmat)[:3, 3]
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1).detach()
    masks = (radii > 0).all(dim=-1)
    colors = spherical_harmonics(sh_degree, means - campos, coeffs, masks=masks)
    colors = torch.clamp_min(colors + 0.5, 0.0)
    return means2d, radii, depths, conics, opac, colors


@torch.no_grad()
def view_error(params, camera, delta_se, background=(1.0, 1.0, 1.0), n_max=64, eps=1e-9,
               sh_degree=3, tile_size=16):
    """Alg. 1 for one view: adds each Gaussian's removal SE (Eq. 4) summed over the view's
    pixels into `delta_se` [N] (float32, in place). Returns the kernel's C_render [H,W,3]."""
    W, H = int(camera["width"]), int(camera["height"])
    means2d, radii, depths, conics, opac, colors = project(params, camera, sh_degree)
    offsets, flatten_ids = tile_lists(means2d, radii, depths, W, H, tile_size)
    bg = torch.tensor(background, dtype=torch.float32)
    (render,) = ext().pop_error(
        means2d.float().contiguous(), conics.float().contiguous(), opac.float().contiguous(),
        colors.float().contiguous(), bg, W, H, int(tile_size), offsets, flatten_ids, int(n_max),
        float(eps), delta_se,
    )
    return render


@torch.no_grad()
def scene_error(params, cameras, **kwargs):
    """Cumulative Δ_SE over all `cameras` (the global buffer of Alg. 1)."""
    delta_se = torch.zeros(len(params["means"]), device=params["means"].device)
    for cam in cameras:
        view_error(params, cam, delta_se, **kwargs)
    return delta_se
