"""JIT-built TrimGS contribution kernel (csrc/trim_contrib.cu) and its gsplat front end.

The official contribution pass (Trim3DGS `render(..., record_transmittance=True)`) traverses the
3DGS tile lists built with the 3-sigma radius ceil(3 sqrt(lambda_max)). gsplat 1.5.3's
`rasterization` uses an opacity-aware, per-axis 3.33-sigma box instead, which changes the set
of covered pixels |P_k| that normalizes Eq. 5. So we call gsplat's projection with
`opacities=None` (no opacity culling), replace the radii by the official radius, and build the
tile lists with gsplat's own `isect_tiles` / `isect_offset_encode`.
"""

import functools
from pathlib import Path

import torch

from . import core

CSRC = Path(__file__).resolve().parent / "csrc"
TILE = 16


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_trimgs_cuda",
        sources=[str(CSRC / "trim_contrib.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def project(params, camera, eps, near_plane=0.2):
    """gsplat projection of the flat Gaussians with the official 3-sigma radius and tiles."""
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

    W, H = int(camera["width"]), int(camera["height"])
    radii, means2d, depths, conics, _ = fully_fused_projection(
        params["means"].detach(), None, params["quats"].detach(),
        core.flat_scales(params["scales"].detach(), eps), camera["viewmat"][None].float(),
        camera["K"][None].float(), W, H, eps2d=0.3, packed=False, near_plane=near_plane,
        opacities=None,
    )
    visible = (radii > 0).all(dim=-1)  # [1,N]
    r = torch.where(visible, core.official_radii(conics), torch.zeros_like(conics[..., 0]))
    radii_off = r.int()[..., None].repeat(1, 1, 2).contiguous()
    tw, th = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii_off, depths, TILE, tw, th)
    offsets = isect_offset_encode(isect_ids, 1, tw, th)
    return {
        "means2d": means2d[0].float().contiguous(),
        "conics": conics[0].float().contiguous(),
        "opacities": params["opacities"].detach().sigmoid().float().contiguous(),
        "isect_offsets": offsets[0].int().contiguous(),
        "flatten_ids": flatten_ids.int().contiguous(),
        "width": W, "height": H, "radii": r[0],
    }


@torch.no_grad()
def contribution(params, camera, eps, gamma=0.5):
    """Per-Gaussian (sum alpha^2g T^(2-2g), covered pixel count, final T) for one view."""
    m = project(params, camera, eps)
    s, cnt, T = ext().trim_contrib(
        m["means2d"], m["conics"], m["opacities"], m["isect_offsets"], m["flatten_ids"],
        m["width"], m["height"], TILE, float(gamma),
    )
    return {"sum": s, "covered": cnt, "T": T, "meta": m}
