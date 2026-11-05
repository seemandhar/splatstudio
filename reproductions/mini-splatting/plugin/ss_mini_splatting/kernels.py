"""JIT-built CUDA statistics kernels (csrc/) and their Python wrappers.

gsplat 1.5.3 exposes the per-view tile intersection lists (`isect_offsets`, `flatten_ids`) and
the projected 2D Gaussians in `rasterization`'s meta, but no per-Gaussian blending-weight sums,
argmax-weight pixel counts or per-pixel argmax ids. `csrc/ms_raster.cu` re-traverses gsplat's
own tile lists (same alpha numerics) and accumulates the statistics of the official
Mini-Splatting rasterizer (`render_imp` / `render_depth`).
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build (first call, cached by torch in ~/.cache/torch_extensions) and import the module."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_mini_splatting_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "ms_raster.cu"), str(CSRC / "simple_knn.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _flat(meta):
    """Single-camera, non-packed gsplat meta -> contiguous kernel inputs."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("Mini-Splatting statistics expect one non-packed camera")
    return (
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        int(meta["width"]),
        int(meta["height"]),
        int(meta["tile_size"]),
    )


@torch.no_grad()
def blend_stats(meta, weights=True):
    """Per-Gaussian (accum_weights, area_proj, area_max) and per-pixel (argmax id, T).

    accum_weights_i = sum over pixels of w_i = alpha_i T_i; area_proj_i = #pixels with w_i > 0
    in the blend; area_max_i = S_i of Eq. 1 (#pixels whose max-weight Gaussian is i).
    """
    w, cnt, mx, argmax, T = ext().ms_stats(*_flat(meta), bool(weights))
    return {"accum_weights": w, "area_proj": cnt, "area_max": mx, "argmax": argmax, "T": T}


def quat_to_rotmat(quats):
    """wxyz (any norm) -> local-to-world rotation matrices [N,3,3]."""
    w, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


@torch.no_grad()
def depth_points(meta, params, camera):
    """Per-pixel mid-point of the max-weight Gaussian (official render_depth) and final T."""
    rot_wl = quat_to_rotmat(params["quats"].detach()).transpose(1, 2).reshape(-1, 9)
    inv_scale = 1.0 / (3.0 * params["scales"].detach().exp())  # official uses the 3-sigma ellipsoid
    c2w = torch.linalg.inv(camera["viewmat"].float()).contiguous()
    K = camera["K"].float().cpu()
    pts, T, argmax = ext().ms_depth(
        *_flat(meta),
        params["means"].detach().float().contiguous(),
        rot_wl.float().contiguous(),
        inv_scale.float().contiguous(),
        c2w,
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )
    return {"points": pts, "T": T, "argmax": argmax}


@torch.no_grad()
def knn_dist2(points):
    """simple-knn distCUDA2: mean squared distance to the 3 (approximate) nearest neighbours."""
    return ext().dist_cuda2(points.detach().float().contiguous())


@torch.no_grad()
def project(params, camera):
    """gsplat projection + tile intersection with the renderer's settings (classic, eps2d 0.3)."""
    from gsplat import rasterization

    _, _, meta = rasterization(
        means=params["means"].detach(),
        quats=params["quats"].detach(),
        scales=params["scales"].detach().exp(),
        opacities=params["opacities"].detach().sigmoid(),
        colors=params["sh0"].detach(),
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        sh_degree=0,
        packed=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    return meta
