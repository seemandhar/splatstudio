"""C1 global significance (Eq. 3/4 as in the official code) and the significance prune rule.

`hit_scores` runs the JIT-built CUDA port of the official `renderCUDA_count` (csrc/
significance.cu) on gsplat 1.5.3 projections/tile lists; `v_importance` and `prune_mask` are
`prune.py:calculate_v_imp_score` and `GaussianModel.prune_gaussians` of the official code.
"""

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
        name="ss_lightgaussian_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "significance.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def project(params, camera):
    """gsplat projection exactly as `rasterization` does it (classic, non-packed, eps2d 0.3,
    near 0.01, far 1e10, opacity-aware radii) for one camera."""
    from gsplat.cuda._wrapper import fully_fused_projection

    viewmat, K = camera["viewmat"], camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    opac = params["opacities"].detach().sigmoid().contiguous()
    radii, means2d, depths, conics, _ = fully_fused_projection(
        params["means"].detach(), None, F.normalize(params["quats"].detach(), dim=-1),
        params["scales"].detach().exp(), viewmat[None], K[None], W, H, eps2d=0.3,
        near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False, opacities=opac,
    )
    return means2d[0], radii[0], depths[0], conics[0], opac


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
def view_scores(params, camera, count, score, tile_size=16):
    """Adds one view's hit counts / sum(opacity * T) into `count` (int32) / `score` (float32)."""
    W, H = int(camera["width"]), int(camera["height"])
    means2d, radii, depths, conics, opac = project(params, camera)
    offsets, flatten_ids = tile_lists(means2d, radii, depths, W, H, tile_size)
    ext().significance(
        means2d.float().contiguous(), conics.float().contiguous(), opac.float().contiguous(),
        W, H, int(tile_size), offsets, flatten_ids, count, score,
    )


@torch.no_grad()
def hit_scores(params, cameras):
    """`prune.py:prune_list`: (gaussian_list, imp_list) summed over all training cameras."""
    n = len(params["means"])
    dev = params["means"].device
    count = torch.zeros(n, dtype=torch.int32, device=dev)
    score = torch.zeros(n, dtype=torch.float32, device=dev)
    for cam in cameras:
        view_scores(params, cam, count, score)
    return count, score


@torch.no_grad()
def v_importance(score, log_scales, v_pow=0.1, kth_fraction=0.9):
    """`calculate_v_imp_score`: score * (V / V_sorted_desc[int(0.9 N)]) ** v_pow, V = prod scale.

    The paper's Eq. 4 clips V/V_max90 at 1; the official code does not (code followed).
    """
    volume = torch.prod(log_scales.detach().exp(), dim=1)
    index = int(len(volume) * kth_fraction)
    kth = torch.sort(volume, descending=True).values[index]
    return torch.pow(volume / kth, v_pow) * score


@torch.no_grad()
def prune_mask(v, percent):
    """`GaussianModel.prune_gaussians`: remove v <= sorted(v)[int(percent (N - 1))]."""
    sorted_v = torch.sort(v).values
    value = sorted_v[int(percent * (len(sorted_v) - 1))]
    return v <= value


@torch.no_grad()
def global_significance(params, cameras, v_pow=0.1):
    count, score = hit_scores(params, cameras)
    return v_importance(score, params["scales"], v_pow), count, score
