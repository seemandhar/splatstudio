"""SG: significance-score pruning (paper §3.2.2, Eq. 4), adopted from LightGaussian.

Eq. 4 is LightGaussian's Eq. 3; it is implemented as the LightGaussian official code
(github.com/VITA-Group/LightGaussian prune.py, train_densify_prune.py):
  imp_j = sum over training-view pixels of opacity_j * T (CUDA `renderCUDA_count`, here the
          kernel of the native LightGaussian port, csrc/significance.cu),
  v_j   = (V_j / V_sorted_desc[int(0.9 N)])^0.1 * imp_j,  V = prod(scale),
  prune v <= sorted(v)[int(p (N-1))], p_k = first * decay^k.
Trick-GS prunes "at lower image scales": views are projected at the current training resolution
and low-pass, with the GM-masked opacity and scale (what the training renderer rasterizes).
"""

import functools
import math
from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_trick_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "significance.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def project(means, quats, scales, opacities, camera, eps2d=0.3):
    """gsplat projection as `rasterization` does it (classic, non-packed) for one camera."""
    from gsplat.cuda._wrapper import fully_fused_projection

    W, H = int(camera["width"]), int(camera["height"])
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, F.normalize(quats, dim=-1), scales, camera["viewmat"][None],
        camera["K"][None], W, H, eps2d=float(eps2d), near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, packed=False, calc_compensations=False, opacities=opacities,
    )
    return means2d[0], radii[0], depths[0], conics[0]


def tile_lists(means2d, radii, depths, width, height, tile_size=16):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d[None], radii[None], depths[None], tile_size, tw, th, packed=False, n_images=1
    )
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
    return offsets.int().contiguous(), flatten_ids.int().contiguous()


@torch.no_grad()
def hit_scores(means, quats, scales, opacities, cameras, eps2d=0.3, tile_size=16):
    """(count int32 [N], score float32 [N]) summed over `cameras`."""
    n = len(means)
    count = torch.zeros(n, dtype=torch.int32, device=means.device)
    score = torch.zeros(n, dtype=torch.float32, device=means.device)
    opac = opacities.float().contiguous()
    for cam in cameras:
        W, H = int(cam["width"]), int(cam["height"])
        means2d, radii, depths, conics = project(means, quats, scales, opac, cam, eps2d)
        offsets, flatten_ids = tile_lists(means2d, radii, depths, W, H, tile_size)
        ext().significance(
            means2d.float().contiguous(), conics.float().contiguous(), opac, W, H,
            int(tile_size), offsets, flatten_ids, count, score,
        )
    return count, score


@torch.no_grad()
def v_importance(score, scales, v_pow=0.1, kth_fraction=0.9):
    """`calculate_v_imp_score` (scales activated)."""
    volume = torch.prod(scales, dim=1)
    kth = torch.sort(volume, descending=True).values[int(len(volume) * kth_fraction)]
    return torch.pow(volume / kth, v_pow) * score


@torch.no_grad()
def prune_mask(v, percent):
    """`GaussianModel.prune_gaussians`: v <= sorted(v)[int(percent (N - 1))]."""
    value = torch.sort(v).values[int(percent * (len(v) - 1))]
    return v <= value
