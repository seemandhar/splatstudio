"""Octree construction and LOD selection, verbatim numerics of Octree-GS scene/gaussian_model.py
(github.com/city-super/Octree-GS @ 9db64fe): set_level, set_coarse_interval, octree_sample, create_from_pcd (box and
voxel size), map_to_int_level ('round'), weed_out, set_anchor_mask. Pure torch; runs on CPU or CUDA.

Levels are stored as float tensors (exact small integers); comparisons `level <= int_level` match the official
int tensors.
"""

import math

import numpy as np
import torch


def set_level(points, cam_centers, dist_ratio=0.999, fork=2, levels=-1, init_level=-1, scale=1.0):
    """set_level: per-camera r_d / (1-r_d) distance quantiles, then the same quantiles over all of them.
    Returns (standard_dist, levels K, init_level) with K = round(log2(d_max/d_min)/log2(fork)) + 1."""
    all_dist = []
    for c in cam_centers:
        dist = torch.sqrt(torch.sum((points - c) ** 2, dim=1))
        dmax = torch.quantile(dist, dist_ratio)
        dmin = torch.quantile(dist, 1 - dist_ratio)
        all_dist.append(torch.stack([dmin, dmax]).float() * scale)
    all_dist = torch.cat(all_dist)
    dist_max = torch.quantile(all_dist, dist_ratio)
    dist_min = torch.quantile(all_dist, 1 - dist_ratio)
    if levels == -1:
        levels = torch.round(torch.log2(dist_max / dist_min) / math.log2(fork)).int().item() + 1
    if init_level == -1:
        init_level = int(levels / 2)
    return float(dist_max), int(levels), int(init_level)


def octree_box(points, extend=1.1, fork=2, base_layer=10):
    """create_from_pcd: scalar box over all coordinates (min/max of the whole point array, times `extend`),
    voxel_size = box_d / fork^base_layer, init_pos = (box_min, box_min, box_min).
    Returns (voxel_size [float32 0-d tensor], init_pos [3])."""
    box_min = torch.min(points) * extend
    box_max = torch.max(points) * extend
    box_d = box_max - box_min
    voxel_size = box_d / (float(fork) ** base_layer)
    init_pos = torch.stack([box_min, box_min, box_min]).float()
    return voxel_size.float(), init_pos


def level_size(voxel_size, fork, level):
    """voxel_size / fork^level computed like the official float32 tensor expression."""
    v = voxel_size if torch.is_tensor(voxel_size) else torch.tensor(voxel_size, dtype=torch.float32)
    return v / (float(fork) ** level)


def octree_sample(points, init_pos, voxel_size, levels, fork=2):
    """octree_sample: per level L, unique(round((P - init_pos)/size_L)) * size_L + init_pos."""
    pos, lev = [], []
    for cur_level in range(levels):
        cur_size = level_size(voxel_size, fork, cur_level).to(points.device)
        new = torch.unique(torch.round((points - init_pos) / cur_size), dim=0) * cur_size + init_pos
        pos.append(new)
        lev.append(torch.full((len(new),), float(cur_level), device=points.device))
    return torch.cat(pos), torch.cat(lev)


def int_level(pred_level, max_level):
    """map_to_int_level with dist2level='round' (the script setting): clamp(round(pred), 0, max_level)."""
    return torch.clamp(torch.round(pred_level).int(), min=0, max=int(max_level))


def view_frequency(positions, levels_of, cam_centers, standard_dist, levels, fork=2, scale=1.0,
                   chunk_elems=1 << 25):
    """weed_out's visible_count / n_cams: fraction of training cameras for which an anchor's level is selected
    (round LOD from the raw anchor position, no LOD bias)."""
    n, nc = len(positions), len(cam_centers)
    out = torch.zeros(n, device=positions.device)
    if n == 0:
        return out
    rows = max(1, chunk_elems // max(nc, 1))
    for s in range(0, n, rows):
        p = positions[s : s + rows]
        dist = torch.sqrt(torch.sum((p[:, None, :] - cam_centers[None]) ** 2, dim=-1)) * scale  # [m, C]
        pred = torch.log2(standard_dist / dist) / math.log2(fork)
        il = int_level(pred, levels - 1)
        out[s : s + rows] = (levels_of[s : s + rows, None] <= il).int().sum(dim=1).float()
    return out / nc


def weed_out(positions, levels_of, cam_centers, standard_dist, levels, threshold, fork=2):
    """weed_out: keep anchors whose view frequency exceeds `threshold`; also returns the mean view frequency of
    the *input* anchors (used as the adaptive threshold when visible_threshold < 0)."""
    freq = view_frequency(positions, levels_of, cam_centers, standard_dist, levels, fork)
    keep = freq > threshold
    mean = float(freq.mean()) if len(freq) else 0.0
    return keep, mean


def coarse_intervals(levels, init_level, coarse_iter=10000, coarse_factor=1.5):
    """set_coarse_interval: geometric intervals N_{i-1} = omega N_i (q = 1/omega), cumulative, summing to
    coarse_iter, one per level above init_level."""
    out = []
    num_level = levels - 1 - init_level
    if num_level > 0:
        q = 1 / coarse_factor
        a1 = coarse_iter * (1 - q) / (1 - q**num_level)
        temp = 0
        for i in range(num_level):
            interval = a1 * q**i + temp
            temp = interval
            out.append(interval)
    return out


def coarse_index(intervals, step, init_level, levels, progressive=True, training=True):
    """set_anchor_mask's coarse_index (number of usable levels): searchsorted(intervals, it) + 1 + init_level while
    training progressively, K otherwise."""
    if progressive and training:
        return int(np.searchsorted(intervals, step)) + 1 + init_level
    return levels


def anchor_mask(anchor, level, extra_level, cam_center, voxel_size, standard_dist, fork, max_level, scale=1.0):
    """set_anchor_mask (Eq. 3, code 'round' variant): LOD distance from the anchor shifted by half its level's
    voxel, pred = log2(std/d)/log2(fork) + Delta L, selected iff L_j <= clamp(round(pred), 0, max_level)."""
    level = level.reshape(-1)
    extra_level = extra_level.reshape(-1)
    pos = anchor + ((voxel_size / 2) / (float(fork) ** level))[:, None]
    dist = torch.sqrt(torch.sum((pos - cam_center) ** 2, dim=1)) * scale
    pred = torch.log2(standard_dist / dist) / math.log2(fork) + extra_level
    return level <= int_level(pred, max_level)
