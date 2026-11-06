"""Aggregate small Gaussians into coarse-level Gaussians (C2; paper Alg. 1, supp. Sec. 2).

Code form of MS-GS @ 50e5040 scene/gaussian_model.py:insert_large_gaussians:
  * positions normalised by the 3DGS camera extent B; per axis, values > 1 map to 2 - 1/x
    (the official code leaves the negative side linear -- kept);
  * voxel size 0.02 * level / 4 in that space; voxel index floor(p / size) (Open3D-ML
    VoxelPooling(position_fn='center', feature_fn='average'));
  * every raw parameter (xyz, sh, logit opacity, raw quaternion, log scale) is averaged per voxel,
    together with the per-Gaussian minimum pixel coverage S_min;
  * scale *= target_px / clip(mean S_min, 0.25, 2.0); opacity unchanged; new level; S range = -1.
"""

import torch

RAW_FIELDS = ("means", "sh0", "shN", "opacities", "quats", "scales")


@torch.no_grad()
def contract(means, extent):
    rel = means / extent
    return torch.where(rel > 1, 2 - 1 / rel, rel)


@torch.no_grad()
def voxel_aggregate(raw, min_px, extent, level, *, voxel_base=0.02, target_px=2.0,
                    scale_clip=(0.25, 2.0)):
    """raw: dict of the chosen Gaussians' raw parameters. Returns dict of new Gaussians."""
    n = len(raw["means"])
    if n == 0:
        return None
    size = voxel_base * (level / 4)
    keys = torch.floor(contract(raw["means"], extent) / size).long()
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    m = len(counts)

    def pool(t):
        flat = t.reshape(n, -1).float()
        out = torch.zeros(m, flat.shape[1], device=flat.device, dtype=flat.dtype)
        out.index_add_(0, inverse, flat)
        out /= counts[:, None].to(flat.dtype)
        return out.reshape(m, *t.shape[1:])

    new = {k: pool(raw[k]) for k in RAW_FIELDS}
    avg_px = pool(min_px)
    factor = target_px / avg_px.clamp(*scale_clip)
    new["scales"] = new["scales"] + torch.log(factor)[:, None]
    new["ms_level"] = torch.full((m,), float(level), device=avg_px.device)
    new["ms_max_px"] = torch.full((m,), -1.0, device=avg_px.device)
    new["ms_min_px"] = torch.full((m,), -1.0, device=avg_px.device)
    return new
