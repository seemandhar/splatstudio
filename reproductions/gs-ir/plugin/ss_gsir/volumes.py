"""Occlusion and irradiance volumes of GS-IR (official gs-ir/gs_ir/{__init__,volumes}.py and
gs-ir/src/{occlusion,irradiance}_kernel.cu).

* occlusion volume: dense id grid [R, R, R] (int32, -1 = empty before dilation) + sparse SH
  coefficients [n, 16, 1] of the baked *visibility*; recovery = masked trilinear interpolation
  (supp. Eq. 6) at x + n * bound / R, then the cosine-weighted hemisphere average of the
  clamped SH (CUDA, ``sh_reconstruction``).
* irradiance volume: learnable [R, R, R, 9, 1] SH grid; trilinear (unmasked, clamped) of the
  coefficients at x, evaluated at the normal, clamped at 0. Gradients reach the coefficients.
"""

import torch

from .cubemap import gather_rows
from .cuda import ext

EPS = 1e-4
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_components(bands, d):
    """components_from_spherical_harmonics(degree=bands, directions) -> [..., bands^2]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    out = [torch.full_like(x, C0)]
    if bands > 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if bands > 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z,
                C2[4] * (xx - yy)]
    if bands > 3:
        out += [C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    if bands > 4:
        raise ValueError("bands must be <= 4")
    return torch.stack(out, dim=-1)


_OFFS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def _corners(points, aabb, res):
    lo, hi = aabb[:3], aabb[3:]
    grid = (hi - lo) / (res - 1.0)
    n = ((points - lo) / grid).clamp(0.0, res - 1.0)
    f = n.floor()
    o = n - f
    q = f.long()
    return lo, grid, f, o, q


def _trilinear_weights(o):
    ws = []
    for dx, dy, dz in _OFFS:
        wx = o[:, 0] if dx else 1 - o[:, 0]
        wy = o[:, 1] if dy else 1 - o[:, 1]
        wz = o[:, 2] if dz else 1 - o[:, 2]
        ws.append(wx * wy * wz)
    return torch.stack(ws, dim=1)  # [P, 8]


def _flat_ids(q, res):
    ids = []
    for dx, dy, dz in _OFFS:
        x = (q[:, 0] + dx).clamp(max=res - 1)
        y = (q[:, 1] + dy).clamp(max=res - 1)
        z = (q[:, 2] + dz).clamp(max=res - 1)
        ids.append((x * res + y) * res + z)
    return torch.stack(ids, dim=1)  # [P, 8]


def masked_trilinear(coeffs, ids, aabb, points, normals):
    """sparse_interpolate_coefficients: coeffs [n, d2] (or [n, d2, 1]), ids [R, R, R] -> [P, d2]."""
    res = ids.shape[0]
    lo, grid, f, o, q = _corners(points, aabb, res)
    w = _trilinear_weights(o)
    masks = []
    for dx, dy, dz in _OFFS:
        corner = (f + torch.tensor([dx, dy, dz], device=f.device, dtype=f.dtype)) * grid + lo
        masks.append(((corner - points) * normals).sum(-1) > 0)
    w = w * torch.stack(masks, dim=1).float()
    s = w.sum(1, keepdim=True)
    w = w / torch.where(s == 0, torch.full_like(s, EPS), s)
    cid = ids.reshape(-1)[_flat_ids(q, res)].long()  # [P, 8]
    c = coeffs.reshape(coeffs.shape[0], -1)
    return (c[cid] * w[..., None]).sum(1)


def recon_occlusion(coeffs, ids, bound, points, normals, bands=4, samples=256):
    """gs_ir.recon_occlusion -> visibility [P, 1] (0 for zero normals)."""
    res = ids.shape[0]
    aabb = torch.tensor([-bound] * 3 + [bound] * 3, device=points.device, dtype=points.dtype)
    shifted = points + normals * (bound / float(res))
    c = masked_trilinear(coeffs, ids, aabb, shifted, normals)  # [P, d2]
    rough = torch.ones(len(points), device=points.device, dtype=torch.float32)
    return ext().sh_reconstruction(c[:, None, :].contiguous().float(),
                                   normals.contiguous().float(), rough, samples, bands, True)


def query_irradiance(coeffs, aabb, points, normals):
    """IrradianceVolumes.query_irradiance: coeffs [R, R, R, d2, C] -> [P, C] (>= 0)."""
    res = coeffs.shape[0]
    d2, ch = coeffs.shape[3], coeffs.shape[4]
    bands = int(round(d2**0.5))
    with torch.no_grad():
        comps = sh_components(bands, normals)  # [P, d2]
        _, _, _, o, q = _corners(points, aabb, res)
        w = _trilinear_weights(o)
        cid = _flat_ids(q, res)
    flat = coeffs.reshape(-1, d2, ch)
    interp = (gather_rows(flat, cid) * w[..., None, None]).sum(1)  # [P, d2, C]
    return (interp * comps[..., None]).sum(1).clamp(min=0.0)


@torch.no_grad()
def dilate_ids(ids):
    """Fill empty (-1) cells from their 6-neighbours until none are empty (official
    dialate_occlusion_ids; neighbours visited +z, +y, +x, -x, -y, -z with the last valid one
    winning, here with snapshot (Jacobi) updates)."""
    if bool((ids >= 0).sum() == 0):
        raise ValueError("cannot dilate an occlusion grid without any baked cell")
    ids = ids.clone()
    order = [(2, 1), (1, 1), (0, 1), (0, -1), (1, -1), (2, -1)]
    while bool((ids < 0).any()):
        snap = ids
        new = ids.clone()
        for axis, sgn in order:
            nb = _shift_clamped(snap, axis, sgn)
            new = torch.where((snap < 0) & (nb >= 0), nb, new)
        ids = new
    return ids


def _shift_clamped(t, axis, sgn):
    """value at index i + sgn along axis, clamped to the grid."""
    n = t.shape[axis]
    idx = (torch.arange(n, device=t.device) + sgn).clamp(0, n - 1)
    return t.index_select(axis, idx)
