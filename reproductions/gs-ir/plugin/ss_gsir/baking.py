"""Occlusion baking of GS-IR (official baking.py), with gsplat as the cube-face rasterizer.

Grid: ``res`` points per axis over [-bound, bound]^3, spacing 2 bound / (res - 1). Probes are
the 8 grid corners of the cell containing every Gaussian centre (indices clamped to the grid,
so Gaussians outside the box mark boundary cells). For every probe p the Gaussians with
|mu - p|_inf < ``valid`` are rendered into six 90-degree faces (res_cube^2, official camera
rotations, near plane 0.2 = the 3DGS rasterizer's frustum cull, alpha-normalised z-depth);
depth := depth * [alpha > 0.5] * |ray|; the cube is read back on a 256 x 512 lat-long grid
(nearest); visibility = 1 - [depth < threshold] + [depth == 0]; the 16 SH coefficients are
sum(visibility * Y_lm * dOmega). Afterwards empty grid cells inherit a neighbour's id.

Probes are processed in spatial blocks; each block renders a batch of probes in one gsplat
call over the union crop, with every probe's own crop expressed by zeroing the opacity of the
Gaussians outside its box (gsplat skips alpha < 1/255, so this equals cropping).
"""

import time

import torch
from gsplat import rasterization

from . import cubemap
from .volumes import dilate_ids, sh_components

# official baking.py camera-to-world rotations (OpenCV camera axes as columns), one per face
FACE_C2W = (
    ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
)


def face_rotations(device):
    return torch.tensor(FACE_C2W, device=device, dtype=torch.float32)  # [6, 3, 3]


def probe_mask(means, res, bound):
    """[res]^3 bool: cells marked by the 8 corners around each Gaussian centre."""
    grid = 2.0 * bound / (res - 1)
    q = torch.div(means + bound, grid, rounding_mode="floor").long()
    mask = torch.zeros(res, res, res, dtype=torch.bool, device=means.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                a = (q + torch.tensor([dx, dy, dz], device=q.device)).clamp(0, res - 1)
                mask[a[:, 0], a[:, 1], a[:, 2]] = True
    return mask


def face_cameras(positions, cube_res):
    """positions [B, 3] -> viewmats [B, 6, 4, 4] (world-to-camera), Ks [B, 6, 3, 3]."""
    dev = positions.device
    rot = face_rotations(dev)  # c2w rotations
    b = len(positions)
    view = torch.zeros(b, 6, 4, 4, device=dev)
    rt = rot.transpose(1, 2)  # w2c rotation
    view[:, :, :3, :3] = rt[None]
    view[:, :, :3, 3] = -(rt[None] @ positions[:, None, :, None])[..., 0]
    view[:, :, 3, 3] = 1.0
    f = cube_res / 2.0  # fov 90: W / (2 tan 45)
    K = torch.tensor([[f, 0.0, cube_res / 2.0], [0.0, f, cube_res / 2.0], [0.0, 0.0, 1.0]],
                     device=dev)
    return view, K.expand(b, 6, 3, 3).contiguous()


class LatLong:
    """Precomputed lat-long readback of a depth cube + SH projection weights."""

    def __init__(self, cube_res, bands, device, h=256, w=512):
        dirs, solid = cubemap.latlong_dirs(h, w, device, centered=False)
        face, s, t = cubemap.dir_to_face(dirs)
        tx = (s * cube_res).floor().long().clamp(0, cube_res - 1)
        ty = (t * cube_res).floor().long().clamp(0, cube_res - 1)
        self.index = ((face * cube_res + ty) * cube_res + tx).reshape(-1)  # nearest texel
        comps = sh_components(bands, dirs)  # [h, w, d2]
        self.weights = (solid * comps).reshape(-1, comps.shape[-1])  # [h*w, d2]
        self.cube_res = cube_res
        ys, xs = torch.meshgrid(torch.arange(cube_res, device=device, dtype=torch.float32),
                                torch.arange(cube_res, device=device, dtype=torch.float32),
                                indexing="ij")
        c = cube_res / 2.0
        rays = torch.stack(((xs - c + 0.5) / c, (ys - c + 0.5) / c, torch.ones_like(xs)), -1)
        self.norm = rays.norm(dim=-1)  # [res, res]

    def coefficients(self, depth, alpha, threshold):
        """depth / alpha [B, 6, r, r] -> SH coefficients of visibility [B, d2]."""
        d = depth * (alpha > 0.5).float() * self.norm
        d = d.reshape(d.shape[0], -1)[:, self.index]  # [B, h*w]
        vis = (1.0 - (d < threshold).float()) + (d == 0).float()
        return vis @ self.weights


@torch.no_grad()
def render_probe_depths(means, quats, scales, opac, positions, valid, cube_res, near):
    """Depth / alpha cubes [B, 6, r, r] of the probes at ``positions`` [B, 3]."""
    lo = positions.min(0).values - valid
    hi = positions.max(0).values + valid
    sel = ((means > lo) & (means < hi)).all(1)
    b = len(positions)
    if not bool(sel.any()):
        z = torch.zeros(b, 6, cube_res, cube_res, device=positions.device)
        return z, z
    m, q, s, o = means[sel], quats[sel], scales[sel], opac[sel]
    inside = ((m[None] - positions[:, None]).abs() < valid).all(-1)  # [B, n]
    opac_b = o[None].expand(b, -1) * inside.float()
    view, K = face_cameras(positions, cube_res)
    n = len(m)
    out, alpha, _ = rasterization(
        means=m[None].expand(b, n, 3).contiguous(), quats=q[None].expand(b, n, 4).contiguous(),
        scales=s[None].expand(b, n, 3).contiguous(), opacities=opac_b.contiguous(),
        colors=torch.zeros(b, n, 1, device=m.device), viewmats=view, Ks=K, width=cube_res,
        height=cube_res, near_plane=near, packed=False, render_mode="ED",
    )
    return out[..., 0], alpha[..., 0]


@torch.no_grad()
def bake_occlusion(params, res=256, bound=16.0, valid=1.5, threshold=0.4, bands=4,
                   cube_res=256, near=0.2, batch=16, block=4, log=None):
    """-> dict(ids [res]^3 int32, coefficients [n, bands^2, 1], stats)."""
    t0 = time.time()
    means = params["means"].detach().float()
    quats = params["quats"].detach().float()
    scales = params["scales"].detach().float().exp()
    opac = params["opacities"].detach().float().sigmoid()
    dev = means.device
    mask = probe_mask(means, res, bound)
    cells = torch.nonzero(mask)  # C order == official torch.where order
    n = len(cells)
    ids = torch.full((res, res, res), -1, dtype=torch.int32, device=dev)
    ids[cells[:, 0], cells[:, 1], cells[:, 2]] = torch.arange(n, dtype=torch.int32, device=dev)
    grid = 2.0 * bound / (res - 1)
    positions = cells.float() * grid - bound
    ll = LatLong(cube_res, bands, dev)
    coeffs = torch.zeros(n, bands * bands, device=dev)
    # spatial blocks of block^3 cells, batches of `batch` probes inside each block
    key = cells // block
    nb = res // block + 1
    order = torch.argsort((key[:, 0] * nb + key[:, 1]) * nb + key[:, 2], stable=True)
    bkey = ((key[:, 0] * nb + key[:, 1]) * nb + key[:, 2])[order]
    starts = torch.nonzero(torch.cat([torch.ones(1, dtype=torch.bool, device=dev),
                                      bkey[1:] != bkey[:-1]])).flatten().tolist() + [n]
    done = 0
    for a, z in zip(starts[:-1], starts[1:]):
        for i in range(a, z, batch):
            j = min(i + batch, z)
            idx = order[i:j]
            depth, alpha = render_probe_depths(means, quats, scales, opac, positions[idx],
                                               valid, cube_res, near)
            coeffs[idx] = ll.coefficients(depth, alpha, threshold)
            done += j - i
        if log is not None and done and (done // 50000) != ((done - (z - a)) // 50000):
            log(f"gs-ir baking: {done}/{n} probes, {time.time() - t0:.0f} s")
    ids = dilate_ids(ids)
    stats = {"probes": int(n), "res": res, "bound": bound, "threshold": threshold,
             "valid": valid, "seconds": round(time.time() - t0, 1),
             "mean_visibility_dc": float(coeffs[:, 0].mean() / (4 * torch.pi * 0.28209479177387814))
             if n else 0.0}
    return {"ids": ids, "coefficients": coeffs[..., None].contiguous(), "stats": stats}
