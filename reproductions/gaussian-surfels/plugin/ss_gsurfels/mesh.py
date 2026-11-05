"""Volumetric cutting and screened-Poisson meshing (paper §3.4) — port of the official render.py
(render_set, name == 'train'), GaussianModel.to_occ_grid, utils/image_utils.{depth2wpos,
resample_points, grid_prune} and utils/general_utils.poisson_mesh.

Extra dependency: pymeshlab (not in the splatstudio env; installed with
``pip install --target ~/models/gsurfels-pylib pymeshlab``; found via PYTHONPATH or the
sys.path fallback in ``_import_pymeshlab``).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .raster import utils_extension

PYLIB = Path(os.environ.get("GSURFELS_PYLIB", "~/models/gsurfels-pylib")).expanduser()


def _import_pymeshlab():
    try:
        import pymeshlab
    except ImportError:
        if str(PYLIB) not in sys.path:
            sys.path.append(str(PYLIB))
        import pymeshlab
    return pymeshlab


@torch.no_grad()
def occupancy_grid(params, grid_dim_max=512, bound=None, cutoff=0.0):
    """GaussianModel.to_occ_grid: bbox of the means +-10 % (or ``bound``), cubic voxels."""
    xyz = params["means"].detach().float()
    if bound is None:
        xyz_min, xyz_max = xyz.min(0)[0], xyz.max(0)[0]
        ln = xyz_max - xyz_min
        xyz_min, xyz_max = xyz_min - ln * 0.1, xyz_max + ln * 0.1
    else:
        xyz_min, xyz_max = (b.float().clone() for b in bound)
    ln = xyz_max - xyz_min
    grid_len = ln.max() / grid_dim_max
    grid_dim = (ln / grid_len + 0.5).to(torch.int32)
    grid = utils_extension().gaussian2occgrid(
        xyz_min.contiguous(), xyz_max.contiguous(), grid_len.reshape(()).contiguous(),
        grid_dim.contiguous(), xyz.contiguous(),
        F.normalize(params["quats"].detach().float(), dim=-1).contiguous(),
        params["scales"].detach().float().exp().contiguous(),
        params["opacities"].detach().float().sigmoid()[:, None].contiguous(),
        torch.tensor([cutoff], dtype=torch.float32, device=xyz.device))
    return grid, -xyz_min, 1 / grid_len, grid_dim


def grid_prune(grid, shift, scale, dim, pts, thrsh=1.0):
    """utils/image_utils.grid_prune: keep points whose voxel holds > thrsh accumulated G*o."""
    cord = ((pts + shift) * scale).to(torch.long)
    out = (torch.le(cord, 0) | torch.gt(cord, dim - 1)).any(1)
    cord = torch.maximum(torch.minimum(cord, (dim - 1).long()), torch.zeros_like(cord))
    keep = grid[cord[:, 0], cord[:, 1], cord[:, 2]] > thrsh
    return keep & ~out


def depth2wpos(depth, camera):
    """utils/image_utils.depth2wpos (incl. the official fx/fy swap) -> [3,H,W] world points."""
    K = camera["K"].float()
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    H, W = depth.shape[-2:]
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                          torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    d = depth[0]
    cam = torch.stack([(u - cx) * d / fy, (v - cy) * d / fx, d], -1)
    w2c = camera["viewmat"].float()
    world = (cam - w2c[:3, 3]) @ w2c[:3, :3]
    return world.permute(2, 0, 1)


def resample_points(camera, depth, normal, mask):
    """utils/image_utils.resample_points: [M, 6] world xyz + world normal (colour dropped)."""
    m = mask[0].to(torch.bool)
    wpos = depth2wpos(depth, camera).permute(1, 2, 0)[m]
    n_cam = normal.permute(1, 2, 0)[m]
    n_world = n_cam @ camera["viewmat"][:3, :3].float()
    return torch.cat([wpos, n_world], -1)


def nn_sqdist(query, ref):
    """Squared distance to the nearest reference point (pytorch3d knn_points semantics)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(ref).query(query, k=1, workers=16)
    return d.astype(np.float64) ** 2


def poisson_mesh(path, vtx, normal, depth=10, thrsh=3e-5, threads=16):
    """general_utils.poisson_mesh: screened Poisson, prune vertices with squared NN distance to
    the samples > thrsh, close holes (<= 300 edges). meshlab's default thread count spins on
    this 256-core host, so threads is set explicitly (does not change the reconstruction)."""
    ml = _import_pymeshlab()
    ms = ml.MeshSet()
    v = vtx.cpu().numpy().astype(np.float64)
    ms.add_mesh(ml.Mesh(vertex_matrix=v, v_normals_matrix=normal.cpu().numpy().astype(
        np.float64)))
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True,
                                                         samplespernode=1.5, threads=threads)
    vert = ms.current_mesh().vertex_matrix()
    face = ms.current_mesh().face_matrix()
    ms.save_current_mesh(str(path) + "_plain.ply")
    q = nn_sqdist(vert, v)
    ms.add_mesh(ml.Mesh(vertex_matrix=vert, face_matrix=face, v_scalar_array=q))
    ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
    ms.meshing_remove_selected_vertices()
    ms.meshing_close_holes(maxholesize=300)
    ms.save_current_mesh(str(path) + "_pruned.ply")
    return {"plain_vertices": int(len(vert)),
            "pruned_vertices": int(ms.current_mesh().vertex_number()),
            "pruned_faces": int(ms.current_mesh().face_number())}


@torch.no_grad()
def extract(run, data, device="cuda", poisson_depth=10, cut=True, grid_dim=512):
    """render.py render_set('train'): black background, full resolution, SH degree 3."""
    from splatstudio.data import load_dataset, on_device

    from . import sidecar
    from .raster import render

    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location=device, weights_only=False)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    ds = load_dataset(data, 1)
    tab = sidecar.table(data)
    grid = shift = scale = dims = None
    if cut:
        grid, shift, scale, dims = occupancy_grid(params, grid_dim)
    bg = torch.zeros(3, device=device)
    pts, n_raw, n_cut = [], 0, 0
    for name in ds["splits"]["train"]:
        fr = on_device(ds["frames"][name], device)
        out = render(params, fr, 3, bg)
        opac, depth = out["opac"], out["depth"]
        mask_vis = opac > 1e-5
        normal = F.normalize(out["normal"], dim=0) * mask_vis
        idx = tab.index(fr["viewmat"]) if tab is not None else None
        mask_gt = tab.view(idx, 1, device)["mask"] if idx is not None else torch.ones_like(opac)
        mask_clip = (depth > 0) & (depth < 20)
        p = resample_points(fr, depth, normal, mask_vis * mask_gt * mask_clip)
        n_raw += len(p)
        if cut:
            p = p[grid_prune(grid, shift, scale, dims, p[:, :3], thrsh=1.0)]
        n_cut += len(p)
        pts.append(p.cpu())
    pts = torch.cat(pts, 0)
    out_dir = Path(run) / "mesh"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / (f"poisson_mesh_{poisson_depth}" + ("" if cut else "_nocut"))
    stats = poisson_mesh(stem, pts[:, :3], pts[:, 3:6], poisson_depth)
    stats.update({"points_raw": n_raw, "points_after_cut": n_cut, "cut": cut,
                  "grid_dims": dims.tolist() if dims is not None else None})
    return Path(str(stem) + "_pruned.ply"), stats
