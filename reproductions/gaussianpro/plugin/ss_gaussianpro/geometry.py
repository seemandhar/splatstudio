"""Pure-torch geometry of the official GaussianPro code (github.com/kcheng1021/GaussianPro @ 86bb048).

Conventions: cameras are SplatStudio dicts (OpenCV world-to-camera ``viewmat`` [4,4], pixel ``K``);
images HWC; depth maps [H,W] in camera z. Official equivalents are cited per function.
"""

import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from gsplat.utils import normalized_quat_to_rotmat

SH_C0 = 0.28209479177387814
INVALID_DEPTH = 300.0  # train.py marker for filtered propagated depth
INVALID_NORMAL = -10.0  # train.py marker for filtered propagated normal


def shortest_axis_normals(quats, log_scales):
    """Eq. 4 / gaussian_renderer.render(return_normal): rotation column of the smallest scale."""
    rot = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))  # [N,3,3], columns = axes
    idx = torch.argmin(log_scales, dim=1)
    return rot[torch.arange(len(idx), device=idx.device), :, idx]


def camera_normals(means, quats, log_scales, viewmat):
    """Per-Gaussian camera-frame normal flipped towards the camera (official render()).

    ``normal * ((((view_dir * normal).sum(-1) < 0) * 1 - 0.5) * 2)`` then ``R_w2c @ normal``.
    """
    normal = shortest_axis_normals(quats, log_scales)
    c2w = torch.linalg.inv(viewmat)
    view_dir = means - c2w[:3, 3]
    sign = (((view_dir * normal).sum(dim=-1) < 0).float() - 0.5) * 2
    normal = normal * sign[:, None]
    return normal @ viewmat[:3, :3].T


def camera_depths(means, viewmat):
    """Eq. 3: z of W mu + t (official ``means3D_depth``)."""
    return means @ viewmat[2, :3] + viewmat[2, 3]


def threshold(step, begin, end, max_thr, min_thr):
    """train.py:131 linear annealing of the abs-rel depth error threshold."""
    return max_thr - (max_thr - min_thr) * (step - begin) / (end - begin)


def bilinear_sampler(img, coords):
    """utils/graphics_utils.bilinear_sampler (grid_sample, align_corners=True, zero padding)."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def reproject_with_depth(depth_ref, K_ref, c2w_ref, depth_src, K_src, c2w_src):
    """utils/graphics_utils.reproject_with_depth for batch 1 ([H,W] depths, c2w extrinsics)."""
    H, W = depth_ref.shape
    dev = depth_ref.device
    y_ref, x_ref = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                  indexing="ij")
    x_ref, y_ref = x_ref.reshape(-1).float(), y_ref.reshape(-1).float()
    ones = torch.ones_like(x_ref)
    xyz_ref = torch.linalg.inv(K_ref) @ (torch.stack((x_ref, y_ref, ones)) * depth_ref.reshape(1, -1))
    xyz_src = (torch.linalg.inv(c2w_src) @ c2w_ref @ torch.cat((xyz_ref, ones[None])))[:3]
    K_xyz_src = K_src @ xyz_src
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    x_src = xy_src[0].reshape(H, W).float()
    y_src = xy_src[1].reshape(H, W).float()
    sampled = bilinear_sampler(depth_src.view(1, 1, H, W),
                               torch.stack((x_src, y_src), dim=-1).view(1, H, W, 2))
    xyz_src = torch.linalg.inv(K_src) @ (torch.cat((xy_src, ones[None])) * sampled.reshape(1, -1))
    xyz_reproj = (torch.linalg.inv(c2w_ref) @ c2w_src @ torch.cat((xyz_src, ones[None])))[:3]
    depth_reproj = xyz_reproj[2].reshape(H, W).float()
    K_xyz_reproj = K_ref @ xyz_reproj
    xy_reproj = K_xyz_reproj[:2] / K_xyz_reproj[2:3]
    return (depth_reproj, xy_reproj[0].reshape(H, W).float(), xy_reproj[1].reshape(H, W).float(),
            x_src, y_src)


def check_geometric_consistency(depth_ref, K_ref, c2w_ref, depth_src, K_src, c2w_src,
                                thre1=2.0, thre2=0.01):
    """utils/graphics_utils.check_geometric_consistency (App. A Eq. 15-16 per target view).

    Valid where the forward-backward reprojection error < thre1 px and the relative depth
    error < thre2. Returns the boolean mask [H,W]. The depth maps must have the same size
    (official assumption).
    """
    if depth_ref.shape != depth_src.shape:
        raise ValueError("geometric consistency check expects equal-size depth maps (official)")
    H, W = depth_ref.shape
    dev = depth_ref.device
    y_ref, x_ref = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                  indexing="ij")
    depth_reproj, x2d, y2d, _, _ = reproject_with_depth(depth_ref, K_ref, c2w_ref,
                                                         depth_src, K_src, c2w_src)
    dist = torch.sqrt((x2d - x_ref) ** 2 + (y2d - y_ref) ** 2)
    relative = torch.abs(depth_reproj - depth_ref) / depth_ref
    return torch.logical_and(dist < thre1, relative < thre2)


def backproject(depth, K, viewmat, mask, stride):
    """GaussianModel.densify_from_depth_propagation: world points of masked pixels on the
    ``[::stride, ::stride]`` grid. Returns (points [M,3], downsampled mask [h,w])."""
    H, W = depth.shape
    dev = depth.device
    y, x = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    coords = torch.stack([x.float(), y.float(), torch.ones_like(depth)], dim=-1).view(-1, 3)
    cam = (torch.linalg.inv(K) @ coords.T).T * depth.reshape(-1, 1)
    c2w = torch.linalg.inv(viewmat)
    world = (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]
    world = world.view(H, W, 3)[::stride, ::stride]
    sub_mask = mask[::stride, ::stride]
    return world[sub_mask], sub_mask


# --- COLMAP shared-point neighbour graph (paper App. A, `load_pairs_relation`) -----------------

def read_colmap_image_points(path):
    """images.bin -> {image name: set of observed POINT3D_IDs (!= -1)}."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4 + 32 + 24 + 4)  # image_id, qvec, tvec, camera_id
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            data = f.read(24 * npts)
            ids = struct.unpack("<" + "ddq" * npts, data)[2::3]
            out[name.decode()] = {i for i in ids if i != -1}
    return out


def shared_point_neighbours(names, observations, count=4):
    """For each training view: up to ``count`` other training views ranked by the number of
    shared SfM points (descending, ties by index), score > 0 (official pair loader)."""
    sets = [observations[n] for n in names]
    result = []
    for i, si in enumerate(sets):
        scores = [(len(si & sj), j) for j, sj in enumerate(sets) if j != i]
        ranked = sorted((s for s in scores if s[0] > 0), key=lambda s: (-s[0], s[1]))
        result.append([j for _, j in ranked[:count]])
    return result


def find_sparse_dir(data_root, configured=None):
    """COLMAP model holding the image tracks. Searches ``configured``, ``<data>/sparse/0``,
    ``<data>/sparse`` and the converter's source scene ``<data>/../../360_v2|tandt_db/<scene>``."""
    import json

    root = Path(data_root).resolve()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates += [root / "sparse" / "0", root / "sparse"]
    try:
        meta = json.loads((root / "transforms.json").read_text())
        scene = meta.get("source", {}).get("scene", root.name)
        sub = meta.get("source", {}).get("colmap", "0")
        for base in root.parent.parent.iterdir():
            if base.is_dir():
                candidates += [base / scene / "sparse" / sub, base / scene / "sparse"]
    except (OSError, ValueError):
        pass
    for c in candidates:
        if (c / "images.bin").is_file():
            return c
    raise FileNotFoundError(f"No COLMAP images.bin found for {root} (set density config 'sparse_dir')")
