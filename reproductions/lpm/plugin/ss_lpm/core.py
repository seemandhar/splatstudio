"""Pure LPM geometry/selection functions, ported from the official code
(https://github.com/Surrey-UP-Lab/GS-LPM @7c06026, files under `lpm/`; Inria 3DGS licence).

Conventions: images HWC float in [0,1]; cameras are SplatStudio dicts (OpenCV world-to-camera
`viewmat`, pixel `K`); regions are integer boxes (x0, y0, x1, y1); a 3D zone is a 6-vector
(xmin, ymin, zmin, xmax, ymax, zmax).
"""

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------- C1 error map
def error_map(image, gt, quantile=0.4, patch=16, fill=0.85):
    """Official `get_errormap(..., "diff")`: brightness-normalised L1, 40% quantile mask,
    16x16 patches kept when >85% of their pixels are above the quantile. Returns HxW float."""
    h, w = image.shape[:2]
    img = image / (image.mean() + 0.01)
    ref = gt / (gt.mean() + 0.01)
    err = (img - ref).abs().sum(dim=-1)  # sum over channels
    mask = err > torch.quantile(err, quantile)
    ph = (h + patch - 1) // patch * patch - h
    pw = (w + patch - 1) // patch * patch - w
    mask = torch.nn.functional.pad(mask, (0, pw, 0, ph), mode="constant", value=0)
    patches = mask.unfold(0, patch, patch).unfold(1, patch, patch)
    significant = patches.sum(dim=(2, 3)) > patch * patch * fill
    full = significant.repeat_interleave(patch, 0).repeat_interleave(patch, 1)
    return full[:h, :w].float()


# ---------------------------------------------------------------- C2 region mapping
def error_regions(emap):
    """Bounding boxes (x0, y0, x0+w, y0+h) of 8-connected components of the error map."""
    m = ((emap > 0).cpu().numpy().astype(np.uint8)) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [
        (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
        )
        for i in range(1, n)
    ]


def _bbox(points):
    if points.numel() == 0:
        return None
    lo, hi = points.min(dim=0).values, points.max(dim=0).values
    return (int(lo[0].item()), int(lo[1].item()), int(hi[0].item()), int(hi[1].item()))


def paired_regions(emap, pts_cur, pts_ref):
    """Official `get_paired_regions`: for each error component, the int bbox of the matched
    keypoints inside it (inclusive test) in the current view and of their matches in the
    referred view."""
    cur, ref = [], []
    for x0, y0, x1, y1 in error_regions(emap):
        inside = (
            (pts_cur[:, 0] >= x0) & (pts_cur[:, 0] <= x1)
            & (pts_cur[:, 1] >= y0) & (pts_cur[:, 1] <= y1)
        )
        r1 = _bbox(pts_ref[inside])
        if r1 is not None:
            cur.append(_bbox(pts_cur[inside]))
            ref.append(r1)
    return cur, ref


# ---------------------------------------------------------------- C3 zone projection
def camera_center(camera):
    vm = camera["viewmat"]
    return -vm[:3, :3].T @ vm[:3, 3]


def pixel_rays(camera, xs, ys):
    """World rays (origin, unnormalised direction) through pixel centres (x+0.5, y+0.5);
    official `set_rays_od` uses pix2ndc(v) = (2v+1)/S - 1, i.e. the same pixel centres."""
    vm, K = camera["viewmat"].float(), camera["K"].float()
    pix = torch.stack([xs.float() + 0.5, ys.float() + 0.5, torch.ones_like(xs, dtype=torch.float32)], -1)
    d_cam = pix @ torch.linalg.inv(K).T
    d = d_cam @ vm[:3, :3]  # = (R^T d_cam^T)^T
    o = camera_center(camera).expand_as(d)
    return torch.cat([o, d], dim=-1)


def perimeter_rays(camera, region):
    """Official `gather_rays`: top row, bottom row (y1-1), left col, right col (x1-1)."""
    x0, y0, x1, y1 = region
    dev = camera["viewmat"].device
    xr = torch.arange(x0, x1, device=dev)
    yr = torch.arange(y0, y1, device=dev)
    xs = torch.cat([xr, xr, torch.full_like(yr, x0), torch.full_like(yr, x1 - 1)])
    ys = torch.cat([torch.full_like(xr, y0), torch.full_like(xr, y1 - 1), yr, yr])
    return pixel_rays(camera, xs, ys)


def closest_points(rays_a, rays_b):
    """Official `get_rays_intersection`: closest points of paired lines, both returned."""
    oa, da, ob, db = rays_a[:, :3], rays_a[:, 3:], rays_b[:, :3], rays_b[:, 3:]
    cross = torch.cross(da, db, dim=1)
    den = (cross * cross).sum(dim=1)
    ta = (torch.cross(ob - oa, db, dim=1) * cross).sum(dim=1) / den
    tb = (torch.cross(oa - ob, da, dim=1) * cross).sum(dim=1) / den
    return torch.cat([oa + ta[:, None] * da, ob + tb[:, None] * db * (-1)], dim=0)


def region_zone(cam_cur, cam_ref, r0, r1):
    a, b = perimeter_rays(cam_cur, r0), perimeter_rays(cam_ref, r1)
    if not len(a) or not len(b):
        return None
    diff = len(a) - len(b)
    if diff > 0:
        b = torch.cat([b, b[0].repeat(diff, 1)])
    elif diff < 0:
        a = torch.cat([a, a[0].repeat(-diff, 1)])
    pts = closest_points(a, b)
    return torch.cat([pts.min(dim=0).values, pts.max(dim=0).values])


def zones3d(cam_cur, cam_ref, regions_cur, regions_ref):
    out = []
    for r0, r1 in zip(regions_cur, regions_ref):
        if (r0[2] - r0[0]) * (r0[3] - r0[1]) > 0 and (r1[2] - r1[0]) * (r1[3] - r1[1]) > 0:
            z = region_zone(cam_cur, cam_ref, r0, r1)
            if z is not None:
                out.append(z)
    return out


# ---------------------------------------------------------------- C4 localized densification
def in_zones(xyz, zones):
    """Strictly inside any axis-aligned zone (official `get_points_mask_per_region`)."""
    mask = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    for z in zones:
        z = z.to(xyz.device)
        mask |= ((z[:3] < xyz) & (xyz < z[3:])).all(dim=1)
    return mask


def densify_masks(grads, max_scale, in_zone, grad_threshold, size_limit, grad_ratio=0.5):
    """Official `lpm_densify_and_clone/split` selection, on the pre-clone population.
    (Clones receive zero gradient in the official split step, so they are never split and the
    split mask restricted to originals is identical.) Returns clone, split, n_extra."""
    high = grads >= grad_threshold
    lower = grads >= grad_threshold * grad_ratio
    small = max_scale <= size_limit
    zone_small = in_zone & small & lower
    zone_large = in_zone & ~small & lower
    clone = (high & small) | zone_small
    split = (high & ~small) | zone_large
    n_extra = int((zone_small & ~high).sum()) + int((zone_large & ~high).sum())
    return clone, split, n_extra


def prune_mask(logits, scales_max, n_extra, min_opacity, world_limit):
    """Official `lpm_densify_and_prune` prune: sigma < min_opacity, the n_extra lowest-logit
    points (<= the n_extra-th smallest logit), and (when `world_limit`) scale > limit."""
    mask = torch.sigmoid(logits) < min_opacity
    if n_extra:
        thr = torch.topk(logits, n_extra, largest=False).values[-1]
        mask |= logits <= thr
    if world_limit is not None:
        mask |= scales_max > world_limit
    return mask


# ---------------------------------------------------------------- C5 localized reset
def cone_masks(cam_pos, zone, xyz):
    """Official `get_points_in_single_cone`: (in front of zone inside the camera cone, inside
    the zone sphere of radius min-extent/2)."""
    lo, hi = zone[:3], zone[3:]
    center = (lo + hi) / 2
    radius = (lo - hi).abs().min() / 2
    inside = (xyz - center).norm(dim=1) <= radius
    height = (cam_pos - center).norm()
    cos_half = torch.cos(torch.atan(radius / height))
    axis = (center - cam_pos) / height
    v = xyz - cam_pos
    vlen = v.norm(dim=1)
    front = ((v / vlen[:, None]) * axis).sum(dim=1) >= cos_half
    front &= vlen <= height
    d = height - radius
    if d > 0:
        front &= vlen <= d
    return front, inside


def reset_mask(cam_pos, zones, xyz, logits, top_ratio=0.7):
    """Official `reset_localized_points`: Gaussians in front of a zone whose opacity logit is at
    least the k-th largest among the zone's inside points, k = int(0.7 * n_inside)."""
    front = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    inside = torch.zeros_like(front)
    for z in zones:
        f, i = cone_masks(cam_pos, z.to(xyz.device), xyz)
        front |= f
        inside |= i
    n = int(inside.sum())
    if n == 0:
        return front & False
    k = int(n * top_ratio)
    if k == 0:
        return front & False
    thr = torch.topk(logits[inside], k, largest=True).values[-1]
    return front & (logits >= thr)


# ---------------------------------------------------------------- neighbours
def center_ray_dirs(cameras):
    """World direction of the ray through pixel (W//2, H//2) (official `camera_center_rays`)."""
    dirs = []
    for c in cameras:
        dev = c["viewmat"].device
        x = torch.tensor([int(c["width"]) // 2], device=dev)
        y = torch.tensor([int(c["height"]) // 2], device=dev)
        dirs.append(pixel_rays(c, x, y)[0, 3:])
    return torch.stack(dirs)


def neighbour_mask(dirs, angle_deg):
    """Official `find_angles`: 0 < angle(center rays) < angle_deg."""
    n = dirs.norm(dim=1)
    cos = (dirs @ dirs.T) / torch.outer(n, n)
    ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    return (ang < angle_deg) & (ang > 0)
