"""Pure functions of MVGS (arXiv 2410.02103), matched to the official code
https://github.com/xiaobiaodu/MVGS @ 36d073a (train.py, scene/gaussian_model.py, run_360.py,
utils/camera_utils.py). Everything here is stateless and unit-tested on CPU.
"""

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------
# Cross-intrinsic guidance (Sec. 3.2, Eq. 9; run_360.py)
# --------------------------------------------------------------------------------------------


def check_stages(stages):
    """stages: sequence of (views M, official -r flag, iterations). -r -1 = training resolution."""
    out = []
    for stage in stages:
        if len(stage) != 3:
            raise ValueError("each stage is [views, resolution_flag, iterations]")
        m, r, n = int(stage[0]), float(stage[1]), int(stage[2])
        if m < 1 or n < 1 or not (r == -1 or r > 0):
            raise ValueError(f"invalid stage {stage}")
        out.append((m, r, n))
    if not out:
        raise ValueError("at least one stage is required")
    return tuple(out)


def total_steps(stages):
    return sum(n for _, _, n in stages)


def stage_at(stages, step):
    """1-based global step -> (stage index, 1-based stage-local iteration).

    Every official stage is a separate train.py run restored from the previous checkpoint with
    `first_iter = 0`, so its iteration counter restarts at 1.
    """
    if step < 1:
        raise ValueError("steps are 1-based")
    start = 0
    for k, (_, _, n) in enumerate(stages):
        if step <= start + n:
            return k, step - start
        start += n
    raise RuntimeError(
        f"step {step} is beyond the MVGS stage schedule ({total_steps(stages)} steps); "
        "run ss-train with --steps equal to the schedule length"
    )


def stage_size(width, height, flag, full_width_ratio):
    """Render/target size of a stage.

    The official loader resizes the full-resolution image to (round(W_full / r), round(H_full / r))
    for r in {1, 2, 4, 8} and to width 1600 for r = -1 (utils/camera_utils.py:loadCam). Our
    training images are the r = -1 images, so W_full = W * full_width_ratio.
    """
    if flag == -1:
        return int(width), int(height)
    return int(round(width * full_width_ratio / flag)), int(round(height * full_width_ratio / flag))


def scaled_camera(camera, width, height):
    """Same field of view at a new pixel size (Eq. 9: f/s, c/s), as a resize with unchanged FoV."""
    if width == camera["width"] and height == camera["height"]:
        return camera
    K = camera["K"].clone()
    K[0] *= width / camera["width"]
    K[1] *= height / camera["height"]
    return {**camera, "K": K, "width": int(width), "height": int(height)}


def resize_image(image, width, height):
    """HWC float image -> (height, width), antialiased bicubic (PIL-like), clamped to [0, 1]."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    x = image.permute(2, 0, 1)[None]
    y = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return y[0].permute(1, 2, 0).clamp(0, 1)


# --------------------------------------------------------------------------------------------
# Multi-view augmented densification (Sec. 3.3.1, Eq. 10; train.py:135-148)
# --------------------------------------------------------------------------------------------


def camera_center(viewmat):
    return torch.linalg.inv(viewmat)[:3, 3]


def mvad_threshold(centers, beta, tau=1.0):
    """beta/2 if any pair of unit-normalised camera centres is farther apart than tau, else beta.

    Official: camera_t = C / ||C||; diffs over all ordered pairs i != j; `torch.any(diffs > 1)`.
    """
    centers = torch.as_tensor(centers, dtype=torch.float32)
    if len(centers) < 2:
        return float(beta)
    t = centers / torch.linalg.vector_norm(centers, dim=-1, keepdim=True)
    d = torch.cdist(t[None], t[None])[0]
    off = ~torch.eye(len(t), dtype=torch.bool, device=t.device)
    return float(beta) * 0.5 if bool((d[off] > tau).any()) else float(beta)


# --------------------------------------------------------------------------------------------
# Cross-ray densification (Sec. 3.3.2, supp. Alg. 1; train.py:151-176, gaussian_model.py:376-510)
# --------------------------------------------------------------------------------------------


def max_loss_window(l1map, h_box=20, w_box=40):
    """Window of largest mean L1 (official sliding loop). l1map: HWC.

    Window starts: range(0, H - h_box, h_box) x range(0, W - w_box, w_box); first strictly
    greater maximum with initial maximum 0. Returns (loss, [row, col, row + h_box, col + w_box]).
    """
    h, w = l1map.shape[:2]
    ni, nj = len(range(0, h - h_box, h_box)), len(range(0, w - w_box, w_box))
    if ni == 0 or nj == 0:
        return 0.0, [0, 0, h_box, w_box]
    gray = l1map.mean(dim=-1)[None, None]
    pooled = F.avg_pool2d(gray, (h_box, w_box), stride=(h_box, w_box))[0, 0, :ni, :nj]
    best = float(pooled.max())
    if not best > 0:
        return 0.0, [0, 0, h_box, w_box]
    flat = int(torch.nonzero(pooled.flatten() == pooled.max())[0])
    i, j = (flat // nj) * h_box, (flat % nj) * w_box
    return best, [i, j, i + h_box, j + w_box]


def pixel_rays(camera, rows, cols):
    """World rays through pixel centres (x + 0.5, y + 0.5): official set_rays_od (pix2ndc).

    Returns origins, unit directions [..., 3]. OpenCV camera (x right, y down, z forward).
    """
    K, viewmat = camera["K"], camera["viewmat"]
    rows = torch.as_tensor(rows, dtype=K.dtype, device=K.device)
    cols = torch.as_tensor(cols, dtype=K.dtype, device=K.device)
    x = (cols + 0.5 - K[0, 2]) / K[0, 0]
    y = (rows + 0.5 - K[1, 2]) / K[1, 1]
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    c2w = torch.linalg.inv(viewmat)
    d = d_cam @ c2w[:3, :3].T
    d = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    return c2w[:3, 3].expand_as(d), d


def corner_rays(camera, box):
    """Rays at the official corner order: top-left, bottom-right, bottom-left, top-right.

    Official indexing ray[0, :, box[0], box[1]] etc., i.e. (row, col) with box = [r0, c0, r1, c1].
    """
    r0, c0, r1, c1 = box
    return pixel_rays(camera, [r0, r1, r1, r0], [c0, c1, c0, c1])


def intersect_lines(o1, d1, o2, d2, eps=1e-6):
    """Midpoint of the closest points of two lines (official intersect_lines), batched.

    Returns points [..., 3] and a validity mask (False where ||d1 x d2|| < eps: parallel).
    """
    d1 = d1 / torch.linalg.vector_norm(d1, dim=-1, keepdim=True)
    d2 = d2 / torch.linalg.vector_norm(d2, dim=-1, keepdim=True)
    cross = torch.linalg.cross(d1, d2, dim=-1)
    n2 = (cross * cross).sum(-1)
    valid = n2.sqrt() >= eps
    diff = o2 - o1
    t1 = (torch.linalg.cross(diff, d2, dim=-1) * cross).sum(-1) / n2.clamp_min(1e-30)
    t2 = (torch.linalg.cross(diff, d1, dim=-1) * cross).sum(-1) / n2.clamp_min(1e-30)
    p = 0.5 * ((o1 + t1[..., None] * d1) + (o2 + t2[..., None] * d2))
    return p, valid


def sort_views(losses):
    """Indices sorted by window loss, descending, stable (Python `sorted(..., reverse=True)`)."""
    return sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)


def cross_ray_boxes(cameras, boxes, losses, box_index="official"):
    """Axis-aligned 3D cuboids [K, 6] (xmin, ymin, zmin, xmax, ymax, zmax) from view pairs.

    box_index="official" reproduces gaussian_model.py:470 `box1 = boxes[j]` where j enumerates
    `cams[i+1:]` (the second camera's rays use the box of sorted view j); "paper" uses the box of
    the second camera itself (sorted view i+1+j).
    """
    if box_index not in ("official", "paper"):
        raise ValueError("box_index must be 'official' or 'paper'")
    order = sort_views(losses)
    cams = [cameras[i] for i in order]
    bxs = [boxes[i] for i in order]
    out = []
    for i in range(len(cams)):
        for j in range(len(cams) - i - 1):
            k = i + 1 + j
            o0, d0 = corner_rays(cams[i], bxs[i])
            o1, d1 = corner_rays(cams[k], bxs[j] if box_index == "official" else bxs[k])
            p, valid = intersect_lines(o0, d0, o1, d1)
            if not bool(valid.all()):
                continue
            out.append(torch.cat([p.min(dim=0).values, p.max(dim=0).values]))
    if not out:
        return torch.zeros(0, 6)
    return torch.stack(out)


def in_boxes(means, boxes, chunk=16):
    """Gaussians strictly inside any cuboid (official densify_and_clone mask)."""
    mask = torch.zeros(len(means), dtype=torch.bool, device=means.device)
    boxes = boxes.to(means.device, means.dtype)
    for s in range(0, len(boxes), chunk):
        b = boxes[s : s + chunk]
        inside = (means[None] > b[:, None, :3]) & (means[None] < b[:, None, 3:])
        mask |= inside.all(dim=-1).any(dim=0)
    return mask
