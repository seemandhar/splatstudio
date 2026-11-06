"""Image-space helpers of GS-IR: depth-derived normals, view directions, TV losses."""

import torch
import torch.nn.functional as F

DEPTH_THRESH = 0.01  # forward.cu depthmapToNormalCUDA


def median_blur3(depth):
    """kornia.filters.median_blur(depth, (3, 3)) on [H, W] (zero padding)."""
    h, w = depth.shape
    cols = F.unfold(depth[None, None], 3, padding=1)  # [1, 9, H*W]
    return cols.median(dim=1).values.reshape(h, w)


def normal_from_depth(depth, K, c2w_rot):
    """Port of depthmapToNormalCUDA on the median-filtered depth [H, W] -> world normals
    [H, W, 3]; zero where the 5x5 window has depth < 0.01 or within 2 px of the border."""
    d = median_blur3(depth)
    h, w = d.shape
    dev = d.device
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=d.dtype),
                            torch.arange(w, device=dev, dtype=d.dtype), indexing="ij")
    ray = torch.stack(((xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, torch.ones_like(xs)), -1)
    pos = ray * d[..., None]
    pad = F.pad(pos.permute(2, 0, 1), (1, 1, 1, 1), mode="replicate").permute(1, 2, 0)
    dpad = F.pad(d[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    cen = pos
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    dl, dr = dpad[1:-1, :-2], dpad[1:-1, 2:]
    du, dd = dpad[:-2, 1:-1], dpad[2:, 1:-1]
    ddx = torch.where(((dl - d).abs() < (dr - d).abs())[..., None], cen - left, right - cen)
    ddy = torch.where(((dd - d).abs() < (du - d).abs())[..., None], cen - down, up - cen)
    n = F.normalize(torch.cross(ddx, ddy, dim=-1), dim=-1, eps=1e-12)
    valid = (d >= DEPTH_THRESH).float()
    ok = -F.max_pool2d(-valid[None, None], 5, stride=1, padding=2)[0, 0] > 0.5
    border = torch.zeros_like(ok)
    border[2:-2, 2:-2] = True
    ok = ok & border
    world = n @ c2w_rot.T
    return torch.where(ok[..., None], world, torch.zeros_like(world))


def camera_rays(K, height, width, device):
    """Canonical (unnormalised) camera rays [H, W, 3] through pixel centres."""
    ys, xs = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                            torch.arange(width, device=device, dtype=torch.float32),
                            indexing="ij")
    return torch.stack(((xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1],
                        torch.ones_like(xs)), -1)


def view_geometry(camera, depth):
    """Official train.py: view_dirs = -(c2w_R normalize(ray)); points = cam + (-view_dirs) z."""
    viewmat = camera["viewmat"]
    c2w = torch.linalg.inv(viewmat)
    rays = F.normalize(camera_rays(camera["K"], camera["height"], camera["width"],
                                   viewmat.device), dim=-1)
    dirs = rays @ c2w[:3, :3].T
    view_dirs = -dirs
    points = dirs * depth + c2w[:3, 3]
    return view_dirs, points, c2w


def tv_loss(gt, pred):
    """train.py get_tv_loss (pad=1, step=1): gt [3, H, W], pred [C, H, W]."""
    wh = torch.exp(-(gt[:, 1:, :] - gt[:, :-1, :]).abs().mean(dim=0, keepdim=True))
    ww = torch.exp(-(gt[:, :, 1:] - gt[:, :, :-1]).abs().mean(dim=0, keepdim=True))
    th = (pred[:, 1:, :] - pred[:, :-1, :]).pow(2)
    tw = (pred[:, :, 1:] - pred[:, :, :-1]).pow(2)
    return (th * wh).mean() + (tw * ww).mean()


def masked_tv_loss(mask, gt, pred):
    """train.py get_masked_tv_loss (no erosion): mask [1, H, W]."""
    wh = torch.exp(-(gt[:, 1:, :] - gt[:, :-1, :]).abs().mean(dim=0, keepdim=True))
    ww = torch.exp(-(gt[:, :, 1:] - gt[:, :, :-1]).abs().mean(dim=0, keepdim=True))
    th = (pred[:, 1:, :] - pred[:, :-1, :]).pow(2)
    tw = (pred[:, :, 1:] - pred[:, :, :-1]).pow(2)
    m = mask.float()
    mh = m[:, 1:, :] * m[:, :-1, :]
    mw = m[:, :, 1:] * m[:, :, :-1]
    return (th * wh * mh).mean() + (tw * ww * mw).mean()
