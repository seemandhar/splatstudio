"""Pure multi-view geometry of PGSR (official train.py multi-view block, utils/graphics_utils,
scene/gaussian_model.get_points_from_depth / get_points_depth_in_depth_map, loss_utils.lncc,
get_img_grad_weight, scene/__init__ neighbour graph). OpenCV w2c ``viewmat`` convention.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .raster import camera_rays, intrinsics


def gray(image_hwc):
    """process_image: 0.299 R + 0.587 G + 0.114 B. [H,W]."""
    return 0.299 * image_hwc[..., 0] + 0.587 * image_hwc[..., 1] + 0.114 * image_hwc[..., 2]


def image_grad_weight(image_hwc):
    """get_img_grad_weight: max over x/y of channel-mean central differences, min-max
    normalised to [0,1], border padded with 1. Returns [H,W]."""
    img = image_hwc.permute(2, 0, 1)
    _, hd, wd = img.shape
    bottom, top = img[..., 2:hd, 1:wd - 1], img[..., 0:hd - 2, 1:wd - 1]
    right, left = img[..., 1:hd - 1, 2:wd], img[..., 1:hd - 1, 0:wd - 2]
    gx = torch.mean(torch.abs(right - left), 0, keepdim=True)
    gy = torch.mean(torch.abs(top - bottom), 0, keepdim=True)
    g, _ = torch.max(torch.cat((gx, gy), dim=0), dim=0)
    g = (g - g.min()) / (g.max() - g.min())
    return F.pad(g[None, None], (1, 1, 1, 1), mode="constant", value=1.0).squeeze()


def neighbour_graph(viewmats, max_num=8, max_angle=30.0, min_dis=0.01, max_dis=1.5):
    """scene/__init__.py: per training view, other views sorted by np.lexsort((angles, diss))
    (primary key distance, ties by angle), kept if angle < max_angle deg and
    min_dis < distance < max_dis, first ``max_num``."""
    c2w = torch.linalg.inv(viewmats.float())
    centers = c2w[:, :3, 3]
    rays = F.normalize(c2w[:, :3, 2], dim=-1)  # [0,0,1] @ R^T with R = c2w rotation
    diss = torch.norm(centers[:, None] - centers[None], dim=-1).cpu().numpy()
    tmp = torch.sum(rays[:, None] * rays[None], dim=-1)
    angles = (torch.arccos(tmp) * 180 / 3.14159).cpu().numpy()
    out = []
    for i in range(len(viewmats)):
        order = np.lexsort((angles[i], diss[i]))
        keep = ((angles[i][order] < max_angle) & (diss[i][order] > min_dis)
                & (diss[i][order] < max_dis))
        out.append([int(j) for j in order[keep][:max_num]])
    return out


def _pixels(height, width, device):
    iy, ix = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    return torch.stack([ix, iy], dim=-1).float()  # [H,W,2] (x, y)


def depth_to_world(camera, depth):
    """get_points_from_depth: rays (integer pixels, Cx = W/2) * z-depth -> world [H*W,3]."""
    pts = (camera_rays(camera)[: depth.shape[0], : depth.shape[1]] * depth[..., None])
    c2w = torch.linalg.inv(camera["viewmat"].float())
    return pts.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]


def to_cam(camera, pts):
    vm = camera["viewmat"].float()
    return pts @ vm[:3, :3].T + vm[:3, 3]


def project(camera, pts_cam):
    K = intrinsics(camera)
    return torch.stack([pts_cam[:, 0] * K[0, 0] / pts_cam[:, 2] + K[0, 2],
                        pts_cam[:, 1] * K[1, 1] / pts_cam[:, 2] + K[1, 2]], -1)


def sample_depth(camera, depth, pts_cam):
    """get_points_depth_in_depth_map (scale 1): bilinear, align_corners, border padding."""
    W, H = int(camera["width"]), int(camera["height"])
    proj = project(camera, pts_cam)
    mask = ((proj[:, 0] > 0) & (proj[:, 0] < W) & (proj[:, 1] > 0) & (proj[:, 1] < H)
            & (pts_cam[:, 2] > 0.1))
    grid = proj.clone()
    grid[..., 0] /= (W - 1) / 2
    grid[..., 1] /= (H - 1) / 2
    grid = (grid - 1).view(1, -1, 1, 2)
    z = F.grid_sample(depth[None, None], grid, mode="bilinear", padding_mode="border",
                      align_corners=True)[0, :, :, 0]
    return z, mask


def forward_backward_error(cam_r, depth_r, cam_n, depth_n):
    """Official geometric consistency: p_r -> X (ref plane depth) -> neighbour camera ->
    depth from the neighbour plane depth map -> X' -> reprojected p_r'. Returns
    (pixel_noise [H*W], in-view mask [H*W])."""
    H, W = depth_r.shape
    pixels = _pixels(H, W, depth_r.device).reshape(-1, 2)
    pts = depth_to_world(cam_r, depth_r)
    pts_n = to_cam(cam_n, pts)
    map_z, mask = sample_depth(cam_n, depth_n, pts_n)
    pts_n = pts_n / pts_n[:, 2:3] * map_z.squeeze()[..., None]
    c2w_n = torch.linalg.inv(cam_n["viewmat"].float())
    pts_back = pts_n @ c2w_n[:3, :3].T + c2w_n[:3, 3]
    proj = project(cam_r, to_cam(cam_r, pts_back)).float()
    return torch.norm(proj - pixels, dim=-1), mask


def occlusion_weights(noise, mask, threshold=1.0, occlusion_aware=True):
    """w = 1/exp(phi) if phi < threshold (and in view) else 0; detached."""
    if occlusion_aware:
        mask = mask & (noise < threshold)
        w = (1.0 / torch.exp(noise)).detach()
    else:
        w = torch.ones_like(noise)
    w = torch.where(mask, w, torch.zeros_like(w))
    return w, mask


def relative_pose(cam_r, cam_n):
    """ref -> neighbour: R = R_n R_r^T, t = -R t_r + t_n (w2c)."""
    vr, vn = cam_r["viewmat"].float(), cam_n["viewmat"].float()
    R = vn[:3, :3] @ vr[:3, :3].T
    t = -R @ vr[:3, 3] + vn[:3, 3]
    return R, t


def homography(cam_r, cam_n, normal, distance, ncc_scale=1.0):
    """H_rn = K_n (R_rn - T_rn n_r^T / d_r) K_r^{-1} per pixel. normal [B,3], distance [B]."""
    R, t = relative_pose(cam_r, cam_n)
    B = distance.shape[0]
    Hm = R[None] - torch.matmul(t[None, :, None].expand(B, 3, 1),
                                normal[:, :, None].permute(0, 2, 1)) / distance[..., None, None]
    Kn = intrinsics(cam_n, ncc_scale)
    Kr_inv = torch.linalg.inv(intrinsics(cam_r, ncc_scale))
    return torch.matmul(Kn[None].expand(B, 3, 3), Hm) @ Kr_inv


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing="xy")[::-1], dim=-1).view(1, -1, 2)


def patch_warp(Hm, uv):
    B, P = uv.shape[:2]
    Hm = Hm.view(B, 3, 3)
    homo = torch.cat((uv, torch.ones((B, P, 1), device=uv.device)), dim=-1)
    g = torch.einsum("bik,bpk->bpi", Hm, homo).reshape(B, P, 3)
    return g[..., :2] / (g[..., 2:] + 1e-10)


def sample_gray(gray_hw, pix):
    """grid_sample(align_corners=True, zero padding) at pixel coordinates [B,P,2]."""
    H, W = gray_hw.shape
    g = pix.clone()
    g[..., 0] = 2 * g[..., 0] / (W - 1) - 1.0
    g[..., 1] = 2 * g[..., 1] / (H - 1) - 1.0
    out = F.grid_sample(gray_hw[None, None], g.reshape(1, -1, 1, 2), align_corners=True)
    return out.reshape(pix.shape[0], pix.shape[1])


def lncc(ref, nea):
    """loss_utils.lncc: 1 - cross^2 / (var_r var_n + 1e-8), clamped [0,2]; mask ncc < 0.9."""
    bs, tps = nea.shape
    ps = int(np.sqrt(tps))
    ref_nea = (ref * nea).view(bs, 1, ps, ps)
    ref = ref.view(bs, 1, ps, ps)
    nea = nea.view(bs, 1, ps, ps)
    filters = torch.ones(1, 1, ps, ps, device=ref.device)
    pad = ps // 2

    def box(t):
        return F.conv2d(t, filters, stride=1, padding=pad)[:, :, pad, pad]

    ref_sum, nea_sum = box(ref), box(nea)
    ref2_sum, nea2_sum, ref_nea_sum = box(ref.pow(2)), box(nea.pow(2)), box(ref_nea)
    ref_avg, nea_avg = ref_sum / tps, nea_sum / tps
    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum
    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = torch.clamp(1 - cc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    return ncc, ncc < 0.9


def ncc_terms(cam_r, cam_n, gray_r, gray_n, normal, distance, pixels, weights, patch_size=3,
              ncc_scale=1.0):
    """Official photometric block for sampled reference pixels [B,2] (integer coords).

    normal [B,3] / distance [B]: rendered (camera-frame) plane parameters at those pixels
    (carry gradients). Returns (weighted ncc of kept pixels [K], keep mask [B])."""
    with torch.no_grad():
        offsets = patch_offsets(patch_size, pixels.device)
        ori = pixels.reshape(-1, 1, 2) / ncc_scale + offsets.float()
        ref_val = sample_gray(gray_r, ori)
    Hm = homography(cam_r, cam_n, normal, distance, ncc_scale)
    grid = patch_warp(Hm.reshape(-1, 3, 3), ori)
    nea_val = sample_gray(gray_n, grid)
    ncc, mask = lncc(ref_val, nea_val)
    mask = mask.reshape(-1)
    ncc = ncc.reshape(-1) * weights
    return ncc[mask], mask
