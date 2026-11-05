"""Pure-torch TrimGS math (CPU-testable): flat Gaussians, culling box, N-way split,
contribution aggregation/trimming, depth normals and the normal regularization terms.

Official code: https://github.com/YuxueYang1204/TrimGS @ 6da2f212, `Trim3DGS/` (Inria 3DGS
licence, non-commercial research use). Line references are to that tree.
"""

import math

import torch
import torch.nn.functional as F
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- flat Gaussians (C0)
def flat_scales(log_scales, eps):
    """FlattenGaussianModel.get_scaling (gaussian_model.py:510): axis 0 fixed at eps."""
    return torch.cat(
        (torch.full_like(log_scales[:, :1], eps), log_scales[:, 1:].exp()), dim=1
    )


def rotmats(quats):
    return normalized_quat_to_rotmat(F.normalize(quats, dim=-1))


def gaussian_normals(quats):
    """FlattenGaussianModel.get_normal: first rotation column (the flat axis)."""
    return rotmats(quats)[..., 0]


@torch.no_grad()
def pin_flat_axis(params, eps):
    params["scales"].data[:, 0] = math.log(eps)


def pretrain_means_lr(lr0, step, pretrain_steps, final_ratio=0.01):
    """3DGS get_expon_lr_func(1.6e-4, 1.6e-6, max_steps=30000) for steps <= pretrain_steps;
    afterwards the fine-tune's constant lr (tune.py cmdline position_lr_init = final)."""
    return lr0 * final_ratio ** (min(step, pretrain_steps) / pretrain_steps)


# ---------------------------------------------------------------- scale-driven densify (C2)
def official_radii(conics):
    """3DGS preprocess radius: ceil(3 sqrt(lambda_max(cov2d))), cov2d = conic^-1 (incl. +0.3)."""
    a, b, c = conics.unbind(-1)
    det = (a * c - b * b).clamp_min(1e-30)
    ca, cb, cc = c / det, -b / det, a / det  # inverse of [[a,b],[b,c]]
    mid = 0.5 * (ca + cc)
    lam = mid + torch.sqrt((mid * mid - (ca * cc - cb * cb)).clamp_min(0.1))
    return torch.ceil(3.0 * torch.sqrt(lam))


def culling_mask(means, centers, expansion=2.0):
    """tune.py:culling. NOTE: the z bounds use span_x (official code, reproduced verbatim)."""
    lo, hi = centers.min(0).values, centers.max(0).values
    span = (hi - lo) * expansion
    c = centers.mean(0)
    sx, sy = span[0], span[1]
    return (
        (means[:, 0] > c[0] - sx / 2) & (means[:, 0] < c[0] + sx / 2)
        & (means[:, 1] > c[1] - sy / 2) & (means[:, 1] < c[1] + sy / 2)
        & (means[:, 2] > c[2] - sx / 2) & (means[:, 2] < c[2] + sx / 2)
    )


def scale_split_mask(scales_eff, max_radii, scene_mask, percent_dense, extent, scale_factor,
                     max_screen_size):
    """densify_and_scale_split(no_grad=True): (max scale > tau_s or radii > max_screen) & box."""
    m = scales_eff.max(dim=1).values > percent_dense * extent * scale_factor
    if max_screen_size:
        m = m | (max_radii > max_screen_size)
    return m & scene_mask


@torch.no_grad()
def split_n(params, optimizers, mask, n=3, eps=1e-8, generator=None):
    """N-way split of FlattenGaussianModel.densify_and_scale_split (gaussian_model.py:653):
    means mu + R (eps_i * s), eps ~ N(0, I) with the flat scale s (axis 0 = eps); scales
    s / (0.8 n); other fields copied; originals removed; new Adam moments zero (cat zeros)."""
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    if len(sel) == 0:
        return 0
    device = mask.device
    s = flat_scales(params["scales"][sel], eps)
    R = rotmats(params["quats"][sel])
    noise = torch.randn(n, len(sel), 3, device=device, generator=generator)
    samples = torch.einsum("nij,nj,bnj->bni", R, s, noise)  # [n, S, 3]

    def param_fn(name, p):
        if name == "means":
            new = (p[sel][None] + samples).reshape(-1, 3)
        elif name == "scales":
            new = torch.log(s / (0.8 * n)).repeat(n, 1)
            new[:, 0] = math.log(eps)
        else:
            new = p[sel].repeat(n, *[1] * (p.dim() - 1))
        return torch.nn.Parameter(torch.cat([p[rest], new]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], torch.zeros((n * len(sel), *v.shape[1:]), device=device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    return int(len(sel))


@torch.no_grad()
def remove(params, optimizers, mask):
    sel = torch.where(~mask)[0]

    def param_fn(name, p):
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return v[sel]

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    return int(mask.sum())


# ---------------------------------------------------------------- trimming (C1)
def normalized_contribution(contrib_sum, covered):
    """gaussian_renderer: transmittance_sum / (num_covered_pixels + 1e-6) (Eq. 5)."""
    return contrib_sum / (covered.float() + 1e-6)


def official_topk_insert(top_list, trans):
    """Literal replay of tune.py:prune_low_contribution_gaussians' "top-K" list.

    The list starts as K copies of the first view's value; a later view is inserted only when it
    exceeds top_list[0] (the current maximum), shifting the others down. So the list holds the
    last K record-breaking values (padded by view 0), not a true top-K; this is what the
    official code averages (Eq. 6 with V = these K entries)."""
    K = len(top_list)
    if top_list[0] is None:
        return [trans.clone() for _ in range(K)]
    m = trans > top_list[0]
    if m.any():
        for i in range(K - 1):
            top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
        top_list[0][m] = trans[m]
    return top_list


def trim_mask(contribution, ratio):
    """prune contribution < quantile(contribution, ratio) (tune.py:72-74)."""
    if contribution.numel() <= 16_000_000:
        tile = torch.quantile(contribution, ratio)
    else:  # torch.quantile input limit; same linear interpolation via numpy
        import numpy as np

        tile = torch.tensor(np.quantile(contribution.cpu().numpy(), ratio)).to(contribution)
    return contribution < tile


# ---------------------------------------------------------------- normal regularization (C3)
def sobel_mask(depth, thresh=0.05, dilation=1):
    """utils/image_utils.py:compute_gradient (Sobel / 3, replicate pad) + min-pool erosion."""
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=depth.device,
                      dtype=depth.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    d = F.pad(depth[None, None], (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(d, kx) / 3, F.conv2d(d, ky) / 3
    w = (torch.sqrt(gx**2 + gy**2) < thresh).float()
    if dilation > 0:
        w = -F.max_pool2d(-w, dilation * 2 + 1, stride=1, padding=dilation)
    return w[0, 0]


def backproject(depth, K, c2w):
    """World points of a camera-z depth map [H,W] at pixel centres (+0.5)."""
    H, W = depth.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype) + 0.5,
        torch.arange(W, device=depth.device, dtype=depth.dtype) + 0.5,
        indexing="ij",
    )
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = torch.stack((x, y, depth), dim=-1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def depth_normals(points, cam_center, dilation=2):
    """tune.py:normal_regularization: two cross products on a replicate-padded point map,
    flipped toward the camera. points [H,W,3] -> normals [H,W,3]."""
    p = F.pad(points.permute(2, 0, 1)[None], (dilation,) * 4, mode="replicate")
    d, d2 = dilation, 2 * dilation
    vec1 = p[:, :, d2:, d2:] - p[:, :, :-d2, :-d2]
    vec2 = p[:, :, :-d2, d2:] - p[:, :, d2:, :-d2]
    n1 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
    vec1 = p[:, :, d:-d, d2:] - p[:, :, d:-d, :-d2]
    vec2 = p[:, :, :-d2, d:-d] - p[:, :, d2:, d:-d]
    n2 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
    n = F.normalize(n1 + n2, p=2, dim=0)  # [3,H,W]
    to_cam = F.normalize(cam_center.view(3, 1, 1) - points.permute(2, 0, 1)[None], p=2, dim=1)[0]
    n = n * torch.sign((n * to_cam).sum(0, keepdim=True))
    return n.permute(1, 2, 0)


def face_camera(normals, means, cam_center):
    to_cam = F.normalize(cam_center[None] - means, p=2, dim=1)
    return normals * torch.sign((normals * to_cam).sum(1, keepdim=True))


def normal_map_loss(rendered_normal, depth_normal, mask):
    """mean(|(N_G+1)/2 - (N_D+1)/2| * M) over 3 x H x W (Eq. 8, code form)."""
    pred = F.normalize(rendered_normal, p=2, dim=-1)
    return (torch.abs((pred + 1) / 2 - (depth_normal + 1) / 2) * mask[..., None]).mean()


def smooth_loss(gs_normal, valid, knn_idx):
    """kNN normal smoothness of tune.py:normal_regularization (sign detached)."""
    vn = gs_normal[valid]
    nb = gs_normal[knn_idx[valid]]  # [V,K,3]
    nb = nb * torch.sign((nb * vn[:, None]).sum(-1, keepdim=True)).detach()
    mean = (F.normalize(nb.mean(1), p=2, dim=-1) + 1) / 2
    return torch.abs((vn + 1) / 2 - mean).mean()


def knn_indices(points, k=16):
    """Exact k nearest neighbours (self included, like pytorch3d.knn_points) via scipy."""
    from scipy.spatial import cKDTree

    pts = points.detach().float().cpu().numpy()
    _, idx = cKDTree(pts).query(pts, k=k, workers=-1)
    return torch.from_numpy(idx.astype("int64")).to(points.device)
