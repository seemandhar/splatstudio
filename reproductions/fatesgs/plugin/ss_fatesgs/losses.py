"""FatesGS losses (paper Eq. 8, 9, 14, 16, 17), numerics of the official code
(train.py, utils/loss_utils.py, utils/feat_utils.py, commit a721d318)."""

import torch
import torch.nn.functional as F

from . import sidecar
from .raster import depths_to_points

PHOTO_DEFAULTS = {"lambda_dssim": 0.2}
REG_DEFAULTS = {"lambda_dist": 10000.0, "lambda_normal": 0.05, "dist_from": 3000,
                "normal_from": 7000}
RANK_DEFAULTS = {"weight": 10.0, "patch_size": 32, "margin": 1e-4,
                 "crops": [[576, 768], [544, 736]]}
SMOOTH_DEFAULTS = {"weight": 1.0, "edge_margin": 1e-2, "margin": 1e-4}
FEAT_DEFAULTS = {"weight": 1.5, "levels": [1, 2], "diff_threshold": 0.5, "visibility": "official"}


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return {**defaults, **cfg}


# ---------------------------------------------------------------- L_c (Eq. 17)
def _gaussian_window(size=11, sigma=1.5, device=None):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(3, 1, size, size).contiguous()


def ssim_same(img1, img2):
    """utils/loss_utils.ssim (3DGS): 11x11 Gaussian, padding = 5 ('same'), mean."""
    x, y = (t.permute(2, 0, 1)[None] for t in (img1, img2))
    w = _gaussian_window(device=x.device)
    mu1, mu2 = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    s11 = F.conv2d(x * x, w, padding=5, groups=3) - mu1**2
    s22 = F.conv2d(y * y, w, padding=5, groups=3) - mu2**2
    s12 = F.conv2d(x * y, w, padding=5, groups=3) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
            / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))).mean()


def create_photometric(spec):
    cfg = _check(spec.config, PHOTO_DEFAULTS, "photometric")
    lam = float(cfg["lambda_dssim"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        return (1 - lam) * (rgb - target).abs().mean() + lam * (1 - ssim_same(rgb, target))

    return loss


# ---------------------------------------------------------------- L_d, L_n (Eq. 17)
def create_surfel_reg(spec):
    cfg = _check(spec.config, REG_DEFAULTS, "surfel-reg")

    def loss(params, outputs, target):
        step = outputs["step"]
        ld = cfg["lambda_dist"] if step > cfg["dist_from"] else 0.0
        ln = cfg["lambda_normal"] if step > cfg["normal_from"] else 0.0
        # train.py: normal_error = 1 - (rend_normal * surf_normal).sum(channel); both unnormalised
        normal_err = 1 - (outputs["normal"] * outputs["surface_normal"]).sum(-1)
        return ld * outputs["distortion"].mean() + ln * normal_err.mean()

    return loss


# ---------------------------------------------------------------- L_r (Eq. 8)
def center_crop(x, h, w):
    """torchvision CenterCrop on [..., H, W] (top = int(round((H - h) / 2)))."""
    H, W = x.shape[-2:]
    top, left = int(round((H - h) / 2.0)), int(round((W - w) / 2.0))
    return x[..., top: top + h, left: left + w]


def patchify(img, p):
    """[H,W] -> [n_patches, p*p] (F.unfold, stride p, remainder cropped)."""
    return F.unfold(img[None, None], p, stride=p)[0].transpose(0, 1).contiguous()


def patched_rank(surf, mono, p, margin, perm=None):
    sp, mp = patchify(surf, p), patchify(mono, p)
    length = sp.shape[1] // 2 * 2
    if perm is None:
        perm = torch.randperm(length, device=surf.device)
    sp, mp = sp[:, perm], mp[:, perm]
    h = length // 2
    # relu( sgn(mono_a - mono_b) * (surf_b - surf_a) + m ), mean over pairs and patches
    return torch.clamp_min(torch.sign(mp[:, :h] - mp[:, h:]) * (sp[:, h:] - sp[:, :h]) + margin,
                           0).mean()


def depth_rank_loss(surf, mono, crops, p, margin, perms=None):
    total = 0.0
    for k, (h, w) in enumerate(crops):
        total = total + 0.5 * patched_rank(center_crop(surf, h, w), center_crop(mono, h, w), p,
                                           margin, None if perms is None else perms[k])
    return total


def create_rank(spec):
    cfg = _check(spec.config, RANK_DEFAULTS, "depth-rank")

    def loss(params, outputs, target):
        tab = sidecar.table()
        i = tab.index(outputs["camera"]["viewmat"])
        mono = tab.mono_depth(i, outputs["depth"].device)
        return cfg["weight"] * depth_rank_loss(outputs["depth"][..., 0], mono, cfg["crops"],
                                               int(cfg["patch_size"]), float(cfg["margin"]))

    return loss


# ---------------------------------------------------------------- L_s (Eq. 9)
def tv_loss(surf, mono, edge_margin, margin):
    """utils/loss_utils.TVLoss on [H,W] maps."""
    h = torch.clamp_min((surf[1:, :] - surf[:-1, :]).abs() - margin, 0) * (
        (mono[1:, :] - mono[:-1, :]).abs() < edge_margin).float()
    w = torch.clamp_min((surf[:, 1:] - surf[:, :-1]).abs() - margin, 0) * (
        (mono[:, 1:] - mono[:, :-1]).abs() < edge_margin).float()
    return h.mean() + w.mean()


def create_smooth(spec):
    cfg = _check(spec.config, SMOOTH_DEFAULTS, "depth-smooth")

    def loss(params, outputs, target):
        tab = sidecar.table()
        i = tab.index(outputs["camera"]["viewmat"])
        mono = tab.mono_depth(i, outputs["depth"].device)
        return cfg["weight"] * tv_loss(outputs["depth"][..., 0], mono, cfg["edge_margin"],
                                       cfg["margin"])

    return loss


# ---------------------------------------------------------------- L_f (Eq. 14-15)
def visibility(pts, src_center, uv):
    """Eq. 15 visibility exactly as the official get_feat_loss_corr computes it:
    sort points by distance to o_s, torch.unique(sorted_uv, dim=0) (which returns groups in
    lexicographic uv order), cumsum of the group counts as "first index of each group" in the
    distance-sorted list. NOTE: because unique(dim=0) sorts, these indices are not the nearest
    point of each pixel in general — this reproduces the code, not the paper's intent."""
    dists = torch.linalg.vector_norm(pts - src_center[None], dim=-1)
    _, order = torch.sort(dists)
    suv = uv[order]
    _, cnts = torch.unique(suv, sorted=False, return_counts=True, dim=0)
    cnts = torch.cat((torch.zeros(1, dtype=torch.long, device=uv.device), cnts))
    first = torch.cumsum(cnts, dim=0)[:-1]
    vis_sorted = torch.zeros_like(order)
    vis_sorted[first] = 1
    _, inv = torch.sort(order)
    return vis_sorted[inv].float()


def visibility_zbuffer(pts, src_center, uv):
    """The paper's Eq. 15 (true z-buffer: nearest point per rounded source pixel). Used only by
    the ``visibility: zbuffer`` option / probes."""
    dists = torch.linalg.vector_norm(pts - src_center[None], dim=-1)
    order = torch.argsort(dists, stable=True)
    _, inv, cnts = torch.unique(uv[order], return_inverse=True, return_counts=True, dim=0)
    first = torch.full((len(cnts),), len(order), device=uv.device, dtype=torch.long)
    first = first.scatter_reduce(0, inv, torch.arange(len(order), device=uv.device), "amin")
    vis_sorted = torch.zeros(len(order), device=uv.device)
    vis_sorted[first] = 1.0
    vis = torch.zeros_like(vis_sorted)
    vis[order] = vis_sorted
    return vis


def project(pts, viewmat, K):
    """World points [m,3] -> pixel coords [m,2] (official idx_world2cam / idx_cam2img)."""
    cam = pts @ viewmat[:3, :3].T + viewmat[:3, 3]
    img = cam @ K.T
    return img[:, :2] / (img[:, 2:3] + 1e-9)


def sample_feat(feat, grid_px):
    """normalize_for_grid_sample + bilinear grid_sample (align_corners False, zero padding)."""
    h, w = feat.shape[-2:]
    size = torch.tensor([w, h], dtype=grid_px.dtype, device=grid_px.device)
    g = (grid_px / size * 2 - 1).clamp(-1.1, 1.1)
    inr = ((g <= 1) & (g >= -1)).all(dim=-1)
    out = F.grid_sample(feat[None], g[None, :, None, :], mode="bilinear", padding_mode="zeros",
                        align_corners=False)[0, :, :, 0]
    return out, inr  # [C,m], [m]


def feature_level_loss(pts, ref, srcs, feat_ref, feat_srcs, scale, diff_threshold,
                       vis_fn=visibility):
    """get_feat_loss_corr for one reference view and one level (mean over sources and points)."""
    ref_uv = project(pts, ref["viewmat"], ref["K"])
    f_ref, in_ref = sample_feat(feat_ref, ref_uv / scale)
    n_ref = f_ref.norm(dim=0).clamp(min=1e-9)
    terms = []
    for src, f_src_map in zip(srcs, feat_srcs):
        uv = project(pts, src["viewmat"], src["K"])
        center = torch.linalg.inv(src["viewmat"])[:3, 3]
        vis = vis_fn(pts.detach(), center, uv.detach().round().long())
        f_src, in_src = sample_feat(f_src_map, uv / scale)
        corr = (f_ref * f_src).sum(0) / n_ref / f_src.norm(dim=0).clamp(min=1e-9)
        cl = (1 - corr).abs()
        valid = (in_ref & in_src).float()
        diff = (cl < diff_threshold).float()
        terms.append(cl * valid * diff * vis)
    return torch.stack(terms).mean()


def feature_loss(depth, camera, tab, levels, diff_threshold, vis_fn=visibility):
    i = tab.index(camera["viewmat"])
    feats = tab.features(depth.device)
    mask = depth.reshape(-1) > 0  # use_mask False: mask = surf_depth > 0
    if mask.sum() == 0:
        return depth.sum() * 0.0
    pts = depths_to_points(camera, depth)[mask]
    srcs = []
    for j in tab.pairs[i]:
        srcs.append({"viewmat": tab.viewmats[j].to(depth.device), "K": camera["K"].float()})
    ref = {"viewmat": camera["viewmat"].float(), "K": camera["K"].float()}
    total, wsum = 0.0, 0.0
    for scale in levels:
        lv = scale - 1
        f_ref = feats[lv][i]
        f_src = [feats[lv][j] for j in tab.pairs[i]]
        total = total + (1.0 / scale) * feature_level_loss(pts, ref, srcs, f_ref, f_src,
                                                           float(scale), diff_threshold, vis_fn)
        wsum += 1.0 / scale
    return total / wsum


def create_feature(spec):
    cfg = _check(spec.config, FEAT_DEFAULTS, "feature")
    vis_fn = {"official": visibility, "zbuffer": visibility_zbuffer}[cfg["visibility"]]

    def loss(params, outputs, target):
        tab = sidecar.table()
        return cfg["weight"] * feature_loss(outputs["depth"][..., 0], outputs["camera"], tab,
                                            [int(v) for v in cfg["levels"]],
                                            float(cfg["diff_threshold"]), vis_fn)

    return loss
