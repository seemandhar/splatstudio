"""DET-GS losses (arXiv 2508.04099 §III). Numerics the paper leaves open follow DNGaussian's
official utils/loss_utils.py (same Global-Local normalisation): see paper.md §3."""

import random

import torch
import torch.nn.functional as F

from . import priors
from .renderer import hard_depth

DEPTH_DEFAULTS = {"gamma": 0.1, "eta": 1.0, "every": 5, "patch_min": 5, "patch_max": 20,
                  "omega": 0.95, "error_tolerance": 0.2, "space": "depth", "seed": 0}
EDGE_DEFAULTS = {"beta": 0.1, "eps": 1e-8}
TV_DEFAULTS = {"phi": 0.8, "tau_edge": 1e-2, "tau_smooth": 1e-4}


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return {**defaults, **cfg}


# ------------------------------------------------------------------ C1 (Eq. 5-7)
def patchify(x, p):
    """[1,1,H,W] -> [n_patches, p*p], non-overlapping (remainder cropped)."""
    return F.unfold(x, kernel_size=p, stride=p).permute(0, 2, 1).reshape(-1, p * p)


def normalize(x, std=None):
    """Eq. 5 (std=None: patch std) / Eq. 6 (std = whole-image std). delta = 1e-2 * std(x)
    (DNGaussian normalize())."""
    mean = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True) if std is None else std
    return (x - mean) / (s + 1e-2 * x.reshape(-1).std())


def margin_l2(a, b, margin):
    """Error-tolerant L2 (§IV-B): mean squared error over entries with |a - b| > margin."""
    diff = a - b
    sel = diff.abs() > margin
    if not bool(sel.any()):
        return diff.sum() * 0.0
    return diff[sel].square().mean()


def patch_loss(depth, target, p, margin):
    return margin_l2(normalize(patchify(depth, p)), normalize(patchify(target, p)), margin)


def image_loss(depth, target, p, margin):
    return margin_l2(normalize(patchify(depth, p), std=depth.std().detach()),
                     normalize(patchify(target, p), std=target.std().detach()), margin)


def hierarchical_depth_loss(depth, target, p_local, p_global, gamma, eta, margin):
    """Eq. 7. depth, target: [1,1,H,W]."""
    return gamma * patch_loss(depth, target, p_local, margin) + eta * image_loss(
        depth, target, p_global, margin)


def mono_target(disparity):
    """Depth Anything V2 predicts affine-invariant disparity; the affine-invariant normalisation
    needs a depth-ordered map, so the sign is flipped (DNGaussian uses 255 - disparity)."""
    return -disparity


def to_space(depth, disparity, space, eps=1e-6):
    """(rendered, target) in the supervision space. "depth" (default, DNGaussian-consistent):
    rendered z vs -disparity. "disparity": 1/z vs disparity (official-3DGS depth-regularisation style).
    Not stated by the paper (fidelity gap)."""
    if space == "depth":
        return depth, mono_target(disparity)
    if space == "disparity":
        return 1.0 / depth.clamp_min(eps), disparity
    raise ValueError(f"det-gs-depth: unknown space {space!r}")


def create_depth(spec):
    cfg = _check(spec.config, DEPTH_DEFAULTS, "det-gs-depth")
    space = str(cfg["space"])
    to_space(torch.ones(1), torch.ones(1), space)  # validate early
    every, lo, hi = int(cfg["every"]), int(cfg["patch_min"]), int(cfg["patch_max"])
    rng = random.Random(int(cfg["seed"]))

    def loss(params, outputs, target):
        step = int(outputs["step"])
        if step % every != 0:
            return params["means"].sum() * 0.0
        cam = outputs["camera"]
        disp, _ = priors.table().lookup(cam["viewmat"], params["means"].device)
        d, t = to_space(hard_depth(params, cam, cfg["omega"]), disp[None, None], space)
        return hierarchical_depth_loss(d, t, rng.randint(lo, hi), rng.randint(lo, hi),
                                       float(cfg["gamma"]), float(cfg["eta"]),
                                       float(cfg["error_tolerance"]))

    return loss


# ------------------------------------------------------------------ C2 (Eq. 8-11)
CROSS = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])


def edge_aware_depth_loss(depth, edges, eps=1e-8):
    """depth [H,W] rendered depth, edges [H,W] bool (Canny). Eq. 10-11."""
    m = (~edges).to(depth.dtype)[None, None]
    d = depth[None, None]
    k = CROSS.to(depth)[None, None]
    mean = F.conv2d(d * m, k, padding=1) / (F.conv2d(m, k, padding=1) + eps)  # Eq. 10
    num = (m * (d - mean).square()).sum()
    return num / m.sum().clamp_min(1.0)  # Eq. 11, P = number of non-edge pixels


def create_edge(spec):
    cfg = _check(spec.config, EDGE_DEFAULTS, "det-gs-edge")
    beta, eps = float(cfg["beta"]), float(cfg["eps"])

    def loss(params, outputs, target):
        _, edges = priors.table().lookup(outputs["camera"]["viewmat"], outputs["rgb"].device)
        depth = (outputs["depth"] * outputs["alpha"])[..., 0]  # Eq. 3 (not alpha-normalised)
        return beta * edge_aware_depth_loss(depth, edges, eps)

    return loss


# ------------------------------------------------------------------ C3 (Eq. 12-15)
def edge_preserving_tv(pred, gt, tau_edge=1e-2, tau_smooth=1e-4):
    """pred, gt: [H,W,C]. Eq. 13-15; mean over the pixel grid and channels."""
    gh_p, gh_g = pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1]
    gv_p, gv_g = pred[1:] - pred[:-1], gt[1:] - gt[:-1]
    mh, mv = (gh_g.abs() < tau_edge).to(pred.dtype), (gv_g.abs() < tau_edge).to(pred.dtype)
    th = (mh * (gh_p.abs() - tau_smooth).clamp_min(0)).sum()
    tv = (mv * (gv_p.abs() - tau_smooth).clamp_min(0)).sum()
    return (th + tv) / pred.numel()


def create_tv(spec):
    cfg = _check(spec.config, TV_DEFAULTS, "det-gs-tv")
    phi, te, ts = float(cfg["phi"]), float(cfg["tau_edge"]), float(cfg["tau_smooth"])

    def loss(params, outputs, target):
        return phi * edge_preserving_tv(outputs["rgb"], target, te, ts)

    return loss
