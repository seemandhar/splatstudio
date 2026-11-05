"""AGS-Mesh-specific probes: DNC prior filtering, depth switch, ANR normal masking, surface normals."""

import math

import numpy as np
import torch

from .dnc import OPENGL_TO_OPENCV, dnc_frame, knn_normals


def _component(method, name):
    from splatstudio.registry import resolve

    spec = next(c for c in method.components if c.name == name)
    return spec, resolve(spec.factory)(spec)


# ----------------------------------------------------------------------------- DNC (C1)
def _plane_depth(n_cam, c, H, W, f, cx, cy):
    """z-depth of the plane n . X = c seen through pixel centres (+0.5)."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5, indexing="xy")
    rays = np.stack(((u - cx) / f, (v - cy) / f, np.ones_like(u)), -1)
    return c / (rays @ n_cam)


def _encode(n):
    return np.clip(np.round((n + 1) / 2 * 255), 0, 255).astype(np.uint8)


def dnc_filter(method, device):
    """Synthetic frame: a slanted wall (consistent prior), a block whose sensor depth is corrupted
    (a different plane, like a LiDAR edge/far-field error), and a block whose monocular prior is
    wrong by 35 deg. DNC must keep the consistent wall and reject both bad blocks; KNN normals
    must equal an independent numpy PCA (smallest-eigenvalue eigenvector)."""
    H, W, f = 64, 80, 60.0
    cx, cy = W / 2, H / 2
    n_wall = np.array([0.25, -0.15, -1.0])
    n_wall /= np.linalg.norm(n_wall)  # camera frame (OpenCV), facing the camera (-z)
    depth = _plane_depth(n_wall, -2.0 * 1.0, H, W, f, cx, cy)
    n_bad = np.array([-0.6, 0.4, -1.0])
    n_bad /= np.linalg.norm(n_bad)
    d_bad = _plane_depth(n_bad, -2.0, H, W, f, cx, cy)
    blk_d = (slice(8, 32), slice(8, 32))
    depth[blk_d] = d_bad[blk_d]
    prior = np.broadcast_to(n_wall, (H, W, 3)).copy()
    ang = math.radians(35.0)
    rot = np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]])
    blk_n = (slice(32, 56), slice(48, 72))
    prior[blk_n] = (rot @ n_wall)
    # camera at the world origin: OpenGL c2w whose OpenCV version is the identity
    c2w_gl = np.linalg.inv(OPENGL_TO_OPENCV)
    mask, dbg = dnc_frame(depth.astype(np.float32), _encode(prior), c2w_gl, (f, f, cx, cy),
                          knn=200, tau=10.0, return_debug=True)
    conf = 1.0 - mask.astype(np.float64) / 255.0 > 0  # plugin confidence rule
    inner_d = (slice(14, 26), slice(14, 26))
    inner_n = (slice(38, 50), slice(54, 66))
    clean = np.ones((H, W), bool)
    clean[2:38, 2:38] = False  # corrupted block + KNN reach around it
    clean[26:62, 42:78] = False
    clean[:4], clean[-4:], clean[:, :4], clean[:, -4:] = False, False, False, False
    kept_clean = float(conf[clean].mean())
    rej_d = float((~conf[inner_d]).mean())
    rej_n = float((~conf[inner_n]).mean())
    if kept_clean < 0.97:
        raise AssertionError(f"DNC rejects consistent wall pixels ({kept_clean:.3f} kept)")
    if rej_d < 0.9 or rej_n < 0.9:
        raise AssertionError(f"DNC keeps unreliable priors (depth {rej_d:.3f}, normal {rej_n:.3f})")
    # independent PCA normal for a few points vs Open3D KNN normals (up to sign)
    from .dnc import backproject

    pts = backproject(depth.astype(np.float32), f, f, cx, cy, np.eye(4))
    o3d_n = knn_normals(pts, 200)
    rng = np.random.default_rng(0)
    worst = 1.0
    for i in rng.choice(len(pts), 12, replace=False):
        d2 = ((pts - pts[i]) ** 2).sum(1)
        nb = pts[np.argsort(d2, kind="stable")[:200]].astype(np.float64)
        w, v = np.linalg.eigh(np.cov((nb - nb.mean(0)).T))
        worst = min(worst, abs(float(v[:, 0] @ o3d_n[i])))
    if worst < 0.999:
        raise AssertionError(f"KNN normals differ from the smallest-eigenvector PCA ({worst})")
    # orientation: depth normals must face the camera (flip against the ray)
    nd = dbg["normal_depth"]
    if (nd[clean] @ np.array([0.0, 0, -1.0]) <= 0).any():
        raise AssertionError("depth normals not flipped towards the camera")
    return {"kept_consistent": kept_clean, "rejected_bad_depth": rej_d,
            "rejected_bad_prior": rej_n, "pca_abs_cos_min": worst,
            "rejected_total": float((~conf).mean())}


# ------------------------------------------------------------------------- depth (C1)
def _edge_aware_loop(p, d, img, valid):
    H, W = p.shape
    lx, ly = [], []
    for i in range(H):
        for j in range(W):
            if not valid[i, j]:
                continue
            ll = math.log(1 + abs(float(p[i, j] - d[i, j])))
            if j < W - 1:
                lx.append(math.exp(-float((img[i, j] - img[i, j + 1]).abs().mean())) * ll)
            if i < H - 1:
                ly.append(math.exp(-float((img[i, j] - img[i + 1, j]).abs().mean())) * ll)
    return sum(lx) / len(lx) + sum(ly) / len(ly)


def depth_schedule(method, device):
    spec, fn = _component(method, "ags-depth")
    g = torch.Generator(device="cpu").manual_seed(0)
    H, W = 9, 11
    pred = (torch.rand(H, W, 1, generator=g) * 3).to(device).requires_grad_()
    gt = (torch.rand(H, W, 1, generator=g) * 3 + 0.2).to(device)
    gt[0, :2] = 0.05  # invalid sensor depth (< 0.1 m)
    conf = torch.ones(H, W, 1)
    conf[4:7, 3:8] = 0.0  # DNC-rejected block
    conf[2, 2] = 1 - 250 / 255  # JPEG value 250 -> still confident (> 0)
    conf = conf.to(device)
    rgb = torch.rand(H, W, 3, generator=g).to(device)
    view = {"depth": gt, "confidence": conf}
    img = rgb.clamp(min=10 / 255).cpu()
    p, d = pred.detach().cpu()[..., 0], gt.cpu()[..., 0]
    c = conf.cpu()[..., 0]
    before = fn({}, {"depth": pred, "dn_view": view, "step": 7000}, rgb)  # ns 6999: raw depth
    exp_before = 0.2 * _edge_aware_loop(p, d, img, d > 0.1)
    after = fn({}, {"depth": pred, "dn_view": view, "step": 7001}, rgb)  # ns 7000: D_f
    exp_after = 0.2 * _edge_aware_loop(p, d, img, (d > 0.1) & (c > 0))
    for v, e, tag in ((before, exp_before, "raw"), (after, exp_after, "filtered")):
        if abs(v.item() - e) > 1e-5 * max(1.0, e):
            raise AssertionError(f"{tag} depth loss {v.item()} != independent {e}")
    gb = torch.autograd.grad(before, pred)[0][..., 0].cpu()
    ga = torch.autograd.grad(after, pred)[0][..., 0].cpu()
    if gb[4:7, 3:8].abs().min() == 0 or ga[4:7, 3:8].abs().max() != 0:
        raise AssertionError("DNC switch at ns=7000 must remove gradients of rejected pixels only")
    if ga[2, 2] == 0:
        raise AssertionError("confidence 1-250/255 > 0 must stay supervised (official rule)")
    return {"raw": before.item(), "filtered": after.item(), "weight": 0.2, "switch_ns": 7000}


# -------------------------------------------------------------------------- ANR (C2)
def _edges_loop(gm, thr=0.01):
    """Independent find_edges: loops over the zero-padded Laplacian of 1/(g+1e-6), 3x3 dilation."""
    C, H, W = gm.shape
    inv = 1.0 / (gm.double() + 1e-6)
    e = torch.zeros(C, H, W, dtype=torch.bool)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                acc = -4 * inv[c, i, j]
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < H and 0 <= jj < W:
                        acc += inv[c, ii, jj]
                e[c, i, j] = acc > thr
    out = torch.zeros_like(e)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                out[c, i, j] = e[c, max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].any()
    return out


def normal_anr(method, device):
    spec, fn = _component(method, "ags-normal")
    g = torch.Generator(device="cpu").manual_seed(2)
    H, W = 8, 9
    D = torch.tensor([1.0, -1.0, -1.0])
    prior01 = torch.rand(H, W, 3, generator=g)
    gt_gl = (2 * prior01 - 1) * D  # official GL-frame prior
    surf = torch.nn.functional.normalize(torch.randn(H, W, 3, generator=g), dim=-1)
    surf = torch.where((surf * gt_gl).sum(-1, keepdim=True) > 0, -surf, surf)  # disagree
    gt_unit = torch.nn.functional.normalize(gt_gl, dim=-1)
    agree = torch.zeros(H, W, dtype=torch.bool)
    agree[1:4, 2:6] = True
    surf[agree] = gt_unit[agree]  # angle(surf, gt) small only if |gt| ~ 1: use exact agreement
    gt_gl[agree] = gt_unit[agree]
    prior01 = (gt_gl * D + 1) / 2
    pred01 = torch.rand(H, W, 3, generator=g).to(device).requires_grad_()
    base = {"normal": pred01, "surface_normal": surf.to(device),
            "dn_view": {"normal": prior01.to(device)}}
    z = fn({}, {**base, "step": 7001}, None)  # ns 7000: lambda 0
    if z.item() != 0.0:
        raise AssertionError("normal loss must be off until ns > 7000")
    gt_c = gt_gl.permute(2, 0, 1).double()
    s_c = surf.permute(2, 0, 1).double()
    p_c = ((2 * pred01.detach().cpu() - 1) * D).permute(2, 0, 1).double()
    g_term = (p_c - gt_c).abs().mean().item()
    keep_e = ~_edges_loop(gt_c.float())
    exp_mid = 0.1 * ((s_c[keep_e] - gt_c[keep_e]).abs().mean().item() + g_term)
    mid = fn({}, {**base, "step": 7002}, None)  # ns 7001: edge-masked surface term
    keep_a = torch.zeros(H, W, dtype=torch.bool)
    for i in range(H):
        for j in range(W):
            cosv = float(torch.clamp((s_c[:, i, j] * gt_c[:, i, j]).sum(), -1, 1))
            keep_a[i, j] = math.acos(cosv) <= 0.1
    if not torch.equal(keep_a, agree):
        raise AssertionError("fixture: ANR keep set should be exactly the agreeing pixels")
    exp_late = 0.1 * ((s_c[:, keep_a] - gt_c[:, keep_a]).abs().mean().item() + g_term)
    late = fn({}, {**base, "step": 15001}, None)  # ns 15000: ANR confidence mask
    for v, e, tag in ((mid, exp_mid, "edge-masked"), (late, exp_late, "ANR")):
        if abs(v.item() - e) > 1e-5:
            raise AssertionError(f"{tag} normal loss {v.item()} != independent {e}")
    if abs(late.item() - 0.1 * g_term) > 1e-6:
        raise AssertionError("agreeing pixels have zero surface error: ANR term must vanish")
    gr = torch.autograd.grad(late, pred01)[0]
    if not torch.count_nonzero(gr):
        raise AssertionError("Gaussian-normal term must reach the rendered normals")
    return {"ns7000": z.item(), "edge_masked": mid.item(), "anr": late.item(),
            "edges_excluded": int((~keep_e).sum()), "anr_kept": int(keep_a.sum())}


# -------------------------------------------------------------------- surface normals
def surface_normals(method, device):
    from .renderer import surface_normals as sn

    H, W, f = 20, 24, 30.0
    n_cv = np.array([0.3, -0.2, -1.0])
    n_cv /= np.linalg.norm(n_cv)
    depth = _plane_depth(n_cv, -1.5, H, W, f, W / 2, H / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float64)
    out = sn(torch.from_numpy(depth)[..., None], K)
    want = torch.from_numpy(n_cv * np.array([1.0, -1.0, -1.0]))
    err = (out[1:-1, 1:-1] - want).abs().max().item()
    if err > 1e-6:
        raise AssertionError(f"surface normal of a plane off by {err} (GL frame, facing camera)")
    if out[0].abs().max() != 0 or out[:, -1].abs().max() != 0:
        raise AssertionError("pcd_to_normal pads a zero border")
    return {"max_error": err, "normal_gl": want.tolist()}
