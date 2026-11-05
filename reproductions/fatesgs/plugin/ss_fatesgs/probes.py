"""Mechanism probes for the FatesGS port. Each raises AssertionError when the mechanism is
removed or wrong."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from . import losses
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import Density


def _params(n, device, scale=0.05):
    from torch import nn
    p = nn.ParameterDict({
        "means": nn.Parameter(torch.zeros(n, 3, device=device)),
        "scales": nn.Parameter(torch.full((n, 3), math.log(scale), device=device)),
        "quats": nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": nn.Parameter(torch.full((n,), 2.0, device=device)),
        "sh0": nn.Parameter(torch.full((n, 1, 3), 0.5, device=device)),
        "shN": nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })
    opts = {k: torch.optim.Adam([v], lr=1e-3, eps=1e-15) for k, v in p.items()}
    for v in p.values():
        v.grad = torch.zeros_like(v)
    for o in opts.values():
        o.step()
    return p, opts


def _camera(device, W=64, H=48, f=60.0):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": W, "height": H}


# ------------------------------------------------------------------ renderer
def surfel_renderer(method, device):
    from .raster import render
    p, _ = _params(1, device, scale=1.0)
    with torch.no_grad():
        p["means"][0] = torch.tensor([0.0, 0.0, 2.0], device=device)
        p["opacities"][0] = 4.0
    cam = _camera(device)
    out = render(p, cam, 0, torch.zeros(3, device=device))
    d = out["depth"][24, 32, 0].item()
    assert abs(d - 2.0) < 1e-3, f"plane depth {d} != 2"
    n = F.normalize(out["normal"][24, 32], dim=-1)
    assert abs(abs(n[2].item()) - 1) < 1e-3, f"plane normal {n.tolist()}"
    a = out["alpha"][24, 32, 0].item()
    assert 0.9 < a <= 1.0, a
    sn = out["surface_normal"][24, 32]
    assert abs(abs(sn[2].item()) - a) < 1e-2, "surf normal = depth normal * alpha"
    loss = out["rgb"].sum() + out["depth"].sum() + out["distortion"].sum()
    loss.backward()
    g2 = out["info"]["means2d"].grad
    for k in ("means", "opacities", "scales", "quats", "sh0"):
        assert torch.isfinite(p[k].grad).all() and p[k].grad.abs().sum() > 0, k
    assert g2 is not None and g2.abs().sum() > 0, "no means2d gradient"
    assert p["scales"].grad[:, 2].abs().sum() == 0, "third scale must be unused"
    return {"depth": d, "alpha": a, "normal_z": n[2].item(),
            "means2d_grad": g2.abs().sum().item()}


# ------------------------------------------------------------------ density
def density_recipe(method, device):
    spec = SimpleNamespace(config={})
    den = Density(spec)
    p, opts = _params(4, device, scale=0.001)
    with torch.no_grad():
        p["scales"][1, :2] = math.log(0.5)     # big -> split
        p["scales"][1, 2] = math.log(1e-3)
        p["opacities"][3] = -3.0               # sigmoid ~0.047 < 0.1 -> pruned
        p["means"][1] = torch.tensor([1.0, 2.0, 3.0], device=device)
    den.initialize(p, opts, scene_scale=2.0)
    st = den.state
    st["grad_accum"][:] = torch.tensor([1e-3, 1e-3, 1e-5, 1e-5], device=device)
    st["denom"][:] = 1
    den.densify_and_prune(p, opts, None)
    last = st["last"]
    assert last["clone"] == 1 and last["split"] == 1, last
    # 4 + 1 clone + 2 split - 1 split original - 1 pruned = 5
    assert len(p["means"]) == 5, len(p["means"])
    split_pts = p["means"][-2:]
    assert (split_pts[:, 2] - 3.0).abs().max() < 1e-6, "split must stay in the surfel plane"
    exp = math.log(0.5 / 1.6)
    assert (p["scales"][-2:, :2] - exp).abs().max() < 1e-5, "split scale / (0.8*2)"
    assert (p["opacities"].sigmoid() >= 0.1).all()
    # opacity reset at step 1000: min(o, 0.1); xyz LR schedule
    n = len(p["means"])
    radii = torch.ones(n, device=device, dtype=torch.int32)
    m2d = torch.zeros(n, 3, device=device)
    m2d.grad = torch.zeros(n, 3, device=device)
    den.after_backward(p, opts, 999, {"info": {"means2d": m2d, "radii": radii}})
    assert p["opacities"].sigmoid().min() > 0.5
    den.after_backward(p, opts, 1000, {"info": {"means2d": m2d, "radii": radii}})
    assert (p["opacities"].sigmoid() - 0.1).abs().max() < 1e-5, "reset to min(o, 0.1)"
    lr15k = den.xyz_lr(15000) / 2.0
    assert abs(lr15k - 1.6e-5) / 1.6e-5 < 1e-6, lr15k
    assert abs(opts["means"].param_groups[0]["lr"] - den.xyz_lr(1001)) < 1e-12
    # size pruning only after the first reset interval, threshold 20 px
    st["max_radii"][:] = 0
    st["max_radii"][0] = 25
    st["grad_accum"][:] = 0
    den.densify_and_prune(p, opts, DENSITY_DEFAULTS["max_screen_size"])
    assert st["last"]["prune"] >= 1
    return {"after_first": 5, "lr_at_15000_over_extent": lr15k}


# ------------------------------------------------------------------ L_c, L_d, L_n
def surfel_reg_formula(method, device="cpu"):
    torch.manual_seed(0)
    x, y = torch.rand(20, 24, 3), torch.rand(20, 24, 3)
    s_same = losses.ssim_same(x, y)
    # independent: explicit zero padding then valid conv equals 'same' conv
    w = losses._gaussian_window()
    xp, yp = (F.pad(t.permute(2, 0, 1)[None], (5, 5, 5, 5)) for t in (x, y))
    mu1, mu2 = F.conv2d(xp, w, groups=3), F.conv2d(yp, w, groups=3)
    v1 = F.conv2d(xp * xp, w, groups=3) - mu1**2
    v2 = F.conv2d(yp * yp, w, groups=3) - mu2**2
    cv = F.conv2d(xp * yp, w, groups=3) - mu1 * mu2
    ref = (((2 * mu1 * mu2 + 1e-4) * (2 * cv + 9e-4)) / ((mu1**2 + mu2**2 + 1e-4)
                                                          * (v1 + v2 + 9e-4))).mean()
    assert abs(s_same - ref) < 1e-6
    assert abs(losses.ssim_same(x, x) - 1) < 1e-5
    reg = losses.create_surfel_reg(SimpleNamespace(config={}))
    out = {"normal": torch.ones(8, 8, 3) / 3 ** 0.5, "surface_normal": torch.zeros(8, 8, 3),
           "distortion": torch.full((8, 8, 1), 1e-3)}
    vals = {s: float(reg(None, {**out, "step": s}, None)) for s in (3000, 3001, 7000, 7001)}
    assert vals[3000] == 0 and abs(vals[3001] - 10.0) < 1e-6
    assert abs(vals[7000] - 10.0) < 1e-6 and abs(vals[7001] - 10.05) < 1e-6, vals
    return {"ssim_same": float(s_same), "reg": vals}


# ------------------------------------------------------------------ L_r
def rank_formula(method, device="cpu"):
    torch.manual_seed(0)
    H, W, P, m = 40, 72, 8, 1e-4
    mono = torch.rand(H, W)
    surf = torch.rand(H, W)
    perm = torch.randperm(P * P)
    got = losses.patched_rank(surf, mono, P, m, perm)
    # independent loop
    vals = []
    for py in range(H // P):
        for px in range(W // P):
            sp = surf[py * P:(py + 1) * P, px * P:(px + 1) * P].reshape(-1)[perm]
            mp = mono[py * P:(py + 1) * P, px * P:(px + 1) * P].reshape(-1)[perm]
            h = P * P // 2
            for j in range(h):
                s = torch.sign(mp[j] - mp[h + j])
                vals.append(max(float(s * (sp[h + j] - sp[j]) + m), 0.0))
    ref = sum(vals) / len(vals)
    assert abs(float(got) - ref) < 1e-6, (float(got), ref)
    consistent = float(losses.patched_rank(mono * 3 + 1, mono, P, m, perm))
    reversed_ = float(losses.patched_rank(-mono, mono, P, m, perm))
    assert consistent <= m + 1e-9 and reversed_ > 100 * consistent, (consistent, reversed_)
    full = losses.depth_rank_loss(torch.rand(581, 777), torch.rand(581, 777),
                                  losses.RANK_DEFAULTS["crops"], 32, m)
    assert torch.isfinite(full)
    return {"loop_match": ref, "consistent": consistent, "reversed": reversed_}


# ------------------------------------------------------------------ L_s
def smooth_formula(method, device="cpu"):
    torch.manual_seed(0)
    surf, mono = torch.rand(12, 15), torch.rand(12, 15) * 0.02
    got = float(losses.tv_loss(surf, mono, 1e-2, 1e-4))
    hv = [max(abs(float(surf[i + 1, j] - surf[i, j])) - 1e-4, 0)
          * (abs(float(mono[i + 1, j] - mono[i, j])) < 1e-2)
          for i in range(11) for j in range(15)]
    wv = [max(abs(float(surf[i, j + 1] - surf[i, j])) - 1e-4, 0)
          * (abs(float(mono[i, j + 1] - mono[i, j])) < 1e-2)
          for i in range(12) for j in range(14)]
    ref = sum(hv) / len(hv) + sum(wv) / len(wv)
    assert abs(got - ref) < 1e-6
    edge = torch.zeros(12, 15)
    edge[:, 7:] = 1.0  # strong edge in mono depth switches the term off across it
    step = torch.zeros(12, 15)
    step[:, 7:] = 0.5
    assert float(losses.tv_loss(step, edge, 1e-2, 1e-4)) == 0.0
    assert float(losses.tv_loss(step, torch.zeros(12, 15), 1e-2, 1e-4)) > 0
    return {"loop_match": ref}


# ------------------------------------------------------------------ L_f
class _FakeTable:
    def __init__(self, viewmats, feats):
        self.viewmats = viewmats
        self.feats = feats
        self.pairs = [[1], [0]]

    def index(self, viewmat):
        return int((self.viewmats - viewmat[None]).abs().amax(dim=(1, 2)).argmin())

    def features(self, device):
        return self.feats


def feature_consistency(method, device="cpu"):
    torch.manual_seed(0)
    W, H, f, z0 = 80, 60, 70.0, 3.0
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    v0 = torch.eye(4)
    v1 = torch.eye(4)
    v1[0, 3] = -0.4  # second camera shifted by +0.4 in x
    tex = torch.randn(8, 1, 40, 40)

    def world_feat(xy):  # smooth random field on the plane z = z0, x,y in [-2, 2]
        g = (xy / 2.0).clamp(-1, 1)
        return F.grid_sample(tex.permute(1, 0, 2, 3), g[None, :, None, :], align_corners=True,
                             mode="bilinear")[0, :, :, 0]

    def view_feat(viewmat, scale):
        h, w = math.ceil(H / scale), math.ceil(W / scale)
        gx, gy = torch.meshgrid(torch.arange(w).float(), torch.arange(h).float(), indexing="xy")
        uv = torch.stack([gx, gy], -1).reshape(-1, 2) * scale
        c2w = torch.linalg.inv(viewmat)
        rays = torch.cat([uv, torch.ones(len(uv), 1)], -1) @ torch.linalg.inv(K).T
        pts = rays * z0 + c2w[:3, 3]
        return world_feat(pts[:, :2]).reshape(8, h, w)

    feats = [[view_feat(v0, 1), view_feat(v1, 1)], [view_feat(v0, 2), view_feat(v1, 2)]]
    tab = _FakeTable(torch.stack([v0, v1]), feats)
    cam = {"K": K, "viewmat": v0, "width": W, "height": H}
    true = losses.feature_loss(torch.full((H, W), z0), cam, tab, [1, 2], 0.5)
    shifted = losses.feature_loss(torch.full((H, W), z0 * 1.15), cam, tab, [1, 2], 0.5)
    assert float(true) < 0.5 * float(shifted), (float(true), float(shifted))
    d = torch.full((H, W), z0, requires_grad=True)
    losses.feature_loss(d * 1.1, cam, tab, [1, 2], 0.5).backward()
    assert d.grad.abs().sum() > 0, "feature loss must reach depth"
    # visibility bookkeeping: official (code) vs paper z-buffer on a crafted case
    uv = torch.tensor([[5, 5], [5, 5], [0, 9]])
    pts = torch.tensor([[0.0, 0, 1], [0, 0, 2], [0, 0, 3]])
    off = losses.visibility(pts, torch.zeros(3), uv).tolist()
    zb = losses.visibility_zbuffer(pts, torch.zeros(3), uv).tolist()
    assert zb == [1.0, 0.0, 1.0] and off == [1.0, 1.0, 0.0], (off, zb)
    return {"true_depth": float(true), "shifted_depth": float(shifted),
            "visibility_official": off, "visibility_zbuffer": zb}
