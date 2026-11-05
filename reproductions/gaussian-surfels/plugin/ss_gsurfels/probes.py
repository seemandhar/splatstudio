"""Mechanism probes for the Gaussian Surfels port. Each raises AssertionError when the mechanism
is removed or wrong."""

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import losses
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import Density, scale_opacity
from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .renderer import Renderer


def _params(n, device):
    p = nn.ParameterDict({
        "means": nn.Parameter(torch.zeros(n, 3, device=device)),
        "scales": nn.Parameter(torch.full((n, 3), math.log(0.05), device=device)),
        "quats": nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": nn.Parameter(torch.full((n,), 2.0, device=device)),
        "sh0": nn.Parameter(torch.full((n, 1, 3), 0.5, device=device)),
        "shN": nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })
    opts = {k: torch.optim.Adam([v], lr=1e-3, eps=1e-15) for k, v in p.items()}
    for v in p.values():
        v.grad = torch.ones_like(v) * 1e-3
    for o in opts.values():
        o.step()
        o.zero_grad()
    return p, opts


def _camera(device, W=64, H=48, f=60.0):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": W, "height": H}


def _quat_from_normal(n):
    """Rotation whose third column is n (axis-angle from z)."""
    z = torch.tensor([0.0, 0, 1])
    n = F.normalize(n, dim=0)
    axis = torch.linalg.cross(z, n)
    s, c = axis.norm(), torch.dot(z, n)
    if s < 1e-8:
        return torch.tensor([1.0, 0, 0, 0])
    ang = torch.atan2(s, c)
    axis = axis / s
    return torch.cat([torch.cos(ang / 2)[None], axis * torch.sin(ang / 2)])


# ------------------------------------------------------------------ renderer (C1)
def surfel_raster(method, device):
    from .raster import render
    dev = torch.device(device)
    slope = 0.3
    nrm = torch.tensor([-slope, 0.0, 1.0])
    q = _quat_from_normal(nrm)
    xs, ys = torch.meshgrid(torch.linspace(-1.2, 1.2, 49), torch.linspace(-0.9, 0.9, 37),
                            indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    pts = torch.stack([xs, ys, 2.0 + slope * xs], -1)
    n = len(pts)
    p, _ = _params(n, dev)
    with torch.no_grad():
        p["means"].copy_(pts.to(dev))
        p["quats"].copy_(q.to(dev).repeat(n, 1))
        p["scales"].copy_(torch.tensor([math.log(0.08), math.log(0.08), -1e10],
                                       device=dev).repeat(n, 1))
        p["opacities"].fill_(4.0)
    cam = _camera(dev)
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        out = render(p, cam, 0, bg)
    f, W, H = 60.0, 64, 48
    v, u = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    a = (u + 0.5 - W / 2) / f  # kernel pixel centres: ndc2Pix = ((ndc + 1) W - 1) / 2
    truth = 2.0 / (1 - slope * a)  # ray (a, b, 1) t meets z = 2 + slope x
    depth = out["depth"][0].cpu()
    inner = (slice(12, 36), slice(16, 48))
    err = (depth[inner] - truth[inner]).abs().max().item()
    # measured 1.0e-3 with per-pixel depth; 4.3e-2 with config.perpix_depth = 0 (centre depth)
    assert err < 5e-3, f"per-pixel surfel depth error {err}"
    nmap = F.normalize(out["normal"], dim=0)[:, 24, 32].cpu()
    expect = -F.normalize(nrm, dim=0)  # dual_visible: normals face the camera
    assert torch.allclose(nmap, expect, atol=1e-3), f"normal {nmap.tolist()} vs {expect.tolist()}"
    # zero thickness: the third scale has no effect
    with torch.no_grad():
        p["scales"][:, 2] = math.log(0.5)
        out2 = render(p, cam, 0, bg)
    assert torch.equal(out2["color"], out["color"]) and torch.equal(out2["depth"], out["depth"])

    # x10 rendered-normal gradient: one camera-facing surfel, L = sum N_x; tilting about y
    # changes the footprint only to second order, so d L / d q_y comes from the normal path.
    s, _ = _params(1, dev)
    with torch.no_grad():
        s["means"][0] = torch.tensor([0.0, 0.0, 2.0], device=dev)
        s["scales"][0] = torch.tensor([math.log(0.3), math.log(0.3), -1e10], device=dev)
        s["opacities"].fill_(0.5)

    def objective():
        return render(s, cam, 0, bg)["normal"][0].sum()

    s["quats"].grad = None
    objective().backward()
    analytic = s["quats"].grad[0, 2].item()
    eps = 1e-3
    vals = []
    for sign in (1, -1):
        with torch.no_grad():
            s["quats"][0] = torch.tensor([1.0, 0.0, sign * eps, 0.0], device=dev)
            vals.append(objective().item())
    numeric = (vals[0] - vals[1]) / (2 * eps)
    ratio = analytic / numeric
    assert abs(abs(ratio) - 10.0) < 1.0, f"normal-gradient scale {ratio} (official x10)"
    return {"depth_max_err": err, "normal": nmap.tolist(), "normal_grad_ratio": ratio}


# ------------------------------------------------------------------ density (C6)
def density_recipe(method, device):
    dev = torch.device(device)
    n = 8
    p, opts = _params(n, dev)
    spec = SimpleNamespace(config={**DENSITY_DEFAULTS, "use_sidecar_init": False})
    d = Density(spec)
    d.initialize(p, opts, 1.0)
    d.extent = 1.0
    d.train_keys = {"a", "b"}
    st = d.state
    logs = {0: (0.05, 0.05), 1: (0.6, 0.05), 2: (1e-5, 1e-5), 4: (0.005, 0.005),
            5: (0.005, 0.005), 6: (0.05, 0.02)}
    with torch.no_grad():
        p["scales"][:, :2] = math.log(0.005)
        p["scales"][:, 2] = -1e10
        for i, (a, b) in logs.items():
            p["scales"][i, :2] = torch.tensor([math.log(a), math.log(b)])
        p["opacities"].fill_(1.0)
        p["opacities"][0] = math.log(0.05 / 0.95)
        p["means"][:] = torch.arange(n, device=dev, dtype=torch.float32)[:, None]
        p["quats"][6] = _quat_from_normal(torch.tensor([0.0, 1.0, 1.0])).to(dev)
    st["denom"][:] = 1
    st["xyz_accum"][:] = 0
    st["xyz_accum"][[4, 5, 6]] = 1e-3
    st["scale_accum"][:] = 0
    st["opac_accum"][:] = p["opacities"].detach()
    st["opac_accum"][5] = 3.0  # mean logit > 2 -> not cloned
    st["seen"][:] = True
    st["seen"][3] = False
    d.views_seen = {"a"}  # window incomplete: invisible prune must wait
    d.adaptive_prune(p, opts)
    assert st["last"]["invisible"] == 0 and len(p["means"]) == 5, st["last"]
    kept = p["means"][:, 0].round().long().tolist()
    assert kept == [3, 4, 5, 6, 7], kept
    d.adaptive_densify(p, opts)
    # clone 4 (not 5); split 6 into 2 in-plane children with s / 1.6, third scale -1e10
    assert st["last"]["clone"] == 1 and st["last"]["split"] == 1, st["last"]
    assert len(p["means"]) == 5 + 1 + 1, len(p["means"])
    child = p["scales"][-2:].detach()
    assert torch.allclose(child[:, :2].exp(), torch.tensor([0.05, 0.02], device=dev) / 1.6,
                          rtol=1e-5), child
    assert (child[:, 2] == -1e10).all()
    nrm = torch.tensor([0.0, 1.0, 1.0], device=dev) / math.sqrt(2)
    off = p["means"][-2:].detach() - torch.tensor([6.0, 6, 6], device=dev)
    assert (off @ nrm).abs().max() < 1e-5, "split samples must stay in the surfel plane"
    assert float(st["denom"].abs().sum()) == 0.0, "statistics restart after densify"
    # invisible prune once every training view was seen
    st["seen"] = torch.ones(len(p["means"]), dtype=torch.bool, device=dev)
    st["seen"][0] = False
    d.views_seen = {"a", "b"}
    before = len(p["means"])
    d.adaptive_prune(p, opts)
    assert st["last"]["invisible"] == 1 and len(p["means"]) == before - 1, st["last"]
    # opacity reset x0.12 with zeroed moments, at (step - 1) % 1000 == 0 only
    o0 = p["opacities"].detach().sigmoid().clone()
    scale_opacity(p, opts, 0.12)
    assert torch.allclose(p["opacities"].sigmoid(), o0 * 0.12, rtol=1e-4)
    assert float(opts["opacities"].state[p["opacities"]]["exp_avg"].abs().sum()) == 0.0
    for step, fired in ((1000, False), (1001, True)):
        o_before = p["opacities"].detach().sigmoid().clone()
        m2d = torch.zeros(len(p["means"]), 3, device=dev, requires_grad=True)
        m2d.grad = torch.zeros_like(m2d)
        outputs = {"info": {"means2d": m2d,
                            "radii": torch.zeros(len(p["means"]), dtype=torch.int32, device=dev)},
                   "camera": {"viewmat": torch.eye(4, device=dev)}}
        d.interval = 10**9
        d.after_backward(p, opts, step, outputs)
        changed = not torch.allclose(p["opacities"].sigmoid(), o_before)
        assert changed == fired, (step, changed)
    lr0, lr1 = d.xyz_lr(0), d.xyz_lr(15000)
    assert math.isclose(lr0, 1.6e-4) and math.isclose(lr1, 1.6e-6), (lr0, lr1)
    return {"xyz_lr": [lr0, lr1], "final_count": len(p["means"])}


# ------------------------------------------------------------------ cutting (C5)
def _occ_reference(pos, R, scale, opac, pmin, pmax, step, dims):
    """Independent numpy version of utils/cuda_utils.cu gaussian2occgrid_kernel (one surfel)."""
    grid = np.zeros(dims, np.float64)
    cutoff = 1.0 / 255.0
    rad = max(scale[0], scale[1]) * math.sqrt(-2 * math.log(cutoff / opac)) + step * 0.5
    r = np.arange(-rad, rad, 0.5 * step, dtype=np.float32)
    dx, dy, dz = (np.float32(pos[i]) + r for i in range(3))
    X, Y, Z = np.meshgrid(dx, dy, dz, indexing="ij")
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    assert (P >= pmin).all() and (P < pmax).all()
    dif = P - pos
    d2p = dif @ R[:, 2]
    keep = np.abs(d2p) <= step * 0.5
    dif = dif[keep] - np.outer(d2p[keep], R[:, 2])
    u = (dif @ R[:, 0]) / scale[0]
    v = (dif @ R[:, 1]) / scale[1]
    a = np.exp(-(u * u + v * v) / 2) * opac
    ok = a >= cutoff
    cell = ((P[keep][ok] - pmin) / step).astype(np.int64)
    np.add.at(grid, (cell[:, 0], cell[:, 1], cell[:, 2]), a[ok])
    return grid


def cutting_mesh(method, device):
    from .mesh import grid_prune, occupancy_grid
    dev = torch.device(device)
    p, _ = _params(1, dev)
    q = _quat_from_normal(torch.tensor([0.2, 0.1, 1.0]))
    with torch.no_grad():
        p["means"][0] = torch.tensor([0.013, -0.021, 0.007], device=dev)
        p["quats"][0] = q.to(dev)
        p["scales"][0] = torch.tensor([math.log(0.1), math.log(0.05), -1e10], device=dev)
        p["opacities"][0] = math.log(0.9 / 0.1)
    bound = (torch.tensor([-0.5, -0.5, -0.5], device=dev), torch.tensor([0.5, 0.5, 0.5],
                                                                          device=dev))
    grid, shift, scale, dims = occupancy_grid(p, 64, bound)
    from gsplat.utils import normalized_quat_to_rotmat
    R = normalized_quat_to_rotmat(F.normalize(p["quats"], dim=-1))[0].detach().cpu().numpy()
    ref = _occ_reference(p["means"][0].detach().cpu().numpy(), R, [0.1, 0.05], 0.9,
                         np.array([-0.5] * 3, np.float32), np.array([0.5] * 3, np.float32),
                         1.0 / 64, tuple(dims.tolist()))
    g = grid.cpu().numpy()
    err = float(np.abs(g - ref).max())
    assert err < 1e-3 * max(1.0, ref.max()), f"occupancy grid differs from reference: {err}"
    pts = torch.tensor([[0.013, -0.021, 0.007], [0.4, 0.4, 0.4], [0.013, -0.021, 0.3]],
                       device=dev)
    keep = grid_prune(grid, shift, scale, dims, pts, thrsh=1.0).tolist()
    assert keep == [True, False, False], keep
    return {"grid_max": float(g.max()), "grid_err": err, "keep": keep}


# ------------------------------------------------------------------ schedule (C6)
def schedule_recipe(method, device):
    from .raster import scaled_camera
    r = Renderer(SimpleNamespace(config={}))
    got = {c: r.factor(c) for c in (1, 2016, 2017, 5040, 5041, 15000)}
    assert got == {1: 4, 2016: 4, 2017: 2, 5040: 2, 5041: 1, 15000: 1}, got
    cam = _camera("cpu", W=1600, H=1200, f=2892.0)
    c4 = scaled_camera(cam, 4)
    assert (c4["width"], c4["height"]) == (400, 300)
    assert torch.allclose(c4["K"][0], cam["K"][0] / 4)
    assert RENDERER_DEFAULTS["random_background"] is True
    assert RENDERER_DEFAULTS["eval_background"] == [0.0, 0.0, 0.0]
    return {"factors": got}


# ------------------------------------------------------------------ losses (C2-C4)
def _plane_depth(H=40, W=56, f=50.0, slope=0.3):
    v, u = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    a = (u - W / 2) / f
    cam = {"K": torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), "width": W,
           "height": H}
    return (2.0 / (1 - slope * a))[None], cam


def consistency_formula(method, device):
    depth, cam = _plane_depth()
    mask = torch.ones_like(depth, dtype=torch.bool)
    n = losses.depth2normal(depth, mask, cam)
    expect = -F.normalize(torch.tensor([-0.3, 0.0, 1.0]), dim=0)
    inner = n[:, 2:-2, 2:-2].reshape(3, -1)
    err = (inner - expect[:, None]).abs().max().item()
    assert err < 1e-4, f"depth2normal plane error {err}"
    cons = losses.cos_loss(n, n).item()
    assert cons < 1e-5, cons
    tilted = F.normalize(n + torch.tensor([0.3, 0.0, 0.0])[:, None, None], dim=0)
    l_t = losses.cos_loss(tilted, n).item()
    assert l_t > 1e-2, l_t
    cfg = losses.CONS_DEFAULTS
    w = [losses.consistency_weight(s, cfg) for s in (1, 7500, 15000)]
    assert math.isclose(w[0], 0.01 + 0.1 * 2 / 15000) and math.isclose(w[1], 0.11)
    assert math.isclose(w[2], 0.11)
    return {"plane_err": err, "tilted_loss": l_t, "weights": w}


def normal_prior_formula(method, device):
    g = torch.Generator().manual_seed(0)
    N = F.normalize(torch.randn(3, 9, 11, generator=g), dim=0)
    M = F.normalize(torch.randn(3, 9, 11, generator=g), dim=0)
    w = (torch.rand(1, 9, 11, generator=g) > 0.3).float()
    got = losses.cos_loss(N, M * w, weight=w).item()
    vals = []
    for i in range(9):
        for j in range(11):
            c = float((N[:, i, j] * M[:, i, j] * w[0, i, j] * w[0, i, j]).sum())
            if c < 1.0:
                vals.append(1 - c)
    ref = sum(vals) / len(vals)
    assert abs(got - ref) < 1e-5, (got, ref)
    mask = torch.ones(1, 9, 11, dtype=torch.bool)
    curv = losses.normal2curv(N, mask)
    i, j = 4, 5
    lap = (N[:, i - 1, j] + N[:, i + 1, j] + N[:, i, j - 1] + N[:, i, j + 1] - 4 * N[:, i, j])
    assert abs(curv[0, i, j].item() - lap.abs().sum().item()) < 1e-5
    flat = losses.normal2curv(torch.tensor([0.0, 0, -1])[:, None, None].expand(3, 9, 11), mask)
    assert flat.abs().max() == 0
    cfg = losses.PRIOR_DEFAULTS
    ws = [losses.prior_weight(s, cfg) for s in (0, 15000)]
    assert math.isclose(ws[0], 0.04) and math.isclose(ws[1], 0.02)
    return {"cos": got, "weights": ws}


def opacity_formula(method, device):
    o = torch.tensor([0.005, 0.3, 0.5, 0.7, 0.995])
    got = losses.opacity_term(o, losses.OPAC_DEFAULTS).item()
    w = torch.tensor([0.0, 0.01, 0.01, 1.0, 0.0])
    ref = (torch.exp(-((o - 0.5) ** 2) / 0.05) * w).mean().item()
    assert abs(got - ref) < 1e-7, (got, ref)
    return {"value": got}
