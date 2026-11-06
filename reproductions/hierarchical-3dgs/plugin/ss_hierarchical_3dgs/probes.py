"""Executable checks; each raises if its mechanism is removed or wrong."""

import math

import torch

from . import hier
from .density import (
    DEFAULTS,
    HierarchicalDensity,
    clone_select,
    expon_lr,
    split_select,
)
from .optim import FrozenAdam, SparseAdam, StepMask
from .renderer import DEFAULTS as RENDER_DEFAULTS


def _unavailable(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


class _Spec:
    def __init__(self, config=None):
        self.config = config or {}


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """OpenCV world-to-camera matrix (x right, y down, z forward)."""
    eye, target, up = (torch.tensor(v, dtype=torch.float32) for v in (eye, target, up))
    z = torch.nn.functional.normalize(target - eye, dim=0)
    x = torch.nn.functional.normalize(torch.linalg.cross(z, up), dim=0)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])
    V = torch.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def cameras(device, n=8, radius=6.0, size=64):
    K = torch.tensor([[60.0, 0, size / 2], [0, 60.0, size / 2], [0, 0, 1]], device=device)
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        V = look_at((radius * math.cos(a), radius * math.sin(a), 1.0)).to(device)
        out.append({"K": K, "viewmat": V, "width": size, "height": size})
    return out


def synthetic_leaves(device, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = {
        "means": torch.rand(n, 3, generator=g) * 2 - 1,
        "scales": torch.log(torch.rand(n, 3, generator=g) * 0.04 + 0.01),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.logit(torch.rand(n, generator=g) * 0.8 + 0.1),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.05,
        hier.FIELD: torch.zeros(n, hier.WIDTH),
    }
    return {k: v.to(device) for k, v in p.items()}


def covariance(log_scales, quats):
    from .density import quat_to_rotmat

    R = quat_to_rotmat(quats.double())
    S = torch.diag_embed(log_scales.double().exp() ** 2)
    return R @ S @ R.transpose(1, 2)


def _children(nodes, i):
    s, c = int(nodes[i, 5]), int(nodes[i, 6])
    return range(s, s + c)


def _leaf_descendants(nodes):
    """node -> list of leaf Gaussian indices below it (iterative post-order)."""
    m = len(nodes)
    out = [None] * m
    order, stack = [], [0]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(_children(nodes, i))
    for i in reversed(order):
        if int(nodes[i, 0]) == 0:
            out[i] = [int(nodes[i, 2])]
        else:
            out[i] = [j for c in _children(nodes, i) for j in out[c]]
    return out


def hierarchy_build(method, device):
    """Eq. (merge): every interior node equals the w' = o*S weighted merge of its leaf
    descendants (weights compose, so this holds after compaction), falloff = sum w' / S_p,
    tree/box consistency and leaf preservation, for the vendored creator."""
    _unavailable(device)
    leaves = synthetic_leaves(device)
    h = hier.build(leaves, cameras(device))
    nodes, boxes = h["nodes"].long(), h["boxes"]
    pos, ls, rot = h["means"].double(), h["scales"].double(), h["rots"].double()
    alpha = h["alpha"][:, 0].double()
    n_leaf_nodes = int((nodes[:, 0] == 0).sum())
    if n_leaf_nodes != h["valid_leaves"] or h["valid_leaves"] != len(leaves["means"]):
        raise AssertionError("leaf nodes do not match the input leaves")
    # leaves keep their trained attributes (positions are stored in float32)
    lp = leaves["means"].double().cpu()
    leaf_g = nodes[nodes[:, 0] == 0, 2]
    d = torch.cdist(pos[leaf_g], lp).min(dim=1).values
    if float(d.max()) > 1e-6:
        raise AssertionError("leaf positions changed")
    for i in range(1, len(nodes)):
        par = int(nodes[i, 1])
        if i not in _children(nodes, par):
            raise AssertionError("parent/child links inconsistent")
        if not bool((boxes[i, 0, :3] >= boxes[par, 0, :3] - 1e-2).all()
                    and (boxes[i, 1, :3] <= boxes[par, 1, :3] + 1e-2).all()):
            raise AssertionError("child box outside parent box")
    surf = lambda s: s[:, 0] * s[:, 1] + s[:, 0] * s[:, 2] + s[:, 1] * s[:, 2]  # noqa: E731
    sc = ls.exp()
    w = alpha * surf(sc)
    cov = covariance(ls, rot)
    desc = _leaf_descendants(nodes)
    worst = {"mean": 0.0, "weight": 0.0, "cov": 0.0}
    checked = 0
    for i in range(len(nodes)):
        if int(nodes[i, 0]) == 0:
            continue
        L = torch.tensor(desc[i])
        g = int(nodes[i, 2])
        wl = w[L] / w[L].sum()
        mu = (wl[:, None] * pos[L]).sum(0)
        dd = pos[L] - mu
        S = (wl[:, None, None] * (cov[L] + dd[:, :, None] * dd[:, None, :])).sum(0)
        ext = float(sc[L].max())
        worst["mean"] = max(worst["mean"], float((mu - pos[g]).norm()) / ext)
        worst["weight"] = max(worst["weight"], abs(float(w[L].sum() / w[g]) - 1))
        worst["cov"] = max(worst["cov"], float((S - cov[g]).norm() / S.norm()))
        checked += 1
    if worst["mean"] > 0.05 or worst["weight"] > 0.05 or worst["cov"] > 0.05:
        raise AssertionError(f"interior nodes are not the Eq. (merge) of their leaves: {worst}")
    if checked < 50:
        raise AssertionError("too few interior nodes checked")
    return {"leaves": len(leaf_g), "nodes": len(nodes), "gaussians": len(pos),
            "interior_checked": checked, "anchors": len(h["anchors"]),
            "max_rel_errors_vs_eq_merge": worst}


def _size(box, campos):
    lo, hi = box[0, :3].double(), box[1, :3].double()
    if bool(((campos >= lo) & (campos <= hi)).all()):
        return math.inf
    closest = torch.maximum(lo, torch.minimum(hi, campos))
    return float(box[0, 3]) / float((campos - closest).norm())


def cut_selection(method, device):
    """Sec. 3.2: the selected nodes form a cut (every leaf covered exactly once), each cut node
    satisfies eps(n) < tau <= eps(parent) (leaves excepted), coarser targets give smaller cuts,
    and t / K equal an independent evaluation of the interpolation rule."""
    _unavailable(device)
    leaves = synthetic_leaves(device, n=600, seed=1)
    h = hier.build(leaves, cameras(device))
    blob = hier.pack(h["nodes"], h["boxes"], h["anchors"], len(h["means"])).to(device)
    nodes, boxes, _ = hier.unpack({hier.FIELD: blob})
    nodes_c, boxes_c = nodes.long().cpu(), boxes.cpu()
    gauss_node = {int(nodes_c[i, 2]): i for i in range(len(nodes_c))}
    desc = _leaf_descendants(nodes_c)
    cam = cameras(device, n=1, radius=3.0)[0]
    campos = torch.linalg.inv(cam["viewmat"])[:3, 3]
    cut = hier.Cut()
    sizes, worst_t = [], 0.0
    for tau in (0.0, 0.02, 0.05, 0.2, 1.0):
        r, p, t, k = cut.select(nodes, boxes, len(h["means"]), tau, campos)
        ids = [gauss_node[int(g)] for g in r.cpu()]
        covered = sorted(j for i in ids for j in desc[i])
        if covered != sorted(int(x) for x in nodes_c[nodes_c[:, 0] == 0, 2]):
            raise AssertionError(f"tau={tau}: selection is not a cut of the hierarchy")
        cp = campos.double().cpu()
        for idx, i in enumerate(ids):
            si = _size(boxes_c[i], cp)
            par = int(nodes_c[i, 1])
            sp = _size(boxes_c[par], cp) if par >= 0 else math.inf
            if int(nodes_c[i, 0]) > 0 and not si < tau * (1 + 1e-4):
                raise AssertionError("interior cut node larger than the target")
            if par >= 0 and not sp >= tau * (1 - 1e-4):
                raise AssertionError("cut node whose parent already satisfies the target")
            if int(p[idx]) != (int(nodes_c[par, 2]) if par >= 0 else int(nodes_c[i, 2])):
                if par >= 0:
                    raise AssertionError("parent index of a cut node is wrong")
            # independent computeTsIndexed
            if par < 0 or sp > 2 * tau:
                tt = 1.0
            else:
                start = max(0.5 * sp, si)
                diff = sp - start
                tt = 1.0 if diff <= 0 else max(1 - max(0.0, tau - start) / diff, 0.0)
            worst_t = max(worst_t, abs(tt - float(t[idx])))
            kk = int(nodes_c[par, 6]) if par >= 0 else 1
            if int(k[idx]) != kk:
                raise AssertionError("num_node_kids differs from the parent's child count")
        sizes.append(len(r))
    if worst_t > 1e-3:
        raise AssertionError(f"interpolation weights differ from Eq. (t_interp) rule: {worst_t}")
    if not all(a >= b for a, b in zip(sizes, sizes[1:])) or sizes[0] <= sizes[-1]:
        raise AssertionError(f"cut size not decreasing with the target: {sizes}")
    return {"cut_sizes": sizes, "max_t_error": worst_t}


def alpha_interpolation(method, device):
    """Eq. (alphap): K co-located copies of a parent rendered at t = 0 with num_node_kids = K
    reproduce the parent exactly; without the alpha' correction (K = 1) they do not."""
    _unavailable(device)
    from .raster import _Rasterize, raster_settings

    cam = cameras(device, n=1, radius=4.0)[0]
    s = raster_settings(cam, torch.zeros(3, device=device), 0)
    mean = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    sc = torch.tensor([[0.3, 0.2, 0.25]], device=device)
    q = torch.tensor([[1.0, 0, 0, 0]], device=device)
    sh = torch.zeros(1, 16, 3, device=device)
    sh[:, 0] = torch.tensor([1.0, 0.5, -0.5], device=device)
    op = torch.tensor([[0.8]], device=device)

    def draw(K, kids):
        rep = lambda x: x.repeat(K, *([1] * (x.dim() - 1)))  # noqa: E731
        m = rep(mean)
        img, _, _ = _Rasterize.apply(m, torch.zeros_like(m), rep(sh), rep(op), rep(sc), rep(q),
                                     s, torch.zeros(K, device=device),
                                     torch.full((K,), kids, dtype=torch.int32, device=device))
        return img

    parent = draw(1, 1)
    errs = {}
    for K in (2, 3):
        diff = (draw(K, K) - parent).abs()
        errs[K] = (float(diff.max()), float(diff.mean()))
        wrong = float((draw(K, 1) - parent).abs().max())
        # the kernel skips alpha < 1/255 per Gaussian, so K faint children differ from the
        # parent by up to ~K/255 in the far tails; the bulk must match.
        if errs[K][0] > 3.0 / 255 or errs[K][1] > 1e-3:
            raise AssertionError(f"K={K}: children at t=0 differ from the parent ({errs[K]})")
        if wrong < 5e-2:
            raise AssertionError("probe insensitive: uncorrected alpha gave the same image")
    return {"max_mean_abs_error_K2": errs[2], "max_mean_abs_error_K3": errs[3]}


def hierarchy_optimization(method, device):
    """Sec. 4.1: after the build, training renders a random-granularity cut (log-uniform target
    in [limit_min, limit_max)), gradients reach the parents through the interpolation, and the
    optimizer leaves the anchors (leaves) untouched while interior nodes move."""
    _unavailable(device)
    from torch import nn

    from .renderer import Renderer

    leaves = synthetic_leaves(device, n=500, seed=2)
    params = nn.ParameterDict({k: nn.Parameter(v.clone()) for k, v in leaves.items()})
    optimizers = {k: torch.optim.Adam([p], lr=0.001) for k, p in params.items()}
    dens = HierarchicalDensity(_Spec({"leaves_steps": 1}))
    cams = cameras(device, n=8, radius=3.0)
    dens.set_training_cameras(cams)
    dens.initialize(params, optimizers, 3.0)
    with torch.no_grad():
        params["opacities"].copy_(leaves["opacities"])
    dens.build_hierarchy(params, optimizers)
    if not isinstance(optimizers["means"], FrozenAdam) or not hier.is_built(params):
        raise AssertionError("hierarchy phase not entered")
    rend = Renderer(_Spec())
    _, _, anchor_mask = hier.unpack(params)
    before = {k: params[k].detach().clone() for k in ("means", "sh0", "opacities")}
    thresholds, parent_grad = [], 0.0
    for it in range(40):
        out = rend.render(params, cams[it % len(cams)], 3)
        thresholds.append(out["info"]["threshold"])
        target = torch.zeros_like(out["rgb"])
        loss = (out["rgb"] - target).abs().mean() + out["rgb"].mean()
        loss.backward()
        parent_grad += float(params["sh0"].grad.abs().sum())
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
    lo, hi = RENDER_DEFAULTS["limit_min"], RENDER_DEFAULTS["limit_max"]
    th = torch.tensor(thresholds)
    if float(th.min()) < lo or float(th.max()) >= hi:
        raise AssertionError("granularity sample outside [limit_min, limit_max)")
    moved = {k: (params[k].detach() - before[k]).abs().reshape(len(before[k]), -1).sum(1)
             for k in before}
    if any(float(v[anchor_mask].max()) > 0 for v in moved.values()):
        raise AssertionError("anchor Gaussians changed during hierarchy optimization")
    interior_moved = int((moved["sh0"][~anchor_mask] > 0).sum())
    if interior_moved == 0:
        raise AssertionError("no interior node was optimized")
    if parent_grad == 0:
        raise AssertionError("no gradient reached the hierarchy")
    return {"anchors": int(anchor_mask.sum()), "interior_moved": interior_moved,
            "threshold_geomean": float(th.log().mean().exp()),
            "expected_geomean": math.sqrt(lo * hi)}


def leaf_training(method, device):
    """Sec. 5.2 + train_single.py: MAX (not mean) screen-gradient statistic, the official
    selection rule grad*radius*sigma^0.2 >= 0.015 & sigma > 0.15, split scale /1.6, opacity-only
    prune, reset to min(sigma, 0.01), big-Gaussian shrink (off by default as in the README.md:550
    eval protocol, on with skip_scale_big_gauss False), sparse Adam (invisible rows frozen),
    position lr schedule 2e-5 -> 2e-7 (x extent) applied by after_backward over the runtime's own
    means-lr decay and restarted for the post phase, extent = 1.1 x q0.9 camera distance."""
    import numpy as np
    from torch import nn

    cfg = dict(DEFAULTS)
    dens = HierarchicalDensity(_Spec())
    n = 6
    params = nn.ParameterDict({
        "means": nn.Parameter(torch.zeros(n, 3)),
        "scales": nn.Parameter(torch.log(torch.full((n, 3), 0.01))),
        "quats": nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)),
        "opacities": nn.Parameter(torch.zeros(n)),
        "sh0": nn.Parameter(torch.zeros(n, 1, 3)), "shN": nn.Parameter(torch.zeros(n, 15, 3)),
        hier.FIELD: nn.Parameter(torch.zeros(n, hier.WIDTH))})
    optimizers = {k: torch.optim.Adam([p]) for k, p in params.items()}
    dens.initialize(params, optimizers, 1.0)
    if abs(float(params["opacities"].sigmoid()[0]) - 0.01) > 1e-6:
        raise AssertionError("initial opacity is not 0.01")
    # statistic: max over iterations, not mean
    for g in (0.3, 0.1, 0.2):
        m2 = torch.zeros(n, 3, requires_grad=True)
        (m2[:, 0] * g).sum().backward()
        dens._accumulate({"means2d": m2, "radii": torch.full((n,), 4)})
    if not torch.allclose(dens.accum, torch.full((n,), 0.3)):
        raise AssertionError(f"densification statistic is not the maximum: {dens.accum}")
    # selection rule against an independent evaluation
    grads = torch.tensor([0.01, 0.01, 0.002, 0.01, 0.01, 0.05])
    radii = torch.tensor([2.0, 2.0, 2.0, 2.0, 0.0, 1.0])
    op = torch.tensor([0.5, 0.1, 0.5, 0.5, 0.5, 0.9])
    ms = torch.tensor([1.0, 1.0, 1.0, 1e-5, 1.0, 1.0])
    ref_split = torch.tensor([True, False, False, False, False, True])
    ref_clone = torch.tensor([False, False, False, True, False, False])
    if not torch.equal(split_select(grads, radii, op, ms, cfg, 1.0), ref_split):
        raise AssertionError("split selection differs from the official rule")
    if not torch.equal(clone_select(grads, radii, op, ms, cfg, 1.0), ref_clone):
        raise AssertionError("clone selection differs from the official rule")
    # densify: Gaussian 0 is split into two copies of scale s/1.6
    with torch.no_grad():
        params["opacities"].copy_(torch.logit(torch.tensor([0.5, 0.5, 0.004, 0.5, 0.5, 0.5])))
    dens.accum = torch.tensor([1.0, 0, 0, 0, 0, 0])
    dens.max_radii = torch.tensor([1.0, 0, 0, 0, 0, 0])
    dens._densify(params, optimizers, 600)
    ev = dens.state["events"][-1]
    if (ev["split"], ev["clone"], ev["pruned"], ev["n_after"]) != (1, 0, 1, 6):
        raise AssertionError(f"unexpected densification result {ev}")
    if not torch.allclose(params["scales"][-1].exp(), torch.full((3,), 0.01 / 1.6), rtol=1e-4):
        raise AssertionError("split scale is not s / (0.8 * 2)")
    dens.reset_opacity(params, optimizers)
    if float(params["opacities"].sigmoid().max()) > 0.01 + 1e-6:
        raise AssertionError("opacity reset is not min(sigma, 0.01)")
    # big-Gaussian shrink: off by default (--skip_scale_big_gauss), x0.8 when enabled
    if not cfg["skip_scale_big_gauss"]:
        raise AssertionError("big-Gaussian shrink must default to off (README.md:550)")
    step = cfg["densify_until"] + 1  # leaf phase, no densification/reset/build
    for skip, expect in ((True, 0.05), (False, 0.04)):
        dens.cfg["skip_scale_big_gauss"] = skip
        with torch.no_grad():
            params["scales"][0] = math.log(0.05)
        # runtime.py: means lr decayed by the runtime *before* after_backward
        optimizers["means"].param_groups[0]["lr"] = 123.0
        dens.after_backward(params, optimizers, step, {"info": {}})
        if abs(float(params["scales"][0, 0].exp()) - expect) > 1e-6:
            raise AssertionError(f"big-Gaussian shrink with skip={skip}: expected {expect}")
        lr = optimizers["means"].param_groups[0]["lr"]
        if abs(lr / (expon_lr(step + 1, 2e-5, 2e-7, 30000) * dens.extent) - 1) > 1e-9:
            raise AssertionError(f"after_backward did not override the runtime means lr: {lr}")
    dens.cfg["skip_scale_big_gauss"] = True
    # post phase restarts the position schedule (train_post.py update_learning_rate(1))
    dens._set_position_lr(optimizers, cfg["leaves_steps"] + 1)
    lr_post = optimizers["means"].param_groups[0]["lr"]
    if abs(lr_post / (expon_lr(1, 2e-5, 2e-7, 30000) * dens.extent) - 1) > 1e-9:
        raise AssertionError("position lr is not restarted for hierarchy optimization")
    # extent: 1.1 x 0.9-quantile (not max) of camera distances, set after initialize
    radii_c = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0]
    cams = [{"viewmat": look_at((r * math.cos(i), r * math.sin(i), 0.0))}
            for i, r in enumerate(radii_c)]
    centres = np.array([[r * math.cos(i), r * math.sin(i), 0.0] for i, r in enumerate(radii_c)])
    ref = 1.1 * np.quantile(np.linalg.norm(centres - centres.mean(0), axis=1), 0.9)
    dens.set_training_cameras(cams)
    if abs(dens.extent / ref - 1) > 1e-6 or dens.state["extent"] != dens.extent:
        raise AssertionError(f"extent {dens.extent} != 1.1 * q0.9 = {ref}")
    lr = optimizers["means"].param_groups[0]["lr"]
    if abs(lr / (expon_lr(1, 2e-5, 2e-7, 30000) * ref) - 1) > 1e-6:
        raise AssertionError("position lr not rescaled to the quantile extent")
    # sparse Adam: rows without opacity gradient keep parameters and moments
    p = nn.Parameter(torch.zeros(4, 2))
    o = nn.Parameter(torch.zeros(4))
    mask = StepMask({"opacities": o})
    opt = SparseAdam([p], lr=0.1, mask=mask)
    for it, rows in enumerate(([0, 1, 2, 3], [0, 1])):
        mask.begin(it)
        o.grad = torch.zeros(4)
        o.grad[rows] = 1.0
        p.grad = torch.ones(4, 2)
        opt.step()
    if float(p[2:].sub(-0.1).abs().max()) > 1e-6 or float(p[:2].sub(-0.2).abs().max()) > 1e-3:
        raise AssertionError(f"sparse Adam did not freeze invisible rows: {p}")
    lr0 = expon_lr(1, 2e-5, 2e-7, 30000)
    lr1 = expon_lr(30000, 2e-5, 2e-7, 30000)
    if abs(lr0 / 2e-5 - 1) > 1e-3 or abs(lr1 / 2e-7 - 1) > 1e-6:
        raise AssertionError("position lr schedule is not 2e-5 -> 2e-7")
    return {"max_statistic": 0.3, "split_scale": 0.01 / 1.6, "sparse_adam_rows_frozen": 2,
            "shrink_default": "off", "extent_q90": float(ref)}
