"""Mechanism probes for the BOGausS port. Each raises AssertionError when its mechanism is
removed or wrong; the references are written independently of core.py / optim.py / ops.py."""

import math
from dataclasses import replace

import torch

from . import core, ops
from .density import BOGausSDensity
from .optim import BOGausSAdam


def _density(method, **config):
    spec = next(c for c in method.components if c.kind == "density")
    return BOGausSDensity(replace(spec, config={**spec.config, "log": False, **config}))


def _scene(n, device, seed=0, depth=3.0, spread=0.6, scale=0.08, opacity=None):
    from splatstudio.state import initialize

    from . import method as make

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += depth
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(make(), pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_(torch.full((n, 3), math.log(scale), device=device)
                               + 0.3 * torch.randn(n, 3, generator=g).to(device))
        q = torch.randn(n, 4, generator=g)
        params["quats"].copy_(torch.nn.functional.normalize(q, dim=-1).to(device))
        o = torch.rand(n, generator=g) * 0.8 + 0.1 if opacity is None else opacity
        params["opacities"].copy_(torch.logit(torch.as_tensor(o, dtype=torch.float32)).to(device))
    return params, opts


def _camera(device, w=40, h=32, f=30.0, shift=0.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    viewmat = torch.eye(4, device=device)
    viewmat[0, 3] = shift
    return {"K": K, "viewmat": viewmat, "width": w, "height": h}


# ------------------------------------------------------------------ C1 precision
def precision(method, device):
    g = torch.Generator().manual_seed(0)
    means = (torch.rand(50, 3, generator=g) - 0.5) * 0.5
    means[:, 2] += 2.0
    cams = []
    for k in range(3):
        ang = 0.2 * (k - 1)
        R = torch.tensor([[math.cos(ang), 0, math.sin(ang)], [0, 1.0, 0],
                          [-math.sin(ang), 0, math.cos(ang)]])
        vm = torch.eye(4)
        vm[:3, :3] = R
        vm[0, 3] = 0.1 * k
        f = 40.0 + 10 * k
        cams.append({"viewmat": vm, "K": torch.tensor([[f, 0, 32.0], [0, f * 1.1, 24.0],
                                                       [0, 0, 1.0]]), "width": 64, "height": 48})
    sigma_uv, alpha = 1.5, 2.0
    got, seen = core.precision_delta(means.to(device), cams, alpha, sigma_uv, margin=10.0)
    # independent reference: explicit per-camera pseudo-inverse lateral precision matrices
    ref = []
    for x in means.double():
        prec = torch.zeros(3, 3, dtype=torch.float64)
        for cam in cams:
            R, t, K = cam["viewmat"][:3, :3].double(), cam["viewmat"][:3, 3].double(), cam["K"]
            z = float((R @ x + t)[2])
            D = torch.diag(torch.tensor([float(K[0, 0]) ** 2, float(K[1, 1]) ** 2, 0.0],
                                        dtype=torch.float64))
            prec += R.T @ D @ R / (z * z * sigma_uv**2)
        ref.append(alpha * float(torch.trace(prec)) ** -0.5)
    ref = torch.tensor(ref, dtype=torch.float32)
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-4, atol=1e-7)
    if not bool((seen == 3).all()):
        raise AssertionError("All Gaussians should be seen by the three wide cameras")
    twice, _ = core.precision_delta(means.to(device), cams + cams, alpha, sigma_uv, margin=10.0)
    torch.testing.assert_close(twice.cpu(), ref / math.sqrt(2), rtol=1e-4, atol=1e-7)
    far = means.clone()
    far[:, 2] *= 2.0
    one = [{**cams[0], "viewmat": torch.eye(4)}]
    d1, _ = core.precision_delta(means.to(device), one, alpha, sigma_uv, margin=10.0)
    d2, _ = core.precision_delta(far.to(device), one, alpha, sigma_uv, margin=10.0)
    torch.testing.assert_close((d2 / d1).cpu(), torch.full((50,), 2.0), rtol=1e-4, atol=1e-5)
    return {"median_delta": float(got.median()), "max_rel_err":
            float(((got.cpu() - ref) / ref).abs().max())}


# ------------------------------------------------------------------ C2 optimizer
def unbiased_sparse_adam(method, device):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, 3, device=device))
    p0 = p.detach().clone()
    opt = BOGausSAdam([p], lr=0.01, mode="position")
    opt.visible = torch.tensor([True, False, True, False], device=device)
    opt.delta = torch.tensor([1.0, 1.0, 0.5, 2.0], device=device)
    g1, g2 = torch.randn(4, 3, device=device), torch.randn(4, 3, device=device)
    p.grad = g1.clone()
    opt.step()
    st = opt.state[p]
    if st["t"].tolist() != [1.0, 0.0, 1.0, 0.0]:
        raise AssertionError("Lifespan t must count only visible updates")
    exp1 = p0 - 0.01 * opt.delta[:, None] * torch.sign(g1)
    exp1[1], exp1[3] = p0[1], p0[3]
    torch.testing.assert_close(p.detach(), exp1, rtol=1e-5, atol=1e-6)
    # second step, Gaussian 1 visible for the first time: unbiased (= its gradient sign)
    opt.visible = torch.tensor([True, True, False, False], device=device)
    p.grad = g2.clone()
    opt.step()
    m0 = g1[0] + 0.5 * (g2[0] - g1[0])
    v0 = g1[0] ** 2 + 0.5 * (g2[0] ** 2 - g1[0] ** 2)
    exp0 = exp1[0] - 0.01 * m0 / (v0.sqrt() + 1e-15)
    torch.testing.assert_close(p.detach()[0], exp0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p.detach()[1], p0[1] - 0.01 * torch.sign(g2[1]), rtol=1e-5,
                               atol=1e-6)
    torch.testing.assert_close(p.detach()[2], exp1[2])  # invisible: frozen, no momentum drift
    # torch Adam on the same sequence differs (dense momentum moves invisible rows)
    q = torch.nn.Parameter(p0.clone())
    adam = torch.optim.Adam([q], lr=0.01, eps=1e-15)
    for g in (g1, g2):
        q.grad = g.clone()
        adam.step()
    if float((q.detach() - p.detach()).abs().max()) < 1e-3:
        raise AssertionError("BOGausS optimizer behaves like dense Adam")
    return {"t": st["t"].tolist(), "diff_vs_adam": float((q.detach() - p.detach()).abs().max())}


def linear_size(method, device):
    sigma0 = torch.tensor([[0.5, 0.2, 0.1], [0.3, 0.3, 0.3]], device=device)
    p = torch.nn.Parameter(sigma0.log())
    opt = BOGausSAdam([p], lr=0.1, mode="size", min_size_factor=1.0)
    opt.visible = torch.tensor([True, True], device=device)
    opt.delta = torch.tensor([0.5, 0.01], device=device)
    grad_log = torch.tensor([[1.0, -1.0, 1.0], [2.0, -3.0, 0.5]], device=device)
    p.grad = grad_log.clone()
    opt.step()
    # first unbiased step: sigma - eta * Delta * sign(dL/dsigma), clamped at Delta
    lin = sigma0 - 0.1 * opt.delta[:, None] * torch.sign(grad_log / sigma0)
    lin = torch.maximum(lin, opt.delta[:, None].expand_as(lin))
    torch.testing.assert_close(p.detach().exp(), lin, rtol=1e-5, atol=1e-6)
    # a log-space update would give sigma * exp(-eta * sign): must differ
    logu = sigma0 * torch.exp(-0.1 * torch.sign(grad_log))
    if float((logu - p.detach().exp()).abs().max()) < 1e-3:
        raise AssertionError("Size update is logarithmic")
    if not bool((p.detach().exp()[0] >= 0.5 - 1e-6).all()):
        raise AssertionError("Minimal size Delta not enforced")
    return {"sigma_after": p.detach().exp().tolist()}


# ------------------------------------------------------------------ C3 inheritance
def state_inheritance(method, device):
    n = 6
    params, opts = _scene(n, device, seed=4)
    d = _density(method)
    d.initialize(params, opts, 1.0)
    d.state["delta"] = torch.full((n,), 1e-4, device=device)
    for name, opt in opts.items():
        p = params[name]
        opt.state[p] = {"exp_avg": torch.randn_like(p), "exp_avg_sq": torch.rand_like(p),
                        "t": torch.arange(n, device=device).float() + 5}
    before = {k: {s: v.clone() for s, v in opts[k].state[params[k]].items()} for k in params}
    sel = torch.tensor([1, 4], device=device)
    ops.split(params, opts, d.state, sel, 0.3, alpha_t=1.0, alpha_g=0.2)
    for k in params:
        p = params[k]
        if not p.requires_grad or not p.is_leaf:
            raise AssertionError(f"{k}: split froze the parameter")
        st = opts[k].state[p]
        b = before[k]
        for idx_new, idx_m in ((1, 1), (4, 4), (n, 1), (n + 1, 4)):
            torch.testing.assert_close(st["exp_avg"][idx_new], 0.2 * b["exp_avg"][idx_m])
            torch.testing.assert_close(st["exp_avg_sq"][idx_new], 0.04 * b["exp_avg_sq"][idx_m])
            torch.testing.assert_close(st["t"][idx_new], b["t"][idx_m])
        for keep in (0, 2, 3, 5):
            torch.testing.assert_close(st["exp_avg"][keep], b["exp_avg"][keep])
    return {"count": len(params["means"])}


# ------------------------------------------------------------------ C4/C6 statistics kernel
def _brute_stats(meta, err):
    m2d, con, op = (meta[k][0].detach().cpu() for k in ("means2d", "conics", "opacities"))
    err = err.detach().cpu()
    offs, ids = meta["isect_offsets"][0].flatten().tolist(), meta["flatten_ids"].tolist()
    ts, W, H = meta["tile_size"], meta["width"], meta["height"]
    tw = meta["tile_width"]
    n = m2d.shape[0]
    out = {k: [0.0] * n for k in ("blend", "se", "count")}
    for i in range(H):
        for j in range(W):
            tile = (i // ts) * tw + (j // ts)
            end = offs[tile + 1] if tile + 1 < len(offs) else len(ids)
            T, px, py = 1.0, j + 0.5, i + 0.5
            for k in range(offs[tile], end):
                gid = ids[k]
                dx, dy = float(m2d[gid, 0]) - px, float(m2d[gid, 1]) - py
                c = con[gid].tolist()
                s = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
                a = min(0.999, float(op[gid]) * math.exp(-s))
                if s < 0 or a < 1 / 255:
                    continue
                if T * (1 - a) <= 1e-4:
                    break
                out["blend"][gid] += a * T
                out["se"][gid] += a * T * float(err[i, j])
                out["count"][gid] += 1
                T *= 1 - a
    return {k: torch.tensor(v) for k, v in out.items()}


def stats_kernel(method, device):
    from gsplat import rasterization

    from .kernels import pixel_stats

    params, _ = _scene(30, device, seed=1)
    cam = _camera(device)
    with torch.no_grad():
        _, alpha, meta = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["sh0"], params["shN"]), dim=1),
            viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=cam["width"],
            height=cam["height"], sh_degree=0, packed=False,
        )
    err = torch.rand(cam["height"], cam["width"], generator=torch.Generator().manual_seed(2))
    got = pixel_stats(meta, err.to(device))
    ref = _brute_stats(meta, err)
    for k in ref:
        torch.testing.assert_close(got[k].cpu(), ref[k], rtol=1e-4, atol=1e-3)
    if float(ref["count"].sum()) < 200:
        raise AssertionError("Probe scene too sparse")
    torch.testing.assert_close(got["blend"].sum().cpu(), alpha.sum().cpu(), rtol=1e-3, atol=1e-2)
    return {"pairs": float(ref["count"].sum())}


# ------------------------------------------------------------------ C4 pruning
def prune(method, device):
    n = 1000
    params, opts = _scene(n, device, seed=5)
    d = _density(method, budget=n // 2)  # target below N after pruning: no splits
    d.initialize(params, opts, 1.0)
    g = torch.Generator().manual_seed(6)
    pmax = torch.rand(n, generator=g).to(device) * 0.2 + 0.03
    low = torch.randperm(n, generator=g)[:25].to(device)
    pmax[low] = torch.rand(25, generator=g).to(device) * 0.019
    d.state["pmax"] = pmax.clone()
    d.state["delta"] = torch.full((n,), 1e-3, device=device)
    ids = torch.arange(n, device=device, dtype=torch.float32)
    with torch.no_grad():
        params["means"][:, 0] = ids  # tag each Gaussian by its index
    d.densify(params, opts, 1000)
    kept = params["means"][:, 0].detach().round().long()
    removed = sorted(set(range(n)) - set(kept.tolist()))
    expected = sorted(low[torch.argsort(pmax[low])[:10]].tolist())  # cap 1% of 1000
    if removed != expected:
        raise AssertionError(f"Pruned {removed}, expected lowest-P_prune {expected}")
    return {"pruned": len(removed)}


# ------------------------------------------------------------------ C5 split
def split(method, device):
    n = 3
    params, opts = _scene(n, device, seed=7)
    with torch.no_grad():
        params["scales"].copy_(torch.tensor([[0.1, 0.4, 0.05]] * n, device=device).log())
        params["quats"].copy_(torch.tensor([[1.0, 0, 0, 0]] * n, device=device))
        params["opacities"].copy_(torch.logit(torch.tensor([0.2, 0.6, 0.95], device=device)))
    state = {"delta": torch.full((n,), 1e-4, device=device)}
    m0 = params["means"].detach().clone()
    o0 = params["opacities"].detach().sigmoid().clone()
    ops.split(params, opts, state, torch.arange(n, device=device), 0.3)
    if len(params["means"]) != 2 * n:
        raise AssertionError("Each split must produce exactly two children (no clone)")
    shift = torch.tensor([0.0, 0.3 * 0.4, 0.0], device=device)
    torch.testing.assert_close(params["means"].detach()[:n], m0 + shift)
    torch.testing.assert_close(params["means"].detach()[n:], m0 - shift)
    s = params["scales"].detach().exp()
    torch.testing.assert_close(s[:, [0, 2]], torch.tensor([[0.1, 0.05]] * 2 * n, device=device))
    # adaptation must preserve the mother's alpha profile better than unadapted children
    x = torch.linspace(-4.5, 4.5, 181, dtype=torch.float64)
    out = {}
    for i, o in enumerate(o0.tolist()):
        ratio = float(s[i, 1]) / 0.4
        oc = float(params["opacities"].detach()[i].sigmoid())
        target = o * torch.exp(-0.5 * x * x)

        def prof(r, c):
            g1 = c * torch.exp(-0.5 * ((x - 0.3) / r) ** 2)
            g2 = c * torch.exp(-0.5 * ((x + 0.3) / r) ** 2)
            return 1 - (1 - g1) * (1 - g2)

        e_adapt = float((prof(ratio, oc) - target).square().sum())
        e_naive = float((prof(1.0, o) - target).square().sum())
        e_3dgs = float((prof(1 / 1.6, o) - target).square().sum())
        if not (e_adapt < 0.25 * e_naive and e_adapt < e_3dgs) or not (oc < o and ratio < 1):
            raise AssertionError(f"SplitAdaptation does not preserve density (o={o})")
        out[f"o{o:.2f}"] = {"ratio": ratio, "o_child": oc, "err": e_adapt, "naive": e_naive}
    return out


# ------------------------------------------------------------------ C6 priority
def split_priority(method, device):
    n = 8
    params, opts = _scene(n, device, seed=8, opacity=torch.full((n,), 0.5))
    d = _density(method, budget=n + 2, densify_until=1000, densify_from=500)
    d.initialize(params, opts, 1.0)
    d.state["delta"] = torch.full((n,), 1e-4, device=device)
    d.state["pmax"][:] = 1.0
    # camera A: large total error, B: small; absolute SE favours 0,1 (camera A),
    # relative SE favours 5,6 (dominant in camera B)
    se_a = torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0, 0, 0, 3.0], device=device)
    se_b = torch.tensor([0, 0, 0, 0, 0, 0.9, 0.8, 0.1], device=device)
    for se in (se_a, se_b):
        d.state["psnr"] += core.snr_priority_increment(se)
    absolute = torch.topk(se_a + se_b, 2).indices.sort().values.tolist()
    with torch.no_grad():
        params["means"][:, 0] = torch.arange(n, device=device).float()
    d.densify(params, opts, 1000)
    children = sorted(params["means"].detach()[n:, 0].round().long().tolist())
    if children != [5, 6] or absolute == [5, 6]:
        raise AssertionError(f"Split {children}; expected P_SNR top-2 [5, 6]")
    return {"split": children, "absolute_se_would_split": absolute}


# ------------------------------------------------------------------ schedule
def schedule(method, device):
    d = _density(method)
    c = d.cfg
    steps = core.densify_steps(c["densify_from"], c["densify_until"], c["densify_interval"])
    if steps[0] != 1000 or steps[-1] != 15000 or len(steps) != 29:
        raise AssertionError("Densification must run every 500 steps from 1000 to 15000")
    S, B = 1000, 30000
    t = [core.count_target(s, S, B, 500, 15000, 500) for s in steps]
    if t[-1] != B or any(b < a for a, b in zip(t, t[1:])) or t[0] <= S:
        raise AssertionError("Growth targets must rise monotonically to the budget")
    # no opacity reset: a densification at step 3000 without prune/split leaves opacities
    n = 20
    params, opts = _scene(n, device, seed=9)
    d = _density(method, budget=n)
    d.initialize(params, opts, 1.0)
    d.state["pmax"][:] = 1.0
    d.state["delta"] = torch.full((n,), 1e-4, device=device)
    o0 = params["opacities"].detach().clone()
    radii = torch.ones(1, n, 2, device=device)
    d.after_backward(params, opts, 3000, {"info": {"radii": radii}, "step": 3000})
    torch.testing.assert_close(params["opacities"].detach(), o0)
    if d.history[-1]["step"] != 3000:
        raise AssertionError("Densification did not run at step 3000")
    return {"densify_steps": len(steps), "first_targets": t[:3]}
