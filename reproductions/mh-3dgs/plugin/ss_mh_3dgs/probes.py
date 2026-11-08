"""Mechanism probes for MH-3DGS. Each raises if its mechanism is removed or altered."""

import math
from collections import defaultdict

import numpy as np
import torch
from splatstudio import ComponentSpec

from . import core
from .density import MHDensity, append_gaussians, check_trainable
from .losses import create_regularizers


def _density(config=None):
    return MHDensity(ComponentSpec("mh-density", "density", "x", (), (), config or {}))


def _scene(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) - 0.5
    pts[:, 2] += 3
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    from splatstudio.methods import make_method
    from splatstudio.state import initialize

    params, opts = initialize(make_method("x"), pts.to(device), cols.to(device))
    with torch.no_grad():
        params["opacities"].copy_(torch.randn(n, generator=g).to(device))
        params["scales"].copy_((torch.rand(n, 3, generator=g) * -3 - 2).to(device))
    return params, opts


def importance_formula(method, device="cpu"):
    c = dict(core.DEFAULTS)
    g = torch.Generator().manual_seed(1)
    n, h, w = 500, 23, 31
    means = (torch.rand(n, 3, generator=g) * 3 - 1).double()
    opa = torch.rand(n, generator=g).double()
    smap = torch.rand(3, h, w, generator=g).double()
    lmap = torch.rand(3, h, w, generator=g).double()
    s, l1 = core.lookup_world_xy(means, smap, lmap)
    got = core.importance(opa, s, l1, c).numpy()
    # Independent numpy evaluation of the official gaussian_model.py code.
    m, sm, lm = means.numpy(), smap.numpy(), lmap.numpy()
    xs = np.clip(np.trunc(m[:, 0] * (w - 1)).astype(int), 0, w - 1)
    ys = np.clip(np.trunc(m[:, 1] * (h - 1)).astype(int), 0, h - 1)
    ss = sm[:, ys, xs].mean(0)
    ll = lm[:, ys, xs].mean(0)

    def z(x):
        return (x - x.mean()) / (x.std(ddof=1) + 1e-6)

    ref = 1 / (1 + np.exp(-(0.8 * z(opa.numpy()) + 0.5 * z(ss) + 0.5 * z(ll))))
    err = float(np.abs(got - ref).max())
    if err > 1e-9:
        raise AssertionError(f"importance differs from the official formula: {err}")
    # Lookup must be the world-xy one: a projected lookup would not match `ss`.
    if not np.allclose(s.numpy(), ss):
        raise AssertionError("pixel lookup is not the official world-xy lookup")
    return {"max_abs_error": err, "n": n}


def voxel_acceptance(method, device="cpu"):
    c = dict(core.DEFAULTS)
    g = torch.Generator().manual_seed(2)
    v = 0.02
    # Existing: 4 points in voxel (0,0,0), none in voxel (10,0,0); random background points.
    crowd = torch.tensor([[0.005, 0.005, 0.005]]) + torch.rand(4, 3, generator=g) * 0.01
    other = torch.rand(300, 3, generator=g) * 2 + 1
    existing = torch.cat([crowd, other])
    # Independent dict-loop voxel counts (official compute_voxel_counts).
    counts = defaultdict(int)
    for p in existing.double().numpy():
        counts[tuple(math.floor(x / v) for x in p)] += 1
    queries = torch.cat([torch.rand(50, 3, generator=g) * 4 - 1, crowd[:1]])
    got = core.voxel_counts_at(existing, queries, v).tolist()
    ref = [counts.get(tuple(math.floor(x / v) for x in q), 0) for q in queries.double().numpy()]
    if got != ref:
        raise AssertionError("voxel counts differ from the dict-loop reference")
    # Acceptance frequency in a crowded (c=4) and an empty voxel with I = 0.7.
    trials = 200000
    imp = torch.full((trials,), 0.7)
    rates = {}
    for cnt in (0, 4):
        rho = core.acceptance(imp, torch.full((trials,), cnt), c)
        acc = (torch.rand(trials, generator=g) < rho).float().mean().item()
        exp = core.expected_accept(0.7, cnt, c)
        if abs(acc - exp) > 0.01:
            raise AssertionError(f"acceptance {acc} != expected {exp} for count {cnt}")
        rates[cnt] = acc
    if not rates[4] < 0.1 * rates[0] * 1.2:
        raise AssertionError("crowding penalty missing")
    return {"accept_empty": rates[0], "accept_crowded4": rates[4],
            "expected": [core.expected_accept(0.7, 0, c), core.expected_accept(0.7, 4, c)]}


def birth_proposals(method, device="cpu"):
    c = dict(core.DEFAULTS)
    params, opts = _scene(400, "cpu")
    # Allocate Adam moments.
    loss = sum(p.sum() for p in params.values())
    loss.backward()
    for o in opts.values():
        o.step()
        o.zero_grad()
    before = {k: v.detach().clone() for k, v in params.items()}
    gen = torch.Generator().manual_seed(3)
    denom = torch.zeros(400)
    denom[:200] = 1
    imp = torch.rand(400, generator=gen)
    means = params["means"].detach()
    scales = params["scales"].exp().detach()
    # Statistical std check with huge voxels off: many proposals from one parent.
    one = torch.zeros(400)
    one[7] = 1.0
    cfg = dict(c, lambda_v=0.0, accept_shift=-50.0)  # accept everything
    offs = []
    for _ in range(4000):
        idx, prop, _, _ = core.propose(means, scales, denom, one, 1, 10.0, means, 0.02, cfg, gen)
        offs.append(prop - means[idx])
    offs = torch.cat(offs)
    expect = 10.0 * scales[7] / (1 + math.log(2.0))  # denom[7] = 1
    ratio = (offs.std(0) / expect)
    if not torch.all((ratio > 0.95) & (ratio < 1.05)):
        raise AssertionError(f"proposal std ratio {ratio.tolist()}")
    idx, prop, n_prop, _ = core.propose(means, scales, denom, imp, 300, 2.0, means, 0.02, c, gen)
    if n_prop > 300 or len(idx) == 0:
        raise AssertionError("unexpected proposal count")
    append_gaussians(params, opts, idx, prop)
    check_trainable(params)
    n0 = 400
    if len(params["means"]) != n0 + len(idx):
        raise AssertionError("population did not grow by the accepted count")
    for k, p in params.items():
        if not torch.equal(p[:n0].detach(), before[k]):
            raise AssertionError(f"parents changed: {k}")
        if k == "means":
            if not torch.equal(p[n0:].detach(), prop):
                raise AssertionError("child means are not the proposals")
        elif not torch.equal(p[n0:].detach(), before[k][idx]):
            raise AssertionError(f"child {k} not copied from parent")
        st = opts[k].state[p]
        if not (st["exp_avg"][n0:] == 0).all() or not (st["exp_avg_sq"][n0:] == 0).all():
            raise AssertionError("child Adam moments not zero")
    return {"std_ratio": ratio.tolist(), "proposed": n_prop, "accepted": len(idx)}


def relocation(method, device="cuda"):
    if not str(device).startswith("cuda"):
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("gsplat relocation kernel needs CUDA")
    from gsplat.relocation import compute_relocation

    params, opts = _scene(300, device)
    with torch.no_grad():
        params["opacities"][:50] = -10.0  # dead
    loss = sum(p.sum() for p in params.values())
    loss.backward()
    for o in opts.values():
        o.step()
        o.zero_grad()
    with torch.no_grad():
        params["opacities"][:50] = -10.0
    d = _density()
    d.initialize(params, opts)
    from gsplat.strategy.ops import relocate

    alive_before = params["means"][50:].detach().clone()
    torch.manual_seed(0)
    dead = params["opacities"].sigmoid() <= d.c["opacity_threshold"]
    relocate(params, opts, {}, dead, d._binoms(params["means"].device), min_opacity=0.005)
    check_trainable(params)
    if len(params["means"]) != 300:
        raise AssertionError("relocation changed the population size")
    if (params["opacities"].sigmoid() <= 0.005).any():
        raise AssertionError("dead Gaussians remain")
    # Each revived Gaussian sits on an alive one.
    dist = (params["means"][:50].detach()[:, None] - alive_before[None]).abs().amax(-1).min(1).values
    if dist.max() > 1e-6:
        raise AssertionError("dead Gaussians not moved onto alive ones")
    # Formula check against the MCMC kernel for N=2 vs closed form 1-(1-o)^(1/2).
    o = torch.tensor([0.6], device=device)
    s = torch.ones(1, 3, device=device)
    no, _ = compute_relocation(o, s, torch.tensor([2], device=device), d._binoms(device))
    if abs(float(no) - (1 - (1 - 0.6) ** 0.5)) > 1e-5:
        raise AssertionError("relocation opacity formula")
    return {"relocated": 50, "max_dist": float(dist.max())}


def schedules(method, device="cpu"):
    c = dict(core.DEFAULTS)
    events = [s for s in range(1, 30001) if core.schedule(s, c) is not None]
    if events[0] != 600 or events[-1] != 14900 or len(events) != 144:
        raise AssertionError(f"event grid {events[0]}..{events[-1]} ({len(events)})")
    for step in (600, 7500, 14900):
        p = (step - 500) / 14500
        e = core.schedule(step, c)
        ref = {"coarse_scale": 10 * (1 - p) + 5 * p, "fine_scale": 2 * (1 - p) + p,
               "voxel_size": 0.02 * (1 - p) + 0.005 * p,
               "coarse_batch": max(1, int((1 - p) * 4500)),
               "fine_batch": max(1, int((1 - p) * 16000))}
        for k, v in ref.items():
            if abs(e[k] - v) > 1e-12:
                raise AssertionError(f"{k} at {step}: {e[k]} != {v}")
    n = 161
    k = core.subset_size(n, core.schedule(600, c)["progress"])
    if k != int(n * (1 - 100 / 14500)):
        raise AssertionError("subset size")
    rr = core.round_robin(n, 600, k)
    if rr[0] != (600 * k) % n or len(set(rr)) != k:
        raise AssertionError("round robin")
    if core.subset_size(n, 0.9999) != 1:
        raise AssertionError("subset floor at 1")
    return {"events": len(events), "k_at_600": k, "k_at_14900":
            core.subset_size(n, core.schedule(14900, c)["progress"])}


def multiview_maps(method, device="cuda"):
    if not str(device).startswith("cuda"):
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("rendering needs CUDA")
    from .renderer import render

    params, opts = _scene(2000, device)
    cams, imgs = [], []
    g = torch.Generator().manual_seed(4)
    for i in range(5):
        vm = torch.eye(4)
        vm[0, 3] = 0.1 * (i - 2)
        cams.append({"K": torch.tensor([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]], device=device),
                     "viewmat": vm.to(device), "width": 64, "height": 48})
        imgs.append(torch.rand(48, 64, 3, generator=g))
    d = _density()
    d.initialize(params, opts)
    d.set_training_frames(cams, imgs)
    views = [1, 3, 4]
    with torch.no_grad():
        s_agg, l_agg = d.error_maps(params, views, 0)
        bg = torch.zeros(3, device=device)
        ref_s = ref_l = 0
        for v in views:
            rgb = render(params, cams[v], 0, bg)["rgb"].clamp(0, 1).permute(2, 0, 1)
            gt = imgs[v].to(device).permute(2, 0, 1)
            ref_s = ref_s + core.ssim_map(rgb, gt) / 3
            ref_l = ref_l + (rgb - gt).abs() / 3
        single_s, _ = d.error_maps(params, [1], 0)
    es, el = (s_agg - ref_s).abs().max().item(), (l_agg - ref_l).abs().max().item()
    if es > 1e-5 or el > 1e-5:
        raise AssertionError(f"aggregated maps differ: {es} {el}")
    if (single_s - s_agg).abs().max().item() < 1e-4:
        raise AssertionError("maps do not depend on the view subset")
    # SSIM map sanity: identical images give 1.
    one = core.ssim_map(imgs[0].permute(2, 0, 1).to(device), imgs[0].permute(2, 0, 1).to(device))
    if (one - 1).abs().max().item() > 1e-4:
        raise AssertionError("SSIM map of identical images is not 1")
    return {"max_err_ssim": es, "max_err_l1": el, "shape": list(s_agg.shape)}


def regularizers(method, device="cpu"):
    spec = ComponentSpec("r", "loss", "x", (), (), {"opacity_weight": 0.01, "scale_weight": 0.01})
    fn = create_regularizers(spec)
    params, _ = _scene(50, "cpu")
    loss = fn(params, {}, None)
    loss.backward()
    o, s = params["opacities"].detach(), params["scales"].detach()
    go = 0.01 * torch.sigmoid(o) * (1 - torch.sigmoid(o)) / 50
    gs = 0.01 * s.exp() / 150
    eo = (params["opacities"].grad - go).abs().max().item()
    es = (params["scales"].grad - gs).abs().max().item()
    if eo > 1e-9 or es > 1e-9:
        raise AssertionError(f"regulariser gradients differ {eo} {es}")
    return {"opacity_grad_err": eo, "scale_grad_err": es}


def mh_step_end_to_end(method, device="cuda"):
    """Runs MHDensity.after_backward at a densification step on a fixture with dead Gaussians:
    relocation must revive them, the population must grow by the births, statistics reset,
    and the optimizer state stays aligned."""
    if not str(device).startswith("cuda"):
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("rendering needs CUDA")
    from splatstudio.methods import make_method
    from splatstudio.state import check_state

    from .renderer import render

    params, opts = _scene(3000, device)
    with torch.no_grad():
        params["opacities"][:100] = -10.0
    cams, imgs = [], []
    g = torch.Generator().manual_seed(5)
    for i in range(4):
        vm = torch.eye(4)
        vm[0, 3] = 0.1 * (i - 2)
        cams.append({"K": torch.tensor([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]], device=device),
                     "viewmat": vm.to(device), "width": 64, "height": 48})
        imgs.append(torch.rand(48, 64, 3, generator=g))
    d = _density()
    d.initialize(params, opts)
    d.set_training_frames(cams, imgs)
    out = render(params, cams[0], 0, torch.zeros(3, device=device))
    out["rgb"].mean().backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    with torch.no_grad():
        params["opacities"][:100] = -10.0
    n0 = len(params["means"])
    d.after_backward(params, opts, 600, out)
    check_state(make_method("x"), params, opts)
    check_trainable(params)
    h = d.history[-1]
    if h["dead_relocated"] != 100:
        raise AssertionError(f"relocation not executed in _step: {h}")
    if (params["opacities"][:100].sigmoid() <= 0.005).any():
        raise AssertionError("dead Gaussians remain after _step")
    if h["born"] <= 0 or len(params["means"]) != n0 + h["born"]:
        raise AssertionError(f"births not appended: {h}")
    if len(d.denom) != len(params["means"]) or d.denom.any():
        raise AssertionError("statistics not reset after births")
    n1 = len(params["means"])
    d.after_backward(params, opts, 650, out)  # off-grid: no event
    if len(params["means"]) != n1:
        raise AssertionError("event fired off the densification grid")
    return {k: h[k] for k in ("views", "dead_relocated", "born", "gaussians")}
