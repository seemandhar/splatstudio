"""Mechanism probes. Each one compares against an independently written formula or
implementation and raises if the mechanism is removed or wrong."""

import math

import torch
from splatstudio import ComponentSpec
from splatstudio.state import check_state, sample_scene

from . import core
from .density import DEFAULTS, ImprovedGSDensity
from .optim import Gate, GatedAdam


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _density(method=None, **cfg):
    spec = ComponentSpec("improvedgs-density", "density", "x", ("gradient.screen.inria_abs",),
                         (), {**DEFAULTS, "budget": 1000, **cfg})
    return ImprovedGSDensity(spec)


def _random_scene(n, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    means[:, 2] += 4.0
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.log(torch.rand(n, 3, generator=g) * 0.15 + 0.03)),
        "quats": torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(n, 4, generator=g),
                                                                  dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.rand(n, generator=g) * 0.8 + 0.1)),
        "sh0": torch.nn.Parameter(torch.randn(n, 1, 3, generator=g) * 0.5),
        "shN": torch.nn.Parameter(torch.randn(n, 15, 3, generator=g) * 0.05),
    }).to(device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[80.0, 0, 48], [0, 80.0, 32], [0, 0, 1]], device=device),
              "width": 96, "height": 64}
    return params, camera


# ---- renderer --------------------------------------------------------------------------------
def raster_vendored(method, device):
    _require_cuda()
    from gsplat import rasterization

    from .raster import rasterize

    params, camera = _random_scene(200, device)
    bg = torch.zeros(3, device=device)
    color, radii, _, _, means2D = rasterize(params, camera, 3, bg)
    ref, _, _ = rasterization(
        params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        torch.cat((params["sh0"], params["shN"]), 1), camera["viewmat"][None], camera["K"][None],
        camera["width"], camera["height"], sh_degree=3, packed=False)
    diff = (color.permute(1, 2, 0) - ref[0]).abs()
    if float(diff.mean()) > 5e-3 or float(diff.max()) > 0.1:
        raise AssertionError(f"vendored RGB differs from gsplat: mean {diff.mean():.4g}")
    target = torch.rand_like(color)
    (color - target).square().mean().backward()
    g = means2D.grad
    if g is None or not torch.count_nonzero(g[:, 2:4]):
        raise AssertionError("no absolute screen gradient in channels 2:4")
    if torch.any(g[:, 2:4] + 1e-9 < g[:, 0:2].abs()):
        raise AssertionError("absolute accumulation smaller than |signed sum|")
    return {"rgb_mean_abs_diff_vs_gsplat": float(diff.mean()), "visible": int((radii > 0).sum()),
            "abs_over_signed": float(g[:, 2:4].norm() / g[:, 0:2].norm().clamp_min(1e-12))}


def _edge_loop(image):
    """Independent edge map: explicit 8-neighbour Laplacian on the rounded gray image."""
    rgb = (image.double() * 255).round()
    gray = ((299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2]) / 1000).round()
    H, W = gray.shape
    pad = torch.zeros(H + 2, W + 2, dtype=torch.float64)
    pad[1:-1, 1:-1] = gray
    out = 8 * gray
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out = out - pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    out = out.clamp(0, 255) / 255
    return (out - out.min()) / (out.max() - out.min())


def eas_score(method, device):
    _require_cuda()
    from gsplat import rasterization

    from .raster import blend_weights

    # Edge map vs explicit loop, and an analytic case.
    img = torch.rand(20, 30, 3, generator=torch.Generator().manual_seed(1))
    em = core.edge_map(img)
    ref = _edge_loop(img)
    if float((em.double() - ref).abs().max()) > 1e-5:
        raise AssertionError("edge_map differs from the explicit Laplacian")
    flat = core.edge_map(torch.full((8, 8, 3), 0.5))  # zero padding: only the border responds
    if torch.count_nonzero(flat[1:-1, 1:-1]) or float(flat[0, 0]) != 1.0:
        raise AssertionError("constant image: interior must have no edges, corners maximal")

    params, camera = _random_scene(150, device, seed=3)
    H, W = camera["height"], camera["width"]
    w = torch.rand(H, W, device=device, generator=torch.Generator(device).manual_seed(2))
    accum, radii = blend_weights(params, camera, w)
    # Independent: d/dc_i sum_p w_p C_p with scalar per-Gaussian colours c_i = sum_p w_p T alpha.
    colors = torch.zeros(len(params["means"]), 1, device=device, requires_grad=True)
    img, _, _ = rasterization(
        params["means"].detach(), params["quats"].detach(), params["scales"].detach().exp(),
        params["opacities"].detach().sigmoid(), colors, camera["viewmat"][None],
        camera["K"][None], W, H, packed=False)
    (img[0, ..., 0] * w).sum().backward()
    ref = colors.grad[:, 0]
    rel = float((accum - ref).abs().sum() / ref.abs().sum().clamp_min(1e-12))
    if rel > 0.02:
        raise AssertionError(f"edge-weighted blend weights differ from reference ({rel:.3g})")
    # Score prefers the Gaussian that covers edge pixels.
    w2 = torch.zeros(H, W, device=device)
    w2[:, : W // 2] = 1.0
    a2, r2 = blend_weights(params, camera, w2)
    left = params["means"][:, 0] < -0.3
    right = params["means"][:, 0] > 0.3
    if not float(a2[left & (r2 > 0)].mean()) > float(a2[right & (r2 > 0)].mean()):
        raise AssertionError("edge weights do not steer the score")
    return {"blend_weight_rel_err_vs_gsplat": rel, "visible": int((radii > 0).sum())}


# ---- densification ---------------------------------------------------------------------------
def _cpu_scene(n, seed=0):
    params, camera = _random_scene(n, "cpu", seed)
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    return params, opts


def eas_sampling(method, device):
    pool = core.EdgeViewPool(25, 10, seed=0)
    sizes = [len(pool.sample()) for _ in range(6)]
    if sizes != [10, 10, 5, 10, 10, 5]:
        raise AssertionError(f"view pool sizes {sizes}")
    d = _density()
    d.cameras = [None] * 25
    d.pool = core.EdgeViewPool(25, 10, seed=0)
    full = {s: len(d.eas_views(s)) for s in (3400, 3500, 6400, 9400)}
    if full != {3400: 25, 3500: 10, 6400: 25, 9400: 10}:
        raise AssertionError(f"full-view schedule wrong: {full}")
    # Sampling: qualifiers only, frequency proportional to score.
    n = 400
    params, opts = _cpu_scene(n)
    d = _density(budget=10**6)
    d.initialize(params, opts, 1.0)
    scores = torch.linspace(0.01, 1.0, n)
    qual = torch.zeros(n, dtype=torch.bool)
    qual[::2] = True
    counts = torch.zeros(n)
    for t in range(200):
        d.gen.manual_seed(t)
        w = scores.clone()
        w[~qual] = 0
        idx = torch.multinomial(w, 20, replacement=False, generator=d.gen)
        counts[idx] += 1
    if counts[~qual].sum() != 0:
        raise AssertionError("non-qualifier selected")
    hi, lo = counts[qual][-50:].mean(), counts[qual][:50].mean()
    if not hi > 3 * lo:
        raise AssertionError("selection frequency does not follow the score")
    # End-to-end through the density: selected parents are qualifiers.
    before = params["means"].detach().clone()
    k = d._long_axis_split(params, opts, scores, 30, qual)
    kept = len(params["means"]) - 2 * k
    survivors = params["means"][:kept]
    removed = ~torch.isin(before[:, 0], survivors[:, 0])
    if k != 30 or bool((removed & ~qual).any()):
        raise AssertionError("LAS parents are not all qualifiers")
    return {"pool_sizes": sizes, "full_view_steps": full, "freq_ratio_top_bottom": float(hi / lo)}


def las_geometry(method, device):
    params, opts = sample_scene(method, "cpu")
    n = len(params["means"])
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():  # allocate real Adam moments first
        o.step()
        o.zero_grad()
    with torch.no_grad():
        params["scales"][0] = torch.log(torch.tensor([0.3, 0.1, 0.05]))
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)  # 90 deg about z: x -> y
        params["quats"][0] = torch.tensor([c, 0.0, 0.0, s])
        params["opacities"][0] = torch.logit(torch.tensor(0.5))
    parent = params["means"][0].detach().clone()
    d = _density()
    d.initialize(params, opts, 1.0)
    sel_score = torch.zeros(n)
    sel_score[0] = 1.0
    qual = torch.zeros(n, dtype=torch.bool)
    qual[0] = True
    k = d._long_axis_split(params, opts, sel_score, 1, qual)
    check_state(method, params, opts)
    if k != 1 or len(params["means"]) != n + 1:
        raise AssertionError("LAS must replace one parent by two children")
    ch = params["means"][-2:].detach()
    offs = ch - parent
    expect = torch.tensor([0.0, 3 * 0.45 * 0.3, 0.0])
    torch.testing.assert_close(offs[0], expect, atol=1e-5, rtol=0)
    torch.testing.assert_close(offs[1], -expect, atol=1e-5, rtol=0)
    sc = params["scales"][-2:].detach().exp()
    r = math.sqrt(1 - 0.45**2)
    torch.testing.assert_close(sc[0], torch.tensor([0.3 * 0.55, 0.1 * r, 0.05 * r]), atol=1e-6,
                               rtol=1e-5)
    op = params["opacities"][-2:].detach().sigmoid()
    torch.testing.assert_close(op, torch.full((2,), 0.3), atol=1e-6, rtol=1e-5)
    for name, p in params.items():
        m = opts[name].state[p]["exp_avg"][-2:]
        if torch.count_nonzero(m):
            raise AssertionError(f"{name}: children moments must be zero")
    return {"child_offset": offs[0].tolist(), "child_scales": sc[0].tolist(),
            "child_opacity": float(op[0])}


def growth_control(method, device):
    n_max = 3_000_000
    for it in (500, 600, 1000, 5000, 9000, 14400, 14500, 14900):
        p = min(max((it - 500) / 14000, 0.0), 1.0)
        ref = n_max if p >= 1 else max(int(math.sqrt(p) * n_max), 1)
        if core.budget(it, n_max) != ref:
            raise AssertionError(f"budget({it}) wrong")
    n = 300
    params, opts = _cpu_scene(n)
    d = _density(budget=1000, use_eas=False)
    d.initialize(params, opts, 1.0)
    step = 4000
    target = core.budget(step, 1000)
    d.stats["accum_abs"][:] = 1.0  # every Gaussian qualifies
    d.stats["denom"][:] = 1.0
    with torch.no_grad():
        params["opacities"][:] = 0.0
    d._densify(params, opts, step)
    if len(params["means"]) != target:
        raise AssertionError(f"count {len(params['means'])} != budget {target}")
    d.stats["accum_abs"][:] = 1.0
    d.stats["denom"][:] = 1.0
    d._densify(params, opts, step)  # already at budget: no growth
    if len(params["means"]) != target:
        raise AssertionError("densification exceeded the budget")
    return {"budget_at_4000": target, "count": len(params["means"])}


def rap_schedule(method, device):
    n = 1000
    params, opts = _cpu_scene(n)
    with torch.no_grad():
        params["opacities"][:] = torch.logit(torch.linspace(0.005, 0.9, n))
    d = _density(use_eas=False)
    d.initialize(params, opts, 1.0)
    g = torch.Generator().manual_seed(0)
    events = {}
    for step in range(1, 15001):
        if step > 3000 and step % 3000 == 150:  # simulated, uneven opacity recovery after each reset
            with torch.no_grad():
                m = len(params["opacities"])
                params["opacities"][:] = torch.logit(torch.rand(m, generator=g) * 0.8 + 0.1)
        before = len(params["means"])
        op_before = float(params["opacities"].detach().sigmoid().max())
        d._rap(params, opts, step)
        after = len(params["means"])
        op_after = float(params["opacities"].detach().sigmoid().max())
        if after != before:
            events[step] = ("prune", before - after, before)
        elif op_after < op_before - 1e-6:
            events[step] = ("reset", op_after)
    resets = sorted(s for s, e in events.items() if e[0] == "reset")
    prunes = sorted(s for s, e in events.items() if e[0] == "prune")
    if resets != [3000, 6000, 9000, 12000]:
        raise AssertionError(f"resets at {resets}")
    if prunes != [300, 3300, 6300]:
        raise AssertionError(f"prunes at {prunes}")
    if any(abs(events[s][1] - 0.05) > 1e-5 for s in resets):
        raise AssertionError("reset value is not 0.05")
    early = 0.02
    expect_early = int((torch.linspace(0.005, 0.9, n) < early).sum())
    if events[300][1] != expect_early:
        raise AssertionError("early prune must remove exactly opacity < 0.02")
    fracs = [events[s][1] / events[s][2] for s in (3300, 6300)]
    if any(abs(f - 0.2) > 0.01 for f in fracs):
        raise AssertionError(f"RAP must prune the lowest 20 %: {fracs}")
    return {"resets": resets, "prunes": prunes, "early_pruned": events[300][1],
            "rap_fractions": fracs}


def mu_gated_adam(method, device):
    rule = core.mu_update
    expect = {1: True, 19999: True, 20000: True, 20001: False, 20005: True, 23995: True,
              24000: True, 24005: False, 24020: True, 29980: True, 30000: False}
    got = {s: rule(s) for s in expect}
    if got != expect:
        raise AssertionError(f"MU schedule {got}")
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(4))
    gate = Gate(lambda s: s % 5 == 0)
    opt = GatedAdam(torch.optim.Adam([p], lr=0.1, eps=1e-15), gate)
    ref = torch.optim.Adam([q], lr=0.1, eps=1e-15)
    grads = [torch.tensor([1.0, -2.0, 3.0, 0.5]) * (i + 1) for i in range(5)]
    for i, g in enumerate(grads, start=1):
        gate.step = i
        p.grad = g.clone() if p.grad is None else p.grad + g
        opt.step()
        opt.zero_grad()
        if i < 5 and torch.count_nonzero(p):
            raise AssertionError("parameter changed on a skipped step")
    q.grad = sum(grads)
    ref.step()
    torch.testing.assert_close(p.detach(), q.detach())
    if opt.skipped != 4:
        raise AssertionError("skip count wrong")
    return {"schedule": {str(k): v for k, v in got.items()}, "skipped": opt.skipped}


def lr_schedule(method, device):
    params, opts = sample_scene(method, "cpu")
    d = _density()
    d.initialize(params, opts, 2.0)
    if abs(opts["opacities"].param_groups[0]["lr"] - 0.025) > 1e-12:
        raise AssertionError("opacity lr is not 0.025")
    if not isinstance(opts["means"], GatedAdam):
        raise AssertionError("optimizers were not gated for MU")
    out = {}
    for s in (1, 7000, 15000, 29999):
        lr = d.xyz_lr(s)
        t = s / 30000
        ref = 2.0 * 4e-5 * (2e-6 / 4e-5) ** t
        if abs(lr - ref) > 1e-12 * max(1.0, ref) + 1e-15:
            raise AssertionError(f"lr({s}) = {lr} != {ref}")
        out[str(s)] = lr
    if abs(opts["means"].param_groups[0]["lr"] - d.xyz_lr(1)) > 1e-15:
        raise AssertionError("initial means lr not set")
    return {"means_lr": out}
