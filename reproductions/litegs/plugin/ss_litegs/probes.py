"""Mechanism probes for the LiteGS port. Each raises AssertionError when the mechanism is removed or
wrong; the references below are written independently of density.py / kernels.py."""

import math
from dataclasses import replace

import torch

from . import schedule
from .density import LiteDensity, growth_budget, variance_score
from .optim import LiteAdam, Visibility


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _density(method, **config):
    spec = _spec(method, "density")
    return LiteDensity(replace(spec, config={**spec.config, **config}))


def _scene(n, device, seed=0, opacity_max=0.85, scale=0.08):
    from splatstudio.state import initialize

    from . import method as make

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * 0.8
    pts[:, 2] += 3.0
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(make(), pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_((math.log(scale) + 0.3 * torch.randn(n, 3, generator=g)).to(device))
        q = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
        params["quats"].copy_(q.to(device))
        o = torch.rand(n, generator=g) * (opacity_max - 0.1) + 0.1
        params["opacities"].copy_(torch.logit(o).to(device))
        params["shN"].copy_((0.1 * torch.randn(n, 15, 3, generator=g)).to(device))
    return params, opts


def _camera(device, w=24, h=20, f=22.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": w, "height": h}


# ---------------------------------------------------------------------------- C1: CUDA moments
def stats_kernel(method, device):
    if not str(device).startswith("cuda"):
        raise AssertionError("litegs-stats-kernel requires CUDA")
    from gsplat import rasterization

    from .kernels import opacity_grad_moments

    schedule.reset()
    params, opts = _scene(14, device)
    cam = _camera(device)
    density = _density(method, target_primitives=100)
    colors = density._colors(params, cam, 3, torch.ones(14, dtype=torch.bool, device=device))
    o = params["opacities"].detach().sigmoid().clone().requires_grad_(True)

    def render(op):
        img, alpha, meta = rasterization(
            params["means"].detach(), params["quats"].detach(), params["scales"].detach().exp(),
            op, colors[None], cam["viewmat"][None], cam["K"][None], cam["width"], cam["height"],
            sh_degree=None, packed=False)
        return img[0] + (1 - alpha[0]), meta

    # colours used by the kernel reproduce gsplat's own SH path
    img_sh, alpha_sh, _ = rasterization(
        params["means"].detach(), params["quats"].detach(), params["scales"].detach().exp(),
        o.detach(), torch.cat((params["sh0"], params["shN"]), 1).detach(), cam["viewmat"][None],
        cam["K"][None], cam["width"], cam["height"], sh_degree=3, packed=False)
    rgb, meta = render(o)
    assert torch.allclose(img_sh[0] + (1 - alpha_sh[0]), rgb.detach(), atol=1e-5)
    H, W = rgb.shape[:2]
    gen = torch.Generator(device="cpu").manual_seed(3)
    v = torch.randn(H, W, 3, generator=gen).to(device)
    m = opacity_grad_moments(meta, colors, rgb.detach(), v)
    # independent: per-pixel autograd through gsplat's backward
    S = torch.zeros(14, device=device)
    M = torch.zeros(14, device=device)
    for i in range(H):
        for j in range(W):
            (g,) = torch.autograd.grad((rgb[i, j] * v[i, j]).sum(), o, retain_graph=True)
            S += g.square()
            M += g
    (g_total,) = torch.autograd.grad((rgb * v).sum(), o)
    # independent: count and blend weight by a per-pixel front-to-back loop (gsplat numerics)
    m2d, con, op = (meta[k][0].detach().cpu() for k in ("means2d", "conics", "opacities"))
    offs, ids = meta["isect_offsets"][0].flatten().tolist(), meta["flatten_ids"].tolist()
    ts, tw = meta["tile_size"], meta["tile_width"]
    cnt, wt = torch.zeros(14), torch.zeros(14)
    for i in range(H):
        for j in range(W):
            tile = (i // ts) * tw + (j // ts)
            end = offs[tile + 1] if tile + 1 < len(offs) else len(ids)
            T = 1.0
            for k in ids[offs[tile]:end]:
                dx, dy = float(m2d[k, 0]) - (j + 0.5), float(m2d[k, 1]) - (i + 0.5)
                a, b, c = (float(x) for x in con[k])
                sig = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                alpha = min(0.999, float(op[k]) * math.exp(-sig))
                if sig < 0 or alpha < 1 / 255:
                    continue
                if T * (1 - alpha) <= 1e-4:
                    break
                cnt[k] += 1
                wt[k] += T * alpha
                T *= 1 - alpha
    assert torch.allclose(m["sum"], M, rtol=1e-3, atol=1e-4), (m["sum"], M)
    assert torch.allclose(m["sum"], g_total, rtol=1e-3, atol=1e-4)
    assert torch.allclose(m["sq_sum"], S, rtol=1e-3, atol=1e-5), (m["sq_sum"], S)
    assert torch.equal(m["count"].cpu(), cnt)
    assert torch.allclose(m["weight"].cpu(), wt, rtol=1e-4, atol=1e-5)
    # the variance term is informative: S differs from M^2/C for overlapping Gaussians
    gap = (m["sq_sum"] - m["sum"].square() / m["count"].clamp_min(1)).abs().max().item()
    assert gap > 1e-4
    return {"max_S_err": (m["sq_sum"] - S).abs().max().item(),
            "max_M_err": (m["sum"] - M).abs().max().item(), "fragments": int(cnt.sum()),
            "variance_gap": gap}


def variance_score_probe(method, device):
    # two primitives with 4 fragments each: A conflicting (sum 2), B consistent (sum 4)
    ga = torch.tensor([2.0, -1.0, 2.0, -1.0])
    gb = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sq = torch.stack([ga.square().sum(), gb.square().sum()])
    sm = torch.stack([ga.sum(), gb.sum()])
    cnt = torch.tensor([4.0, 4.0])
    o = torch.tensor([0.5, 0.5])
    score = variance_score(sq, sm, cnt, o, 2)
    ref = [max(float(s) / 5 - (float(mm) / 5) ** 2, 0) * 4 * 0.25 for s, mm in zip(sq, sm)]
    assert torch.allclose(score, torch.tensor(ref)), (score, ref)
    assert score[0] > score[1], "conflicting per-pixel gradients must score higher"
    mag = variance_score(sq, sm, cnt, o, 2, kind="magnitude")
    assert mag[1] > mag[0], "magnitude ablation should rank the consistent primitive higher"
    # opacity^2 weighting and nan/negative guards
    s2 = variance_score(sq, sm, cnt, torch.tensor([1.0, 0.5]), 2)
    assert torch.isclose(s2[0], score[0] * 4)
    z = variance_score(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]),
                       torch.tensor([1.0]))
    assert float(z) == 0.0
    return {"score_conflicting": float(score[0]), "score_consistent": float(score[1])}


def densify_probe(method, device):
    schedule.reset()
    n = 40
    params, opts = _scene(n, device, seed=1)
    with torch.no_grad():
        params["sh0"][:, 0, 0].copy_(torch.arange(n, device=device, dtype=torch.float32))
        big = torch.arange(n, device=device) % 2 == 0
        params["scales"][big] = math.log(0.05)   # > 0.01 * extent(=1) -> split
        params["scales"][~big] = math.log(0.005)  # <= 0.01 -> clone
    target = 120
    d = _density(method, target_primitives=target, log=False)
    d.initialize(params, opts, scene_scale=1.0)
    d.set_training_cameras([None] * 10)  # 30000 steps -> 3000 epochs, until 2401
    schedule.reset()
    for p in params.values():
        p.grad = torch.ones_like(p)
    d.visibility.mask = torch.ones(n, dtype=torch.bool, device=device)
    for opt in opts.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    gen = torch.Generator(device="cpu").manual_seed(0)
    weight = torch.rand(n, generator=gen) + 0.1
    weight[[3, 7, 11]] = 0
    count = torch.randint(1, 30, (n,), generator=gen).float()
    count[[5, 6]] = 0  # never seen -> score 0
    sq = torch.rand(n, generator=gen) * 10 + 1
    sm = torch.rand(n, generator=gen)
    d.stats = {"weight": weight.to(device), "count": count.to(device), "sq_sum": sq.to(device),
               "sum": sm.to(device)}
    before = {k: v.detach().clone() for k, v in params.items()}
    epoch = 5
    d.densify(params, opts, epoch)
    cur = (target - n) / (d.until - 3) * (epoch - 3) + n
    budget = min(max(int(cur - n), 1) + 3, n)
    assert budget == growth_budget(epoch, n, 3, n, target, 3, d.until)
    assert len(params["means"]) == n + budget - 3, (len(params["means"]), budget)
    keep = torch.ones(n, dtype=torch.bool)
    keep[[3, 7, 11]] = False
    kept = keep.nonzero()[:, 0].to(device)
    for k in params:
        assert torch.equal(params[k][: len(kept)], before[k][kept]), f"parent {k} changed"
    new = slice(len(kept), None)
    parent = params["sh0"][new, 0, 0].round().long()
    assert not bool(((parent == 5) | (parent == 6)).any()), "sampled a zero-score primitive"
    assert len(torch.unique(parent)) == len(parent), "sampling must be without replacement"
    is_big = big[parent]
    ps = before["scales"][parent]
    assert torch.allclose(params["scales"][new][is_big], ps[is_big] - math.log(1.6), atol=1e-6)
    assert torch.equal(params["scales"][new][~is_big], ps[~is_big])
    assert torch.equal(params["means"][new][~is_big], before["means"][parent][~is_big])
    off = (params["means"][new][is_big] - before["means"][parent][is_big]).norm(dim=-1)
    assert bool((off > 0).all()) and bool((off < 0.05 * 6).all())
    for name, opt in opts.items():
        p = params[name]
        st = opt.state[p]
        assert st["exp_avg"].shape == p.shape
        assert torch.count_nonzero(st["exp_avg"][new]) == 0, "new moments must be zero"
        assert torch.count_nonzero(st["exp_avg"][: len(kept)]) > 0
        assert p.requires_grad and p.is_leaf
    return {"budget": budget, "n_after": len(params["means"]), "split": int(is_big.sum()),
            "clone": int((~is_big).sum())}


def decay_schedule(method, device):
    schedule.reset()
    params, opts = _scene(30, device, seed=2)
    with torch.no_grad():
        params["opacities"][:3] = torch.logit(torch.tensor([0.01, 0.002, 0.999]))
    d = _density(method, target_primitives=100, log=False)
    d.initialize(params, opts, scene_scale=1.0)
    for p in params.values():
        p.grad = torch.ones_like(p)
    d.visibility.mask = torch.ones(30, dtype=torch.bool, device=device)
    for opt in opts.values():
        opt.step()
    o0 = params["opacities"].detach().sigmoid().clone()
    pid = id(params["opacities"])
    d.decay(params, opts)
    o1 = params["opacities"].detach().sigmoid()
    ref = torch.maximum(o0 * 0.5, torch.full_like(o0, 1 / 128))
    assert torch.allclose(o1, ref, atol=1e-6), "decay must halve opacity with a 1/128 floor"
    assert id(params["opacities"]) == pid and params["opacities"].requires_grad
    assert all(len(opt.state) == 0 for opt in opts.values()), "Adam states must be cleared"
    # schedule for garden (161 training views, 30000 steps)
    d.set_training_cameras([None] * 161)
    schedule.reset()
    assert (d.total_epoch, d.until) == (186, 141), (d.total_epoch, d.until)
    dens = [e for e in range(d.total_epoch) if d._stats_epoch(e)]
    dec = [e for e in range(d.total_epoch)
           if schedule.is_decay_epoch(e, 3, d.until, 10)]
    assert dens == list(range(5, 141, 5)) and dec == list(range(10, 141, 10)), (dens, dec)
    assert schedule.epoch_of(161 * 6, 161) == 5 and schedule.epoch_of(161 * 6 + 1, 161) == 6
    # room (272 training views)
    assert schedule.densify_until(schedule.total_epochs(30000, 272), 10) == 81
    return {"garden_densify_epochs": len(dens), "garden_decay_epochs": len(dec),
            "until": d.until}


def adam_probe(method, device):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 3))
    q = torch.nn.Parameter(p.detach().clone())
    vis = Visibility()
    opt = LiteAdam([p], lr=0.01, visibility=vis)
    ref_opt = torch.optim.Adam([q], lr=0.01, eps=1e-15)
    mask = torch.tensor([True, False, True, True, False, True])
    m = torch.zeros(6, 3)
    v = torch.zeros(6, 3)
    ref = p.detach().clone()
    for _ in range(3):
        g = torch.randn(6, 3)
        p.grad = g.clone()
        q.grad = g.clone()
        vis.mask = mask
        opt.step()
        ref_opt.step()
        m[mask] = 0.9 * m[mask] + 0.1 * g[mask]
        v[mask] = 0.999 * v[mask] + 0.001 * g[mask] ** 2
        ref[mask] -= 0.01 * m[mask] / (v[mask].sqrt() + 1e-15)
    assert torch.allclose(p.detach(), ref, atol=1e-6), "LiteAdam != bias-free visible-only Adam"
    assert torch.count_nonzero(opt.state[p]["exp_avg"][~mask]) == 0
    assert not torch.allclose(p.detach(), q.detach(), atol=1e-4), "must differ from torch Adam"
    vis.mask = torch.ones(5, dtype=torch.bool)
    p.grad = torch.ones(6, 3)
    try:
        opt.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("misaligned visibility must raise")
    return {"max_err": (p.detach() - ref).abs().max().item()}


def sh_schedule(method, device):
    from .renderer import create_renderer

    r = create_renderer(_spec(method, "renderer"))
    params, _ = _scene(10, device, seed=4)
    with torch.no_grad():
        params["sh0"].fill_(3.0)  # bright colours -> rgb above 1 with the white background
    cam = _camera(device)
    try:
        schedule.bind(10)
        degrees = {}
        for last in (0, 49, 50, 99, 100, 149, 150, 400):
            schedule.SHARED["step"] = last
            out = r.render(params, cam, sh_degree=3)
            degrees[last + 1] = out["sh_degree"]
        assert degrees == {1: 0, 50: 0, 51: 1, 100: 1, 101: 2, 150: 2, 151: 3, 401: 3}, degrees
        assert float(out["rgb"].max()) <= 1.0 and float(out["rgb_raw"].max()) > 1.0
        with torch.no_grad():
            schedule.SHARED["step"] = 0
            ev = r.render(params, cam, sh_degree=3)
        assert ev["sh_degree"] == 3
    finally:
        schedule.reset()
    return {"degrees": {str(k): v for k, v in degrees.items()}}
