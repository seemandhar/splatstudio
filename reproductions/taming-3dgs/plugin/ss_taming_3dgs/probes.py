"""Mechanism probes for the Taming 3DGS port. Each raises AssertionError when the mechanism is
removed or wrong; the independent references are written separately from density.py/core.py."""

import math
from dataclasses import replace

import torch

from . import core
from .density import TamingDensity
from .optim import IntervalAdam


def _density(method, **config):
    spec = next(c for c in method.components if c.kind == "density")
    return TamingDensity(replace(spec, config={**spec.config, **config}))


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


def _brute_stats(meta, weights):
    """Per-pixel front-to-back loop (gsplat numerics) over each tile's list -> stats + alpha."""
    m2d, con, op = (meta[k][0].detach().cpu() for k in ("means2d", "conics", "opacities"))
    weights = weights.detach().cpu()
    offs, ids = meta["isect_offsets"][0].flatten().tolist(), meta["flatten_ids"].tolist()
    ts, W, H = meta["tile_size"], meta["width"], meta["height"]
    tw = meta["tile_width"]
    n = m2d.shape[0]
    out = {k: [0.0] * n for k in ("saliency", "count", "blend", "dist")}
    alpha_img = torch.zeros(H, W)
    for i in range(H):
        for j in range(W):
            tile = (i // ts) * tw + (j // ts)
            end = offs[tile + 1] if tile + 1 < len(offs) else len(ids)
            T, px, py = 1.0, j + 0.5, i + 0.5
            for k in range(offs[tile], end):
                gid = ids[k]
                dx, dy = float(m2d[gid, 0]) - px, float(m2d[gid, 1]) - py
                c = con[gid].tolist()
                sigma = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
                a = min(0.999, float(op[gid]) * math.exp(-sigma))
                if sigma < 0 or a < 1 / 255:
                    continue
                if T * (1 - a) <= 1e-4:
                    break
                out["saliency"][gid] += float(weights[i, j])
                out["count"][gid] += 1
                out["blend"][gid] += a * T
                out["dist"][gid] += math.hypot(dx, dy)
                T *= 1 - a
            alpha_img[i, j] = 1 - T
    return {k: torch.tensor(v) for k, v in out.items()}, alpha_img


def _rasterize(params, cam):
    from gsplat import rasterization

    with torch.no_grad():
        rgb, alpha, meta = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["sh0"], params["shN"]), dim=1),
            viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=cam["width"],
            height=cam["height"], sh_degree=0, packed=False,
        )
    return rgb[0] + (1 - alpha[0]), alpha[0], meta


def stats_kernel(method, device):
    from .kernels import pixel_stats

    params, _ = _scene(30, device, seed=1)
    cam = _camera(device)
    _, alpha, meta = _rasterize(params, cam)
    weights = torch.rand(cam["height"], cam["width"], generator=torch.Generator().manual_seed(2))
    got = pixel_stats(meta, weights.to(device))
    ref, alpha_ref = _brute_stats(meta, weights)
    for k in ref:
        torch.testing.assert_close(got[k].cpu(), ref[k], rtol=1e-4, atol=1e-3)
    if float(ref["count"].sum()) < 200:
        raise AssertionError("Probe scene too sparse")
    # independent of the tile traversal: per-pixel blend weights sum to gsplat's alpha
    torch.testing.assert_close(alpha_ref, alpha[..., 0].cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got["blend"].sum().cpu(), alpha.sum().cpu(), rtol=1e-3, atol=1e-2)
    return {"pairs": float(ref["count"].sum()), "max_dist_err":
            float((got["dist"].cpu() - ref["dist"]).abs().max())}


def growth_curve(method, device):
    S, B = 1000, 21000
    counts = core.count_array(S, B, 500, 15000, 100)
    N = (15000 - 500) // 100
    ref = [S + (B - S) * (1 - (1 - x / N) ** 2) for x in range(N)]
    if len(counts) != N or counts[0] != S:
        raise AssertionError("Count array length/start wrong")
    if max(abs(a - b) for a, b in zip(counts, ref)) > 1.0:
        raise AssertionError("Count array is not the budget parabola")
    if any(b < a for a, b in zip(counts, counts[1:])) or not (B - 300 < counts[-1] <= B):
        raise AssertionError("Count array must rise monotonically to the budget")
    # the j-th densification call reaches counts[j + 1] (up to the official floor division of
    # the clone/split split) when candidates suffice
    n = 400
    params, opts = _scene(n, device, seed=3, opacity=torch.full((n,), 0.9))
    d = _density(method, budget=n * 3, mode="final_count")
    d.initialize(params, opts, scene_scale=1.0)
    d.state["grad_accum"][:] = 1.0
    d.state["count"][:] = 1.0
    d.densify(params, opts, 600, scores=torch.rand(n, device=device) + 0.1)
    if not d.counts[1] - 1 <= len(params["means"]) <= d.counts[1] or d.counts[1] <= n:
        raise AssertionError(f"Population {len(params['means'])} != target {d.counts[1]}")
    return {"targets": counts[:3], "after_first_call": len(params["means"])}


def _ref_normalize(w, x):
    x = torch.nan_to_num(x.float().cpu(), nan=0.0)
    pos = sorted(x[x > 0].tolist())
    out = torch.zeros_like(x)
    if pos:
        med = pos[(len(pos) - 1) // 2]
        out[x > 0] = w * x[x > 0] / med
    return out


def score(method, device):
    from PIL import Image, ImageFilter

    params, opts = _scene(40, device, seed=4)
    d = _density(method, budget=100)
    d.initialize(params, opts, scene_scale=1.0)
    g = torch.Generator().manual_seed(5)
    d.state["grad_accum"][:] = torch.rand(40, generator=g).to(device) * 1e-3
    d.state["count"][:] = 2.0
    views = []
    for shift in (0.0, 0.1):
        cam = _camera(device, shift=shift)
        img8 = (torch.rand(cam["height"], cam["width"], 3, generator=g) * 255).to(torch.uint8)
        views.append({**cam, "image": img8, "edges": core.edge_map(img8)})
    got = d.score(params, 2500, views).cpu()

    w = core.SCORE_WEIGHTS
    ref = torch.zeros(40)
    for v in views:
        rgb, _, meta = _rasterize(params, v)
        rgb = rgb.cpu()
        gt = v["image"].float() / 255
        l1 = (rgb - gt).abs().mean()
        p_i = 0.8 * l1 + 0.2 * (1 - core.ssim_same(rgb, gt))
        e = torch.from_numpy(__import__("numpy").asarray(
            Image.fromarray(v["image"].numpy()).convert("L").filter(ImageFilter.FIND_EDGES)
        ).copy()).float() / 255
        e = (e - e.min()) / (e.max() - e.min())
        lm = (rgb - gt).abs().mean(-1)
        sal = 50 * (lm - lm.min()) / (lm.max() - lm.min()) + 50 * e
        st, _ = _brute_stats(meta, sal)
        vis = (meta["radii"][0] > 0).all(-1).cpu()
        depth = meta["depths"][0].cpu() * vis
        rad = meta["radii"][0].max(-1).values.float().cpu() * vis
        gr = (d.state["grad_accum"] / d.state["count"]).cpu()
        gi = (_ref_normalize(25, gr) + _ref_normalize(100, params["opacities"].sigmoid())
              + _ref_normalize(5, depth) + _ref_normalize(10, rad)
              + _ref_normalize(25, params["scales"].exp().prod(-1)))
        pi = (_ref_normalize(50, st["dist"]) + _ref_normalize(10, st["saliency"])
              + _ref_normalize(0.1, st["count"]) + _ref_normalize(50, st["blend"]))
        ref += torch.where(vis, w["view_importance"] * p_i * (pi + gi), torch.zeros(40))
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=1e-2)
    # opacity steering: same Gaussian at low opacity must score lower
    with torch.no_grad():
        params["opacities"][0] = torch.logit(torch.tensor(0.9))
    hi = d.score(params, 2500, views)[0]
    with torch.no_grad():
        params["opacities"][0] = torch.logit(torch.tensor(0.02))
    lo = d.score(params, 2500, views)[0]
    if not lo < hi:
        raise AssertionError("Low-opacity Gaussian should get a lower score")
    return {"score_sum": float(got.sum()), "hi": float(hi), "lo": float(lo)}


def sampled_densify(method, device):
    n = 200
    params, opts = _scene(n, device, seed=6, opacity=torch.full((n,), 0.9))
    with torch.no_grad():
        params["scales"][:50] = math.log(0.005)   # clone candidates (<= 0.01 extent)
        params["scales"][50:100] = math.log(0.05)  # split candidates
    # allocate moments
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    d = _density(method, budget=n + 40 * 144)
    d.initialize(params, opts, scene_scale=1.0)
    d.counts = [n, n + 40, n + 80]
    d.state["grad_accum"][:100] = 1.0
    d.state["count"][:] = 1.0
    scores = torch.full((n,), 1e-9, device=device)
    scores[:25] = 1.0
    scores[50:75] = 1.0
    scores[100:] = 5.0  # high score but no gradient qualification
    before = params["means"].detach().clone()
    d.densify(params, opts, 600, scores=scores)
    h = d.history[-1]
    if (h["clones"], h["splits"], h["count"]) != (20, 20, n + 40) or h["clone_candidates"] != 50:
        raise AssertionError(f"Budget split wrong: {h}")
    # clones are appended right after the kept originals; children at the very end
    kept = n - 20
    clone_src = params["means"][kept:kept + 20].detach()
    dist = (clone_src[:, None] - before[None]).norm(dim=-1)
    src = dist.argmin(dim=1)
    if not bool((dist.min(dim=1).values < 1e-6).all()) or not bool((src < 25).all()):
        raise AssertionError("Clones must come from high-score, small, qualified Gaussians")
    kept_src = (params["means"][:kept].detach()[:, None] - before[None]).norm(dim=-1)
    removed = set(range(n)) - set(kept_src.argmin(dim=1).tolist())
    if not removed <= set(range(50, 75)):
        raise AssertionError("Split parents must be high-score large qualified Gaussians")
    for name, p in params.items():
        m = opts[name].state.get(p, {}).get("exp_avg")
        if m is not None and torch.count_nonzero(m[kept:]):
            raise AssertionError("New Gaussians must start with zero Adam moments")
    return h


def guided_prune(method, device):
    torch.manual_seed(0)
    n = 100
    op = torch.full((n,), 0.9)
    op[:40] = 0.001  # prune mask: 40 Gaussians
    params, opts = _scene(n, device, seed=7, opacity=op)
    d = _density(method, budget=n)
    d.initialize(params, opts, scene_scale=1.0)
    d.counts = [n, n]
    scores = torch.ones(n, device=device)
    scores[:20] = 1e3   # high score inside the prune mask -> rarely sampled for removal
    d.densify(params, opts, 1000, scores=scores)
    left = params["opacities"].sigmoid().detach()
    pruned = n - len(left)
    if not 0 < pruned <= 20:
        raise AssertionError(f"Expected up to 50% of the 40 prune candidates, got {pruned}")
    if bool((left > 0.5).sum() != 60):
        raise AssertionError("Only prune-mask Gaussians may be removed")
    params2, opts2 = _scene(n, device, seed=7, opacity=op)
    d2 = _density(method, budget=n)
    d2.initialize(params2, opts2, scene_scale=1.0)
    d2.counts = [n, n]
    d2.calls = 27
    d2.densify(params2, opts2, 1000, scores=scores)
    if len(params2["means"]) != n:
        raise AssertionError("No pruning after densification call 27")
    return {"pruned": pruned}


def high_opacity(method, device):
    params, opts = _scene(8, device, seed=8)
    d = _density(method, budget=8)
    d.initialize(params, opts, scene_scale=1.0)
    if abs(opts["opacities"].param_groups[0]["lr"] - 0.025) > 1e-12:
        raise AssertionError("Opacity lr must be 0.025")

    def train_step():
        before = params["opacities"].detach().clone()
        params["opacities"].sum().backward()
        opts["opacities"].step()
        opts["opacities"].zero_grad(set_to_none=True)
        return float((params["opacities"] - before).abs().max())

    moved = train_step()
    d.after_backward(params, opts, 15000, {})
    frozen = max(train_step() for _ in range(3))
    if moved <= 0 or frozen != 0.0:
        raise AssertionError(f"Opacity must train before 15000 and freeze after ({moved}, {frozen})")
    return {"moved_before": moved, "moved_after": frozen}


def sh_interval(method, device):
    torch.manual_seed(0)
    params, opts = _scene(6, device, seed=9)
    d = _density(method, budget=6)
    d.initialize(params, opts, scene_scale=1.0)
    opt = opts["shN"]
    if not isinstance(opt, IntervalAdam) or abs(opt.param_groups[0]["lr"] - 2.5e-4) > 1e-12:
        raise AssertionError("shN must use IntervalAdam with lr 2.5e-4")
    p = params["shN"]
    ref = torch.nn.Parameter(p.detach().clone())
    ref_opt = torch.optim.Adam([ref], lr=2.5e-4, eps=1e-15)
    per = torch.nn.Parameter(p.detach().clone())
    per_opt = torch.optim.Adam([per], lr=2.5e-4, eps=1e-15)
    acc = torch.zeros_like(ref)
    for it in range(1, 49):
        gr = torch.randn_like(ref)
        p.grad = gr.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        per.grad = gr.clone()
        per_opt.step()
        acc += gr
        if it % 16 == 0:
            ref.grad = acc
            ref_opt.step()
            acc = torch.zeros_like(ref)
    torch.testing.assert_close(p.detach(), ref.detach())
    if torch.allclose(p.detach(), per.detach()):
        raise AssertionError("Interval updates must differ from per-step Adam")
    # replacing the parameter mid-window drops the partial sum (official new nn.Parameter)
    opt2 = IntervalAdam([torch.nn.Parameter(torch.zeros(3))], lr=1.0, every=4)
    q = opt2.param_groups[0]["params"][0]
    q.grad = torch.full((3,), 100.0)
    opt2.step()
    q2 = torch.nn.Parameter(torch.zeros(3))
    opt2.param_groups[0]["params"][0] = q2
    for _ in range(3):
        q2.grad = torch.full((3,), -1.0)
        opt2.step()
    if not bool((q2 > 0).all()):
        raise AssertionError("Accumulated gradient of a replaced parameter must be dropped")
    return {"max_diff_vs_per_step": float((p - per).abs().max())}


def schedule(method, device):
    params, opts = _scene(6, device, seed=10, opacity=torch.full((6,), 0.5))

    class Recorder(TamingDensity):
        def __init__(self, spec):
            super().__init__(spec)
            self.dens_steps = []

        def _accumulate(self, info):
            pass

        def densify(self, params, optimizers, step, scores=None):
            self.dens_steps.append(step)

    spec = next(c for c in method.components if c.kind == "density")
    d = Recorder(replace(spec, config={**spec.config, "budget": 100}))
    d.initialize(params, opts, scene_scale=1.0)
    d.counts = [6] * 200
    resets = []
    for step in range(1, 16001):
        with torch.no_grad():
            params["opacities"].fill_(0.0)
        d.after_backward(params, opts, step, {"info": {}})
        if float(params["opacities"].sigmoid().max()) < 0.011:
            resets.append(step)
    interval = spec.config["densification_interval"]
    expected_interval = {"taming-3dgs": 100, "taming-3dgs-budget": 500}.get(method.name, interval)
    if interval != expected_interval:
        raise AssertionError(f"{method.name}: densification interval {interval}")
    # official train.py: `iteration > 500 and iteration % interval == 0` while iteration < 15000
    want = [s for s in range(1, 15000) if s > 500 and s % interval == 0]
    if d.dens_steps != want or resets != [3000, 6000, 9000, 12000]:
        raise AssertionError(f"Schedule wrong: {d.dens_steps[:3]}..{d.dens_steps[-2:]}, {resets}")
    return {"densify_calls": len(d.dens_steps), "resets": resets}
