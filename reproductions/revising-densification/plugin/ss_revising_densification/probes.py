"""Mechanism probes for the Revising-Densification port. Each raises AssertionError when the
mechanism is removed or wrong; references are written independently of density.py / core.py."""

import math
from dataclasses import replace

import torch

from . import core
from .density import RevisedADC
from .losses import ERROR_KEY, create_aux


def _density(method, **config):
    spec = next(c for c in method.components if c.kind == "density")
    cfg = {**spec.config, "log": False, "max_primitives": 10**7, **config}
    return RevisedADC(replace(spec, config=cfg))


def _aux(method, **config):
    spec = next(c for c in method.components if c.name == "revdens-aux")
    return create_aux(replace(spec, config={**spec.config, **config}))


def _scene(method, n, device, seed=0, depth=3.0, spread=0.6, scale=0.08, opacity=None):
    from splatstudio.state import initialize

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += depth
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_(torch.full((n, 3), math.log(scale)) + 0.3 * torch.randn(
            n, 3, generator=g))
        q = torch.randn(n, 4, generator=g)
        params["quats"].copy_(torch.nn.functional.normalize(q, dim=-1))
        o = torch.rand(n, generator=g) * 0.8 + 0.1 if opacity is None else opacity
        params["opacities"].copy_(torch.logit(torch.as_tensor(o, dtype=torch.float32)))
    return params, opts


def _camera(device, w=40, h=32, f=30.0, center=(0.0, 0.0, 0.0)):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    viewmat = torch.eye(4, device=device)
    viewmat[:3, 3] = -torch.tensor(center, device=device)
    return {"K": K, "viewmat": viewmat, "width": w, "height": h}


def _render(params, cam, colors=None):
    from gsplat import rasterization

    kw = dict(sh_degree=0, colors=torch.cat((params["sh0"], params["shN"]), dim=1))
    if colors is not None:
        kw = dict(sh_degree=None, colors=colors)
    img, alpha, meta = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), viewmats=cam["viewmat"][None],
        Ks=cam["K"][None], width=cam["width"], height=cam["height"], packed=False, **kw)
    return img[0], alpha[0], meta


def _brute(meta, error):
    """Independent per-pixel front-to-back alpha compositing (gsplat numerics):
    returns sum_u Err(u) w_k(u), sum_u w_k(u), alpha image."""
    m2d, con, op = (meta[k][0].detach().cpu() for k in ("means2d", "conics", "opacities"))
    offs, ids = meta["isect_offsets"][0].flatten().tolist(), meta["flatten_ids"].tolist()
    ts, W, H, tw = meta["tile_size"], meta["width"], meta["height"], meta["tile_width"]
    err_img = error.detach().cpu().tolist()
    n = m2d.shape[0]
    e_sum, w_sum = [0.0] * n, [0.0] * n
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
                s = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
                a = min(0.999, float(op[gid]) * math.exp(-s))
                if s < 0 or a < 1 / 255:
                    continue
                if T * (1 - a) <= 1e-4:
                    break
                w = a * T
                e_sum[gid] += err_img[i][j] * w
                w_sum[gid] += w
                T *= 1 - a
            alpha_img[i, j] = 1 - T
    return torch.tensor(e_sum), torch.tensor(w_sum), alpha_img


def _detached(params):
    return {k: v.detach() for k, v in params.items()}


# ----------------------------------------------------------------------------- C1
def error_attribution(method, device):
    from .kernels import error_stats

    params, _ = _scene(method, 30, device, seed=1, scale=0.2)
    cam = _camera(device)
    g = torch.Generator().manual_seed(2)
    error = torch.rand(cam["height"], cam["width"], generator=g).to(device)
    with torch.no_grad():
        _, alpha, meta = _render(params, cam)
    e_k, w_k = error_stats(meta, error)
    ref_e, ref_w, alpha_ref = _brute(meta, error)
    torch.testing.assert_close(alpha_ref, alpha[..., 0].cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(e_k.cpu(), ref_e, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(w_k.cpu(), ref_w, rtol=1e-4, atol=1e-3)
    if float(ref_w.sum()) < 100:
        raise AssertionError("Probe scene too sparse")
    # Paper implementation (Sec. 3.2): L_aux = sum_u stopgrad(Err(u)) R[pi, Phi_ERR](u) with a
    # zero-initialised scalar e_k per primitive; dL_aux/de_k = E^pi_k.
    e = torch.zeros(len(params["means"]), 1, device=device, requires_grad=True)
    img, _, _ = _render(_detached(params), cam, colors=e)
    l_aux = (error[..., None] * img).sum()
    if float(l_aux.detach().abs()) != 0.0:
        raise AssertionError("L_aux must be zero with e_k = 0")
    l_aux.backward()
    torch.testing.assert_close(e_k, e.grad[:, 0], rtol=1e-3, atol=1e-3)
    # ablations: uniform error (blend sum) and unweighted pixel error must both differ
    uniform = w_k * error.mean()
    if float((uniform - e_k).abs().max()) < 1e-2:
        raise AssertionError("Error attribution indistinguishable from a uniform error")
    return {"pairs_weight": float(ref_w.sum()), "E_total": float(e_k.sum()),
            "max_err_vs_brute": float((e_k.cpu() - ref_e).abs().max()),
            "max_err_vs_autograd": float((e_k - e.grad[:, 0]).abs().max()),
            "max_diff_uniform_ablation": float((uniform - e_k).abs().max())}


def error_score_max(method, device):
    from .kernels import error_stats

    params, opts = _scene(method, 30, device, seed=3)
    d = _density(method, opacity_decrement=0.0)
    d.initialize(params, opts, 1.0)
    views = []
    for k, center in enumerate(((0.0, 0.0, 0.0), (0.15, -0.1, 0.2))):
        cam = _camera(device, center=center)
        with torch.no_grad():
            _, _, meta = _render(params, cam)
        err = torch.rand(cam["height"], cam["width"], generator=torch.Generator().manual_seed(k))
        err = err.to(device)
        d.accumulate({"info": meta, ERROR_KEY: err})
        views.append(error_stats(meta, err)[0])
    ref = torch.maximum(views[0], views[1])
    torch.testing.assert_close(d.state["score"], ref)
    if float((views[0] + views[1] - ref).abs().max()) < 1e-3:
        raise AssertionError("Fixture cannot distinguish max from sum")
    missing = False
    try:
        d.accumulate({"info": meta})
    except RuntimeError:
        missing = True
    if not missing:
        raise AssertionError("Missing error map must raise")
    d.densify_and_prune(params, opts, 600)
    if float(d.state["score"].abs().max()) != 0.0 or len(d.state["score"]) != len(params["means"]):
        raise AssertionError("Score not reset after the ADC run")
    return {"score_max": float(ref.max()), "sum_minus_max": float((views[0] + views[1] - ref)
                                                                  .abs().max())}


def _grow_rows(before, after):
    """Indices of source rows (means [N,3] before the run) whose copies were appended to the
    means after the run (clone keeps the mean exactly)."""
    new = after[len(before):]
    old = before
    hits = []
    for row in new:
        idx = torch.nonzero((old == row).all(dim=-1)).flatten().tolist()
        hits.extend(idx)
    return sorted(set(hits))


def error_selection(method, device):
    n = 20
    out = {}
    for mode in ("error", "gradient"):
        params, opts = _scene(method, n, device, seed=4, scale=1e-4, opacity=torch.full((n,), 0.5))
        d = _density(method, score=mode, growth_fraction=1.0)
        d.initialize(params, opts, 1.0)
        st = d.state
        st["score"][:] = 0.05  # below the 0.1 threshold
        st["score"][3] = 5.0  # high error, zero gradient
        st["grad_accum"][7] = 1.0  # high gradient, zero error
        st["count"][:] = 1.0
        before = params["means"].detach().clone()
        row = d.densify_and_prune(params, opts, 600)
        out[mode] = {"grown": _grow_rows(before, params["means"].detach()), **row}
    if out["error"]["grown"] != [3] or out["error"]["candidates"] != 1:
        raise AssertionError(f"Error-driven selection wrong: {out['error']}")
    if out["gradient"]["grown"] != [7]:
        raise AssertionError(f"Gradient ablation should select the high-gradient one: "
                             f"{out['gradient']}")
    return {"error_grown": out["error"]["grown"], "gradient_ablation_grown":
            out["gradient"]["grown"], "error_candidates": out["error"]["candidates"]}


def growth_budget(method, device):
    n = 200
    g = torch.Generator().manual_seed(5)
    scores = torch.rand(n, generator=g) + 0.2  # all above threshold
    res = {}
    # (a) fraction: +floor(0.05 * 200) = 10, the 10 highest
    params, opts = _scene(method, n, device, seed=6, scale=1e-4, opacity=torch.full((n,), 0.5))
    d = _density(method)
    d.initialize(params, opts, 1.0)
    d.state["score"].copy_(scores.to(device))
    before = params["means"].detach().clone()
    row = d.densify_and_prune(params, opts, 600)
    top = sorted(torch.topk(scores, 10).indices.tolist())
    grown = _grow_rows(before, params["means"].detach())
    if row["after"] != 210 or grown != top:
        raise AssertionError(f"Fraction budget wrong: {row}, grown {grown} vs top {top}")
    res["fraction_after"] = row["after"]
    # (b) cap with pruning: cap 200, 8 pruned -> refilled to 200 exactly
    o = torch.full((n,), 0.5)
    o[:8] = 0.001
    params, opts = _scene(method, n, device, seed=6, scale=1e-4, opacity=o)
    d = _density(method, max_primitives=200)
    d.initialize(params, opts, 1.0)
    d.state["score"].copy_(scores.to(device))
    row = d.densify_and_prune(params, opts, 600)
    if row["pruned"] != 8 or row["after"] != 200 or row["budget"] != 8:
        raise AssertionError(f"Cap/refill wrong: {row}")
    res["cap_row"] = {k: row[k] for k in ("pruned", "budget", "after")}
    # (c) cap reached: no growth
    params, opts = _scene(method, n, device, seed=6, scale=1e-4, opacity=torch.full((n,), 0.5))
    d = _density(method, max_primitives=200)
    d.initialize(params, opts, 1.0)
    d.state["score"].copy_(scores.to(device))
    if d.densify_and_prune(params, opts, 600)["after"] != 200:
        raise AssertionError("Grew beyond the global cap")
    # ablation: no growth control -> all 200 candidates grow
    params, opts = _scene(method, n, device, seed=6, scale=1e-4, opacity=torch.full((n,), 0.5))
    d = _density(method, growth_control=False)
    d.initialize(params, opts, 1.0)
    d.state["score"].copy_(scores.to(device))
    abl = d.densify_and_prune(params, opts, 600)["after"]
    if abl != 400:
        raise AssertionError(f"No-control ablation should double: {abl}")
    res["no_control_after"] = abl
    return res


# ----------------------------------------------------------------------------- C2
def clone_opacity(method, device):
    alpha = 0.7
    expect_hat = 1 - math.sqrt(1 - alpha)
    # parameter level: 3 small (clone) + 3 large (split)
    params, opts = _scene(method, 6, device, seed=7, scale=1e-4,
                          opacity=torch.full((6,), alpha))
    with torch.no_grad():
        params["scales"][3:] = math.log(0.5)
    for k in ("opacities", "means"):
        opts[k].state[params[k]]["exp_avg"] = torch.ones_like(params[k])
        opts[k].state[params[k]]["exp_avg_sq"] = torch.ones_like(params[k])
        opts[k].state[params[k]]["step"] = torch.tensor(1.0)
    d = _density(method, opacity_decrement=0.0, growth_fraction=1.0)
    d.initialize(params, opts, 1.0)
    d.state["score"][:] = 1.0
    row = d.densify_and_prune(params, opts, 600)
    o = params["opacities"].sigmoid().detach().cpu()
    # layout: [3 cloned sources, 3 clone copies, 6 split children]
    if row["clone"] != 3 or row["split"] != 3 or len(o) != 12:
        raise AssertionError(f"Unexpected ops {row}")
    torch.testing.assert_close(o[:6], torch.full((6,), expect_hat), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(o[6:], torch.full((6,), alpha), rtol=1e-5, atol=1e-6)
    m = opts["opacities"].state[params["opacities"]]["exp_avg"].cpu()
    if not (torch.all(m[:3] == 1) and torch.all(m[3:] == 0)):
        raise AssertionError("Adam moments misaligned after clone/split")
    # rendered: two coincident clones at a pixel centre leave transmittance 1 - alpha (Sec. 3.3)
    cam = _camera(device, w=16, h=16, f=20.0)
    trans = {}
    for corr in (True, False):
        p1, o1 = _scene(method, 2, device, seed=8, scale=0.02,
                        opacity=torch.tensor([alpha, alpha]))
        with torch.no_grad():
            z = 3.0
            p1["means"][0] = torch.tensor([0.5 * z / 20.0, 0.5 * z / 20.0, z])  # pixel (8, 8)
            p1["means"][1] = torch.tensor([0.0, 0.0, -5.0])  # behind the camera (culled)
        d = _density(method, opacity_decrement=0.0, growth_fraction=1.0,
                     opacity_correction=corr)
        d.initialize(p1, o1, 100.0)  # extent large -> clone
        d.state["score"][:] = 1.0
        d.densify_and_prune(p1, o1, 600)
        with torch.no_grad():
            _, a, _ = _render(p1, cam)
        trans[corr] = 1 - float(a[8, 8, 0])
    if abs(trans[True] - (1 - alpha)) > 1e-4:
        raise AssertionError(f"Corrected clone transmittance {trans[True]} != {1 - alpha}")
    if abs(trans[False] - (1 - alpha) ** 2) > 1e-4:
        raise AssertionError(f"Uncorrected ablation transmittance {trans[False]}")
    return {"alpha_hat": float(o[0]), "split_alpha": float(o[6]),
            "transmittance_corrected": trans[True], "transmittance_ablation": trans[False]}


# ----------------------------------------------------------------------------- C4
def opacity_regularisation(method, device):
    from splatstudio.backends.gsplat import Renderer

    n = 50
    params, opts = _scene(method, n, device, seed=9, opacity=torch.full((n,), 0.5))
    d = _density(method)
    d.initialize(params, opts, 1.0)
    for k in range(3):
        d.densify_and_prune(params, opts, 600 + 100 * k)  # zero scores: no growth
    o = params["opacities"].sigmoid().detach()
    torch.testing.assert_close(o, torch.full_like(o, 0.497), rtol=0, atol=2e-6)
    # never reset (default) vs reset ablation at step 3000
    res = {"after_3_runs": float(o.mean())}
    for mode in ("decrement", "reset"):
        p2, o2 = _scene(method, n, device, seed=9, opacity=torch.full((n,), 0.5))
        d = _density(method, opacity_mode=mode, densify_until=0)
        d.initialize(p2, o2, 1.0)
        d.after_backward(p2, o2, 3000, {})
        res[f"{mode}_opacity_at_3000"] = float(p2["opacities"].sigmoid().mean())
    if abs(res["decrement_opacity_at_3000"] - 0.5) > 1e-6:
        raise AssertionError("Opacity reset happened in decrement mode")
    if abs(res["reset_opacity_at_3000"] - 0.01) > 1e-5:
        raise AssertionError("Reset ablation did not reset")
    # transmittance loss
    renderer = Renderer(next(c for c in method.components if c.kind == "renderer"))
    cam = _camera(device)
    outputs = renderer.render(params, cam, sh_degree=0)
    outputs["step"] = 100
    target = torch.rand(cam["height"], cam["width"], 3, device=device)
    loss = _aux(method)(params, outputs, target)
    ref = 0.1 * (1 - outputs["alpha"].detach()).mean()
    torch.testing.assert_close(loss.detach(), ref)
    loss.backward()
    g = params["opacities"].grad
    if g is None or not float(g.sum()) < 0:
        raise AssertionError("Transmittance loss must push opacities up")
    if ERROR_KEY not in outputs or outputs[ERROR_KEY].requires_grad:
        raise AssertionError("Error map missing or attached")
    res.update(loss=float(loss), opacity_grad_sum=float(g.sum()))
    return res


def error_map(method, device):
    from splatstudio.losses import ssim

    g = torch.Generator().manual_seed(10)
    a = torch.rand(32, 40, 3, generator=g).to(device).requires_grad_(True)
    b = torch.rand(32, 40, 3, generator=g).to(device)
    m = core.error_map(a, b, "ssim")
    if m.shape != (32, 40) or m.requires_grad:
        raise AssertionError("Error map must be [H, W] and detached")
    interior = 1 - m[5:-5, 5:-5].mean()
    ref = ssim(a.detach(), b)
    torch.testing.assert_close(interior, ref, rtol=1e-5, atol=1e-5)
    l1 = core.error_map(a, b, "l1")
    torch.testing.assert_close(l1, (a.detach() - b).abs().mean(-1))
    if float((m - l1).abs().max()) < 1e-3:
        raise AssertionError("SSIM and l1 error maps indistinguishable")
    return {"interior_ssim": float(interior), "runtime_ssim": float(ref)}


# ----------------------------------------------------------------------------- C5
def adc_schedule(method, device):
    d = _density(method)
    runs, acc = [], []
    d.accumulate = lambda outputs: acc.append(1)
    d.densify_and_prune = lambda p, o, step: runs.append(step)
    for step in range(1, 30001):
        d.after_backward({}, {}, step, {})
    if len(runs) != 264 or runs[0] != 600 or runs[-1] != 26900 or len(acc) != 26999:
        raise AssertionError(f"ADC schedule wrong: {len(runs)} runs {runs[:1]}..{runs[-1:]}, "
                             f"{len(acc)} stat steps")
    return {"runs": len(runs), "first": runs[0], "last": runs[-1], "stat_steps": len(acc)}


def cap_table(method, device):
    if core.primitive_cap("baseline", 138766) != (4764685, "mipnerf360/garden"):
        raise AssertionError("garden cap wrong")
    if core.primitive_cap(5000, 100) != (5000, "config"):
        raise AssertionError("int override wrong")
    for bad in (("baseline", 12345), (50, 100), ("auto", 10)):
        try:
            core.primitive_cap(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must raise")
    params, opts = _scene(method, 112627, device, seed=11)
    spec = next(c for c in method.components if c.kind == "density")
    d = RevisedADC(replace(spec, config={**spec.config, "log": False}))
    d.initialize(params, opts, 1.0)
    if d.state["cap"] != 1727096:
        raise AssertionError("room cap not resolved from the initial point count")
    return {"garden": 4764685, "room": d.state["cap"], "scenes": len(core.BASELINE_CAP)}


# ----------------------------------------------------------------------------- base override
def renderer_gradient(method, device):
    """Base renderer-gradient probe run through the full revdens step (render, both losses,
    backward, density hook with the loss-provided error map)."""
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.losses import create_photometric
    from splatstudio.state import check_state
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer = Renderer(next(c for c in method.components if c.kind == "renderer"))
    params, opts = _scene(method, 40, device, seed=12)
    d = _density(method)
    d.initialize(params, opts, 1.0)
    cam = _camera(device)
    outputs = renderer.render(params, cam, sh_degree=0)
    outputs["step"] = 1
    d.before_backward(params, opts, 1, outputs)
    target = (torch.arange(cam["width"], device=device)[None, :, None] % 2).expand(
        cam["height"], cam["width"], 3).float()
    photometric = create_photometric(next(c for c in method.components if c.kind == "loss"))
    total = photometric(params, outputs, target) + _aux(method)(params, outputs, target)
    total.backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    d.after_backward(params, opts, 1, outputs)
    if not float(d.state["score"].max()) > 0:
        raise AssertionError("Density did not accumulate the error score")
    check_state(method, params, opts)
    return {"screen_gradient_norm": float(xy.grad.norm()),
            "score_max": float(d.state["score"].max())}
