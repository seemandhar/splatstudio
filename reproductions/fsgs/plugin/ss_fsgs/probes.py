"""Mechanism probes for the FSGS plugin; each fails if its mechanism is removed or wrong."""

from dataclasses import replace

import numpy as np
import torch
from splatstudio.components import build
from splatstudio.state import check_state, initialize


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _density(method, **overrides):
    from .density import create_density

    spec = _spec(method, "fsgs-density")
    return create_density(replace(spec, config={**spec.config, **overrides}))


def _cluster_scene(method, device):
    """64 points in a unit cube near the origin plus two isolated far points with big scales."""
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(64, 3, generator=g) * 0.5
    far = torch.tensor([[30.0, 0.0, 0.0], [0.0, 40.0, 0.0]])
    points = torch.cat([pts, far]).to(device)
    colors = torch.full_like(points, 0.5)
    params, opts = initialize(method, points, colors)
    with torch.no_grad():
        params["scales"][:] = -3.0
        params["scales"][-2:] = np.log(2.0)  # max scale 2 > extent 1
        params["opacities"][:] = torch.linspace(-7, 2, len(points), device=device)
        params["sh0"].normal_()
        params["shN"].normal_()
    for p in params.values():  # allocate Adam moments so alignment is exercised
        p.grad = torch.ones_like(p) * 1e-3
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    return params, opts


def proximity_unpooling(method, device):
    """C1: compare the density's unpooling to an independent brute-force reference."""
    params, opts = _cluster_scene(method, device)
    ref = {k: v.detach().clone() for k, v in params.items()}
    n0 = len(ref["means"])
    d = _density(method)
    d.initialize(params, opts, scene_scale=1.0)
    added = d.unpool(params, opts)
    # Independent reference: per-source loop over all other points.
    mean = ref["means"].double().cpu().numpy()
    max_scale = np.exp(ref["scales"].double().cpu().numpy()).max(axis=1)
    expected = []
    for i in range(n0):
        if max_scale[i] <= 1.0:
            continue
        dd = ((mean - mean[i]) ** 2).sum(1)
        dd[i] = np.inf
        nn = np.argsort(dd)[:3]
        if dd[nn].mean() > 5.0 * 1.0:
            expected += [(i, j) for j in nn]
    if added != len(expected) or added != 6:
        raise AssertionError(f"unpooling added {added}, reference {len(expected)} (expected 6)")
    new = {k: v[n0:].detach() for k, v in params.items()}
    for row, (i, j) in enumerate(expected):
        mid = (ref["means"][i] + ref["means"][j]) / 2
        if not torch.allclose(new["means"][row], mid, atol=1e-5):
            raise AssertionError("new Gaussian is not at the edge midpoint")
        if not torch.equal(new["scales"][row], ref["scales"][j]) or not torch.equal(
            new["opacities"][row], ref["opacities"][j]
        ):
            raise AssertionError("new scale/opacity not copied from the destination Gaussian")
    if new["sh0"].abs().max() or new["shN"].abs().max():
        raise AssertionError("unpooled SH not zero")
    if not torch.equal(new["quats"], torch.tensor([1.0, 0, 0, 0], device=device).expand(added, 4)):
        raise AssertionError("unpooled rotation not identity")
    check_state(method, params, opts)
    for name, p in params.items():
        if not p.requires_grad:
            raise AssertionError(f"{name} frozen by unpooling")
        st = opts[name].state[p]
        if st["exp_avg"][n0:].abs().max() or st["exp_avg_sq"][n0:].abs().max():
            raise AssertionError("new Gaussians have non-zero Adam moments")
    # Ablations: disabled / after iteration 2000 / no big Gaussian -> nothing added.
    p2, o2 = _cluster_scene(method, device)
    d2 = _density(method, unpool=False)
    d2.initialize(p2, o2, 1.0)
    d2._reset_stats(len(p2["means"]), device)
    r_off = d2.densify_and_prune(p2, o2, step=600)
    p3, o3 = _cluster_scene(method, device)
    d3 = _density(method)
    d3.initialize(p3, o3, 1.0)
    r_late = d3.densify_and_prune(p3, o3, step=2000)
    p4, o4 = _cluster_scene(method, device)
    d4 = _density(method)
    d4.initialize(p4, o4, 1.0)
    r_on = d4.densify_and_prune(p4, o4, step=600)
    # In densify_and_prune the far points are split first (official order), so the 4 children
    # (still max scale > extent, far from the cluster) are the unpooling sources: 4 x 3 edges.
    if r_off["unpooled"] or r_late["unpooled"] or r_on["unpooled"] != 3 * 2 * r_on["far_split"]:
        raise AssertionError(f"unpooling gating wrong: off={r_off} late={r_late} on={r_on}")
    return {"added": added, "on_600": r_on, "off": r_off["unpooled"], "late": r_late["unpooled"]}


def fsgs_densify(method, device):
    """FSGS split rule (no gradient needed), prune threshold and opacity-reset schedule."""
    params, opts = _cluster_scene(method, device)
    d = _density(method, unpool=False)
    d.initialize(params, opts, 1.0)
    n0 = len(params["means"])
    # Mean squared 3-NN distance of the two far points: > 10 * extent -> split (2 -> 4).
    row = d.densify_and_prune(params, opts, step=600)
    if row["far_split"] != 2 or row["split"] != 2 or row["cloned"] != 0:
        raise AssertionError(f"FSGS distance split rule wrong: {row}")
    op = torch.linspace(-7, 2, n0, device=device).sigmoid()
    if row["pruned"] == 0 or row["pruned"] != int((op[:-2] < 0.005).sum()):
        raise AssertionError("prune threshold 0.005 not applied")
    # Ablation: with dist_thres huge, nothing splits.
    p2, o2 = _cluster_scene(method, device)
    d2 = _density(method, unpool=False, dist_thres=1e9)
    d2.initialize(p2, o2, 1.0)
    if d2.densify_and_prune(p2, o2, step=600)["split"] != 0:
        raise AssertionError("split fired without gradient or distance criterion")
    # Opacity reset at 2001, 5001, 8001 only (train.py:166).
    resets = []
    p3, o3 = _cluster_scene(method, device)
    d3 = _density(method, densify_until_iter=0)
    d3.initialize(p3, o3, 1.0)
    for step in (2000, 2001, 2002, 5001, 8001, 9000):
        with torch.no_grad():
            p3["opacities"].fill_(2.0)
        d3.after_backward(p3, o3, step, {"info": {}})
        if float(p3["opacities"].max()) < 2.0:
            resets.append(step)
    if resets != [2001, 5001, 8001]:
        raise AssertionError(f"opacity reset steps {resets}")
    target = float(torch.logit(torch.tensor(0.05)))
    return {"split_row": row, "resets": resets, "reset_logit": target}


def pearson_depth(method, device):
    """C3 train-view loss vs an independent numpy formula; weight schedule; gradient path."""
    from .losses import depth_weight, pearson, train_depth_loss

    g = torch.Generator().manual_seed(1)
    rendered = (torch.rand(12, 16, generator=g) * 3 + 1).double()
    dpt = (torch.rand(12, 16, generator=g) * 20).double()
    r, m = rendered.numpy().ravel(), dpt.numpy().ravel()
    ref = min(1 - np.corrcoef(-m, r)[0, 1], 1 - np.corrcoef(1 / (m + 200.0), r)[0, 1])
    val = float(train_depth_loss(rendered, dpt))
    if abs(val - ref) > 1e-9:
        raise AssertionError(f"depth loss {val} != reference {ref}")
    if abs(float(pearson(rendered, 3 * rendered + 2)) - 1) > 1e-9:
        raise AssertionError("Pearson not invariant to scale/shift")
    ws = [depth_weight(s, 0.05, 0.001, 9500) for s in (1, 9500, 9501, 9502)]
    if ws != [0.05, 0.05, 0.05, 0.001]:
        raise AssertionError(f"depth weight schedule {ws}")
    rend = rendered.clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(train_depth_loss(rend, dpt), rend)
    if not torch.count_nonzero(gr):
        raise AssertionError("no gradient to the rendered depth")
    return {"loss": val, "reference": ref, "weights": ws}


def pseudo_views(method, device):
    """C2: pose generator, gating/ramp and gradient through DPT into the pseudo render colours."""
    from splatstudio.verification import ProbeUnavailable

    from .dpt import DPTUnavailable
    from .losses import pseudo_scale

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")
    sc = [pseudo_scale(s, 2000, 9500, 1, 500) for s in (2000, 2001, 2250, 2500, 9000, 9500)]
    if sc != [0.0, 1 / 500, 0.5, 1.0, 1.0, 0.0]:
        raise AssertionError(f"pseudo gate/ramp wrong: {sc}")
    centers = [[x, 0.0, 0.0] for x in (-0.2, 0.0, 0.2)]
    table = {"probe": {"near": 1.0, "far": 50.0, "train_centers": centers}}
    comps = list(method.components)
    k = next(i for i, c in enumerate(comps) if c.name == "fsgs-density")
    comps[k] = replace(comps[k], config={**comps[k].config, "scene_bounds": table,
                                         "n_pseudo_poses": 200})
    method = replace(method, components=tuple(comps))
    renderer, density, losses = build(method)
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(400, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    pts[:, 2] += 3.0
    params, opts = initialize(method, pts.to(device), torch.rand(400, 3, generator=g).to(device))
    with torch.no_grad():
        params["scales"].fill_(np.log(0.08))
        params["opacities"].fill_(2.0)
    density.initialize(params, opts, 1.0)
    K = torch.tensor([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]], device=device)
    cams = []
    for x in (-0.2, 0.0, 0.2):
        v = torch.eye(4, device=device)
        v[0, 3] = -x  # camera centre at (x, 0, 0)
        cams.append({"viewmat": v, "K": K, "width": 64, "height": 48})
    density.set_training_cameras(cams)
    info = density.pseudo_info
    scale = 1 / 0.75
    focus = 1 / (0.25 / (0.9 * scale) + 0.75 / (5 * 50.0 * scale)) / scale
    if info["scene"] != "probe" or abs(info["focus_distance"] - focus) > 1e-9:
        raise AssertionError(f"pseudo poses ignore the official bounds: {info}")
    poses = density.pseudo_viewmats.double().cpu()
    R = poses[:, :3, :3]
    if (R @ R.transpose(1, 2) - torch.eye(3, dtype=torch.float64)).abs().max() > 1e-4:
        raise AssertionError("pseudo poses are not rotations")
    c = -(R.transpose(1, 2) @ poses[:, :3, 3:4])[..., 0]
    if c[:, 0].abs().max() > 0.2 + 1e-4 or c[:, 1:].abs().max() > 1e-4 or c[:, 0].std() < 0.05:
        raise AssertionError("pseudo camera centres not spread over the training-camera box")
    losses = dict(losses)
    outputs = renderer.render(params, cams[1])
    outputs["step"] = 3000
    density.before_backward(params, opts, 3000, outputs)
    try:
        term = losses["fsgs-pseudo"](params, outputs, torch.rand(48, 64, 3, device=device))
    except DPTUnavailable as exc:
        raise ProbeUnavailable(str(exc)) from exc
    (g_sh0,) = torch.autograd.grad(term, params["sh0"], allow_unused=True)
    # Rendered depth does not depend on colours: a non-zero sh0 gradient proves the gradient
    # flows through DPT(render) (official mode='train').
    if g_sh0 is None or not torch.count_nonzero(g_sh0):
        raise AssertionError("no gradient through DPT into the pseudo-view colours")
    outputs["step"] = 1500
    before = len(density._order)
    off = losses["fsgs-pseudo"](params, outputs, torch.rand(48, 64, 3, device=device))
    if float(off) != 0.0 or len(density._order) != before:
        raise AssertionError("pseudo loss active before start_sample_pseudo")
    return {"loss": float(term), "sh0_grad_norm": float(g_sh0.norm()),
            "focus_distance": info["focus_distance"], "ramp": sc}


def sh_schedule(method, device):
    from .renderer import create_renderer

    r = create_renderer(_spec(method, "fsgs-renderer"))
    seen = {}
    with torch.enable_grad():
        for n in range(1, 2001):
            seen[n] = r.degree()
    expect = {n: min(n // 500, 3) for n in seen}
    if seen != expect:
        bad = [n for n in seen if seen[n] != expect[n]][:5]
        raise AssertionError(f"SH degree schedule differs at renders {bad}")
    with torch.no_grad():
        if r.degree() != 3:
            raise AssertionError("evaluation SH degree is not 3")
    return {"degree_at": {k: seen[k] for k in (499, 500, 1000, 1500)}}
