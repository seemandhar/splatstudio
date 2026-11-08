"""Mechanism probes for EFA-GS. Each fails if its mechanism is removed or altered."""

import math
import random

import torch

from .lfcf import (
    abs_threshold,
    anneal_fraction,
    enlarge_delta,
    is_lfcf_step,
    lfcf_masks,
    log_enlarging_factor,
    normalize_log_minmax,
    shrink_delta,
)
from .mip_filters import compute_filter_3d


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _build(method):
    from splatstudio.components import build

    renderer, density, _ = build(method)
    return renderer, density


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _camera(device, size=32, f=40.0):
    return {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], device=device),
        "width": size,
        "height": size,
    }


# ---------------------------------------------------------------------------- C1 LFCF decision
def lfcf_decision(method, device):
    """Masks equal an independent per-Gaussian loop of Alg. 1 with the official rules."""
    cfg = _component(method, "density").config
    tau, tol = cfg["grad_threshold"], cfg["tolerance"]
    g = torch.Generator().manual_seed(0)
    n = 400
    grads = torch.rand(n, generator=g) * 4 * tau
    grads_abs = torch.rand(n, generator=g) * 1e-3
    prev = torch.rand(n, generator=g) * 4 * tau
    prev_abs = torch.rand(n, generator=g) * 1e-3
    prev_sel = torch.rand(n, generator=g) < 0.7
    q = abs_threshold(grads, grads_abs, tau)
    q_ref = torch.quantile(grads_abs, 1 - float((grads >= tau).float().mean()))
    if not torch.isclose(q, q_ref):
        raise AssertionError(f"abs quantile {q} != torch.quantile {q_ref}")
    selected, enlarged, shrunk, _ = lfcf_masks(grads, grads_abs, tau, q, prev_sel, prev,
                                                prev_abs, tol)
    ref_e, ref_s = [], []
    for i in range(n):
        sel = grads[i] >= tau or grads_abs[i] >= q
        if not sel:
            ref_e.append(False)
            ref_s.append(False)
        elif not prev_sel[i]:
            ref_e.append(True)
            ref_s.append(False)
        else:
            rose = grads[i] > abs(prev[i] - tol) and grads_abs[i] > abs(prev_abs[i] - tol)
            ref_e.append(bool(rose))
            ref_s.append(not rose)
    if enlarged.tolist() != ref_e or shrunk.tolist() != ref_s:
        raise AssertionError("LFCF masks differ from the independent Alg. 1 loop")
    if not (enlarged.any() and shrunk.any()):
        raise AssertionError("fixture must exercise both branches")
    # Removing the gradient-ascent test (everything selected -> shrink) must be detectable.
    _, e2, _, _ = lfcf_masks(grads, grads_abs, tau, q, torch.ones_like(prev_sel),
                             torch.full_like(prev, 1e9), torch.full_like(prev, 1e9), tol)
    if e2.any():
        raise AssertionError("gradient decrease must never enlarge")
    return {"selected": int(selected.sum()), "enlarged": int(enlarged.sum()),
            "shrunk": int(shrunk.sum()), "q_abs": float(q)}


# ------------------------------------------------------------------------ C2 depth strategy
def depth_factor(method, device):
    cfg = _component(method, "density").config
    cmin, cmax = cfg["c_min"], cfg["c_max"]
    interval = torch.tensor([0.001, 0.004, 0.01, 0.1, 0.02])
    theta = normalize_log_minmax(interval)
    lo, hi = math.log(0.001), math.log(0.1)
    ref_theta = [(math.log(v) - lo) / (hi - lo) for v in interval.tolist()]
    torch.testing.assert_close(theta, torch.tensor(ref_theta))
    frac = 0.6
    log_c = log_enlarging_factor(theta, cmin, cmax, frac, True)
    ref = [math.log(t * cmin + (1 - t) * cmax) * frac for t in ref_theta]
    torch.testing.assert_close(log_c, torch.tensor(ref))
    if not math.isclose(float(log_c[0]), math.log(cmax) * frac, rel_tol=1e-6):
        raise AssertionError("nearest Gaussian must get c_max")
    if abs(float(log_c[3]) - math.log(cmin) * frac) > 1e-7:
        raise AssertionError("deepest Gaussian must get c_min")
    flat = log_enlarging_factor(theta, cmin, cmax, frac, False)
    if torch.allclose(flat, log_c):
        raise AssertionError("depth strategy has no effect")
    return {"log_c": log_c.tolist()}


# ------------------------------------------------------------------------ C3 scale strategy
def scale_strategy(method, device):
    cfg = _component(method, "density").config
    log_s = torch.tensor([[-3.0, -5.0, -4.0], [-1.0, -2.0, -6.0]])
    log_c = torch.tensor([[0.3], [0.1]])
    d = enlarge_delta(log_s, log_c, True)
    ref = torch.tensor([[-0.2, 0.3, -0.1], [-2 / 30, -1 / 30, 0.1]])
    torch.testing.assert_close(d, ref)
    if not torch.allclose(d.sum(dim=1), torch.zeros(2), atol=1e-7):
        raise AssertionError("enlarge is not volume preserving")
    m = cfg["split_multiplier"]
    sd = shrink_delta(log_s, log_c, m, True)
    b = -0.5 * (log_c + math.log(m))
    ref_s = torch.cat([torch.stack([b[0, 0], 0 * b[0, 0], 0.5 * b[0, 0]]).unsqueeze(0),
                       torch.stack([b[1, 0], 0.5 * b[1, 0], 0 * b[1, 0]]).unsqueeze(0)])
    torch.testing.assert_close(sd, ref_s)
    iso = enlarge_delta(log_s, log_c, False)
    if torch.allclose(iso, d):
        raise AssertionError("scale strategy has no effect")
    return {"enlarge": d.tolist(), "shrink": sd.tolist()}


# ---------------------------------------------------------------------- C4 schedules
def anneal_schedule(method, device):
    cfg = _component(method, "density").config
    a, b = cfg["densify_from"], cfg["densify_until"]
    vals = {s: anneal_fraction(s, a, b, cfg["pow"], 0.0) for s in (a + 1e-9, (a + b) / 2, b - 100)}
    exp = {a + 1e-9: 1.0, (a + b) / 2: 0.5, b - 100: 100 / (b - a)}
    for s, v in vals.items():
        if not math.isclose(v, exp[s], rel_tol=1e-6, abs_tol=1e-9):
            raise AssertionError(f"anneal({s}) = {v}, expected {exp[s]}")
    steps = [s for s in range(a + 1, b) if s % cfg["densify_every"] == 0
             and is_lfcf_step(s, cfg["interval_times"], cfg["densify_every"])]
    if steps != list(range(600, 15000, 200)):
        raise AssertionError(f"LFCF steps {steps[:3]}..{steps[-3:]} != 600, 800, ..., 14800")
    return {"lfcf_steps": len(steps), "first": steps[0], "last": steps[-1]}


# ---------------------------------------------------------------------- base: filter, GOF
def filter3d(method, device):
    cam = _camera("cpu", 100, 100.0)
    pts = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]])
    f = compute_filter_3d(pts, [cam], 0.2)
    torch.testing.assert_close(f, torch.tensor([5.0, 2.0]) / 100 * math.sqrt(0.2))
    return {"filter": f.tolist()}


def gof_densify(method, device):
    """Clone/split counts of the non-LFCF step equal an independent computation."""
    from splatstudio.state import check_state, sample_scene

    _, density = _build(method)
    params, opts = sample_scene(method, "cpu", n=64)
    with torch.no_grad():
        params["scales"][:32] = math.log(0.001)
        params["scales"][32:] = math.log(0.2)
    density.initialize(params, opts, scene_scale=1.0)
    density.set_training_cameras([_camera("cpu")])
    g = torch.Generator().manual_seed(1)
    s = density.stats
    s["cnt"] = torch.randint(0, 4, (64,), generator=g).float()
    s["g"] = torch.rand(64, generator=g) * 6e-4 * s["cnt"]
    s["g_abs"] = torch.rand(64, generator=g) * 1e-3 * s["cnt"]
    mean = s["g"] / s["cnt"]
    mean[mean.isnan()] = 0
    mean_a = s["g_abs"] / s["cnt"]
    mean_a[mean_a.isnan()] = 0
    tau = density.cfg["grad_threshold"]
    q = torch.quantile(mean_a, 1 - float((mean >= tau).float().mean()))
    hot = (mean >= tau) | (mean_a >= q)
    small = torch.exp(params["scales"]).max(dim=1).values <= density.cfg["percent_dense"]
    n_clone, n_split = int((hot & small).sum()), int((hot & ~small).sum())
    out = density.gof_densify(params, opts)
    if out["cloned"] != n_clone or out["split"] != n_split:
        raise AssertionError(f"GOF densify {out} != clone {n_clone} split {n_split}")
    if len(params["means"]) != 64 + n_clone + n_split:
        raise AssertionError("population after GOF densify is wrong")
    check_state(method, params, opts)
    if not all(p.requires_grad and p.is_leaf for p in params.values()):
        raise AssertionError("a field lost requires_grad")
    return {"cloned": n_clone, "split": n_split, "q_abs": float(q)}


# ---------------------------------------------------------------------- CUDA probes
def _drive(method, device, steps, n=300, tau=None):
    from splatstudio.state import check_state, initialize

    renderer, density = _build(method)
    gen = torch.Generator(device="cpu").manual_seed(5)
    pts = (torch.rand(n, 3, generator=gen) - 0.5) * torch.tensor([1.5, 1.5, 2.0])
    pts[:, 2] += 3.0
    params, opts = initialize(method, pts.to(device), torch.rand(n, 3, generator=gen).to(device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.03))
        params["opacities"].fill_(2.0)
    cam = _camera(device)
    density.initialize(params, opts, scene_scale=4.0)
    density.set_training_cameras([cam])
    if tau is not None:
        density.cfg["grad_threshold"] = tau
    # Isolate LFCF/GOF from opacity pruning so row indices stay comparable (probe instance only).
    density.cfg["prune_opacity"] = 0.0
    target = (torch.arange(32, device=device)[None, :, None] % 3 / 2).expand(32, 32, 3).float()
    random.seed(0)
    torch.manual_seed(0)
    snapshots = {}
    for step in steps:
        out = renderer.render(params, cam, sh_degree=0)
        density.before_backward(params, opts, step, out)
        (out["rgb"] - target).abs().mean().backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        snapshots[step] = (params["scales"].detach().clone(),
                           opts["scales"].state[params["scales"]]["exp_avg"].clone())
        density.after_backward(params, opts, step, out)
        check_state(method, params, opts)
    return renderer, density, params, opts, snapshots


def lfcf_gpu_step(method, device):
    """Drive the controller across two LFCF steps on a rendered fixture."""
    _require_cuda(device)
    from splatstudio.registry import load as load_method

    if not _component(method, "density").config["lfcf"]:
        _, bd, _, _, _ = _drive(method, device, range(501, 801), tau=1e-7)
        if bd.state["lfcf_steps"] != 0 or bd.state["gof_steps"] != 3:
            raise AssertionError(f"LFCF-off variant must run only GOF densification: {bd.state}")
        return {"base_state": bd.state}
    _, density, params, opts, snaps = _drive(method, device, range(501, 801), tau=1e-7)
    st = density.state
    if st["lfcf_steps"] != 2 or st["gof_steps"] != 1:
        raise AssertionError(f"expected LFCF at 600, 800 and GOF at 700, got {st}")
    last = density.last_lfcf
    if last["step"] != 800 or not last["enlarged"].any() or not last["shrunk"].any():
        raise AssertionError("second LFCF step must enlarge and shrink some Gaussians")
    before, adam_before = snaps[800]
    n0 = last["n_before"]
    after = params["scales"].detach()[:n0]
    e, sh = last["enlarged"], last["shrunk"]
    exp_e = before[e] + enlarge_delta(before[e], last["log_c"][e][:, None], True)
    torch.testing.assert_close(after[e], exp_e)
    exp_s = before[sh] + shrink_delta(before[sh], last["log_c"][sh][:, None], 2.0, True)
    torch.testing.assert_close(after[sh], exp_s)
    untouched = ~(e | sh)
    torch.testing.assert_close(after[untouched], before[untouched])
    adam = opts["scales"].state[params["scales"]]["exp_avg"]
    torch.testing.assert_close(adam[:n0], adam_before[:n0])
    if len(params["means"]) != n0 + int(last["split"].sum()):
        raise AssertionError("LFCF children count != split count (N = 1, parents kept)")
    if not all(p.requires_grad and p.is_leaf for p in params.values()):
        raise AssertionError("a field lost requires_grad")
    base = load_method("efa-gs-base")
    _, bd, _, _, _ = _drive(base, device, range(501, 801), tau=1e-7)
    if bd.state["lfcf_steps"] != 0 or bd.state["gof_steps"] != 3:
        raise AssertionError(f"efa-gs-base must not run LFCF: {bd.state}")
    return {"enlarged": int(e.sum()), "shrunk": int(sh.sum()), "split": int(last["split"].sum()),
            "fraction": last["fraction"], "base_state": bd.state}


def renderer_absgrad(method, device):
    """Absolute gradients exist and the density accumulates their scaled L1 norm."""
    _require_cuda(device)
    renderer, density, params, opts, _ = _drive(method, device, [1], n=50)
    cam = _camera(device)
    density.stats = {k: torch.zeros_like(v) for k, v in density.stats.items()}
    out = renderer.render(params, cam, sh_degree=0)
    density.before_backward(params, opts, 2, out)
    target = (torch.arange(32, device=device)[None, :, None] % 2).expand(32, 32, 3).float()
    (out["rgb"] - target).square().mean().backward()
    info = out["info"]
    xy = info["means2d"]
    if xy.grad is None or getattr(xy, "absgrad", None) is None:
        raise AssertionError("renderer does not provide signed and absolute gradients")
    if torch.any(xy.absgrad + 1e-9 < xy.grad.abs()):
        raise AssertionError("absolute accumulation smaller than |signed sum|")
    density.accumulate(info)
    vis = (info["radii"][0] > 0).all(dim=-1)
    sx, sy = cam["width"] / 2, cam["height"] / 2
    ref_abs = (xy.absgrad[0, :, 0].abs() * sx + xy.absgrad[0, :, 1].abs() * sy) * vis
    ref = torch.sqrt((xy.grad[0, :, 0] * sx) ** 2 + (xy.grad[0, :, 1] * sy) ** 2) * vis
    torch.testing.assert_close(density.stats["g_abs"], ref_abs)
    torch.testing.assert_close(density.stats["g"], ref)
    torch.testing.assert_close(density.stats["lff_cnt"], vis.float())
    if not torch.count_nonzero(ref_abs):
        raise AssertionError("no absolute gradient reached the density statistics")
    return {"visible": int(vis.sum()), "g_abs_sum": float(ref_abs.sum())}
