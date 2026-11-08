"""Mechanism probes for RAIN-GS; each fails when its mechanism is removed or altered."""

import math
from dataclasses import replace

import torch
from splatstudio.components import build
from splatstudio.state import check_state, sample_scene

from . import density as D
from . import renderer as R


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _with_config(method, kind, **overrides):
    comps = tuple(
        replace(c, config={**c.config, **overrides}) if c.kind == kind else c
        for c in method.components
    )
    return replace(method, components=comps)


def _camera(device, z=0.0, tx=0.0):
    viewmat = torch.eye(4, device=device)
    viewmat[0, 3] = tx
    viewmat[2, 3] = z
    K = torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=device)
    return {"viewmat": viewmat, "K": K, "width": 32, "height": 32}


def _cameras(device):
    # tvec coordinates span [-1, 4]: lo=-1, hi=4 -> cube [-6, 9]^3 (side 15, centre 1.5).
    return [_camera(device, z=4.0, tx=-1.0), _camera(device, z=3.0, tx=0.5),
            _camera(device, z=2.0, tx=1.0)]


def slv_init(method, device):
    """Population replaced by N=10 points in the official tvec cube with 3-NN scales."""
    params, opts = sample_scene(method, device)
    _, density, _ = build(method)
    cfg = density.cfg
    density.initialize(params, opts, scene_scale=2.0)
    torch.manual_seed(0)
    density.set_training_cameras(_cameras(device))
    check_state(method, params, opts)
    n = len(params["means"])
    if n != cfg["num_init"] or n != 10:
        raise AssertionError(f"SLV init must create 10 Gaussians, got {n}")
    lo, hi = -1.0, 4.0  # independent: min/max over tvec coordinates of _cameras
    side = 3 * (hi - lo)
    start = (lo + hi) / 2 - side / 2
    m = params["means"].detach().double()
    if not ((m >= start - 1e-5) & (m <= start + side + 1e-5)).all():
        raise AssertionError("SLV points outside the official cube")
    if float(m.max() - m.min()) < side / 4:
        raise AssertionError("SLV points do not spread over the cube")
    d = torch.cdist(m, m)
    d.fill_diagonal_(float("inf"))
    expect = d.square().topk(3, dim=1, largest=False).values.mean(1).sqrt().log()
    torch.testing.assert_close(params["scales"].detach()[:, 0].double(), expect,
                               rtol=1e-4, atol=1e-4)
    opa = params["opacities"].detach().sigmoid()
    torch.testing.assert_close(opa, torch.full_like(opa, 0.1))
    rgb = params["sh0"].detach()[:, 0] * D.SH_C0 + 0.5
    if not ((rgb >= 0.5 - 1 / 255 - 1e-4) & (rgb <= 0.5 + D.SH_C0 + 1e-4)).all():
        raise AssertionError("sh0 must come from U[0,1) SH coefficients")
    for name, p in params.items():
        if not (p.is_leaf and p.requires_grad) or opts[name].param_groups[0]["params"][0] is not p:
            raise AssertionError(f"{name}: not a trainable leaf owned by its optimizer")
    return {"gaussians": n, "cube": [start, side],
            "mean_scale": float(params["scales"].exp().mean())}


def lowpass_schedule(method, device):
    """eps2d = clip(HW/(9 pi N), 0.3, 300) at step 1 and every 1000 below 25000; eval 0.3."""
    renderer, _, _ = build(method)
    cfg = renderer.cfg
    if not cfg["progressive_lowpass"]:
        raise AssertionError("progressive low-pass disabled")
    H, W = 840, 1297
    values = {}
    for step, n in [(1, 10), (2, 10), (999, 100), (1000, 100), (1000, 10**7)]:
        renderer.step = step - 1
        values[step, n] = renderer.training_lowpass(H, W, n)
    if abs(values[1, 10] - 300.0) > 1e-9:
        raise AssertionError("N=10 must hit the 300 cap")
    if values[2, 10] != values[1, 10] or values[999, 100] != values[1, 10]:
        raise AssertionError("low-pass must only change every 1000 steps")
    expect = H * W / (9 * math.pi * 100)
    if abs(values[1000, 100] - min(max(expect, 0.3), 300)) > 1e-9:
        raise AssertionError("wrong low-pass formula")
    if abs(values[1000, 10**7] - 0.3) > 1e-12:
        raise AssertionError("low-pass must floor at 0.3")
    mid = H * W / (9 * math.pi * 2000)
    renderer.step = 4999
    if abs(renderer.training_lowpass(H, W, 2000) - mid) > 1e-9:
        raise AssertionError("unclamped value mismatch")
    renderer.step = 24999
    before = renderer.lowpass
    if renderer.training_lowpass(H, W, 50) != before:
        raise AssertionError("no recomputation at/after densify_until (25000)")
    if cfg["eval_lowpass"] != 0.3:
        raise AssertionError("test renders must use 0.3")
    return {"n10": values[1, 10], "n100": values[1000, 100], "n2000": mid}


def lowpass_in_render(method, device):
    """A tiny Gaussian covers far more pixels with the step-1 eps2d (300) than with 0.3."""
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, _, _ = build(method)
    cam = _camera(device)
    p = {
        "means": torch.tensor([[0.0, 0.0, 3.0], [5.0, 5.0, 3.0]], device=device),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * 2, device=device),
        "scales": torch.full((2, 3), math.log(1e-3), device=device),
        "opacities": torch.full((2,), 4.0, device=device),
        "sh0": torch.zeros(2, 1, 3, device=device),
        "shN": torch.zeros(2, 15, 3, device=device),
    }
    p = {k: v.requires_grad_() for k, v in p.items()}
    train = renderer.render(p, cam, sh_degree=0)  # step 1: eps2d = HW / (9 pi N), N=2
    with torch.no_grad():
        test = renderer.render(p, cam, sh_degree=0)
    cov_train = int((train["alpha"] > 0.05).sum())
    cov_test = int((test["alpha"] > 0.05).sum())
    expect = min(max(32 * 32 / (9 * math.pi * 2), 0.3), 300.0)
    if abs(train["info"]["rain_eps2d"] - expect) > 1e-9 or test["info"]["rain_eps2d"] != 0.3:
        raise AssertionError("eps2d not applied as scheduled")
    if cov_train < 20 * max(cov_test, 1):
        raise AssertionError(f"low-pass has no effect on the footprint ({cov_train} vs {cov_test})")
    return {"train_pixels": cov_train, "test_pixels": cov_test}


def _setup_split(method, device, step):
    params, opts = sample_scene(method, device)
    _, density, _ = build(method)
    density.initialize(params, opts, scene_scale=2.0)
    density.set_training_cameras(_cameras(device))
    n = len(params["means"])
    with torch.no_grad():
        params["scales"].copy_(torch.log(torch.full_like(params["scales"], 0.1)))
        params["scales"][0] = math.log(0.005)  # small -> clone candidate (<= 0.01 * 2)
        params["opacities"].fill_(0.0)
        params["means"].copy_(torch.arange(n * 3, device=device, dtype=torch.float32)
                              .reshape(n, 3) / 10 + 1)
    density.state["grad2d"] = torch.zeros(n, device=device)
    density.state["count"] = torch.ones(n, device=device)
    density.state["grad2d"][:2] = 1.0  # #0 clone, #1 split (+ABE), others untouched
    before = {k: v.detach().clone() for k, v in params.items()}
    event = density.densify(params, opts, step)
    check_state(method, params, opts)
    return params, before, event, n


def abe_split(method, device):
    """step <= 10000: split candidate -> 2 children (scale/1.4) + 1 at 0.3*extent*mu."""
    extent = 2.0
    params, before, ev, n = _setup_split(method, device, step=5000)
    if (ev["clone"], ev["split"], ev["abe"]) != (1, 1, 1) or len(params["means"]) != n + 3:
        raise AssertionError(f"unexpected densify counts {ev}")
    # Layout: [rest (n-1) incl. clone source][clone][abe][2 split children]
    abe = params["means"][n]
    torch.testing.assert_close(abe, before["means"][1] * 0.3 * extent)
    torch.testing.assert_close(params["scales"][n], before["scales"][1])
    child = params["scales"][-2:]
    torch.testing.assert_close(child, (before["scales"][1] - math.log(1.4)).expand(2, 3))
    if (params["means"][-2:] - before["means"][1]).norm(dim=-1).max() > 5 * 0.1:
        raise AssertionError("split children not sampled around the parent")
    params2, _, ev2, _ = _setup_split(method, device, step=10100)
    if ev2["abe"] != 0 or len(params2["means"]) != n + 2:
        raise AssertionError("ABE-Split must stop after the warm-up (10000)")
    return {"warmup": ev, "after": ev2}


def warmup_schedule(method, device):
    """Means LR fixed < 10000 then official expon(step - 10000); SH 0/1/2/3 at 4999/5000/6/7k."""
    _, density, _ = build(method)
    cfg = density.cfg
    lr0 = 1.6e-4 * 3.0
    for step in (1, 5000, 9999):
        if D.means_lr(step, lr0, cfg["warmup"]) != lr0:
            raise AssertionError("means LR must be fixed during the warm-up")
    for step in (10000, 20000, 30000):
        t = (step - 10000) / 30000
        expect = math.exp(math.log(lr0) * (1 - t) + math.log(lr0 * 0.01) * t)
        if abs(D.means_lr(step, lr0, cfg["warmup"]) - expect) > 1e-12:
            raise AssertionError("means LR schedule after warm-up differs from official")
    rcfg = _component(method, "renderer").config
    degrees = [R.sh_degree_at(s, rcfg["sh_start"], rcfg["sh_every"])
               for s in (4999, 5000, 5999, 6000, 7000, 30000)]
    if degrees != [0, 1, 1, 2, 3, 3]:
        raise AssertionError(f"SH schedule {degrees}")
    if cfg["densify_until"] != 25000 or abs(cfg["divide_ratio"] * 2 - 1.4) > 1e-12:
        raise AssertionError("densification window / divide ratio differ from official")
    return {"lr_end": D.means_lr(30000, lr0, cfg["warmup"]) / lr0, "sh": degrees}


def renderer_gradient(method, device):
    """Native renderer probe adapted to SLV init: training render supplies screen gradients
    and one full density step keeps the state aligned."""
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    params, opts = sample_scene(method, device)
    density.initialize(params, opts, scene_scale=2.0)
    density.set_training_cameras(_cameras(device))
    cam = _camera(device)
    with torch.no_grad():
        params["means"].copy_(torch.rand_like(params["means"]) - 0.5 + torch.tensor(
            [0.0, 0.0, 3.0], device=device))
    outputs = renderer.render(params, cam, sh_degree=0)
    outputs["step"] = 1
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(32, device=device)[None, :, None] % 2).expand(32, 32, 3).float()
    (outputs["rgb"] - target).square().mean().backward()
    g = outputs["info"]["means2d"].grad
    if g is None or not torch.isfinite(g).all() or not torch.count_nonzero(g):
        raise AssertionError("no finite nonzero screen-space gradient")
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    return {"screen_gradient_norm": float(g.norm()), "eps2d": outputs["info"]["rain_eps2d"]}
