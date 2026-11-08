"""Mechanism probes for the CoR-GS plugin; each fails if its mechanism is removed or wrong."""

from dataclasses import replace

import numpy as np
import torch
from splatstudio.state import check_state, initialize


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _density(method, **overrides):
    from .density import create_density

    spec = _spec(method, "corgs-density")
    return create_density(replace(spec, config={**spec.config, **overrides}))


def _scene(method, device, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) * 2 - 1
    pts[:, 2] += 4.0  # in front of the identity camera
    colors = torch.rand(n, 3, generator=g)
    params, opts = initialize(method, pts.to(device), colors.to(device))
    with torch.no_grad():
        params["scales"][:] = -2.5
        params["opacities"][:] = 0.0
    return params, opts


def _camera(device, w=64, h=48):
    K = torch.tensor([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": w, "height": h}


def _prime_adam(params, opts):
    for p in params.values():
        p.grad = torch.ones_like(p) * 1e-3
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)


# -- C1 co-pruning ---------------------------------------------------------------------------
def co_pruning(method, device):
    """Pruned sets of both fields == brute-force Eq. 3 reference; gated to t>500, t%500==0."""
    params, opts = _scene(method, device, n=120)
    d = _density(method)
    d.initialize(params, opts, scene_scale=1.0)
    f2 = d.field2
    tau = float(d.config["coprune_threshold"])
    with torch.no_grad():  # outliers: far from everything in the other field
        params["means"][:5] += torch.tensor([30.0, 0, 0], device=device) * torch.arange(
            1, 6, device=device)[:, None]
        f2.params["means"][-7:] += torch.tensor([0, -25.0, 0], device=device) * torch.arange(
            1, 8, device=device)[:, None]
        f2.params["means"][10:20] += 0.9 * tau / np.sqrt(3)  # moved 0.9 tau: still matched
    _prime_adam(params, opts)
    _prime_adam(f2.params, f2.optimizers)
    a = params["means"].detach().double().cpu().numpy()
    b = f2.params["means"].detach().double().cpu().numpy()
    da = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min(1)
    db = np.sqrt(((b[:, None] - a[None]) ** 2).sum(-1)).min(1)
    keep_a, keep_b = da <= tau, db <= tau
    if keep_a.all() or keep_b.all():
        raise AssertionError("fixture has no non-matching Gaussians in one field")
    gates = {t: d.coprune_due(t) for t in (500, 999, 1000, 1500, 1501, 10000)}
    if gates != {500: False, 999: False, 1000: True, 1500: True, 1501: False, 10000: True}:
        raise AssertionError(f"co-prune schedule wrong: {gates}")
    row = d.coprune(1000)
    got_a = params["means"].detach().double().cpu().numpy()
    got_b = f2.params["means"].detach().double().cpu().numpy()
    if not (np.allclose(got_a, a[keep_a]) and np.allclose(got_b, b[keep_b])):
        raise AssertionError("co-pruned fields differ from the brute-force Eq. 3 reference")
    check_state(method, params, opts)
    for name, p in f2.params.items():
        st = f2.optimizers[name].state[p]
        if st["exp_avg"].shape != p.shape or not p.requires_grad:
            raise AssertionError(f"field 2 optimizer/param misaligned for {name}")
    off = _density(method, coprune=False)
    off.initialize(params, opts, scene_scale=1.0)
    if off.coprune_due(1000):
        raise AssertionError("coprune=False still schedules co-pruning")
    return {"pruned": row["coprune"], "expected": [int((~keep_a).sum()), int((~keep_b).sum())],
            "tau": tau, "gates": {str(k): v for k, v in gates.items()}}


# -- C2 pseudo-view co-regularisation -----------------------------------------------------------
def _ssim_reference(x, y):
    """Independent SSIM: separable 1-D Gaussian blur via unfold (valid region), HWC inputs."""
    k = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    k = (k / k.sum()).to(x)

    def blur(t):  # t: [C, H, W]
        t = t.unfold(1, 11, 1) @ k  # rows
        return t.unfold(2, 11, 1) @ k  # cols

    a, b = x.permute(2, 0, 1), y.permute(2, 0, 1)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma**2, blur(b * b) - mb**2, blur(a * b) - ma * mb
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))).mean()


class _Stub:
    def __init__(self, field2, camera):
        self.field2, self.camera, self.calls = field2, camera, 0

    def next_pseudo_camera(self):
        self.calls += 1
        return self.camera


def pseudo_coregularization(method, device):
    from .density import clone_field
    from .losses import create_coreg
    from .renderer import rasterize

    params, opts = _scene(method, device)
    f2 = clone_field(params, opts)
    with torch.no_grad():
        f2.params["sh0"].add_(0.3 * torch.randn_like(f2.params["sh0"]))
        f2.params["means"].add_(0.02 * torch.randn_like(f2.params["means"]))
    cam = _camera(device)
    stub = _Stub(f2, cam)
    loss = create_coreg(_spec(method, "corgs-coreg"))
    bg = torch.tensor([0.2, 0.5, 0.7], device=device)
    rgb_main = rasterize(params, cam, 0, bg)["rgb"]
    outs = {"rgb": rgb_main, "corgs": stub, "background": bg, "sh_degree": 0,
            "st_alpha_clamp": True}
    for step in (1, 499, 10001):
        v = loss(params, {**outs, "step": step}, None)
        if float(v) != 0.0 or stub.calls:
            raise AssertionError(f"co-reg active outside [500, 10000] (step {step})")
    value = loss(params, {**outs, "step": 500}, None)
    if stub.calls != 1:
        raise AssertionError("co-reg did not sample one pseudo view")
    with torch.no_grad():
        i1 = rasterize(params, cam, 0, bg)["rgb"].double()
        i2 = rasterize(f2.params, cam, 0, bg)["rgb"].double()
        r = 0.8 * (i1 - i2).abs().mean() + 0.2 * (1 - _ssim_reference(i1, i2))
    w = float(_spec(method, "corgs-coreg").config["weight"])
    expected = w * 2 * float(r)  # R(I1, sg I2) + R(I2, sg I1) have equal values
    if not np.isclose(float(value), expected, rtol=1e-3, atol=1e-6) or expected < 1e-3:
        raise AssertionError(f"co-reg loss {float(value)} != reference {expected}")
    # detach: field-1 term must not reach field 2 and vice versa
    from .losses import coreg_terms

    a = rasterize(params, cam, 0, bg)["rgb"]
    b = rasterize(f2.params, cam, 0, bg)["rgb"]
    r1, _ = coreg_terms(a, b)
    r1.backward()
    leak = any(p.grad is not None and p.grad.abs().sum() > 0 for p in f2.params.values())
    reach = params["sh0"].grad is not None and params["sh0"].grad.abs().sum() > 0
    if leak or not reach:
        raise AssertionError(f"gradient routing wrong (leak={leak}, reach={reach})")
    for p in params.values():
        p.grad = None
    same = clone_field(params, opts)
    stub2 = _Stub(same, cam)
    zero = loss(params, {**outs, "corgs": stub2, "step": 1000}, None)
    if float(zero) > 1e-5:
        raise AssertionError("identical fields give a non-zero co-reg loss")
    return {"loss": float(value), "reference": expected, "identical_fields": float(zero)}


# -- two fields ------------------------------------------------------------------------------
def two_fields(method, device):
    """Θ² is a copy of Θ¹, trained/densified independently, absent from runtime params."""
    from splatstudio.components import build

    renderer, density, losses = build(method)
    params, opts = _scene(method, device, n=300, seed=1)
    with torch.no_grad():
        params["opacities"][:] = 2.0
    density.initialize(params, opts, scene_scale=1.0)
    f2 = density.field2
    if f2 is None:
        raise AssertionError("no second field")
    for name, p in params.items():
        q = f2.params[name]
        if not torch.equal(p, q) or p.data_ptr() == q.data_ptr():
            raise AssertionError(f"field 2 {name} is not an independent copy of field 1")
    if set(f2.optimizers) != set(opts):
        raise AssertionError("field 2 lacks per-field optimizers")
    cam = _camera(device)
    density.pseudo_viewmats = cam["viewmat"][None].clone()
    density.pseudo_K, density.pseudo_size = cam["K"], (cam["width"], cam["height"])
    g = torch.Generator().manual_seed(0)
    target = torch.rand(cam["height"], cam["width"], 3, generator=g).to(device)
    init2 = {k: v.detach().clone() for k, v in f2.params.items()}
    names = [n for n, _ in losses]
    for step in range(1, 6):
        out = renderer.render(params, cam, 0)
        out["step"] = step
        density.before_backward(params, opts, step, out)
        total = sum(loss(params, out, target) for _, loss in losses)
        total.backward()
        if f2.params["sh0"].grad is None:
            raise AssertionError("field 2 receives no gradient")
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        density.after_backward(params, opts, step, out)
    moved = float((f2.params["sh0"] - init2["sh0"]).abs().max())
    if moved == 0.0:
        raise AssertionError("field 2 was not optimised")
    if any(f2.params[k] is params[k] for k in params) or set(params) != set(init2):
        raise AssertionError("field 2 leaked into the runtime params")
    # densify both at a densification step: stats are per field
    density.field1.state["grad2d"][:] = 1.0
    density.field1.state["count"][:] = 1.0
    density.field2.reset_stats()
    n1, n2 = len(params["means"]), len(f2.params["means"])
    r1 = density.densify_and_prune(density.field1, 600)
    r2 = density.densify_and_prune(density.field2, 600)
    if r1["cloned"] + r1["split"] == 0 or r2["cloned"] + r2["split"] != 0:
        raise AssertionError("densification statistics are not per field")
    if len(params["means"]) == n1 or len(f2.params["means"]) > n2:
        raise AssertionError("field sizes did not evolve independently")
    # removing the field-2 loss must break the method (after_backward refuses)
    out = renderer.render(params, cam, 0)
    out["step"] = 6
    density.before_backward(params, opts, 6, out)
    try:
        density.after_backward(params, opts, 6, out)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing field-2 render went unnoticed")
    return {"losses": names, "field2_sh0_moved": moved, "sizes": [len(params["means"]),
                                                                   len(f2.params["means"])]}


# -- protocol --------------------------------------------------------------------------------
def corgs_schedule(method, device):
    from .renderer import create_renderer

    r = create_renderer(_spec(method, "corgs-renderer"))
    with torch.enable_grad():
        seen = {n: r.degree() for n in range(1, 2001)}
        bgs = [r.background("cpu") for _ in range(3)]
    if seen != {n: min(n // 500, 3) for n in seen}:
        raise AssertionError("SH degree schedule differs from min(t // 500, 3)")
    if not all(isinstance(b, torch.Tensor) and b.shape == (3,) for b in bgs) or torch.equal(
            bgs[0], bgs[1]):
        raise AssertionError("training background is not random per render")
    with torch.no_grad():
        if r.degree() != 3 or r.background("cpu") != 0.0:
            raise AssertionError("evaluation must use SH degree 3 and a black background")
    d = _density(method)
    resets = [t for t in range(1, 10001) if d.reset_due(t)]
    if resets != [501, 3501, 6501, 9501]:
        raise AssertionError(f"opacity resets at {resets}")
    copr = [t for t in range(1, 10001) if d.coprune_due(t)]
    expected = list(range(1000, 10001, 500)) if d.config["coprune"] else []
    if copr != expected:
        raise AssertionError("co-prune steps differ from 1000..10000 every 500")
    x = torch.rand(20, 20, 3, dtype=torch.float64)
    y = torch.rand(20, 20, 3, dtype=torch.float64)
    from splatstudio.losses import ssim

    if not torch.isclose(ssim(x, y), _ssim_reference(x, y)):
        raise AssertionError("SSIM reference disagrees with the runtime SSIM")
    return {"resets": resets, "coprune_steps": [copr[0], copr[-1], len(copr)] if copr else [],
            "degree_at": {k: seen[k] for k in (499, 500, 1000, 1500)}}
