"""Mechanism probes for the BBSplat port. Each raises on failure and returns measurements."""

import math

import torch

from . import texture as tex
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import Density, farthest_point_sampling, fibonacci_sphere, noise_coefficient
from .losses import texture_reg
from .reference import render_reference


def _cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        from splatstudio.probes import ProbeUnavailable

        raise ProbeUnavailable("CUDA required")
    return torch.device(device)


def _scene(dev, seed=0):
    """Six billboards at well separated depths in front of a 64x48 camera."""
    g = torch.Generator().manual_seed(seed)
    n = 6
    means = torch.stack([(torch.rand(n, generator=g) - 0.5) * 0.8,
                         (torch.rand(n, generator=g) - 0.5) * 0.6,
                         3.0 + 0.5 * torch.arange(n, dtype=torch.float32)], -1)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.2 * torch.randn(n, 4, generator=g)
    scales = 0.45 + 0.35 * torch.rand(n, 2, generator=g)
    sh = torch.zeros(n, 16, 3)
    sh[:, 0] = (torch.rand(n, 3, generator=g) * 0.6 - 0.3) / 0.28209479177387814
    sh[:, 1:4] = 0.1 * torch.randn(n, 3, 3, generator=g)
    alpha_logit = torch.randn(n, 16, 16, generator=g) + 0.5
    color_logit = torch.randn(n, 3, 16, 16, generator=g) - 1.0
    cam = {"K": torch.tensor([[60.0, 0, 32.0], [0, 60.0, 24.0], [0, 0, 1]]),
           "viewmat": torch.eye(4), "width": 64, "height": 48}
    out = dict(means=means, quats=quats, scales=scales, sh=sh, alpha_logit=alpha_logit,
               color_logit=color_logit)
    return {k: v.to(dev) for k, v in out.items()}, {k: (v.to(dev) if torch.is_tensor(v) else v)
                                                     for k, v in cam.items()}


def _kernel(s, cam, degree, bg, dev):
    from .raster import raster_settings, rasterize

    leaves = {k: v.clone().requires_grad_(True) for k, v in s.items()}
    st = raster_settings(cam, bg, degree)
    color, radii, impact, allmap = rasterize(
        leaves["means"], leaves["sh"], leaves["alpha_logit"].sigmoid(),
        leaves["color_logit"].sigmoid(), leaves["scales"], torch.nn.functional.normalize(
            leaves["quats"], dim=-1), st)
    return leaves, color.permute(1, 2, 0), allmap[1], allmap[0], impact


def _ref(s, cam, degree, bg):
    leaves = {k: v.double().clone().requires_grad_(True) for k, v in s.items()}
    rgb, alpha, dsum, impact = render_reference(
        leaves["means"], leaves["quats"], leaves["scales"], leaves["sh"],
        leaves["alpha_logit"].sigmoid(), leaves["color_logit"].sigmoid(), degree,
        cam["viewmat"].double(), float(cam["K"][0, 0]), float(cam["K"][1, 1]), cam["width"],
        cam["height"], bg.double())
    return leaves, rgb, alpha, dsum, impact


def raster_vs_reference(method, device):
    dev = _cuda(device)
    s, cam = _scene(dev)
    bg = torch.tensor([0.1, 0.2, 0.3], device=dev)
    kl, krgb, kalpha, kdepth, kimp = _kernel(s, cam, 1, bg, dev)
    rl, rrgb, ralpha, rdepth, rimp = _ref(s, cam, 1, bg)
    meas = {}
    for name, a, b in (("rgb", krgb, rrgb), ("alpha", kalpha, ralpha), ("depth", kdepth, rdepth),
                       ("impact", kimp, rimp)):
        err = float((a.double() - b).abs().max())
        scale = float(b.abs().max()) + 1e-12
        meas[f"{name}_max_abs_err"] = err
        if err > 2e-3 * max(1.0, scale):
            raise AssertionError(f"forward {name} differs from reference: {err:.3g} (scale "
                                 f"{scale:.3g})")
    if float(rimp.sum()) <= 0 or float(ralpha.mean()) < 0.2:
        raise AssertionError("probe scene is degenerate (nothing rendered)")
    g = torch.Generator().manual_seed(1)
    w_rgb = torch.randn(krgb.shape, generator=g).to(dev)
    w_a = torch.randn(kalpha.shape, generator=g).to(dev)
    w_d = torch.randn(kdepth.shape, generator=g).to(dev)
    ((krgb * w_rgb).sum() + (kalpha * w_a).sum() + 0.1 * (kdepth * w_d).sum()).backward()
    ((rrgb * w_rgb.double()).sum() + (ralpha * w_a.double()).sum()
     + 0.1 * (rdepth * w_d.double()).sum()).backward()
    for name in ("means", "quats", "scales", "sh", "alpha_logit", "color_logit"):
        a, b = kl[name].grad, rl[name].grad
        if name == "sh":
            a, b = a[:, :4], b[:, :4]  # degree 1 active
        rel = float((a.double() - b).norm() / (b.norm() + 1e-12))
        meas[f"grad_{name}_rel_err"] = rel
        if not math.isfinite(rel) or rel > 2e-2:
            raise AssertionError(f"gradient {name} differs from reference: rel {rel:.3g}")
        if float(b.norm()) == 0:
            raise AssertionError(f"reference gradient of {name} is zero (probe too weak)")
    return meas


def texture_composition(method, device):
    """Eq. 6: one front-facing billboard with constant textures; pixel = alpha*(T^RGB + SH)."""
    from .raster import raster_settings, rasterize

    dev = _cuda(device)
    cam = {"K": torch.tensor([[50.0, 0, 16.0], [0, 50.0, 16.0], [0, 0, 1]], device=dev),
           "viewmat": torch.eye(4, device=dev), "width": 32, "height": 32}
    means = torch.tensor([[0.0, 0.0, 2.0]], device=dev)
    sh = torch.zeros(1, 16, 3, device=dev)
    base = torch.tensor([0.2, 0.3, 0.1], device=dev)
    sh[0, 0] = (base - 0.5) / 0.28209479177387814
    a = torch.full((1, 16, 16), 0.7, device=dev)
    t = torch.tensor([0.25, 0.05, 0.4], device=dev)
    c = t[None, :, None, None].expand(1, 3, 16, 16).contiguous()
    st = raster_settings(cam, torch.zeros(3, device=dev), 0)
    q = torch.tensor([[1.0, 0, 0, 0]], device=dev)
    sc = torch.tensor([[1.0, 1.0]], device=dev)
    with torch.no_grad():
        col = rasterize(means, sh, a, c, sc, q, st)[0][:, 16, 16]
        col0 = rasterize(means, sh, a, torch.zeros_like(c), sc, q, st)[0][:, 16, 16]
    expect = 0.7 * (t + base)
    err = float((col - expect).abs().max())
    if err > 1e-4:
        raise AssertionError(f"colour != alpha*(texture + SH): {col.tolist()} vs {expect.tolist()}")
    if float((col - col0).abs().max()) < 0.1:
        raise AssertionError("texture colour has no effect")
    return {"centre_rgb": col.tolist(), "expected": expect.tolist(), "max_err": err}


def texture_regularizer(method, device="cpu"):
    g = torch.Generator().manual_seed(0)
    n = 5
    al = torch.randn(n, 16, 16, generator=g, requires_grad=True)
    cl = torch.randn(n, 3, 16, 16, generator=g, requires_grad=True)
    impact = torch.tensor([0.0, 5.0, 50.0, 100.0, 400.0])
    got = texture_reg(al, cl, impact, 100.0, 1e-4, 2e-4)
    a0 = 0.5 * torch.tensor(tex.GAUSSIAN_PNG, dtype=torch.float32) / 255
    terms_c, terms_a = [], []
    for i in range(n):  # independent loop form of train.py
        if impact[i] <= 0:
            continue
        w = 100.0 - min(float(impact[i]), 100.0)
        terms_c.append(float(torch.sigmoid(cl[i]).mean()) * w)
        terms_a.append(abs(float((torch.sigmoid(al[i]) - a0).mean())) * w)
    expect = 1e-4 * sum(terms_c) / len(terms_c) + 2e-4 * sum(terms_a) / len(terms_a)
    if abs(float(got) - expect) > 1e-6 * max(1.0, expect):
        raise AssertionError(f"texture reg {float(got)} != {expect}")
    got.backward()
    if float(cl.grad[0].abs().sum()) != 0 or float(cl.grad[3].abs().sum()) != 0:
        raise AssertionError("invisible / saturated-impact billboards must get no reg gradient")
    if float(cl.grad[1].abs().sum()) == 0:
        raise AssertionError("visible low-impact billboard gets no reg gradient")
    zero = texture_reg(al, cl, torch.zeros(n), 100.0, 1e-4, 1e-4)
    return {"value": float(got), "expected": expect, "no_visible": float(zero)}


def mcmc_relocation(method, device="cpu"):
    from splatstudio import ComponentSpec

    torch.manual_seed(0)
    n = 40
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(n, 3)),
        "scales": torch.nn.Parameter(torch.randn(n, 3)),
        "quats": torch.nn.Parameter(torch.randn(n, 4)),
        "opacities": torch.nn.Parameter(torch.zeros(n)),
        "sh0": torch.nn.Parameter(torch.randn(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.randn(n, 15, 3)),
        tex.ALPHA: torch.nn.Parameter(torch.randn(n, 16, 16)),
        tex.COLOR: torch.nn.Parameter(torch.randn(n, 3, 16, 16)),
    })
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
    dead = torch.zeros(n, dtype=torch.bool)
    dead[:7] = True
    with torch.no_grad():
        params[tex.ALPHA][:7] = -12.0  # mean alpha ~ 6e-6 <= 0.005
    d = Density(ComponentSpec("d", "density", "x", (), (), {"cap_max": 50}))
    before = {k: p.detach().clone() for k, p in params.items()}
    ids = {k: id(p) for k, p in params.items()}
    torch.manual_seed(3)
    measured_dead = params[tex.ALPHA].sigmoid().view(n, -1).mean(1) <= 0.005
    if not torch.equal(measured_dead, dead):
        raise AssertionError("dead test (mean alpha <= 0.005) wrong")
    moved = d.relocate(params, opts, dead)
    # independent replay of the sampling with the same seed
    torch.manual_seed(3)
    alive = (~dead).nonzero(as_tuple=True)[0]
    probs = before[tex.ALPHA][alive].sigmoid().view(len(alive), -1).mean(1)
    probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    src = alive[torch.multinomial(probs, 7, replacement=True)]
    cnt = torch.bincount(src)
    for j, di in enumerate(range(7)):
        s = int(src[j])
        a_old = before[tex.ALPHA][s].sigmoid()
        a_new = 1 - (1 - a_old) ** (1.0 / (int(cnt[s]) + 1))
        if not torch.allclose(params[tex.ALPHA][di].sigmoid(), a_new, atol=1e-5):
            raise AssertionError("relocated alpha != 1-(1-a)^(1/(n+1))")
        if not torch.allclose(params[tex.ALPHA][s].sigmoid(), a_new, atol=1e-5):
            raise AssertionError("source alpha not rescaled")
        for k in ("means", "scales", "quats", "sh0", "shN", tex.COLOR):
            if not torch.equal(params[k][di], before[k][s]):
                raise AssertionError(f"{k} not copied from source")
        if not torch.equal(params["scales"][s], before["scales"][s]):
            raise AssertionError("scales must stay unchanged (BBSplat drops the MCMC scale rule)")
    for k, p in params.items():
        if id(p) != ids[k] or not p.requires_grad:
            raise AssertionError(f"{k} parameter replaced or frozen by relocation")
        st = opts[k].state[p]
        if float(st["exp_avg"][src].abs().sum()) != 0:
            raise AssertionError("Adam moments not reset at sampled indices")
    added = d.add_new(params, opts, 50)
    if added != int(1.05 * n) - n or len(params["means"]) != n + added:
        raise AssertionError(f"growth wrong: added {added}")
    for k, p in params.items():
        if not p.requires_grad or opts[k].param_groups[0]["params"][0] is not p:
            raise AssertionError(f"{k}: optimizer misaligned after add")
        if opts[k].state[p]["exp_avg"].shape != p.shape:
            raise AssertionError(f"{k}: moment shape misaligned")
    capped = d.add_new(params, opts, len(params["means"]))
    coef = noise_coefficient()
    if coef != 0.0:
        raise AssertionError(f"official MCMC noise coefficient should underflow to 0, got {coef}")
    return {"relocated": moved, "added": added, "capped_add": capped, "noise_coefficient": coef}


def init_sky_fps(method, device="cpu"):
    g = torch.Generator().manual_seed(0)
    pts = torch.randn(300, 3, generator=g)
    got = farthest_point_sampling(pts, 25)
    sel = [0]
    dist = ((pts - pts[0]) ** 2).sum(-1)
    for _ in range(24):  # brute force FPS
        j = int(torch.argmax(dist))
        sel.append(j)
        dist = torch.minimum(dist, ((pts - pts[j]) ** 2).sum(-1))
    if got.tolist() != sel:
        raise AssertionError("FPS differs from brute-force farthest point sampling")
    sph = fibonacci_sphere(10000)
    r = sph.norm(dim=-1)
    if not torch.allclose(r, torch.ones_like(r), atol=1e-9) or abs(float(sph.mean(0).norm())) > 1e-3:
        raise AssertionError("Fibonacci sphere is not uniform on the unit sphere")
    from splatstudio import ComponentSpec

    cfg = {"max_read_points": 200, "sky_points": 50, "cap_max": 300}
    d = Density(ComponentSpec("d", "density", "x", (), (), cfg))
    n0 = 300
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts.clone()),
        "scales": torch.nn.Parameter(torch.zeros(n0, 3)),
        "quats": torch.nn.Parameter(torch.zeros(n0, 4)),
        "opacities": torch.nn.Parameter(torch.zeros(n0)),
        "sh0": torch.nn.Parameter(torch.zeros(n0, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(n0, 15, 3)),
        tex.ALPHA: torch.nn.Parameter(torch.zeros(n0, 16, 16)),
        tex.COLOR: torch.nn.Parameter(torch.zeros(n0, 3, 16, 16)),
    })
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    d.initialize(params, opts, 2.0)
    n = len(params["means"])
    if n != 250:
        raise AssertionError(f"population {n} != 200 FPS + 50 sky")
    sky = params["means"][:50]
    keep = farthest_point_sampling(pts, 200)
    radius = float(pts[keep].abs().max())
    if not torch.allclose(sky.norm(dim=-1), torch.full((50,), radius), rtol=1e-5):
        raise AssertionError("sky radius != max |xyz| of the subsampled SfM points")
    sky_rgb = params["sh0"][:50, 0] * 0.28209479177387814 + 0.5
    if not torch.allclose(sky_rgb, torch.full_like(sky_rgb, 1 - 0.1 * 0.28209479177387814),
                          atol=1e-6):
        raise AssertionError("sky colour / SH dc offset wrong")
    a = params[tex.ALPHA].sigmoid()[0]
    if not torch.allclose(a, tex.initial_alpha(), atol=1e-6):
        raise AssertionError("alpha init != 0.5 G")
    if not torch.allclose(params[tex.COLOR].sigmoid(), torch.full((n, 3, 16, 16), 0.1),
                          atol=1e-6):
        raise AssertionError("colour texture init != 0.1")
    if not all(p.requires_grad for p in params.values()):
        raise AssertionError("init froze a field")
    return {"population": n, "sky_radius": radius, "zero_texels": int((a == 0).sum())}


def schedule(method, device="cpu"):
    from splatstudio import ComponentSpec

    d = Density(ComponentSpec("d", "density", "x", (), (), {}))
    d.extent = 2.0
    c = DENSITY_DEFAULTS
    lr = d.learning_rates
    checks = {
        "tex_off_499": lr(499)[tex.ALPHA] == 0 and lr(499)[tex.COLOR] == 0,
        "tex_on_500": lr(500)[tex.ALPHA] == c["texture_opacity_lr"]
        and lr(500)[tex.COLOR] == c["texture_color_lr"],
        "tex_on_29999": lr(29999)[tex.COLOR] == c["texture_color_lr"],
        "tex_off_30000": lr(30000)[tex.COLOR] == 0,
        "geometry_on_30000": lr(30000)["means"] > 0 and lr(30000)["sh0"] == c["feature_lr"],
        "frozen_30001": all(lr(30001)[k] == 0 for k in ("means", "sh0", "scales", "quats",
                                                        tex.ALPHA, tex.COLOR)),
        "sh_rest_30001": lr(30001)["shN"] == c["feature_lr"] / 20,
        "no_step_32000": all(v == 0 for v in lr(32000).values()),
        "xyz_lr_1": abs(lr(1)["means"] - 1.6e-4 * 2.0 * 0.01 ** (1 / 30000)) < 1e-12,
        "xyz_lr_30000": abs(lr(30000)["means"] - 1.6e-6 * 2.0) < 1e-12,
        "opacity_unused": lr(1000)["opacities"] == 0,
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise AssertionError(f"schedule checks failed: {bad}")
    return {k: bool(v) for k, v in checks.items()}
