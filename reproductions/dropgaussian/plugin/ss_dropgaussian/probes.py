"""Mechanism probes for DropGaussian; each raises if its mechanism is removed or wrong."""

import math

import torch
from splatstudio.state import sample_scene


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _renderer(method):
    from .renderer import create_renderer

    return create_renderer(_spec(method, "dropgaussian-renderer"))


def _density(method):
    from .density import create_density

    return create_density(_spec(method, "dropgaussian-density"))


def drop_compensation(method, device):
    """Eq. 5: M(i) = 1/(1-r) (kept) or 0 (dropped); keep probability 1-r; E[M] = 1."""
    from .renderer import compensation

    torch.manual_seed(0)
    n, rate = 200_000, 0.15
    m = compensation(n, rate, device)
    expected_kept = 1.0 / (1.0 - rate)  # independent statement of the compensation factor
    kept = m > 0
    if not torch.allclose(m[kept], torch.full_like(m[kept], expected_kept), rtol=1e-6):
        raise AssertionError("kept Gaussians are not scaled by 1/(1-r)")
    if torch.count_nonzero(m[~kept]) != 0:
        raise AssertionError("dropped Gaussians must have zero opacity")
    frac = float(kept.float().mean())
    sigma = math.sqrt(rate * (1 - rate) / n)
    if abs(frac - (1 - rate)) > 5 * sigma:
        raise AssertionError(f"keep fraction {frac} != 1 - r = {1 - rate}")
    mean = float(m.mean())
    if abs(mean - 1.0) > 5 * sigma * expected_kept:
        raise AssertionError(f"compensation does not preserve expected opacity: E[M]={mean}")
    if not torch.equal(compensation(10, 0.0, device), torch.ones(10, device=device)):
        raise AssertionError("r = 0 must keep every Gaussian unscaled")
    m2 = compensation(n, rate, device)
    if torch.equal(m, m2):
        raise AssertionError("the dropout mask must be redrawn for every render")
    return {"rate": rate, "keep_fraction": frac, "kept_factor": float(m[kept][0]),
            "mean_factor": mean}


def drop_schedule(method, device):
    """Eq. 6 boundaries and the renderer-step / runtime-step consistency check."""
    from .renderer import drop_rate

    cfg = _spec(method, "dropgaussian-renderer").config
    gamma, total = float(cfg["gamma"]), int(cfg["total_steps"])
    if (gamma, total) != (0.2, 10000):
        raise AssertionError(f"paper/code schedule is gamma 0.2 over 10000 its, got {gamma}, {total}")
    points = {t: drop_rate(t, gamma, total) for t in (1, 2500, 5000, 10000)}
    official = {t: 0.2 * (t / 10000) for t in points}  # gaussian_renderer/__init__.py:98
    for t, value in points.items():
        if abs(value - official[t]) > 1e-12:
            raise AssertionError(f"r_{t} = {value} != official {official[t]}")
    if not points[1] < points[5000] < points[10000]:
        raise AssertionError("dropping rate must increase with t")
    density = _density(method)
    info = {"means2d": torch.zeros(1, 4, 2, requires_grad=True)}
    density.before_backward({}, {}, 7, {"info": info, "drop": {"step": 7}})
    try:
        density.before_backward({}, {}, 8, {"info": info, "drop": {"step": 7}})
    except RuntimeError:
        pass
    else:
        raise AssertionError("density must reject a renderer step that differs from the runtime")
    return {"rates": points}


def _fixture_scene(device, n=4000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.rand(n, 3, generator=g) - 0.5
    means[:, :2] *= 1.6
    means[:, 2] += 3.0
    params = {
        "means": means,
        "scales": torch.full((n, 3), math.log(0.03)),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.full((n,), torch.logit(torch.tensor(0.6)).item()),
        "sh0": torch.rand(n, 1, 3, generator=g) * 2 - 1,
        "shN": torch.zeros(n, 15, 3),
    }
    params = {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[80.0, 0, 48], [0, 80.0, 48], [0, 0, 1]], device=device),
        "width": 96,
        "height": 96,
    }
    return params, camera


def dropout_in_render(method, device):
    """Training renders drop Gaussians: rgb changes, dropped Gaussians get zero gradient,
    and dropped in-frustum Gaussians stay visible for densification statistics."""
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    torch.manual_seed(0)
    renderer = _renderer(method)
    params, camera = _fixture_scene(device)
    with torch.no_grad():
        full = renderer.render(params, camera, sh_degree=0)["rgb"]
    if renderer.step != 0:
        raise AssertionError("no-grad renders must not advance the dropping schedule")
    renderer.step = 4999  # next training render is t = 5000 -> r = 0.1
    out = renderer.render(params, camera, sh_degree=0)
    drop = out["drop"]
    if drop is None or drop["step"] != 5000 or abs(drop["rate"] - 0.1) > 1e-12:
        raise AssertionError(f"training render did not apply r_5000 = 0.1: {drop}")
    mask = drop["mask"]
    dropped = mask == 0
    n = len(mask)
    if abs(float(dropped.float().mean()) - 0.1) > 5 * math.sqrt(0.09 / n):
        raise AssertionError("drop fraction differs from r")
    diff = float((out["rgb"] - full).abs().max())
    if diff < 1e-3:
        raise AssertionError("training render is identical to the all-Gaussian render")
    out["rgb"].mean().backward()
    g_sh = params["sh0"].grad.abs().sum(dim=(1, 2))
    g_op = params["opacities"].grad.abs()
    if float(g_sh[dropped].max()) != 0.0 or float(g_op[dropped].max()) != 0.0:
        raise AssertionError("dropped Gaussians received a gradient")
    if int((g_sh[~dropped] > 0).sum()) < 0.5 * int((~dropped).sum()):
        raise AssertionError("kept Gaussians received no gradient")
    vis = (out["info"]["visibility_radii"] > 0).all(dim=-1)[0]
    culled = (out["info"]["radii"] > 0).all(dim=-1)[0]
    if bool(culled[dropped].any()):
        raise AssertionError("expected gsplat to cull dropped (zero-opacity) Gaussians")
    if float(vis[dropped].float().mean()) < 0.9:
        raise AssertionError("dropped in-frustum Gaussians must stay visible for statistics")
    return {"rate": drop["rate"], "dropped": int(dropped.sum()), "n": n,
            "max_rgb_change": diff, "dropped_visible_fraction": float(vis[dropped].float().mean())}


def test_time_all_gaussians(method, device):
    """No-grad renders use every Gaussian with unscaled opacity (independent gsplat call)."""
    from gsplat import rasterization
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer = _renderer(method)
    renderer.step = 9999  # schedule at its maximum rate
    params, camera = _fixture_scene(device, seed=1)
    with torch.no_grad():
        got = renderer.render(params, camera, sh_degree=0)
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        ref, alpha, _ = rasterization(
            params["means"], params["quats"], params["scales"].exp(),
            params["opacities"].sigmoid(), colors, camera["viewmat"][None], camera["K"][None],
            camera["width"], camera["height"], sh_degree=0, packed=False,
        )
        ref = ref[0].clamp(0, 1)  # black background: no (1 - alpha) term
    if got["drop"] is not None:
        raise AssertionError("evaluation render applied dropout")
    err = float((got["rgb"] - ref).abs().max())
    if err > 1e-5:
        raise AssertionError(f"evaluation render differs from all-Gaussian render by {err}")
    if renderer.step != 9999:
        raise AssertionError("evaluation render advanced the schedule")
    return {"max_abs_error": err, "mean_alpha": float(alpha.mean())}


def official_densification(method, device):
    """Clone/split/prune on a fixture with the official thresholds; no size pruning; reset."""
    d = _density(method)
    cfg = d.config
    if (cfg["grad_threshold"], cfg["min_opacity"], cfg["densify_until_iter"]) != (
        0.0005, 0.005, 10000
    ):
        raise AssertionError(f"official thresholds changed: {cfg}")
    params, opts = sample_scene(method, device)
    n = len(params["means"])
    with torch.no_grad():
        params["scales"].fill_(math.log(0.005))  # small: <= 0.01 * extent
        params["scales"][1] = math.log(0.05)  # large -> split
        params["scales"][3] = math.log(10.0)  # 10x extent: official keeps it (no size prune)
        params["opacities"].fill_(0.0)
        params["opacities"][4] = torch.logit(torch.tensor(0.004))
        params["opacities"][5] = torch.logit(torch.tensor(0.006))
    d.initialize(params, opts, scene_scale=1.0)
    # Statistics through accumulate(): pixel gradients, NDC scaling (x W/2), visibility radii.
    w = h = 100
    grad = torch.zeros(1, n, 2, device=device)
    grad[0, 0, 0] = 6e-4 / (w / 2)  # -> 6e-4 NDC: clone
    grad[0, 1, 0] = 6e-4 / (w / 2)  # -> split
    grad[0, 2, 0] = 4e-4 / (w / 2)  # below tau (gsplat default 2e-4 would densify it)
    means2d = torch.zeros(1, n, 2, device=device, requires_grad=True)
    means2d.grad = grad
    radii = torch.ones(1, n, 2, dtype=torch.int32, device=device)
    radii[0, 6] = 0  # invisible: not counted
    info = {"means2d": means2d, "width": w, "height": h, "n_cameras": 1,
            "visibility_radii": radii}
    d.accumulate(info)
    if d.state["count"][6] != 0 or d.state["count"][7] != 1:
        raise AssertionError("visibility counting wrong")
    if abs(float(d.state["grad2d"][0]) - 6e-4) > 1e-9:
        raise AssertionError("NDC gradient scaling wrong")
    stats = d.densify_and_prune(params, opts)
    expected = {"cloned": 1, "split": 1, "pruned": 1, "gaussians": n + 1 + 1 - 1}
    if stats != expected:
        raise AssertionError(f"densify stats {stats} != {expected}")
    if not bool((params["scales"].exp().max(dim=1).values > 5.0).any()):
        raise AssertionError("the over-large Gaussian was pruned (official has no size pruning)")
    resets = [s for s in range(1, 12001) if d.resets_opacity(s)]
    if resets != [3000, 6000, 9000]:
        raise AssertionError(f"opacity resets at {resets}, expected [3000, 6000, 9000]")
    return {**stats, "resets": resets}


def _ssim_reference(a, b):
    """Independent SSIM: explicit zero-padded weighted window sums per pixel and channel."""
    size, sigma = 11, 1.5
    r = size // 2
    k = torch.tensor([math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-r, r + 1)],
                     dtype=torch.float64)
    k = k / k.sum()
    win = torch.outer(k, k)
    a, b = a.double(), b.double()
    H, W, C = a.shape
    pa = torch.nn.functional.pad(a.permute(2, 0, 1), (r, r, r, r))
    pb = torch.nn.functional.pad(b.permute(2, 0, 1), (r, r, r, r))
    total = 0.0
    for c in range(C):
        for y in range(H):
            for x in range(W):
                wa = pa[c, y : y + size, x : x + size]
                wb = pb[c, y : y + size, x : x + size]
                mu1, mu2 = (win * wa).sum(), (win * wb).sum()
                s1 = (win * wa * wa).sum() - mu1 * mu1
                s2 = (win * wb * wb).sum() - mu2 * mu2
                s12 = (win * wa * wb).sum() - mu1 * mu2
                c1, c2 = 0.01**2, 0.03**2
                total += float(((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
                               / ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)))
    return total / (C * H * W)


def loss_weights(method, device):
    from .losses import create_photometric

    spec = _spec(method, "photometric")
    if (spec.config["l1_weight"], spec.config["ssim_weight"]) != (1.0, 0.2):
        raise AssertionError(f"official loss weights are L1 1.0, SSIM 0.2: {spec.config}")
    g = torch.Generator().manual_seed(3)
    rgb = torch.rand(14, 12, 3, generator=g)
    target = (rgb + 0.2 * torch.rand(14, 12, 3, generator=g)).clamp(0, 1)
    got = float(create_photometric(spec)({}, {"rgb": rgb.to(device)}, target.to(device)))
    l1 = float((rgb - target).abs().mean())
    ssim = _ssim_reference(rgb, target)
    want = l1 + 0.2 * (1 - ssim)
    if abs(got - want) > 1e-5:
        raise AssertionError(f"loss {got} != L1 + 0.2 (1 - SSIM) = {want}")
    three_dgs = 0.8 * l1 + 0.2 * (1 - ssim)
    if abs(got - three_dgs) < 1e-4:
        raise AssertionError("loss is indistinguishable from 3DGS's 0.8 L1 weighting")
    return {"loss": got, "reference": want, "ssim": ssim}
