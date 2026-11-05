"""Mechanism probes for Mip-Splatting. Each one fails if its mechanism is removed or altered."""

import math

import torch

from .filters import is_filter_step, is_reset_step


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


def _look_at(center, target):
    """OpenCV camera-to-world for a camera at `center` looking at `target` (y down)."""
    forward = torch.nn.functional.normalize(target - center, dim=0)
    up = torch.tensor([0.0, -1.0, 0.0])
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
    down = torch.linalg.cross(forward, right)
    c2w = torch.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, center
    return c2w


def _reference_filter(points, cams, coef, near, margin):
    """Independent per-point, per-camera loop of Eq. 8 with the official visibility rules."""
    f_max = max(c["fx"] for c in cams)
    out, seen = [], []
    for p in points.tolist():
        best = None
        for c in cams:
            rel = [p[i] - c["center"][i] for i in range(3)]
            xc = sum(c["R"][i][0] * rel[i] for i in range(3))  # R = c2w rotation columns
            yc = sum(c["R"][i][1] * rel[i] for i in range(3))
            zc = sum(c["R"][i][2] * rel[i] for i in range(3))
            if zc <= near:
                continue
            u = xc / zc * c["fx"] + c["cx"]
            v = yc / zc * c["fy"] + c["cy"]
            W, H = c["w"], c["h"]
            if -margin * W <= u <= (1 + margin) * W and -margin * H <= v <= (1 + margin) * H:
                best = zc if best is None else min(best, zc)
        out.append(best)
        seen.append(best is not None)
    fallback = max(d for d in out if d is not None)
    return torch.tensor(
        [math.sqrt(coef) * (d if d is not None else fallback) / f_max for d in out]
    ), torch.tensor(seen)


def frequency_bound(method, device):
    from splatstudio.state import sample_scene

    cfg = _component(method, "density").config
    coef, near, margin = cfg["filter_coef"], cfg["near"], cfg["frustum_margin"]
    g = torch.Generator().manual_seed(3)
    cams, frames = [], []
    for i, (dist, f) in enumerate(((4.0, 300.0), (7.0, 500.0), (2.5, 250.0), (9.0, 400.0))):
        angle = 2 * math.pi * i / 4
        center = torch.tensor([dist * math.cos(angle), 0.3 * i, dist * math.sin(angle)])
        c2w = _look_at(center, torch.zeros(3))
        w, h = 320, 240
        K = torch.tensor([[f, 0, w / 2], [0, f * 1.01, h / 2], [0, 0, 1.0]])
        frames.append({"viewmat": torch.linalg.inv(c2w), "K": K, "width": w, "height": h})
        cams.append(
            {"center": center.tolist(), "R": c2w[:3, :3].tolist(), "fx": f, "fy": f * 1.01,
             "cx": w / 2, "cy": h / 2, "w": w, "h": h}
        )
    points = torch.rand(400, 3, generator=g) * 6 - 3
    points[-1] = torch.tensor([0.0, 1000.0, 0.0])  # outside every frustum
    expected, seen = _reference_filter(points, cams, coef, near, margin)
    if seen[-1] or seen.all():
        raise AssertionError("probe fixture must contain an invisible point")

    params, opts = sample_scene(method, device, n=len(points))
    with torch.no_grad():
        params["means"].copy_(points.to(device))
    _, density = _build(method)
    density.initialize(params, opts, 1.0)
    density.set_training_cameras(frames)  # same order as splatstudio.runtime.train
    got = params["filter_3d"].detach().cpu()
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-7)

    # Analytic case: one camera at the origin looking down +z, point at depth 5, f = 100.
    from .filters import compute_filter_3d

    single = [{"viewmat": torch.eye(4), "K": torch.tensor([[100.0, 0, 50], [0, 100.0, 50],
                                                         [0, 0, 1]]), "width": 100,
               "height": 100}]
    analytic = compute_filter_3d(torch.tensor([[0.0, 0.0, 5.0]]), single, coef, near, margin)
    torch.testing.assert_close(analytic, torch.tensor([math.sqrt(0.2) * 5 / 100]))
    if not torch.all(got > 0):
        raise AssertionError("3D filter is inactive")
    return {
        "max_abs_error": float((got - expected).abs().max()),
        "invisible_points": int((~seen).sum()),
        "filter_median": float(got.median()),
        "analytic_filter": float(analytic[0]),
    }


def _rotation(quats):
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def convolution(method, device):
    _require_cuda(device)
    from gsplat import rasterization
    from splatstudio.state import sample_scene

    renderer, _ = _build(method)
    eps2d = _component(method, "renderer").config["kernel_size_2d"]
    params, _ = sample_scene(method, device, n=24)
    g = torch.Generator(device=device).manual_seed(5)
    with torch.no_grad():
        params["scales"].copy_(torch.empty(24, 3, device=device).uniform_(-4.5, -2.0, generator=g))
        params["quats"].copy_(torch.randn(24, 4, device=device, generator=g))
        params["opacities"].copy_(torch.randn(24, device=device, generator=g))
        params["filter_3d"].copy_(torch.empty(24, device=device).uniform_(0.01, 0.06, generator=g))
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device),
        "width": 64,
        "height": 64,
    }
    out = renderer.render(params, camera, sh_degree=0)["rgb"]

    # Independent: full covariance, explicit convolution, determinant-based compensation.
    R = _rotation(params["quats"].detach())
    S2 = torch.diag_embed(params["scales"].detach().exp().square())
    sigma = R @ S2 @ R.transpose(-1, -2)
    f2 = params["filter_3d"].detach().square()[:, None, None] * torch.eye(3, device=device)
    comp = torch.sqrt(torch.linalg.det(sigma.double()) / torch.linalg.det((sigma + f2).double()))
    colors = torch.cat((params["sh0"], params["shN"]), dim=1).detach()
    ref, alpha, _ = rasterization(
        means=params["means"].detach(), quats=params["quats"].detach(),
        scales=params["scales"].detach().exp(),
        opacities=params["opacities"].detach().sigmoid() * comp.float(), colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=64, height=64,
        sh_degree=0, packed=False, eps2d=eps2d, rasterize_mode="antialiased",
        covars=sigma + f2,
    )
    ref = ref[0] + (1 - alpha[0])
    torch.testing.assert_close(out.detach(), ref, atol=2e-4, rtol=1e-3)
    with torch.no_grad():
        saved = params["filter_3d"].clone()
        params["filter_3d"].zero_()
        unfiltered = renderer.render(params, camera, sh_degree=0)["rgb"]
        params["filter_3d"].copy_(saved)
    effect = float((unfiltered - out).abs().max())
    if effect < 1e-2:
        raise AssertionError("3D filter has no visible effect on the probe scene")
    return {"max_abs_error": float((out.detach() - ref).abs().max()),
            "filter_effect_max_abs": effect, "min_compensation": float(comp.min())}


def mip_filter_2d(method, device):
    _require_cuda(device)
    from splatstudio.state import sample_scene

    renderer, _ = _build(method)
    k = 0.1  # paper / official kernel_size for Mip-NeRF 360
    if _component(method, "renderer").config["kernel_size_2d"] != k:
        raise AssertionError("2D Mip filter variance differs from the paper's 0.1")
    params, _ = sample_scene(method, device, n=2)
    f, z, sx, sy, opacity = 20.0, 4.0, 0.1, 0.15, 0.5
    with torch.no_grad():
        params["means"].copy_(torch.tensor([[0.0, 0.0, z], [0.0, 0.0, -5.0]], device=device))
        params["scales"].copy_(torch.tensor([[sx, sy, 0.05]] * 2, device=device).log())
        params["quats"].copy_(torch.tensor([[1.0, 0, 0, 0]] * 2, device=device))
        params["opacities"].fill_(math.log(opacity / (1 - opacity)))
        params["filter_3d"].zero_()
    camera = {
        "viewmat": torch.eye(4, device=device),
        # Principal point on a pixel centre (gsplat samples pixel centres at i + 0.5).
        "K": torch.tensor([[f, 0, 8.5], [0, f, 8.5], [0, 0, 1]], device=device),
        "width": 17,
        "height": 17,
    }
    alpha = renderer.render(params, camera, sh_degree=0)["alpha"][8, 8, 0].item()
    a, c = (f * sx / z) ** 2, (f * sy / z) ** 2  # analytic projected covariance diag
    expected = opacity * math.sqrt(a * c / ((a + k) * (c + k)))
    dilation_only = opacity  # 3DGS dilation: no compensation
    if abs(alpha - expected) > 1e-4:
        raise AssertionError(f"peak alpha {alpha:.6f} != Mip-filter prediction {expected:.6f}")
    return {"peak_alpha": alpha, "expected": expected, "uncompensated": dilation_only}


def filter_aware_reset(method, device):
    from splatstudio.state import sample_scene

    _, density = _build(method)
    value = _component(method, "density").config["opacity_reset_value"]
    params, opts = sample_scene(method, device, n=12)
    # Allocate real Adam moments first, then set the fixture values.
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    with torch.no_grad():
        params["scales"].fill_(math.log(0.02))
        params["filter_3d"].copy_(torch.linspace(0.0, 0.04, 12, device=device))
        params["opacities"].copy_(torch.linspace(-6.0, 3.0, 12, device=device))
    old = params["opacities"].detach().sigmoid().clone()
    density.reset_opacity(params, opts)
    new = params["opacities"].detach().sigmoid()
    # Independent coefficient: sqrt(det Sigma / det(Sigma + f^2 I)) for isotropic s = 0.02.
    s2, f2 = 0.02**2, params["filter_3d"].detach().square()
    coef = (s2 / (s2 + f2)) ** 1.5
    torch.testing.assert_close(new * coef, torch.minimum(old * coef, torch.tensor(value)),
                               rtol=1e-4, atol=1e-7)
    plain = torch.minimum(old, torch.tensor(value))
    if torch.allclose(new, plain, rtol=1e-4, atol=1e-7):
        raise AssertionError("reset is indistinguishable from gsplat's unfiltered reset")
    p = params["opacities"]
    if any(v.abs().max() > 0 for k, v in opts["opacities"].state[p].items() if k != "step"):
        raise AssertionError("opacity Adam moments were not zeroed")
    from splatstudio.state import check_state

    check_state(method, params, opts)
    # Failure regime from the first benchmark: Gaussians much smaller than filter_3d (the
    # float32 det ratio underflows to 0) and saturated opacities must reset to finite values.
    with torch.no_grad():
        params["scales"].copy_(torch.log(torch.tensor(
            [[1e-8] * 3, [1e-4] * 3, [1e-12, 1e-2, 1e-2]] * 4, device=device)))
        params["filter_3d"].fill_(0.004)
        params["opacities"].copy_(torch.tensor([0.0, 20.0, 1.0] * 4, device=device))
    density.reset_opacity(params, opts)
    if not torch.isfinite(params["opacities"]).all():
        raise AssertionError("filter-aware reset produced nonfinite opacities for tiny Gaussians")
    check_state(method, params, opts)
    return {"max_raw_opacity_after": float(new.max()), "plain_reset_max": float(plain.max())}


def schedule(method, device):
    cfg = _component(method, "density").config
    total = cfg["total_steps"]
    got = {s for s in range(1, total + 1)
           if is_filter_step(s, cfg["filter_interval"], 500, cfg["densify_until"], total)}
    official = set(range(600, 15000, 100)) | set(range(15100, total - 100, 100))
    if got != official:
        raise AssertionError(f"filter steps differ: {sorted(got ^ official)[:10]}")
    resets = {s for s in range(1, total + 1)
              if is_reset_step(s, cfg["opacity_reset_every"], cfg["densify_until"])}
    expected = {3000, 6000, 9000, 12000} if cfg["opacity_reset_every"] else set()
    if resets != expected:
        raise AssertionError(f"reset steps {sorted(resets)} != {sorted(expected)}")
    return {"filter_updates": len(got), "first": min(got), "last": max(got),
            "resets": sorted(resets)}


def _probe_camera(device):
    return {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
        "width": 16,
        "height": 16,
    }


def renderer_gradient(method, device):
    """splatstudio.probes:renderer_gradient with the runtime's set_training_cameras hook."""
    _require_cuda(device)
    from splatstudio.state import check_state, sample_scene

    renderer, density = _build(method)
    params, opts = sample_scene(method, device)
    camera = _probe_camera(device)
    density.initialize(params, opts)
    density.set_training_cameras([camera])
    outputs = renderer.render(params, camera, sh_degree=0)
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(16, device=device)[None, :, None] % 2).expand(16, 16, 3).float()
    (outputs["rgb"] - target).square().mean().backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    return {"rgb_shape": list(outputs["rgb"].shape), "screen_gradient_norm": xy.grad.norm().item()}


def single_reset(method, device):
    """Exactly one filter-aware opacity reset at a reset step, none at other steps."""
    _require_cuda(device)
    from splatstudio.state import check_state, sample_scene

    from .filters import filter_coefficient, reset_logits

    cfg = _component(method, "density").config
    every, value = cfg["opacity_reset_every"], cfg["opacity_reset_value"]
    renderer, density = _build(method)
    params, opts = sample_scene(method, device)
    with torch.no_grad():
        params["opacities"].fill_(3.0)  # filtered opacity well above the reset value
        params["scales"].fill_(math.log(0.05))
    camera = _probe_camera(device)
    density.initialize(params, opts)
    density.set_training_cameras([camera])
    # Isolate the reset: no growth/pruning/filter refresh in this probe instance.
    density.strategy.refine_start_iter = 10**9
    coef = filter_coefficient(params["scales"].detach(), params["filter_3d"].detach())
    if not (coef < 0.99).any():
        raise AssertionError("probe fixture needs a non-trivial 3D filter")

    def drive(step):
        before = params["opacities"].detach().clone()
        outputs = renderer.render(params, camera, sh_degree=0)
        density.before_backward(params, opts, step, outputs)
        outputs["rgb"].mean().backward()
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        density.after_backward(params, opts, step, outputs)
        return before, params["opacities"].detach()

    before, after = drive(every - 100)
    if not torch.equal(before, after) or density.resets:
        raise AssertionError("opacity changed at a non-reset step")
    before, after = drive(every)
    expected = reset_logits(params["scales"].detach(), before, params["filter_3d"].detach(), value)
    plain = torch.logit(torch.minimum(before.sigmoid(), torch.tensor(value, device=device)))
    if torch.allclose(after, before):
        raise AssertionError("no opacity reset at the reset step")
    if torch.allclose(after, plain, atol=1e-5):
        raise AssertionError("reset is the unfiltered (or a doubled) 3DGS reset")
    torch.testing.assert_close(after, expected, rtol=1e-5, atol=1e-5)
    if density.resets != 1:
        raise AssertionError(f"{density.resets} resets at one reset step")
    check_state(method, params, opts)
    return {"resets": density.resets, "min_coef": float(coef.min()),
            "filtered_after_max": float((after.sigmoid() * coef).max())}


def camera_hook(method, device):
    from splatstudio.state import sample_scene

    _, density = _build(method)
    params, opts = sample_scene(method, device)
    density.initialize(params, opts)
    try:
        density.before_backward(params, opts, 1, {"info": {}})
    except RuntimeError as exc:
        if "set_training_cameras" not in str(exc):
            raise
    else:
        raise AssertionError("training without training cameras did not fail")
    if torch.count_nonzero(params["filter_3d"]):
        raise AssertionError("filter computed without cameras")
    density.set_training_cameras([_probe_camera(device)])
    if density.filter_updates != 1 or not torch.all(params["filter_3d"] > 0):
        raise AssertionError("set_training_cameras did not compute the 3D filter")
    return {"filter_mean": float(params["filter_3d"].mean())}
