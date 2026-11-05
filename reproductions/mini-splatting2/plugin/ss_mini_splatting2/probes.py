"""Mini-Splatting2 mechanism probes. Each one fails if its mechanism is removed or wrong.

References are written independently of the code under test (brute-force per-pixel blending
in `core.blend_reference`, the general 3DGS-MCMC binomial formula, torch's dense Adam).
`stats_kernel` and the scene helpers follow the Mini-Splatting port's probes (plan #28).
"""

import math
from dataclasses import replace

import torch
from splatstudio.components import build
from splatstudio.registry import resolve
from torch import nn

from . import core, kernels
from .optim import SparseGaussianAdam


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _density(method, **overrides):
    spec = next(c for c in method.components if c.kind == "density")
    if overrides:
        spec = replace(spec, config={**spec.config, **overrides})
    return resolve(spec.factory)(spec)


def _scene(means, scales, opacities, quats=None, colors=None, device="cuda"):
    n = len(means)
    if quats is None:
        quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    if colors is None:
        colors = torch.full((n, 3), 0.5)
    values = {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
        "quats": torch.as_tensor(quats, dtype=torch.float32),
        "opacities": torch.logit(torch.as_tensor(opacities, dtype=torch.float32)),
        "sh0": ((torch.as_tensor(colors, dtype=torch.float32) - 0.5) / core.SH_C0)[:, None, :],
        "shN": torch.zeros(n, 15, 3),
    }
    params = nn.ParameterDict({k: nn.Parameter(v.to(device)) for k, v in values.items()})
    rates = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 0.05, "sh0": 2.5e-3,
             "shN": 1.25e-4}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    return params, opts


def _camera(width, height, f, c2w=None, device="cuda", image=None):
    if c2w is None:
        c2w = torch.eye(4)
    cam = {
        "viewmat": torch.linalg.inv(c2w).to(device),
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device),
        "width": width,
        "height": height,
    }
    if image is not None:
        cam["image"] = image.to(device)
    return cam


def _shift(x, y=0.0):
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([x, y, 0.0])
    return c2w


def _reference_stats(params, cam):
    meta = kernels.project(params, cam)
    return core.blend_reference(meta["means2d"][0].cpu().double(),
                                meta["conics"][0].cpu().double(),
                                meta["opacities"][0].cpu().double(), meta["depths"][0].cpu(),
                                int(cam["width"]), int(cam["height"])), meta


def _random_scene(seed, n, device):
    g = torch.Generator().manual_seed(seed)
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1,
                       torch.rand(n, 1, generator=g) * 3 + 2], 1)
    means[:, :2] *= means[:, 2:3] * 0.45
    scales = torch.rand(n, 3, generator=g) * 0.25 + 0.05
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.8 + 0.15
    return _scene(means, scales, opac, quats, device=device)


def stats_kernel(method, device):
    _require_cuda()
    from gsplat import rasterization

    w, h = 40, 32
    params, _ = _random_scene(3, 24, device)
    cam = _camera(w, h, 40.0, device=device)
    meta = kernels.project(params, cam)
    got = kernels.blend_stats(meta, weights=True)
    ref, _ = _reference_stats(params, cam)
    _, alpha, _ = rasterization(
        params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        params["sh0"], cam["viewmat"][None], cam["K"][None], w, h, sh_degree=0, packed=False)
    agree = (got["argmax"].cpu().long() == ref["argmax"]).float().mean().item()
    err_w = (got["accum_weights"].cpu().double() - ref["accum_weights"]).abs().max().item()
    err_t = (got["T"].cpu().double() - ref["T"]).abs().max().item()
    err_alpha = (1 - got["T"] - alpha[0, ..., 0]).abs().max().item()
    proj = (got["area_proj"].cpu().long() - ref["area_proj"]).abs().sum().item()
    amax = (got["area_max"].cpu().long() - ref["area_max"]).abs().sum().item()
    covered = int((ref["argmax"] >= 0).sum())
    if covered < w * h // 2 or int(got["area_max"].sum()) != int((got["argmax"] >= 0).sum()):
        raise AssertionError("Synthetic scene too sparse or area_max does not sum to coverage")
    if agree < 0.995 or err_w > 1e-3 or err_t > 1e-4 or err_alpha > 1e-5:
        raise AssertionError(f"Kernel disagrees with reference: argmax {agree}, w {err_w}, "
                             f"T {err_t}, alpha {err_alpha}")
    if proj > 2 or amax > 2:
        raise AssertionError(f"Pixel counts differ: area_proj {proj}, area_max {amax}")
    return {"pixels": w * h, "covered": covered, "argmax_agreement": agree,
            "max_weight_error": err_w, "max_T_error": err_t, "gsplat_alpha_error": err_alpha}


def _two_view_density(method, params, opts, w=48, h=40, step_cams="full", **overrides):
    density = _density(method, **overrides)
    density.initialize(params, opts, scene_scale=1.0)
    cams = [_camera(w, h, 40.0, device=params["means"].device),
            _camera(w, h, 40.0, _shift(0.15, 0.05), device=params["means"].device)]
    density.set_training_cameras(cams)
    if step_cams == "full":
        density.half_cameras = density.cameras  # probe the logic at the probe resolution
    return density, cams


def critical_clone(method, device):
    _require_cuda()
    params, opts = _random_scene(5, 40, device)
    # one Gaussian far behind everything: never an argmax -> not critical
    with torch.no_grad():
        params["means"].data[0] = torch.tensor([0.0, 0.0, 30.0], device=device)
    density, cams = _two_view_density(method, params, opts)
    n = len(params["means"])
    # Adam moments (to check copies start at zero while sources keep theirs)
    for opt in opts.values():
        opt.visibility = torch.ones(n, dtype=torch.bool, device=device)
    sum(p.square().sum() for p in params.values()).backward()
    for opt in opts.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    # independent reference of the pass
    imp_w = torch.zeros(n, dtype=torch.float64)
    area = torch.zeros(n, dtype=torch.float64)
    count_vis = torch.zeros(n)
    for cam in cams:
        ref, _ = _reference_stats(params, cam)
        imp_w += ref["accum_weights"]
        area += ref["area_max"].double()
        count_vis += core.cdf_keep_mask(ref["accum_weights"].float(), 0.99).float()
    prune = (count_vis <= 1) | ~core.cdf_keep_mask(imp_w.float(), 0.999)
    imp_c = imp_w.clone()
    imp_c[area == 0] = 0
    critical_ref = core.cdf_keep_mask(imp_c.float(), 0.99)[~prune]
    survivors = (~prune).nonzero().squeeze(1)
    a_old = params["opacities"].detach().sigmoid().cpu()[survivors][critical_ref]
    s_old = params["scales"].detach().exp().cpu()[survivors][critical_ref]
    density._culling_with_clone(params, opts, 1000)
    n_surv, n_crit = int((~prune).sum()), int(critical_ref.sum())
    if len(params["means"]) != n_surv + n_crit or n_crit == 0 or n_crit == n_surv:
        raise AssertionError(f"Population {len(params['means'])} != {n_surv} + {n_crit}")
    a_ref, ratio_ref = core.mcmc_reference(a_old.double(), 2)
    a_ref = a_ref.clamp(min=0.0051)
    a_got = params["opacities"].detach().sigmoid().cpu()
    s_got = params["scales"].detach().exp().cpu()
    src = torch.nonzero(critical_ref).squeeze(1)
    copies = torch.arange(n_surv, n_surv + n_crit)
    err_a = max((a_got[src] - a_ref).abs().max().item(), (a_got[copies] - a_ref).abs().max().item())
    err_s = (s_got[copies] - ratio_ref[:, None].float() * s_old).abs().max().item()
    if err_a > 1e-5 or err_s > 1e-5 * float(s_old.max()):
        raise AssertionError(f"Clone values differ from 3DGS-MCMC: opacity {err_a}, scale {err_s}")
    if 0 in (survivors[src].tolist()):
        raise AssertionError("A never-argmax Gaussian was identified as critical")
    m = opts["means"].state[params["means"]]["exp_avg"]
    if m[copies].abs().max() > 0 or m[src].abs().max() == 0:
        raise AssertionError("Copies must start with zero Adam moments, sources keep theirs")
    return {"survivors": n_surv, "critical": n_crit, "after": len(params["means"]),
            "opacity_error": err_a, "scale_error": err_s}


def visibility_culling(method, device):
    _require_cuda()
    renderer, _, _ = build(method)
    params, opts = _random_scene(7, 60, device)
    density, cams = _two_view_density(method, params, opts)
    n = len(params["means"])
    vp = density._visibility_pass(params, 1000)
    count_rad = torch.zeros(n)
    count_vis = torch.zeros(n)
    for k, cam in enumerate(cams):
        ref, meta = _reference_stats(params, cam)
        vis = core.cdf_keep_mask(ref["accum_weights"].float(), 0.99)
        if not torch.equal(vp["cull"][:, k].cpu(), ~vis):
            raise AssertionError(f"Visibility mask of view {k} differs from the 0.99 CDF")
        count_vis += vis.float()
        count_rad += (meta["radii"][0].cpu() > 0).all(dim=-1).float()
    factor_err = (vp["factor"].cpu() - count_vis / (count_rad + 0.1)).abs().max().item()
    if factor_err > 1e-6:
        raise AssertionError(f"factor_culling mismatch {factor_err}")
    density._set_culling(vp["cull"], vp["factor"])
    density.table.half_res = False
    culled = vp["cull"][:, 0]
    if not culled.any() or culled.all():
        raise AssertionError("Probe scene must cull some but not all Gaussians")
    out = renderer.render(params, cams[0], sh_degree=0)
    keep = (~culled).nonzero().squeeze(1)
    sub = {k: params[k][keep] for k in params}
    from .renderer import MSv2Renderer  # stock render of the subset / full set

    plain = MSv2Renderer.__mro__[1].render(renderer, sub, cams[0], 0)["rgb"]
    full = MSv2Renderer.__mro__[1].render(renderer, params, cams[0], 0)["rgb"]
    err_sub = (out["rgb"] - plain).abs().max().item()
    diff_full = (out["rgb"] - full).abs().max().item()
    if err_sub > 1e-6 or diff_full < 1e-4 or not torch.equal(out["subset_index"], keep):
        raise AssertionError(f"Culled Gaussians reach the rasterizer (sub {err_sub}, "
                             f"full {diff_full})")
    # Culled Gaussians get no gradient and are not updated by the sparse optimizer
    density.before_backward(params, opts, 1000, out)
    out["rgb"].square().sum().backward()
    g = params["means"].grad
    if g[culled].abs().max() > 0:
        raise AssertionError("Culled Gaussians received gradient")
    before = params["means"].detach().clone()
    for opt in opts.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    moved = (params["means"].detach() - before).abs().sum(dim=-1) > 0
    if moved[culled].any():
        raise AssertionError("Culled Gaussians were updated")
    # stats pass prunes count_vis <= 1 (a Gaussian visible in at most one view)
    return {"culled_view0": int(culled.sum()), "gaussians": n, "factor_error": factor_err,
            "subset_render_error": err_sub, "full_render_difference": diff_full}


def sparse_adam(method, device):
    _require_cuda()
    g = torch.Generator().manual_seed(11)
    p0 = torch.randn(10, 3, generator=g)
    vis = torch.tensor([True, False] * 5)
    results = {}
    for dev in ("cpu", device):
        p = nn.Parameter(p0.clone().to(dev))
        q = nn.Parameter(p0[vis].clone().to(dev))
        sparse = SparseGaussianAdam([p], lr=0.01, eps=1e-15)
        dense = torch.optim.Adam([q], lr=0.01, eps=1e-15)
        for t in range(5):
            grad = torch.randn(10, 3, generator=g).to(dev)
            p.grad = grad.clone()
            q.grad = grad[vis.to(dev)].clone()
            sparse.visibility = vis.to(dev)
            sparse.step()
            dense.step()
        err = (p.detach()[vis.to(dev)] - q.detach()).abs().max().item()
        frozen = torch.equal(p.detach()[~vis.to(dev)].cpu(), p0[~vis])
        m = sparse.state[p]["exp_avg"]
        if err > 1e-6 or not frozen or m[~vis.to(dev)].abs().max() > 0:
            raise AssertionError(f"{dev}: sparse Adam mismatch {err} or invisible rows moved")
        results[f"{dev}_max_error"] = err
    return results


def depth_reinit(method, device):
    _require_cuda()
    renderer, _, _ = build(method)
    gx, gy = torch.meshgrid(torch.linspace(-1.5, 1.5, 4), torch.linspace(-1.2, 1.2, 4),
                            indexing="ij")
    grid = torch.stack([gx.flatten(), gy.flatten(), torch.full((16,), 3.0)], 1)
    params, opts = _scene(grid, [[0.6, 0.6, 0.005]] * 16, [0.8] * 16, device=device)
    density = _density(method, num_depth_factor=10.0)
    density.initialize(params, opts, scene_scale=1.0)
    w, h = 40, 30
    colors = [torch.tensor([0.9, 0.2, 0.1]), torch.tensor([0.1, 0.3, 0.8])]
    cams = [_camera(w, h, 30.0, c, device=device, image=col.expand(h, w, 3).clone())
            for c, col in zip((torch.eye(4), _shift(0.3)), colors)]
    density.set_training_cameras([{k: c[k] for k in ("K", "viewmat", "width", "height")}
                                  for c in cams])
    for cam in cams:  # warm-up renders cache the half-resolution targets
        out = renderer.render(params, cam, sh_degree=0)
        if out["rgb"].shape[:2] != (h // 2, w // 2):
            raise AssertionError("Warm-up render is not at half resolution")
        density._cache_target(out, 1)
    n0 = len(params["means"])
    step = density.cfg["depth_reinit_iter"]
    density._set_culling(torch.zeros(n0, 2, dtype=torch.bool, device=device),
                         torch.ones(n0, device=device))
    density._depth_reinit(params, opts, step)
    n = len(params["means"])
    want = 2 * core.depth_samples_per_view(h // 2, w // 2, 2, n0 * 10.0)
    z_err = (params["means"][:, 2] - 3.0).abs().max().item()
    rgb = params["sh0"][:, 0] * core.SH_C0 + 0.5
    col_ok = min(((rgb - c.to(device)).abs().max(dim=1).values < 0.5 / 255 + 1e-4).sum().item()
                 for c in colors)
    opac = params["opacities"].sigmoid()
    fresh = all(len(o.state) == 0 for o in opts.values())
    if n != want or z_err > 1e-3 or col_ok < n // 3 or density.table.cull is not None:
        raise AssertionError(f"Bad reinit: n {n}/{want}, z {z_err}, colour matches {col_ok}")
    if (opac - 0.1).abs().max() > 1e-5 or params["shN"].abs().max() > 0 or not fresh:
        raise AssertionError("Reinit must set opacity 0.1, zero SH rest and reset Adam")
    if density.state["depth_views_fallback"] != 0:
        raise AssertionError("Cached targets were not used")
    return {"points": n, "expected": want, "max_plane_error": z_err,
            "min_points_per_colour": col_ok}


def half_res(method, device):
    _require_cuda()
    from .losses import create_photometric

    renderer, _, _ = build(method)
    params, opts = _scene([[0.0, 0.0, 3.0]], [[0.5, 0.5, 0.5]], [0.8], device=device)
    density = _density(method)
    density.initialize(params, opts, scene_scale=1.0)
    w, h = 41, 31
    img = torch.rand(h, w, 3, generator=torch.Generator().manual_seed(1))
    cam = _camera(w, h, 30.0, device=device, image=img)
    density.set_training_cameras([{k: cam[k] for k in ("K", "viewmat", "width", "height")}])
    other = _camera(w, h, 30.0, _shift(0.2), device=device, image=img)
    loss = create_photometric(next(c for c in method.components if c.kind == "loss"))
    shapes = {}
    for step in (1, density.cfg["warmup_until"], density.cfg["warmup_until"] + 1):
        density._schedule(params, opts, step)
        out = renderer.render(params, cam, sh_degree=0)
        shapes[step] = tuple(out["rgb"].shape[:2])
        value = loss(params, out, cam["image"])
        if not torch.isfinite(value):
            raise AssertionError("Loss not finite")
    half = (int(h / 2), int(w / 2))
    warm = density.cfg["warmup_until"]
    if shapes[1] != half or shapes[warm] != half or shapes[warm + 1] != (h, w):
        raise AssertionError(f"Resolution schedule wrong: {shapes}")
    density._schedule(params, opts, 1)
    if tuple(renderer.render(params, other, sh_degree=0)["rgb"].shape[:2]) != (h, w):
        raise AssertionError("Non-training views must render at full resolution")
    out = renderer.render(params, cam, sh_degree=0)
    tgt = out["loss_target"]
    if tuple(tgt.shape[:2]) != half or (tgt.mean(dim=(0, 1)).cpu() - img.mean(dim=(0, 1))).abs().max() > 0.05:
        raise AssertionError("Half-resolution loss target missing or wrong")
    return {"shapes": {str(k): list(v) for k, v in shapes.items()}}


def schedule(method, device):
    density = _density(method)
    cfg = density.cfg
    ev = {}
    for s in range(1, cfg["total_steps"] + 1):
        for e in core.events(s, cfg):
            ev.setdefault(e, []).append(s)
    clone_ref = [s for s in range(500, 3000, 250) if s != 2000]
    dens_ref = [s for s in range(600, 3000, 100) if s != 2000]
    if (ev.get("culling_with_clone") != clone_ref or ev.get("densify") != dens_ref
            or ev.get("depth_reinit") != [2000] or ev.get("simplify1") != [3000]
            or ev.get("simplify2") != [8000] or ev.get("culling_off") != [13000]):
        raise AssertionError(f"Event schedule differs from config/fast: "
                             f"{ {k: v[:3] + v[-1:] for k, v in ev.items()} }")
    order = core.events(1000, cfg)
    if order != ["densify", "culling_with_clone"]:
        raise AssertionError(f"Densify must precede the clone pass at 1000: {order}")
    params, opts = _scene(torch.rand(6, 3) + torch.tensor([0, 0, 3.0]), torch.full((6, 3), 0.1),
                          torch.full((6,), 0.5), device="cpu")
    density.initialize(params, opts, scene_scale=2.0)
    lr0 = density.lr_init
    checks = {"opacity_lr": opts["opacities"].param_groups[0]["lr"]}
    if abs(checks["opacity_lr"] - 0.025) > 1e-12:
        raise AssertionError("Opacity LR must be 0.025 (official 0.05/2)")
    for step in (1, 9000, 18000):
        density._schedule(params, opts, step)
        got = opts["means"].param_groups[0]["lr"]
        ref = lr0 * 0.01 ** (step / 30000)
        if abs(got - ref) > 1e-12 * max(1.0, ref):
            raise AssertionError(f"LR at {step}: {got} vs {ref} (30000-step horizon)")
        checks[f"lr_{step}"] = got
    p = params["shN"]
    for step, active in ((3000, 0), (3999, 0), (4000, 3), (5000, 8), (6000, 15)):
        p.data.normal_()
        density._schedule(params, opts, step)
        if active < 15 and p.data[:, active:].abs().max() > 0:
            raise AssertionError(f"Inactive SH bands not zeroed at step {step}")
        if active and p.data[:, :active].abs().max() == 0:
            raise AssertionError(f"Active SH bands wrongly zeroed at step {step}")
        checks[f"sh_coeffs_{step}"] = active
    density._schedule(params, opts, 3000)
    if not density.table.half_res:
        raise AssertionError("Step 3000 must still be half resolution")
    density._schedule(params, opts, 3001)
    if density.table.half_res:
        raise AssertionError("Step 3001 must be full resolution")
    checks["events"] = {k: len(v) for k, v in ev.items()}
    if not math.isclose(checks["lr_18000"], lr0 * 0.01 ** 0.6, rel_tol=1e-9):
        raise AssertionError("Final LR must be lr0 * 0.01^0.6")
    return checks
