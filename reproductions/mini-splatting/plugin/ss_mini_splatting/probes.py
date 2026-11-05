"""Mini-Splatting mechanism probes. Each one fails if its mechanism is removed or wrong.

Reference computations here are written independently of the CUDA kernels (brute-force
per-pixel blending, density maximum along a ray from the world-space precision matrix,
Rodrigues rotations instead of quaternion formulas).
"""

import math
from dataclasses import replace

import torch
from splatstudio.components import build
from splatstudio.registry import resolve
from torch import nn

from . import core, kernels


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


def _rodrigues(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    k = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return torch.linalg.matrix_exp(k * angle)


def stats_kernel(method, device):
    _require_cuda()
    from gsplat import rasterization

    g = torch.Generator().manual_seed(3)
    n, w, h = 24, 40, 32
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1, torch.rand(n, 1, generator=g) * 3
                       + 2], 1)
    means[:, :2] *= means[:, 2:3] * 0.45
    scales = torch.rand(n, 3, generator=g) * 0.25 + 0.05
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.8 + 0.15
    params, _ = _scene(means, scales, opac, quats, device=device)
    cam = _camera(w, h, 40.0, device=device)
    meta = kernels.project(params, cam)
    got = kernels.blend_stats(meta, weights=True)
    ref = core.blend_reference(meta["means2d"][0].cpu().double(), meta["conics"][0].cpu().double(),
                               meta["opacities"][0].cpu().double(), meta["depths"][0].cpu(), w, h)
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
        raise AssertionError("Synthetic scene too sparse or S_i does not sum to covered pixels")
    if agree < 0.995 or err_w > 1e-3 or err_t > 1e-4 or err_alpha > 1e-5:
        raise AssertionError(f"Kernel disagrees with reference: argmax {agree}, w {err_w}, "
                             f"T {err_t}, alpha {err_alpha}")
    if proj > 2 or amax > 2:
        raise AssertionError(f"Pixel counts differ: area_proj {proj}, area_max {amax}")
    return {"pixels": w * h, "covered": covered, "argmax_agreement": agree,
            "max_weight_error": err_w, "max_T_error": err_t, "gsplat_alpha_error": err_alpha,
            "area_proj_abs_diff": proj, "area_max_abs_diff": amax}


def depth_midpoint(method, device):
    _require_cuda()
    w, h, f = 48, 40, 50.0
    rot = _rodrigues([0.3, 1.0, 0.5], 0.9)
    axis = torch.tensor([0.3, 1.0, 0.5]) / torch.tensor([0.3, 1.0, 0.5]).norm()
    quat = torch.cat([torch.tensor([math.cos(0.45)]), axis * math.sin(0.45)])[None]
    mean = torch.tensor([[0.1, -0.05, 3.0]])
    scale = torch.tensor([[0.9, 0.3, 0.06]])
    c2w = torch.eye(4)
    c2w[:3, :3] = _rodrigues([0, 1, 0], 0.15).float()
    c2w[:3, 3] = torch.tensor([-0.2, 0.1, 0.0])
    params, _ = _scene(mean, scale, [0.95], quat, device=device)
    cam = _camera(w, h, f, c2w, device=device)
    out = kernels.depth_points(kernels.project(params, cam), params, cam)
    hit = (out["argmax"] == 0).cpu()
    if int(hit.sum()) < 50:
        raise AssertionError("Gaussian covers too few pixels")
    worst, offset = 0.0, 0.0
    origin = c2w[:3, 3].double()
    for i, j in hit.nonzero().tolist():
        d_cam = torch.tensor([(j + 0.5 - w / 2) / f, (i + 0.5 - h / 2) / f, 1.0], dtype=torch.float64)
        d = c2w[:3, :3].double() @ d_cam
        d = d / d.norm()
        p_ref, _ = core.midpoint_reference(origin, d, mean[0].double(), rot, scale[0].double())
        p = out["points"][i, j].cpu().double()
        worst = max(worst, (p - p_ref).norm().item())
        offset = max(offset, (p_ref - mean[0].double()).norm().item())
    if worst > 2e-3:
        raise AssertionError(f"Mid-point differs from the density maximum by {worst}")
    if offset < 0.05:
        raise AssertionError("Probe geometry cannot distinguish mid-points from centres")
    return {"pixels": int(hit.sum()), "max_point_error": worst, "max_offset_from_centre": offset}


def _blur_run(method, device, **overrides):
    renderer, _, _ = build(method)
    density = _density(method, **overrides)
    # Big opaque Gaussian in front of two small hidden ones; zero photometric gradient.
    params, opts = _scene([[0, 0, 3.0], [0.1, 0, 5.0], [-0.1, 0.05, 5.0]],
                          [[1.5, 1.5, 1.5], [0.02, 0.02, 0.02], [0.02, 0.02, 0.02]],
                          [0.9, 0.5, 0.5], device=device)
    density.initialize(params, opts, scene_scale=100.0)
    cam = _camera(64, 48, 40.0, device=device)
    step = density.cfg["densify_from"] + density.cfg["densify_every"]
    out = renderer.render(params, cam, sh_degree=0)
    out["step"] = step
    density.before_backward(params, opts, step, out)
    (out["rgb"] * 0).sum().backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    density.after_backward(params, opts, step, out)
    return params


def blur_split(method, device):
    _require_cuda()
    on = _blur_run(method, device)
    off = _blur_run(method, device, theta_blur=1.0)
    n_on, n_off = len(on["means"]), len(off["means"])
    big_on = on["scales"].exp().max().item()
    if n_off != 3:
        raise AssertionError(f"Without blur split nothing may change (got {n_off})")
    if n_on != 4 or big_on > 1.5 / 1.6 + 1e-4:
        raise AssertionError(f"Blur split did not split the dominant Gaussian ({n_on}, {big_on})")
    return {"gaussians_with_blur_split": n_on, "gaussians_without": n_off,
            "largest_scale_after_split": big_on}


def depth_reinit(method, device):
    _require_cuda()
    renderer, _, _ = build(method)
    density = _density(method, num_depth=600)
    params, opts = _scene([[0.0, 0.0, 3.0]], [[2.0, 2.0, 0.005]], [0.8], device=device)
    density.initialize(params, opts, scene_scale=1.0)
    w, h = 40, 30
    colors = [torch.tensor([0.9, 0.2, 0.1]), torch.tensor([0.1, 0.3, 0.8])]
    c2w1 = torch.eye(4)
    c2w1[:3, 3] = torch.tensor([0.3, 0.0, 0.0])
    cams = [_camera(w, h, 30.0, c, device=device, image=col.expand(h, w, 3).clone())
            for c, col in zip((torch.eye(4), c2w1), colors)]
    density.set_training_cameras([{k: c[k] for k in ("K", "viewmat", "width", "height")}
                                  for c in cams])
    for cam in cams:
        density._cache_target(renderer.render(params, cam, sh_degree=0), 1)
    density._depth_reinit(params, opts, density.cfg["reinit_every"])
    n = len(params["means"])
    want = 2 * core.depth_samples_per_view(h, w, 2, 600)
    z_err = (params["means"][:, 2] - 3.0).abs().max().item()
    spread = params["means"][:, 0].std().item()
    rgb = params["sh0"][:, 0] * core.SH_C0 + 0.5
    # target images are cached as 8-bit (lossless for the 8-bit dataset images)
    col_err = min(((rgb - c.to(device)).abs().max(dim=1).values < 0.5 / 255 + 1e-5).sum().item()
                  for c in colors)
    opac = params["opacities"].sigmoid()
    fresh = all(len(o.state) == 0 for o in opts.values())
    # Mini-Splatting-D draws with replacement and keeps unique pixels: fewer than `want`
    count_ok = (0.7 * want <= n <= want) if density.cfg["depth_with_replacement"] else n == want
    if not count_ok or z_err > 1e-3 or spread < 0.2 or col_err < n // 3:
        raise AssertionError(f"Bad reinit: n {n}/{want}, z {z_err}, spread {spread}, "
                             f"colour matches {col_err}")
    if (opac - 0.1).abs().max() > 1e-5 or params["shN"].abs().max() > 0 or not fresh:
        raise AssertionError("Reinit must set opacity 0.1, zero SH rest and reset Adam")
    return {"points": n, "expected": want, "max_plane_error": z_err, "x_spread": spread,
            "min_points_per_colour": col_err}


def simplification(method, device):
    _require_cuda()
    density = _density(method)
    gx, gy = torch.meshgrid(torch.tensor([-0.6, -0.2, 0.2, 0.6]), torch.tensor([-0.4, 0.0, 0.4]),
                            indexing="ij")
    # distinct depths: no ties in the front-to-back order
    vis = torch.stack([gx.flatten(), gy.flatten(), 3.0 + 0.01 * torch.arange(12.0)], 1)
    hidden = torch.tensor([[0.0, 0.0, 8.0], [0.2, 0.1, 8.0], [-0.2, -0.1, 8.0]])
    wall = torch.tensor([[0.0, 0.0, 6.0]])
    means = torch.cat([vis, wall, hidden])
    scales = torch.cat([torch.full((12, 3), 0.25), torch.full((1, 3), 0.6),
                        torch.full((3, 3), 0.05)])
    opac = torch.cat([torch.full((12,), 0.7), torch.tensor([0.99]), torch.full((3,), 0.9)])
    params, opts = _scene(means, scales, opac, device=device)
    density.initialize(params, opts, scene_scale=1.0)
    cams = [_camera(48, 40, 40.0, device=device)]
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.1, 0.05, 0.0])
    cams.append(_camera(48, 40, 40.0, c2w, device=device))
    density.set_training_cameras(cams)
    imp = density._importance(params)
    # independent reference of the importance (outdoor I^2 or indoor I^1) + intersection
    ref = torch.zeros(len(means), dtype=torch.float64)
    seen = torch.zeros(len(means), dtype=torch.float64)
    for cam in cams:
        meta = kernels.project(params, cam)
        r = core.blend_reference(meta["means2d"][0].cpu().double(),
                                 meta["conics"][0].cpu().double(),
                                 meta["opacities"][0].cpu().double(), meta["depths"][0].cpu(),
                                 48, 40)
        seen += r["area_max"].double()
        if density.cfg["imp_metric"] == "outdoor":
            m = r["area_max"] > 0
            ref[m] += r["accum_weights"][m] / r["area_proj"][m].double()
        else:
            ref += r["accum_weights"]
    ref[seen == 0] = 0
    if (imp[13:] != 0).any() or not (imp[:12] > 0).all():
        raise AssertionError("Intersection preserving failed: hidden Gaussians kept importance")
    err = (imp.cpu().double() - ref).abs().max().item() / ref.abs().max().item()
    if err > 1e-3:
        raise AssertionError(f"Importance differs from the reference ({err})")
    nnz = int((imp != 0).sum())
    density._simplify_sample(params, opts, density.cfg["restart_step"])
    kept = params["means"].detach().cpu()
    expected = int(len(means) * density.cfg["sampling_factor"] * (nnz / len(means)))
    if len(kept) != expected or (kept[:, 2] > 7.0).any():
        raise AssertionError(f"Sampling kept {len(kept)} (expected {expected}) or hidden ones")
    if (params["opacities"].sigmoid() - 0.1).abs().max() > 1e-5:
        raise AssertionError("Sampled Gaussians must be reinitialized from their centres")
    n_before = len(params["means"])
    imp2 = density._importance(params)
    density._simplify_prune(params, opts, density.cfg["simp_iteration2"])
    expect2 = int(core.cdf_keep_mask(imp2, density.cfg["prune_cdf"]).sum())
    if len(params["means"]) != expect2 or expect2 >= n_before + 1:
        raise AssertionError("CDF prune mismatch")
    return {"metric": density.cfg["imp_metric"], "importance_rel_error": err,
            "intersected": nnz, "kept_after_sampling": len(kept), "kept_after_prune": expect2}


def schedule(method, device):
    density = _density(method)
    params, opts = _scene(torch.rand(6, 3) + torch.tensor([0, 0, 3.0]), torch.full((6, 3), 0.1),
                          torch.full((6,), 0.5), device="cpu")
    density.initialize(params, opts, scene_scale=2.0)
    lr0 = density.lr_init
    rs = density.cfg["restart_step"]
    checks = {}
    for step, ref_iter in ((rs - 1, rs - 1), (rs + 1, rs + 1 - rs + 5000), (30000, 20000)):
        density._schedule(params, opts, step)
        got = opts["means"].param_groups[0]["lr"]
        ref = lr0 * 0.01 ** (ref_iter / 30000)
        if abs(got - ref) > 1e-12 * max(1.0, ref) + 1e-15:
            raise AssertionError(f"LR at {step}: {got} vs {ref}")
        checks[f"lr_{step}"] = got
    p = params["shN"]
    p.grad = torch.ones_like(p)
    opts["shN"].step()
    for step, active in ((rs + 500, 0), (rs + 1000, 3), (rs + 2000, 8), (rs + 3000, 15)):
        p.data.normal_()
        st = opts["shN"].state[p]
        st["exp_avg"].normal_()
        density._schedule(params, opts, step)
        if active < 15 and (p.data[:, active:].abs().max() > 0
                            or st["exp_avg"][:, active:].abs().max() > 0):
            raise AssertionError(f"Inactive SH bands not zeroed at step {step}")
        if active and p.data[:, :active].abs().max() == 0:
            raise AssertionError(f"Active SH bands wrongly zeroed at step {step}")
        checks[f"sh_coeffs_{step}"] = active
    return checks
