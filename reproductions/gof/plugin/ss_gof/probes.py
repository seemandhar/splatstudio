"""Mechanism probes for the GOF port. Each raises on failure and returns measurements."""

import math

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
NEAR, FAR = 0.2, 100.0  # kernel NEAR_PLANE / FAR_PLANE (auxiliary.h)


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _camera(device, w=32, h=32, f=30.0, viewmat=None):
    return {"viewmat": torch.eye(4, device=device) if viewmat is None else viewmat,
            "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device),
            "width": w, "height": h}


def _quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def _scene(device, n=5, seed=3):
    """Anisotropic, rotated Gaussians in front of an identity camera (filter_3d = 0)."""
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([1.2, 1.2, 1.5]) - torch.tensor(
        [0.6, 0.6, -2.5])
    scales = torch.log(torch.rand(n, 3, generator=g) * 0.35 + 0.05)
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.logit(torch.rand(n, generator=g) * 0.5 + 0.3)
    sh0 = (torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8
    shN = torch.zeros(n, 15, 3)
    params = {"means": means, "scales": scales, "quats": quats, "opacities": opac,
              "sh0": sh0, "shN": shN, "filter_3d": torch.zeros(n)}
    return {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}


def reference_render(params, camera, bg, radii=None):
    """Independent ray-Gaussian volume renderer (paper Eq. 2-7, kernel conventions) in float64.

    Returns rgb, camera-frame normal (unnormalised blend), median depth, alpha, distortion
    (normalised forward value) and the weight-detached unnormalised distortion whose gradient
    the official backward implements. ``radii`` (from the kernel) reproduces only the tile
    culling (16x16 tiles touched by the projected 3-sigma radius), not the image formation.
    """
    from .raster import activated

    scales, opac, rots, sh = activated(params)
    dd = torch.float64
    W, H = int(camera["width"]), int(camera["height"])
    K, V = camera["K"].to(dd), camera["viewmat"].to(dd)
    Rc, tc = V[:3, :3], V[:3, 3]
    mean_c = params["means"].to(dd) @ Rc.T + tc
    R = Rc @ _quat_to_rot(rots.to(dd))            # gaussian -> camera rotation
    s = scales.to(dd)
    dev = mean_c.device
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dd, device=dev),
                            torch.arange(W, dtype=dd, device=dev), indexing="ij")
    ray = torch.stack([(xs + 0.5 - W / 2) / K[0, 0], (ys + 0.5 - H / 2) / K[1, 1],
                       torch.ones_like(xs)], -1).reshape(-1, 3)
    color = (sh[:, 0].to(dd) * C0 + 0.5).clamp_min(0)
    order = torch.argsort(mean_c[:, 2])
    P = ray.shape[0]
    T = torch.ones(P, dtype=dd, device=ray.device)
    alive = torch.ones(P, dtype=torch.bool, device=ray.device)
    rgb = torch.zeros(P, 3, dtype=dd, device=ray.device)
    nrm = torch.zeros(P, 3, dtype=dd, device=ray.device)
    depth = torch.zeros(P, dtype=dd, device=ray.device)
    acc = torch.zeros(P, dtype=dd, device=ray.device)
    d1 = torch.zeros(P, dtype=dd, device=ray.device)
    d2 = torch.zeros(P, dtype=dd, device=ray.device)
    dist = torch.zeros(P, dtype=dd, device=ray.device)
    dist_det = torch.zeros(P, dtype=dd, device=ray.device)
    d1x = torch.zeros(P, dtype=dd, device=ray.device)
    d2x = torch.zeros(P, dtype=dd, device=ray.device)
    tile_x = (xs.reshape(-1) // 16).long()
    tile_y = (ys.reshape(-1) // 16).long()
    gx, gy = (W + 15) // 16, (H + 15) // 16
    for k in order.tolist():
        if mean_c[k, 2] <= 0.2:
            continue
        in_tile = torch.ones(P, dtype=torch.bool, device=dev)
        if radii is not None:
            r = float(radii[k])
            if r <= 0:
                continue
            px = (K[0, 0] * mean_c[k, 0] / mean_c[k, 2] + K[0, 2] - 0.5).item()
            py = (K[1, 1] * mean_c[k, 1] / mean_c[k, 2] + K[1, 2] - 0.5).item()
            x0, x1 = min(gx, max(0, int((px - r) / 16))), min(gx, max(0, int((px + r + 15) / 16)))
            y0, y1 = min(gy, max(0, int((py - r) / 16))), min(gy, max(0, int((py + r + 15) / 16)))
            in_tile = (tile_x >= x0) & (tile_x < x1) & (tile_y >= y0) & (tile_y < y1)
        o_g = (R[k].T @ (-mean_c[k])) / s[k]          # S^-1 R^T (o - p), o = 0 in camera
        r_g = (ray @ R[k]) / s[k]                     # S^-1 R^T r
        A = (r_g * r_g).sum(-1)
        B = (r_g * o_g).sum(-1)
        Cc = (o_g * o_g).sum()
        t = -B / A                                    # Eq. 6
        power = torch.clamp(-0.5 * (Cc - B * B / A), max=0.0)
        alpha = torch.clamp(opac[k].to(dd) * torch.exp(power), max=0.99)
        ok = alive & in_tile & (t > NEAR) & (alpha >= 1 / 255)
        test_T = T * (1 - alpha)
        stop = ok & (test_T < 1e-4)
        alive = alive & ~stop
        ok = ok & ~stop
        w = torch.where(ok, alpha * T, torch.zeros_like(T))
        m = (FAR * t - FAR * NEAR) / ((FAR - NEAR) * t)
        m = torch.where(ok, m, torch.zeros_like(m))
        err = m * m * (1 - T) + d2 - 2 * m * d1
        dist = dist + err * w
        wd, Td = w.detach(), T.detach()
        dist_det = dist_det + (m * m * (1 - Td) + d2x - 2 * m * d1x) * wd
        d1x = d1x + m * wd
        d2x = d2x + m * m * wd
        d1 = d1 + m * w
        d2 = d2 + m * m * w
        sig_inv = (R[k] / (s[k] ** 2)) @ R[k].T       # camera-frame Sigma^-1
        n = -F.normalize(ray @ sig_inv.T, dim=-1)     # n = -R^T S^-1 r_g (paper Sec. 3.3)
        rgb = rgb + w[:, None] * color[k]
        nrm = nrm + w[:, None] * n
        depth = torch.where(ok & (T > 0.5), t, depth)
        acc = acc + w
        T = torch.where(ok, test_T, T)
    rgb = rgb + T[:, None] * bg.to(dd)
    dist = dist / ((1 - T) ** 2 + 1e-7)
    shape = (H, W)
    return {"rgb": rgb.reshape(*shape, 3), "normal_cam": nrm.reshape(*shape, 3),
            "depth": depth.reshape(shape), "alpha": acc.reshape(shape),
            "distortion": dist.reshape(shape), "distortion_detached": dist_det.reshape(shape)}


def _kernel_maps(params, camera, bg):
    from .raster import _Rasterize, activated, raster_settings

    scales, opac, rots, sh = activated(params)
    means2D = torch.zeros_like(params["means"], requires_grad=True)
    s = raster_settings(camera, bg, 0)
    out, radii = _Rasterize.apply(params["means"], means2D, sh.contiguous(),
                                  opac[:, None].contiguous(), scales.contiguous(),
                                  rots.contiguous(), s)
    return out, means2D, radii


def ray_intersection(method, device):
    """C1: official kernel == independent ray-tracing reference (values and gradients), and
    differs from projection-based splatting (gsplat) on the same anisotropic Gaussians."""
    _cuda(device)
    import gsplat

    from .raster import activated

    cam = _camera(device)
    bg = torch.tensor([0.1, 0.2, 0.3], device=device)
    params = _scene(device)
    out, _, radii = _kernel_maps(params, cam, bg)
    ref = reference_render(params, cam, bg, radii)
    err = {
        "rgb": (out[0:3].permute(1, 2, 0) - ref["rgb"]).abs().max().item(),
        "normal": (out[3:6].permute(1, 2, 0) - ref["normal_cam"]).abs().max().item(),
        "alpha": (out[7] - ref["alpha"]).abs().max().item(),
        "distortion": (out[8] - ref["distortion"]).abs().max().item(),
    }
    covered = ref["alpha"] > 0.6
    err["median_depth"] = (out[6] - ref["depth"])[covered].abs().max().item()
    for k, v in err.items():
        if not v < 2e-3:
            raise AssertionError(f"GOF kernel {k} differs from the ray-tracing reference: {v}")
    # Gradients: random linear functional of rgb, normal and depth.
    g = torch.Generator(device=device).manual_seed(0)
    wr = torch.randn(32, 32, 3, generator=g, device=device)
    wn = torch.randn(32, 32, 3, generator=g, device=device)
    wd = torch.randn(32, 32, generator=g, device=device) * 0.1
    names = ("means", "scales", "quats", "opacities", "sh0")

    def grads(loss):
        return torch.autograd.grad(loss, [params[n] for n in names], allow_unused=True)

    out, _, _ = _kernel_maps(params, cam, bg)
    gk = grads((out[0:3].permute(1, 2, 0) * wr).sum() + (out[3:6].permute(1, 2, 0) * wn).sum()
               + (out[6] * wd).sum())
    ref = reference_render(params, cam, bg, radii)
    gr = grads((ref["rgb"] * wr).sum() + (ref["normal_cam"] * wn).sum()
               + (ref["depth"] * wd).sum())
    rel = {}
    for n, a, b in zip(names, gk, gr):
        a = torch.zeros_like(params[n]) if a is None else a
        rel[n] = ((a.double() - b).norm() / b.norm().clamp_min(1e-12)).item()
        if not rel[n] < 2e-2:
            raise AssertionError(f"GOF backward for {n} disagrees with autograd reference: {rel}")
    # Projection-based splatting is a different image formation model.
    scales, opac, rots, sh = activated(params)
    K = cam["K"][None]
    colors = (sh[:, 0] * C0 + 0.5).clamp_min(0)
    img, alp, _ = gsplat.rasterization(params["means"], rots, scales, opac, colors,
                                       cam["viewmat"][None], K, 32, 32, eps2d=0.0)
    img = img[0] + (1 - alp[0]) * bg
    proj_diff = (img - out[0:3].permute(1, 2, 0)).abs().max().item()
    if not proj_diff > 20 * err["rgb"] + 1e-3:
        raise AssertionError("Ray-traced image indistinguishable from projected splatting")
    return {"max_abs_error": err, "grad_rel_error": rel, "gsplat_projection_diff": proj_diff}


def distortion_formula(method, device):
    """C3: distortion map equals the NDC 2DGS form; its gradient is the weight-detached one
    (zero opacity gradient, nonzero means gradient)."""
    _cuda(device)
    cam = _camera(device)
    bg = torch.zeros(3, device=device)
    params = _scene(device, seed=5)
    out, _, radii = _kernel_maps(params, cam, bg)
    loss = out[8].sum()
    g_op, g_mu = torch.autograd.grad(loss, [params["opacities"], params["means"]])
    ref = reference_render(params, cam, bg, radii)
    gr_mu = torch.autograd.grad(ref["distortion_detached"].sum(), params["means"])[0]
    gr_full = torch.autograd.grad(reference_render(params, cam, bg, radii)["distortion"].sum(),
                                  params["opacities"])[0]
    rel = ((g_mu.double() - gr_mu).norm() / gr_mu.norm()).item()
    if g_op.abs().max().item() != 0.0 or gr_full.abs().max().item() < 1e-6:
        raise AssertionError("Distortion weights are not detached in the kernel backward")
    if not (g_mu.abs().max().item() > 0 and rel < 2e-2):
        raise AssertionError(f"Distortion depth gradient disagrees with the reference: {rel}")
    return {"opacity_grad_max": g_op.abs().max().item(), "means_grad_rel_error": rel,
            "undetached_opacity_grad_norm": gr_full.norm().item()}


def opacity_field(method, device):
    """C2: integrate kernel gives O(t) = o G(min(t, t*)) along a ray, min over views."""
    _cuda(device)
    from .raster import integrate
    from .raster import opacity_field as field

    n = 1
    params = {"means": torch.tensor([[0.0, 0.0, 3.0]]), "scales": torch.log(
        torch.tensor([[0.3, 0.2, 0.25]])), "quats": torch.tensor([[1.0, 0, 0, 0]]),
        "opacities": torch.logit(torch.tensor([0.8])), "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 15, 3), "filter_3d": torch.zeros(n)}
    params = {k: v.to(device) for k, v in params.items()}
    cam = _camera(device)
    ts = torch.tensor([2.5, 2.7, 2.85, 2.95, 3.0, 3.2, 3.6], device=device)
    pts = torch.stack([ts * (0.5 / 30.0), ts * (0.5 / 30.0), ts], -1)  # pixel (16, 16) centre
    got = integrate(params, pts, cam)
    # independent: ray through pixel (16,16) centre, x = o + t r with r = (1/60, 1/60, 1)
    r = torch.tensor([0.5 / 30, 0.5 / 30, 1.0], device=device)
    o_g = (-params["means"][0]) / params["scales"][0].exp()
    r_g = r / params["scales"][0].exp()
    t_star = -(o_g @ r_g) / (r_g @ r_g)
    tt = torch.minimum(ts, t_star)
    xg = o_g[None] + tt[:, None] * r_g[None]
    expect = 0.8 * torch.exp(-0.5 * (xg * xg).sum(-1))
    err = (got - expect).abs().max().item()
    if not err < 1e-4:
        raise AssertionError(f"Opacity field differs from o*G(min(t, t*)): {err}")
    if not (got[-1] - got[-3]).abs() < 1e-5 or not got[0] < got[3]:
        raise AssertionError("Opacity must increase to t* and stay constant afterwards")
    # second view from the side sees the points at other depths; min over both views
    # camera at (3, 0, 3) looking along -x: c2w columns x=(0,0,1), y=(0,1,0), z=(-1,0,0)
    rot_c2w = torch.tensor([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]], device=device)
    side = torch.eye(4, device=device)
    side[:3, :3] = rot_c2w.T
    side[:3, 3] = -rot_c2w.T @ torch.tensor([3.0, 0.0, 3.0], device=device)
    cam2 = _camera(device, viewmat=side)
    both = field(params, pts, [cam, cam2])
    single2 = integrate(params, pts, cam2)
    if not torch.allclose(both, torch.minimum(got, single2)):
        raise AssertionError("Opacity field is not the minimum over views")
    return {"max_abs_error": err, "t_star": t_star.item(), "opacity": got.tolist(),
            "min_over_views": both.tolist()}


def normal_consistency(method, device):
    """C4: depth_to_normal recovers an analytic plane normal; L_d / L_n weights and the
    15000-step schedule of the official train.py."""
    from splatstudio.components import build

    from .raster import depth_to_normal

    dev = "cpu"
    cam = _camera(dev, 40, 30, 35.0)
    n_true = F.normalize(torch.tensor([0.3, -0.2, 1.0]), dim=0)  # plane n.X = 2 (camera frame)
    W, H = 40, 30
    ys, xs = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    rays = torch.stack([(xs - W / 2) / 35.0, (ys - H / 2) / 35.0, torch.ones_like(xs)], -1)
    depth = 2.0 / (rays @ n_true)
    nrm = depth_to_normal(cam, depth)
    err = (nrm[1:-1, 1:-1] + n_true).norm(dim=-1).max().item()  # faces the camera: -n
    if not err < 1e-4 or nrm[0].abs().max() != 0:
        raise AssertionError(f"depth_to_normal wrong: {err}")
    _, _, losses = build(method)
    reg = dict(losses)["gof-reg"]
    outs = {"distortion": torch.full((H, W, 1), 0.01, requires_grad=True),
            "normal": -n_true.expand(H, W, 3).clone().requires_grad_(True),
            "surface_normal": nrm}
    values = {}
    for step in (14999, 15000):
        values[step] = reg({}, {**outs, "step": step}, None).item()
    normal_err = (1 - (-n_true.expand(H, W, 3) * nrm).sum(-1)).mean().item()
    expect = 100 * 0.01 + 0.05 * normal_err
    if values[14999] != 0.0 or not math.isclose(values[15000], expect, rel_tol=1e-5):
        raise AssertionError(f"Regulariser schedule/weights wrong: {values} vs {expect}")
    return {"plane_normal_error": err, "loss": values}


def densify_metric(method, device):
    """C5: kernel abs channel >= |signed sum|; cancelling gradients densify only via M."""
    _cuda(device)
    from dataclasses import replace

    from .density import Density

    cam = _camera(device)
    params = _scene(device)
    out, means2D, radii = _kernel_maps(params, cam, torch.zeros(3, device=device))
    target = (torch.arange(32, device=device)[None, :, None] % 2).float().expand(32, 32, 3)
    (out[0:3].permute(1, 2, 0) - target).square().mean().backward()
    g = means2D.grad
    vis = radii > 0
    l1 = g[vis, 0].abs() + g[vis, 1].abs()
    if not (g[vis, 2] + 1e-9 >= l1 * (1 - 1e-4)).all() or not (g[vis, 2] > 0).all():
        raise AssertionError("Abs gradient channel is not the sum of per-pixel norms")

    def run(abs_gradient):
        spec = next(c for c in method.components if c.kind == "density")
        spec = replace(spec, config={**spec.config, "abs_gradient": abs_gradient})

        d = Density(spec)
        n = 4
        p = {"means": torch.zeros(n, 3), "scales": torch.full((n, 3), math.log(0.001)),
             "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
             "opacities": torch.full((n,), 2.0), "sh0": torch.zeros(n, 1, 3),
             "shN": torch.zeros(n, 15, 3), "filter_3d": torch.zeros(n)}
        p = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()})
        opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
        d.initialize(p, opts, 1.0)
        d.stats["grad_accum"][:] = torch.tensor([3e-4, 0.0, 1e-5, 1e-5])  # #1 cancels
        d.stats["grad_accum_abs"][:] = torch.tensor([5e-4, 9e-4, 1e-5, 2e-5])
        d.stats["denom"][:] = 1
        d.densify_and_prune(p, opts, False)
        return d.state["last"]

    with_m, without = run(True), run(False)
    if with_m["clone"] != 2 or without["clone"] != 1:
        raise AssertionError(f"Abs-gradient densification not active: {with_m} {without}")
    return {"abs_vs_l1_min_ratio": (g[vis, 2] / l1.clamp_min(1e-12)).min().item(),
            "clones_with_M": with_m["clone"], "clones_without_M": without["clone"]}


def density_recipe(method, device):
    """Official densify/prune/reset/3D-filter/LR recipe on a CPU fixture."""
    from .density import Density, expon_lr, is_filter_step
    from .filters import compute_filter_3d

    spec = next(c for c in method.components if c.kind == "density")
    d = Density(spec)
    torch.manual_seed(0)
    n = 6
    s = torch.tensor([0.005, 0.005, 0.05, 0.05, 0.005, 0.2])  # extent 1 -> dense 0.01
    p = {"means": torch.randn(n, 3) + torch.tensor([0, 0, 4.0]),
         "scales": torch.log(s)[:, None].repeat(1, 3), "quats": torch.tensor(
             [[1.0, 0, 0, 0]]).repeat(n, 1),
         "opacities": torch.logit(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.04, 0.5])),
         "sh0": torch.randn(n, 1, 3), "shN": torch.zeros(n, 15, 3), "filter_3d": torch.zeros(n)}
    p = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()})
    opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
    sum(v.square().sum() for v in p.values()).backward()
    for o in opts.values():
        o.step()
    cams = [_camera("cpu", 64, 48, 50.0)]
    d.initialize(p, opts, 1.0)
    d.set_training_cameras(cams)
    # independent 3D filter: sqrt(0.2) * z / f for points in the 1.15x frustum
    z = p["means"][:, 2].detach()
    u = p["means"][:, 0].detach() / z * 50 + 32
    v = p["means"][:, 1].detach() / z * 50 + 24
    inside = (u >= -0.15 * 64) & (u <= 1.15 * 64) & (v >= -0.15 * 48) & (v <= 1.15 * 48)
    exp_f = torch.where(inside, z, z[inside].max()) / 50 * math.sqrt(0.2)
    if not torch.allclose(p["filter_3d"].detach(), exp_f, rtol=1e-5):
        raise AssertionError("3D filter differs from the independent formula")
    torch.testing.assert_close(compute_filter_3d(p["means"], cams), exp_f)
    parent_means = p["means"].detach().clone()
    d.stats["grad_accum"][:] = torch.tensor([3e-4, 0, 3e-4, 0, 3e-4, 3e-4])
    d.stats["grad_accum_abs"][:] = torch.tensor([3e-4, 0, 3e-4, 0, 3e-4, 3e-4])
    d.stats["denom"][:] = 1
    d.densify_and_prune(p, opts, True)
    last = d.state["last"]
    # clone: #0, #4 (small); split: #2, #5 (large); prune: #4 & its clone (0.04 < 0.05),
    # #5's children (0.2/1.6 = 0.125 > 0.1 extent: world prune)
    if (last["clone"], last["split"]) != (2, 2):
        raise AssertionError(f"Clone/split selection wrong: {last}")
    if last["after"] != n + 2 + 2 - 4 or last["prune"] != 4:
        raise AssertionError(f"Prune wrong: {last}")
    sc = p["scales"].exp()[:, 0]
    if not torch.isclose(sc, torch.tensor(0.05 / 1.6), rtol=1e-2).sum() == 2:
        raise AssertionError("Split children must have scale s / 1.6")
    small = torch.isclose(sc, torch.tensor(0.005), rtol=1e-2)
    dist = torch.cdist(p["means"][small].detach(), parent_means[:1])[:, 0]
    if int((dist == 0).sum()) != 1 or int(((dist > 0) & (dist < 0.05)).sum()) != 1:
        raise AssertionError("Clone did not resample its position around the parent")
    # filter-aware reset: filtered opacity <= 0.01
    from .filters import filtered_scales_opacities

    d.update_filter(p)
    d.reset_opacity(p, opts)
    _, fo = filtered_scales_opacities(p["scales"], p["opacities"], p["filter_3d"])
    if not (fo <= 0.01 + 1e-6).all() or (opts["opacities"].state[p["opacities"]]["exp_avg"]
                                          != 0).any():
        raise AssertionError("Filter-aware opacity reset wrong")
    sched = {k: is_filter_step(k, 100, 500, 15000, 30000)
             for k in (500, 600, 15000, 15100, 29800, 29900)}
    if sched != {500: False, 600: True, 15000: False, 15100: True, 29800: True, 29900: False}:
        raise AssertionError(f"3D filter schedule wrong: {sched}")
    lrs = [d.xyz_lr(k) for k in (0, 15000, 30000)]
    expect = [1.6e-4, math.sqrt(1.6e-4 * 1.6e-6), 1.6e-6]
    if not all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(lrs, expect)):
        raise AssertionError(f"xyz LR schedule wrong {lrs}")
    if not math.isclose(expon_lr(10, 1.0, 0.01, 0, 0.01, 100), 0.01 ** 0.1, rel_tol=1e-9):
        raise AssertionError("expon_lr wrong")
    return {"last": last, "filter_updates": d.counts["filter_updates"], "xyz_lr": lrs}
