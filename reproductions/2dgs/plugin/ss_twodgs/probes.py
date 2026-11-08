"""Mechanism probes for the 2DGS port. Each raises on failure and returns measurements."""

import math

import torch

C0 = 0.28209479177387814


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _camera(device, w=64, h=48, f=50.0):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device),
            "width": w, "height": h}


def _quat_y(deg):
    a = math.radians(deg) / 2
    return [math.cos(a), 0.0, math.sin(a), 0.0]


def _rotmat(q):
    from gsplat.utils import normalized_quat_to_rotmat

    return normalized_quat_to_rotmat(torch.nn.functional.normalize(q, dim=-1))


def _params(device, means, scales2, quats, opac, grad=False):
    n = len(means)
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    s = torch.cat([t(scales2).log(), torch.full((n, 1), math.log(1e-3), device=device)], 1)
    p = {
        "means": t(means), "scales": s, "quats": t(quats),
        "opacities": torch.logit(t(opac)),
        "sh0": torch.full((n, 1, 3), (0.7 - 0.5) / C0, device=device),
        "shN": torch.zeros(n, 15, 3, device=device),
    }
    if grad:
        for v in p.values():
            v.requires_grad_(True)
    return p


def _render(params, camera, depth_ratio=0.0):
    from .raster import render

    bg = torch.zeros(3, device=params["means"].device)
    return render(params, camera, 0, bg, depth_ratio)


def independent_splat(camera, mean, scale2, quat, opac, low_pass=True):
    """Independent single-splat evaluation (ray-plane intersection) -> alpha, z, normal."""
    dev = mean.device
    W, H = int(camera["width"]), int(camera["height"])
    f = camera["K"][0, 0]
    R = _rotmat(quat[None])[0]
    tu, tv, n = R[:, 0], R[:, 1], R[:, 2]
    if torch.dot(n, mean) > 0:  # face the camera (camera at origin)
        n = -n
    ys, xs = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    # Kernel pixel grid: pix = f X/Z + (W-1)/2 (ndc2pix with integer pixel indices).
    d = torch.stack([(xs - (W - 1) / 2) / f, (ys - (H - 1) / 2) / f, torch.ones_like(xs)], -1)
    t = torch.dot(n, mean) / (d @ n)
    X = t[..., None] * d
    local = X - mean
    u, v = local @ tu / scale2[0], local @ tv / scale2[1]
    rho = u * u + v * v
    if low_pass:
        # Official kernel (forward.cu compute_aabb): the filter centre is the centre of the
        # screen bounding box of the projected 3-sigma ellipse, not the projected mean.
        th = torch.linspace(0, 2 * math.pi, 20001, device=dev)
        ring = mean + 3 * (scale2[0] * torch.cos(th)[:, None] * tu
                           + scale2[1] * torch.sin(th)[:, None] * tv)
        px = f * ring[:, 0] / ring[:, 2] + (W - 1) / 2
        py = f * ring[:, 1] / ring[:, 2] + (H - 1) / 2
        cx, cy = (px.min() + px.max()) / 2, (py.min() + py.max()) / 2
        rho = torch.minimum(rho, 2.0 * ((xs - cx) ** 2 + (ys - cy) ** 2))
    alpha = torch.clamp(opac * torch.exp(-0.5 * rho), max=0.99)
    alpha = torch.where((alpha < 1 / 255) | (t < 0.2), torch.zeros_like(alpha), alpha)
    return alpha, t, n


@torch.no_grad()
def ray_splat(method, device):
    _cuda(device)
    cam = _camera(device)
    out = {}
    # A: large oblique splat -> intersection depth and normal.
    mean, s2, q, o = [0.1, -0.05, 3.0], [0.6, 0.4], _quat_y(40), 0.8
    r = _render(_params(device, [mean], [s2], [q], [o]), cam)
    a_ref, z_ref, n_ref = independent_splat(cam, torch.tensor(mean, device=device),
                                            torch.tensor(s2, device=device),
                                            torch.tensor(q, device=device), o)
    alpha = r["alpha"][..., 0]
    cover = a_ref > 0.02
    err_a = (alpha - a_ref)[cover].abs().max().item()
    err_z = (r["expected_depth"][..., 0] - z_ref)[cover].abs().max().item()
    n_map = r["normal"][cover] / alpha[cover][:, None]
    err_n = (n_map - n_ref).abs().max().item()
    spread = (z_ref[cover] - mean[2]).abs().max().item()
    out.update(oblique_alpha_err=err_a, oblique_depth_err=err_z, normal_err=err_n,
               depth_spread_vs_centre=spread, covered_pixels=int(cover.sum()))
    if err_a > 2e-3 or err_z > 2e-3 or err_n > 2e-3:
        raise AssertionError(f"Kernel disagrees with ray-splat intersection: {out}")
    if spread < 0.1:
        raise AssertionError("Probe set-up cannot distinguish centre depth from intersection depth")
    # B: almost edge-on thin splat -> the low-pass filter dominates.
    mean, s2, q, o = [0.0, 0.0, 3.0], [0.05, 0.01], _quat_y(80), 0.9
    r = _render(_params(device, [mean], [s2], [q], [o]), cam)
    args = (torch.tensor(mean, device=device), torch.tensor(s2, device=device),
            torch.tensor(q, device=device), o)
    a_lp, _, _ = independent_splat(cam, *args)
    a_3d, _, _ = independent_splat(cam, *args, low_pass=False)
    alpha = r["alpha"][..., 0]
    err_lp = (alpha - a_lp).abs().max().item()
    diff_3d = (alpha - a_3d).abs().max().item()
    out.update(lowpass_alpha_err=err_lp, without_lowpass_diff=diff_3d)
    if err_lp > 2e-3:
        raise AssertionError(f"Low-pass filter differs from min(rho3d, 2|d|^2): {out}")
    if diff_3d < 0.1:
        raise AssertionError("Low-pass filter has no effect on an edge-on splat")
    return out


def distortion_kernel(method, device):
    _cuda(device)
    cam = _camera(device)
    q = [1.0, 0.0, 0.0, 0.0]
    p = _params(device, [[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], [[3.0, 3.0], [3.0, 3.0]], [q, q],
                [0.5, 0.5], grad=True)
    r = _render(p, cam)
    a = []
    for i, z in enumerate((2.0, 4.0)):
        ai, _, _ = independent_splat(cam, p["means"][i].detach(), torch.tensor([3.0, 3.0],
                                     device=device), torch.tensor(q, device=device), 0.5)
        a.append(ai)
    w1, w2 = a[0], a[1] * (1 - a[0])
    m = [100.0 / (100.0 - 0.2) * (1 - 0.2 / z) for z in (2.0, 4.0)]
    ref = w1 * w2 * (m[0] - m[1]) ** 2
    dist = r["distortion"][..., 0]
    err = (dist - ref).abs().max().item()
    l1_alt = (2 * w1 * w2 * 2.0).max().item()  # gsplat's L1 raw-depth form would be this large
    dist.sum().backward()
    gz = p["means"].grad[:, 2].tolist()
    with torch.no_grad():
        single = _render(_params(device, [[0.0, 0.0, 2.0]], [[3.0, 3.0]], [q], [0.5]), cam)
    single_max = single["distortion"].abs().max().item()
    out = {"distortion_err": err, "distortion_max": dist.max().item(), "l1_raw_alt": l1_alt,
           "grad_z_near": gz[0], "grad_z_far": gz[1], "single_splat_max": single_max}
    if err > 1e-6 or dist.max().item() < 1e-4:
        raise AssertionError(f"Distortion is not w1 w2 (m1 - m2)^2 in NDC: {out}")
    if single_max > 1e-7:
        raise AssertionError(f"A single splat has nonzero distortion: {out}")
    if not (gz[0] < 0 < gz[1]):
        raise AssertionError(f"Distortion gradient does not pull the splats together: {out}")
    return out


def projected_center_grad(method, device):
    _cuda(device)
    cam = _camera(device)
    W, H, f = cam["width"], cam["height"], float(cam["K"][0, 0])
    # Fronto-parallel splat: the low-pass branch (whose screen-centre gradient also reaches
    # the means) is inactive for a splat many pixels wide, so the identity is exact.
    q = [1.0, 0.0, 0.0, 0.0]
    p = _params(device, [[0.1, -0.05, 3.0], [0.0, 0.0, -3.0]], [[0.4, 0.3], [0.4, 0.3]],
                [q, q], [0.8, 0.8], grad=True)
    r = _render(p, cam)
    g = torch.Generator(device=device).manual_seed(0)
    weight = torch.rand(r["rgb"].shape, generator=g, device=device)
    (r["rgb"] * weight).sum().backward()
    g2 = r["info"]["means2d"].grad
    gm = p["means"].grad
    Z = 3.0
    ref = torch.stack([gm[0, 0] * Z * W / (2 * f), gm[0, 1] * Z * H / (2 * f)])
    err = ((g2[0, :2] - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    radii = r["info"]["radii"].tolist()
    out = {"rel_err": err, "grad2d": g2[0, :2].tolist(), "ref": ref.tolist(), "radii": radii,
           "z_component": g2[0, 2].item(), "culled_grad": g2[1].abs().max().item()}
    if not (radii[0] > 0 and radii[1] == 0):
        raise AssertionError(f"Unexpected visibility: {out}")
    if ref.abs().max().item() < 1e-6 or err > 1e-3:
        raise AssertionError(f"means2D gradient is not the projected 3D-centre gradient: {out}")
    if out["z_component"] != 0.0 or out["culled_grad"] != 0.0:
        raise AssertionError(f"Unexpected means2D gradient entries: {out}")
    return out


def normal_loss(method, device="cpu"):
    from .losses import create_normal
    from .raster import depth_to_normal

    dev = "cpu"
    W, H, f = 40, 30, 30.0
    cam = {"viewmat": torch.eye(4), "K": torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]),
           "width": W, "height": H}
    n0 = torch.nn.functional.normalize(torch.tensor([0.3, -0.2, -1.0]), dim=0)
    p0 = torch.tensor([0.0, 0.0, 3.0])
    ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    d = torch.stack([(xs - W / 2) / f, (ys - H / 2) / f, torch.ones_like(xs)], -1)
    depth = torch.dot(n0, p0) / (d @ n0)
    nm = depth_to_normal(cam, depth)
    inner = nm[1:-1, 1:-1]
    err_n = (inner - n0).abs().max().item()
    border = nm[0].abs().max().item()
    if err_n > 1e-4 or border != 0.0:
        raise AssertionError(f"depth_to_normal does not recover the plane normal ({err_n})")
    loss = create_normal(_spec(method, "2dgs-normal"))
    g = torch.Generator().manual_seed(1)
    normal = torch.randn(H, W, 3, generator=g, device=dev, requires_grad=True)
    alpha = torch.rand(H, W, 1, generator=g)
    surf = nm * alpha
    ref = 0.05 * (1 - (normal * surf).sum(-1)).mean()
    on = loss(None, {"step": 7001, "normal": normal, "surface_normal": surf}, None)
    off = loss(None, {"step": 7000, "normal": normal, "surface_normal": surf}, None)
    on.backward()
    out = {"plane_normal_err": err_n, "loss_err": abs(on.item() - ref.item()),
           "off_value": off.item(), "grad_norm": normal.grad.norm().item()}
    if out["loss_err"] > 1e-6 or off.item() != 0.0 or out["grad_norm"] == 0.0:
        raise AssertionError(f"Normal-consistency loss / schedule wrong: {out}")
    return out


def distortion_loss(method, device="cpu"):
    from .losses import create_distortion

    spec = _spec(method, "2dgs-distortion")
    lam = float(spec.config["lambda_dist"])
    expected = 100.0 if method.name == "2dgs-paperdist" else 0.0
    if lam != expected:
        raise AssertionError(f"lambda_dist {lam} != {expected} for {method.name}")
    loss = create_distortion(spec)
    dist = torch.rand(8, 8, 1, generator=torch.Generator().manual_seed(2), requires_grad=True)
    on = loss(None, {"step": 3001, "distortion": dist}, None)
    off = loss(None, {"step": 3000, "distortion": dist}, None)
    out = {"lambda_dist": lam, "on": on.item(), "off": off.item(), "mean": dist.mean().item()}
    if abs(on.item() - lam * dist.mean().item()) > 1e-5 or off.item() != 0.0 or not on.requires_grad:
        raise AssertionError(f"Distortion loss / schedule wrong: {out}")
    return out


def density_recipe(method, device="cpu"):
    from splatstudio.state import check_state, initialize

    from .density import create_density, split_surfels

    torch.manual_seed(3)
    pts = torch.tensor([[0.0, 0, 3], [0.5, 0, 3], [0, 0.5, 3], [0.5, 0.5, 3], [1, 0, 3],
                        [0, 1, 3]])
    params, opts = initialize(method, pts, torch.full((6, 3), 0.5))
    dens = create_density(_spec(method, "2dgs-density"))
    dens.initialize(params, opts, scene_scale=1.0)
    q = params["quats"].detach()
    if not (bool((q >= 0).all()) and bool((q < 1).all())):
        raise AssertionError("Quaternions are not initialised with torch.rand (uniform [0,1))")
    with torch.no_grad():
        params["scales"][:, :2] = math.log(0.005)
        params["scales"][1, :2] = torch.tensor([math.log(0.05), math.log(0.02)])  # split
        params["scales"][3, :2] = math.log(0.5)  # world-size prune
        params["scales"][:, 2] = -7.0
        params["opacities"][:] = torch.logit(torch.tensor(0.5))
        params["opacities"][2] = torch.logit(torch.tensor(0.01))  # cull
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    dens.accum = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    dens.denom = torch.ones(6)
    dens.densify_and_prune(params, opts, world_prune=False)
    last = dict(dens.state["last"])
    check_state(method, params, opts)
    if (last["clone"], last["split"], last["prune"], last["after"]) != (1, 1, 1, 7):
        raise AssertionError(f"Clone/split/prune counts wrong: {last}")
    dens.accum, dens.denom = torch.zeros(7), torch.ones(7)
    dens.densify_and_prune(params, opts, world_prune=True)
    world = dict(dens.state["last"])
    if world["prune"] != 1 or world["after"] != 6:
        raise AssertionError(f"World-size prune wrong: {world}")
    # In-plane split geometry on a fresh state.
    params2, opts2 = initialize(method, pts, torch.full((6, 3), 0.5))
    with torch.no_grad():
        params2["scales"][:, :2] = torch.log(torch.tensor([0.3, 0.1]))
        params2["scales"][:, 2] = -5.0
    parent = {k: v.detach().clone() for k, v in params2.items()}
    split_surfels(params2, opts2, torch.tensor([True, False, False, False, False, False]))
    check_state(method, params2, opts2)
    R = _rotmat(parent["quats"][:1])[0]
    off = params2["means"][-2:].detach() - parent["means"][0]
    plane_err = (off @ R[:, 2]).abs().max().item()
    scale_err = (params2["scales"][-2:, :2].detach()
                 - (parent["scales"][0, :2] - math.log(1.6))).abs().max().item()
    third_ok = bool((params2["scales"][-2:, 2] == -5.0).all())
    if len(params2["means"]) != 7 or plane_err > 1e-5 or scale_err > 1e-6 or not third_ok:
        raise AssertionError(f"Split is not in-plane / s/1.6: {plane_err}, {scale_err}")
    # Opacity reset at step 3000 through after_backward.
    n = len(params["means"])
    info = {"means2d": torch.zeros(n, 3, requires_grad=True), "radii": torch.ones(n)}
    info["means2d"].grad = torch.zeros(n, 3)
    dens.after_backward(params, opts, 3000, {"info": info})
    max_op = params["opacities"].sigmoid().max().item()
    if max_op > 0.01 + 1e-6 or dens.counts["resets"] != 1:
        raise AssertionError(f"Opacity reset wrong: {max_op}")
    for name, p in params.items():
        if not p.requires_grad or not p.is_leaf:
            raise AssertionError(f"{name} lost requires_grad")
    check_state(method, params, opts)
    return {"first": last, "world": world, "split_plane_err": plane_err,
            "split_scale_err": scale_err, "max_opacity_after_reset": max_op}
