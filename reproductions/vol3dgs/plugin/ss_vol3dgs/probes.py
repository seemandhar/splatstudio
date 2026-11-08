"""Mechanism probes for Vol3DGS. Each raises AssertionError when its mechanism is missing/wrong."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from .reference import (
    PRECIS_INDEX,
    alpha2density,
    camera_gaussians,
    dense_render,
    density2alpha,
    halve_theta,
)

SH_C0 = 0.28209479177387814


def _need_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _renderer(method, **overrides):
    from .renderer import create_renderer

    spec = _component(method, "renderer")
    return create_renderer(replace(spec, config={**spec.config, **overrides}))


def _density(method, **overrides):
    from .density import create_density

    spec = _component(method, "density")
    return create_density(replace(spec, config={**spec.config, **overrides}))


def _camera(device, W=45, H=38, f=40.0):
    K = torch.tensor([[f, 0, W / 2 - 0.3], [0, f * 1.05, H / 2 + 0.4], [0, 0, 1]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": W, "height": H}


def _scene(device, n=24, seed=0, theta=(0.05, 0.7)):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3)
    means[:, :2] = (torch.rand(n, 2, generator=g) - 0.5) * 2.4
    means[:, 2] = 3.0 + torch.rand(n, generator=g) * 3.0
    th = theta[0] + (theta[1] - theta[0]) * torch.rand(n, generator=g)
    params = {
        "means": means,
        "scales": torch.log(0.08 + 0.25 * torch.rand(n, 3, generator=g)),
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.logit(th),
        "sh0": (torch.rand(n, 1, 3, generator=g) - 0.5) * 2.0,
        "shN": torch.zeros(n, 15, 3),
    }
    return {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}


def _dense_from_params(params, camera, dtype=torch.float64):
    """Independent dense pipeline from parameters (SH degree 0, softplus colour)."""
    p = {k: v.to(dtype) for k, v in params.items()}
    mean_c, precis = camera_gaussians(p["means"], p["quats"], p["scales"],
                                      camera["viewmat"].to(dtype))
    kappa = alpha2density(torch.sigmoid(p["opacities"]), p["scales"].exp())
    colors = F.softplus(SH_C0 * p["sh0"][:, 0] + 0.5, beta=10.0)
    feats = torch.cat((colors, mean_c[:, 2:3]), dim=-1)
    rgb, alpha = dense_render(mean_c, precis, kappa, feats, camera["K"].to(dtype),
                              camera["width"], camera["height"])
    return rgb[..., :3], rgb[..., 3:4] / alpha.clamp(min=1e-10), alpha


def _close(name, a, b, rel=2e-3, abs_=1e-6):
    err = (a.double() - b.double()).abs().max().item()
    scale = b.double().abs().max().item()
    if not err <= rel * scale + abs_:
        raise AssertionError(f"{name}: max error {err:.3g} vs scale {scale:.3g}")
    return err


def forward_vs_dense(method, device):
    _need_cuda()
    from gsplat import rasterization

    cam = _camera(device)
    params = _scene(device)
    r = _renderer(method, background="black", radius_sigma=6.0)
    with torch.no_grad():
        out = r.render(params, cam, sh_degree=0)
        rgb, depth, alpha = _dense_from_params(params, cam)
        e_rgb = _close("rgb", out["rgb"], rgb)
        e_a = _close("alpha", out["alpha"], alpha)
        mask = alpha[..., 0] > 0.05
        e_d = _close("depth", out["depth"][mask], depth[mask])
        _splat, splat_alpha, _ = rasterization(
            params["means"], params["quats"], params["scales"].exp(),
            torch.sigmoid(params["opacities"]),
            F.softplus(SH_C0 * params["sh0"][:, 0] + 0.5, beta=10.0),
            cam["viewmat"][None], cam["K"][None], cam["width"], cam["height"])
        gap = (splat_alpha[0] - out["alpha"]).abs().max().item()
    if gap < 0.02:
        raise AssertionError(f"volumetric alpha indistinguishable from splatting ({gap:.3g})")
    if out["alpha"].max().item() < 0.3:
        raise AssertionError("probe scene too transparent to be informative")
    return {"rgb_err": e_rgb, "alpha_err": e_a, "depth_err": e_d, "splat_alpha_gap": gap,
            "max_alpha": out["alpha"].max().item()}


def backward_vs_dense(method, device):
    _need_cuda()
    from .kernels import rasterize_vol

    cam = _camera(device)
    params = _scene(device, seed=1)
    r = _renderer(method, background="black", radius_sigma=6.0)
    g = torch.Generator(device="cpu").manual_seed(7)
    H, W = cam["height"], cam["width"]
    w_rgb = torch.randn(H, W, 3, generator=g).to(device)
    w_a = torch.randn(H, W, 1, generator=g).to(device)

    # (a) kernel level: gradients w.r.t. the kernel inputs
    out = r.render(params, cam, sh_degree=0)
    with torch.no_grad():
        mean_c, precis = camera_gaussians(params["means"], params["quats"], params["scales"],
                                          cam["viewmat"])
        kappa = alpha2density(torch.sigmoid(params["opacities"]), params["scales"].exp())
        cols = torch.rand(len(kappa), 3, generator=g).to(device)
    leaves = [t.clone().requires_grad_(True) for t in
              (mean_c, precis[:, PRECIS_INDEX[0], PRECIS_INDEX[1]], kappa, cols)]
    # reuse the renderer's tile lists through a fresh projection
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

    from .renderer import tile_radii

    with torch.no_grad():
        br, m2d, dep, con, _ = fully_fused_projection(
            params["means"], None, params["quats"], params["scales"].exp(), cam["viewmat"][None],
            cam["K"][None], W, H, eps2d=0.3, near_plane=0.2)
        rad = tile_radii(con, br, m2d, W, H, 6.0)
        tw, th = math.ceil(W / 16), math.ceil(H / 16)
        _, ids, flat = isect_tiles(m2d, rad, dep, 16, tw, th, packed=False, n_images=1)
        offs = isect_offset_encode(ids, 1, tw, th).reshape(th, tw)
    rgb_k, a_k = rasterize_vol(*leaves, cam["K"], offs, flat, W, H, 16)
    ((rgb_k * w_rgb).sum() + (a_k * w_a).sum()).backward()

    ref = [t.detach().double().requires_grad_(True) for t in leaves]
    a00, a01, a02, a11, a12, a22 = ref[1].unbind(-1)
    P = torch.stack((a00, a01, a02, a01, a11, a12, a02, a12, a22), -1).reshape(-1, 3, 3)
    rgb_r, a_r = dense_render(ref[0], P, ref[2], ref[3], cam["K"].double(), W, H)
    ((rgb_r * w_rgb.double()).sum() + (a_r * w_a.double()).sum()).backward()
    errs = {}
    for name, a, b in zip(("means_c", "precis", "kappa", "colors"), leaves, ref):
        errs[name] = _close(f"grad {name}", a.grad, b.grad, rel=5e-3)

    # (b) end to end: parameter gradients of the full renderer vs the dense pipeline
    loss = (out["rgb"] * w_rgb).sum() + (out["alpha"] * w_a).sum()
    names = ("means", "scales", "quats", "opacities", "sh0")
    grads = torch.autograd.grad(loss, [params[k] for k in names])
    p64 = {k: v.detach().double().requires_grad_(True) for k, v in params.items()}
    rgb_d, _, a_d = _dense_from_params(p64, cam)
    loss_d = (rgb_d * w_rgb.double()).sum() + (a_d * w_a.double()).sum()
    grads_d = torch.autograd.grad(loss_d, [p64[k] for k in names])
    for name, a, b in zip(names, grads, grads_d):
        errs[f"param_{name}"] = _close(f"param grad {name}", a, b, rel=5e-3)
    return errs


def analytic_transmittance(method, device):
    _need_cuda()
    W = H = 33
    K = torch.tensor([[30.0, 0, 16.5], [0, 30.0, 16.5], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": W, "height": H}
    s, theta, z = 0.2, 0.5, 5.0
    r = _renderer(method, background="black", radius_sigma=6.0)

    def one(scales, kappa_theta):
        p = {
            "means": torch.tensor([[0.0, 0.0, z]], device=device),
            "scales": torch.tensor([scales], device=device).log(),
            "quats": torch.tensor([[1.0, 0, 0, 0]], device=device),
            "opacities": torch.logit(torch.tensor([kappa_theta], device=device)),
            "sh0": torch.full((1, 1, 3), 2.0, device=device),
            "shN": torch.zeros(1, 15, 3, device=device),
        }
        with torch.no_grad():
            return r.render(p, cam, sh_degree=0)["alpha"][..., 0], p

    alpha, _p = one([s, s, s], theta)
    centre = alpha[16, 16].item()
    if abs(centre - 0.99 * theta) > 2e-4:
        raise AssertionError(f"centre alpha {centre:.5f} != 0.99 theta = {0.99 * theta:.5f}")
    # off-axis: alpha = 1 - exp(-kappa sqrt(2 pi) s exp(-r_perp^2 / 2 s^2)) for an isotropic
    # Gaussian, r_perp = distance of the mean from the ray (independent closed form)
    kappa = -math.log(1 - 0.99 * theta) / s / 2.506
    worst = 0.0
    for j in range(W):
        d = torch.tensor([(j + 0.5 - 16.5) / 30.0, 0.0, 1.0], dtype=torch.float64)
        d = d / d.norm()
        mu = torch.tensor([0.0, 0.0, z], dtype=torch.float64)
        rp2 = (mu - (mu @ d) * d).square().sum().item()
        expect = min(0.99, 1 - math.exp(-kappa * 2.506 * s * math.exp(-rp2 / (2 * s * s))))
        expect = expect if expect >= 1 / 255 else 0.0
        worst = max(worst, abs(alpha[16, j].item() - expect))
    if worst > 5e-4:
        raise AssertionError(f"off-axis alpha deviates from the line integral by {worst:.3g}")
    # z-scale (along the ray) changes alpha volumetrically; splatting ignores it. Keep kappa fixed
    # by choosing theta so that -log(1-0.99 theta) mean(1/s) is unchanged.
    s3 = [s, s, 3 * s]
    mean_inv = (2 / s + 1 / (3 * s)) / 3
    theta3 = (1 - math.exp(-kappa * 2.506 / mean_inv)) / 0.99
    alpha3, _ = one(s3, theta3)
    expect3 = 1 - math.exp(-kappa * 2.506 * 3 * s)
    if abs(alpha3[16, 16].item() - min(0.99, expect3)) > 5e-4:
        raise AssertionError("stretched-Gaussian centre alpha does not follow beta = 3 s")
    if alpha3[16, 16].item() <= centre + 0.1:
        raise AssertionError("alpha does not grow with the extent along the ray")
    return {"centre_alpha": centre, "offaxis_max_err": worst,
            "stretched_centre_alpha": alpha3[16, 16].item()}


def density_reparam(method, device):
    g = torch.Generator().manual_seed(3)
    theta = torch.rand(64, generator=g, dtype=torch.float64) * 0.98 + 0.01
    scales = torch.rand(64, 3, generator=g, dtype=torch.float64) * 0.5 + 0.01
    kappa = alpha2density(theta, scales)
    for i in range(64):
        expect = (-math.log(1 - 0.99 * theta[i].item())
                  * sum(1 / scales[i, k].item() for k in range(3)) / 3 / 2.506)
        if abs(kappa[i].item() - expect) > 1e-9 * max(1.0, expect):
            raise AssertionError("kappa does not follow -log(1-0.99 theta) mean(1/s) / 2.506")
    # official general_utils.py:56 as written: 1 - exp(-x) / 0.99 -> round trip theta - 1/99
    rt = density2alpha(kappa, scales)
    if (rt - (theta - 1 / 99).clamp(0, 0.99999)).abs().max() > 1e-9:
        raise AssertionError("official density2alpha round trip is not theta - 0.0101")
    rt_inv = density2alpha(kappa, scales, "inverse")
    if (rt_inv - theta).abs().max() > 1e-9:
        raise AssertionError("'inverse' density2alpha is not the inverse of alpha2density")
    big = density2alpha(alpha2density(torch.tensor([0.99999], dtype=torch.float64),
                                      scales[:1]), scales[:1])
    if big.item() >= 0.99:
        raise AssertionError("official round trip must never reach the 0.99 split threshold")
    half = halve_theta(theta, scales)
    closed = (1 - torch.sqrt(1 - 0.99 * theta) / 0.99).clamp(0, 0.99999)
    if (half - closed).abs().max() > 1e-9:
        raise AssertionError("official halving is not 1 - sqrt(1 - 0.99 theta) / 0.99")
    half_inv = halve_theta(theta, scales, "inverse")
    if (half_inv - (1 - torch.sqrt(1 - 0.99 * theta)) / 0.99).abs().max() > 1e-9:
        raise AssertionError("'inverse' halving is not (1 - sqrt(1 - 0.99 theta)) / 0.99")
    if not torch.allclose(alpha2density(half_inv, scales), kappa / 2):
        raise AssertionError("'inverse' halved theta does not halve kappa")
    return {"n": 64, "official_roundtrip_offset": (theta - rt).mean().item(),
            "official_halve_offset": (half_inv - half)[theta > 0.05].mean().item()}


def _densify_scene(method, step, **overrides):
    from splatstudio.state import initialize

    n = 6
    pts = torch.zeros(n, 3)
    pts[:, 0] = torch.arange(n, dtype=torch.float32)
    params, opts = initialize(method, pts, torch.full((n, 3), 0.5))
    smax = [0.01, 0.03, 0.01, 0.01, 0.05, 0.01]
    theta = [0.5, 0.4, 0.995, 0.03, 0.5, 0.5]
    grad = [1e-4, 1e-6, 0.0, 0.0, 0.0, 1e-5]
    with torch.no_grad():
        params["scales"].copy_(torch.tensor(smax).log()[:, None].expand(n, 3))
        params["scales"][:, 1] -= math.log(2.0)  # anisotropic: max stays smax
        params["opacities"].copy_(torch.logit(torch.tensor(theta)))
    d = _density(method, **overrides)
    d.initialize(params, opts, scene_scale=1.0)  # ext = 2 -> thresholds 0.02
    d.state["grad_accum"] = torch.tensor(grad) * 3
    d.state["count"] = torch.tensor([3.0, 3.0, 0.0, 3.0, 0.0, 3.0])
    before = {k: v.detach().clone() for k, v in params.items()}
    d.densify_and_prune(params, opts, step)
    return d, params, opts, before


def _official_half(t):
    return (1 - torch.sqrt(1 - 0.99 * t) / 0.99).clamp(0, 0.99999)


def _inverse_half(t):
    return (1 - torch.sqrt(1 - 0.99 * t)) / 0.99


def densify_rules(method, device):
    from splatstudio.state import check_state

    from .density import expon_lr

    # official density2alpha: the theta >= 0.99 split never fires (G2 at 0.995 stays)
    d, params, opts, before = _densify_scene(method, 3200)
    check_state(method, params, opts)
    lc = d.last_counts
    if (lc["clone"], lc["split"], lc["prune"]) != (1, 1, 2):
        raise AssertionError(f"official clone/split/prune counts {lc} != (1, 1, 2)")
    theta = torch.sigmoid(params["opacities"]).detach()
    sc = params["scales"].exp().detach()
    th0 = torch.sigmoid(before["opacities"])
    # survivors: G0, G2, G5, clone(G0), children of G1 (x2)
    expect = torch.stack((th0[0], th0[2], th0[5], _official_half(th0[0]),
                          _official_half(th0[1]), _official_half(th0[1])))
    if len(theta) != 6 or (theta - expect).abs().max() > 1e-5:
        raise AssertionError(f"theta after densification {theta.tolist()} != {expect.tolist()}")
    if not torch.allclose(params["means"][3], before["means"][0]):
        raise AssertionError("clone is not an exact copy")
    child = sc[4:].max(dim=-1).values
    if not torch.allclose(child, before["scales"][[1, 1]].exp().max(dim=-1).values / 1.6,
                          rtol=1e-5):
        raise AssertionError("split children are not scaled by 1/1.6")
    # 'inverse' option: the high-density split fires and halving is exact
    di, pi, oi, bi = _densify_scene(method, 3200, density2alpha="inverse")
    check_state(method, pi, oi)
    if (di.last_counts["clone"], di.last_counts["split"], di.last_counts["prune"]) != (1, 2, 2):
        raise AssertionError(f"inverse-mode counts {di.last_counts} != (1, 2, 2)")
    thi = torch.sigmoid(pi["opacities"]).detach()
    t0 = torch.sigmoid(bi["opacities"])
    expect_i = torch.stack((t0[0], t0[5], _inverse_half(t0[0]),
                            *_inverse_half(t0[[1, 2, 1, 2]])))
    if len(thi) != 7 or (thi - expect_i).abs().max() > 1e-5:
        raise AssertionError("inverse-mode theta after densification is wrong")
    # before prune_after, the large invisible Gaussian survives
    d2, params2, _, _ = _densify_scene(method, 1000)
    if d2.last_counts["prune"] != 1 or len(params2["means"]) != 7:
        raise AssertionError("world-size prune must start only after step 3000")
    # learning rates and schedule
    lrs = {k: opts[k].param_groups[0]["lr"] for k in ("opacities", "sh0", "shN", "means")}
    if abs(lrs["opacities"] - 0.03) > 1e-12 or abs(lrs["sh0"] - 0.003) > 1e-12 or abs(
            lrs["shN"] - 0.00015) > 1e-12:
        raise AssertionError(f"learning rates {lrs} differ from the paper recipe")
    if abs(d.means_lr(30000) - 1e-5) > 1e-12 or abs(d.means_lr(0) - 1.6e-4) > 1e-12:
        raise AssertionError("means LR schedule endpoints differ from 1.6e-4 -> 1e-5")
    mid = expon_lr(15000, 1.6e-4, 1e-5, 30000)
    if abs(mid - math.sqrt(1.6e-4 * 1e-5)) > 1e-12:
        raise AssertionError("means LR is not log-linear")
    return {"counts": lc, "inverse_counts": di.last_counts, "lrs": lrs}


def world_gradient(method, device):
    _need_cuda()
    cam = _camera(device)
    params = _scene(device, n=16, seed=4)
    with torch.no_grad():
        params["means"][0] = torch.tensor([0.0, 0.0, -2.0], device=device)  # behind the camera
    opts = {k: torch.optim.Adam([p], lr=0.0) for k, p in params.items()}
    r = _renderer(method, background="black", radius_sigma=6.0)
    d = _density(method)
    d.initialize(params, opts, scene_scale=1.0)
    out = r.render(params, cam, sh_degree=0)
    d.before_backward(params, opts, 1, out)
    target = torch.rand(cam["height"], cam["width"], 3, generator=torch.Generator().manual_seed(1))
    target = target.to(device)
    (out["rgb"] - target).abs().mean().backward()
    d.after_backward(params, opts, 1, out)
    p64 = {k: v.detach().double().requires_grad_(True) for k, v in params.items()}
    rgb_d, _, _ = _dense_from_params(p64, cam)
    (rgb_d - target.double()).abs().mean().backward()
    expect = p64["means"].grad.norm(dim=-1).float()
    vis = out["info"]["visible"]
    if vis[0] or not vis[1:].any():
        raise AssertionError("visibility mask wrong for the probe scene")
    got = d.state["grad_accum"]
    _close("world gradient statistic", got[vis], expect[vis], rel=5e-3, abs_=1e-9)
    if got[~vis].abs().max() != 0 or d.state["count"][~vis].max() != 0:
        raise AssertionError("invisible Gaussians accumulated statistics")
    if not torch.equal(d.state["count"][vis], torch.ones_like(got[vis])):
        raise AssertionError("visible count not incremented")
    if r.next_step != 2:
        raise AssertionError("density did not forward the step hint to the renderer")
    return {"visible": int(vis.sum()), "max_stat": got.max().item()}


def recipe_colour(method, device):
    _need_cuda()
    W = H = 17
    K = torch.tensor([[20.0, 0, 8.5], [0, 20.0, 8.5], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": W, "height": H}
    raw = -0.3
    p = {
        "means": torch.tensor([[0.0, 0.0, 4.0]], device=device),
        "scales": torch.full((1, 3), math.log(0.3), device=device),
        "quats": torch.tensor([[1.0, 0, 0, 0]], device=device),
        "opacities": torch.logit(torch.tensor([0.8], device=device)),
        "sh0": torch.full((1, 1, 3), (raw - 0.5) / SH_C0, device=device),
        "shN": torch.zeros(1, 15, 3, device=device),
    }
    r = _renderer(method, background="black", radius_sigma=6.0)
    with torch.no_grad():
        out = r.render(p, cam, sh_degree=0)
    expect = math.log1p(math.exp(10 * raw)) / 10 * 0.99 * 0.8
    got = out["rgb"][8, 8, 0].item()
    if abs(got - expect) > 1e-4 or got <= 0:
        raise AssertionError(f"centre colour {got:.6f} != softplus colour {expect:.6f}")
    schedule = {}
    for step, deg in ((1, 0), (1999, 0), (2000, 1), (4000, 2), (5999, 2), (6000, 3)):
        r.set_next_step(step)
        with torch.enable_grad():
            got_deg = r.degree(3)
        if got_deg != deg:
            raise AssertionError(f"SH degree at step {step} is {got_deg}, expected {deg}")
        schedule[step] = got_deg
    with torch.no_grad():
        r.set_next_step(1)
        if r.degree(3) != 3:
            raise AssertionError("evaluation must use the requested SH degree")
    return {"centre_colour": got, "expected": expect, "sh_schedule": schedule}
