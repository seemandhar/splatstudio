"""Mechanism probes for the 3DGabSplat port. Each raises on failure and returns measurements."""

import math
from dataclasses import replace

import torch

from .frequency import gsplat_jacobian, paper_frequency, project_frequencies, quat_scale_to_rs
from .reference import dense_render
from .renderer import FREQ, WEIGHT


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _scene2d(device, n=48, width=61, height=45, seed=0, max_freq=0.25, max_w=0.45):
    """Random 2D primitives + gsplat tile lists (4-sigma boxes cover every alpha >= 1/255)."""
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    g = torch.Generator(device="cpu").manual_seed(seed)
    means2d = torch.rand(n, 2, generator=g) * torch.tensor([width, height])
    ang = torch.rand(n, generator=g) * math.pi
    sx, sy = 1.5 + 5 * torch.rand(n, generator=g), 1.5 + 5 * torch.rand(n, generator=g)
    c, s = torch.cos(ang), torch.sin(ang)
    R = torch.stack((c, -s, s, c), -1).reshape(n, 2, 2)
    cov = R @ torch.diag_embed(torch.stack((sx, sy), -1) ** 2) @ R.transpose(1, 2)
    inv = torch.linalg.inv(cov)
    conics = torch.stack((inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]), -1)
    colors = torch.rand(n, 4, generator=g)  # RGB + depth channel
    opac = 0.2 + 0.75 * torch.rand(n, generator=g)
    depths = torch.rand(n, generator=g) * 5 + 1
    colors[:, 3] = depths
    f = (torch.rand(n, 2, 2, generator=g) * 2 - 1) * max_freq
    w = torch.rand(n, 2, 1, generator=g) * max_w
    gabs = torch.cat((f, w), -1).reshape(n, 6)
    radii = torch.ceil(4.0 * torch.stack((sx, sy), -1).max(-1, keepdim=True).values)
    radii = radii.expand(n, 2).to(torch.int32)
    tile = 16
    tw, th = math.ceil(width / tile), math.ceil(height / tile)
    host = {"means2d": means2d, "conics": conics, "colors": colors, "opac": opac,
            "depths": depths, "gabs": gabs}
    dev = {k: v.to(device) for k, v in host.items()}
    _, isect_ids, flatten_ids = isect_tiles(dev["means2d"][None], radii[None].to(device),
                                            dev["depths"][None], tile, tw, th, packed=False,
                                            n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
    return dev, offsets, flatten_ids, width, height, tile


def gabor_forward_vs_dense(method, device):
    _cuda(device)
    from .kernels import rasterize_gabor

    d, offsets, ids, W, H, tile = _scene2d(device)
    rgb, alpha = rasterize_gabor(d["means2d"], d["conics"], d["colors"], d["opac"], d["gabs"],
                                 offsets, ids, W, H, tile)
    dd = {k: v.double() for k, v in d.items()}
    ref_rgb, ref_alpha = dense_render(dd["means2d"], dd["conics"], dd["colors"], dd["opac"],
                                      dd["depths"], W, H, dd["gabs"])
    gau_rgb, _ = dense_render(dd["means2d"], dd["conics"], dd["colors"], dd["opac"],
                              dd["depths"], W, H, None)
    err_rgb = (rgb.double() - ref_rgb).abs().max().item()
    err_a = (alpha.double() - ref_alpha).abs().max().item()
    gabor_vs_gauss = (ref_rgb[..., :3] - gau_rgb[..., :3]).abs().max().item()
    if err_rgb > 2e-3 or err_a > 2e-4:  # depth channel ~6 in magnitude
        raise AssertionError(f"CUDA Gabor render differs from dense reference ({err_rgb}, {err_a})")
    if gabor_vs_gauss < 0.05:
        raise AssertionError("Gabor modulation has no visible effect on the reference render")
    # with zero weights the Gabor kernel must reduce to gsplat's Gaussian (paper Sec. 3.2.2)
    gabs0 = d["gabs"].clone().reshape(-1, 2, 3)
    gabs0[..., 2] = 0
    rgb0, _ = rasterize_gabor(d["means2d"], d["conics"], d["colors"], d["opac"],
                              gabs0.reshape(-1, 6), offsets, ids, W, H, tile)
    err_gauss = (rgb0.double() - gau_rgb).abs().max().item()
    if err_gauss > 2e-3:
        raise AssertionError(f"w = 0 does not reduce to the Gaussian kernel ({err_gauss})")
    return {"max_abs_err_rgbd": err_rgb, "max_abs_err_alpha": err_a,
            "gabor_minus_gaussian_max": gabor_vs_gauss, "w0_vs_gaussian_err": err_gauss,
            "intersections": int(ids.numel())}


def gabor_backward_vs_dense(method, device):
    _cuda(device)
    from .kernels import rasterize_gabor

    d, offsets, ids, W, H, tile = _scene2d(device, seed=1)
    names = ("means2d", "conics", "colors", "opac", "gabs")
    g = torch.Generator(device="cpu").manual_seed(7)
    v_rgb = torch.randn(H, W, 4, generator=g).to(device)
    v_rgb[..., 3] *= 0.1
    v_a = torch.randn(H, W, 1, generator=g).to(device)
    leaves = {k: d[k].clone().requires_grad_(True) for k in names}
    rgb, alpha = rasterize_gabor(*(leaves[k] for k in names), offsets, ids, W, H, tile)
    ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
    ref = {k: d[k].double().clone().requires_grad_(True) for k in names}
    rr, ra = dense_render(ref["means2d"], ref["conics"], ref["colors"], ref["opac"],
                          d["depths"].double(), W, H, ref["gabs"])
    ((rr * v_rgb.double()).sum() + (ra * v_a.double()).sum()).backward()
    out = {}
    for k in names:
        a, b = leaves[k].grad.double(), ref[k].grad
        rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        out[f"rel_err_{k}"] = rel
        if not torch.isfinite(a).all() or rel > 2e-3:
            raise AssertionError(f"{k} gradient differs from dense autograd (rel {rel})")
    gw = leaves["gabs"].grad.reshape(-1, 2, 3)
    if not torch.count_nonzero(gw[..., :2]) or not torch.count_nonzero(gw[..., 2]):
        raise AssertionError("no gradient reaches the 2D frequencies or weights")
    return out


def _random_gaussian_cam(g):
    """A camera-space Gaussian well inside the frustum (no gsplat tx/ty clamping)."""
    mean = torch.tensor([0.3, -0.2, 4.0], dtype=torch.float64) + 0.2 * torch.randn(3, generator=g,
                                                                                   dtype=torch.float64)
    q = torch.randn(4, generator=g, dtype=torch.float64)
    s = 0.05 + 0.2 * torch.rand(3, generator=g, dtype=torch.float64)
    M = quat_scale_to_rs(q[None], s[None])[0]
    return mean, q, s, M @ M.T


def frequency_projection(method, device):
    """CPU float64: coordinate-free renderer formula vs literal Eq. 9-10 vs z-integration."""
    g = torch.Generator().manual_seed(3)
    K = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], dtype=torch.float64)
    W, H = 320, 240
    worst = {"eq10_e3": 0.0, "eq10_ewa": 0.0, "integration": 0.0}
    naive_err = 0.0
    for _ in range(5):
        mean, q, s, cov = _random_gaussian_cam(g)
        f = torch.randn(1, 3, generator=g, dtype=torch.float64) * 1.5  # cycles / world unit
        J2 = gsplat_jacobian(mean[None], K, W, H)[0]
        cov2 = J2 @ cov @ J2.T
        inv = torch.linalg.inv(cov2)
        conic = torch.stack((inv[0, 0], inv[0, 1], inv[1, 1]))[None]
        ours = project_frequencies(mean[None], q[None], s[None], f[None], torch.eye(
            4, dtype=torch.float64), K, W, H, conic, torch.ones(1, dtype=torch.bool))[0, 0]
        e3 = paper_frequency(mean, cov, f[0], K, torch.tensor([0, 0, 1.0], dtype=torch.float64))
        ewa = paper_frequency(mean, cov, f[0], K, mean / mean.norm())
        worst["eq10_e3"] = max(worst["eq10_e3"], (ours - e3).abs().max().item())
        worst["eq10_ewa"] = max(worst["eq10_ewa"], (ours - ewa).abs().max().item())
        # literal integration in ray space y = J3 x (third row e3): integral over y_z of the
        # 3D Gabor divided by the integral of its envelope = C2 cos(2 pi f2d . u)
        J3 = torch.cat((J2, torch.tensor([[0, 0, 1.0]], dtype=torch.float64)))
        S = torch.linalg.inv(J3 @ cov @ J3.T)
        fp = torch.linalg.inv(J3).T @ f[0]
        zs = torch.linspace(-3, 3, 6001, dtype=torch.float64)
        ratios = []
        us = [torch.tensor([0.0, 0.0], dtype=torch.float64)] + [
            torch.randn(2, generator=g, dtype=torch.float64) * 3 for _ in range(6)]
        for u in us:
            y = torch.cat((u.expand(len(zs), 2), zs[:, None]), dim=1)
            env = torch.exp(-0.5 * torch.einsum("ni,ij,nj->n", y, S, y))
            ratios.append((env * torch.cos(2 * math.pi * y @ fp)).sum() / env.sum())
        for u, r in zip(us[1:], ratios[1:]):
            expect = torch.cos(2 * math.pi * ours @ u)
            worst["integration"] = max(worst["integration"], abs((r / ratios[0] - expect).item()))
            naive = torch.cos(2 * math.pi * fp[:2] @ u)  # projected f without z-correction
            naive_err = max(naive_err, abs((r / ratios[0] - naive).item()))
    if worst["eq10_e3"] > 1e-9 or worst["eq10_ewa"] > 1e-9:
        raise AssertionError(f"renderer frequency differs from paper Eq. 9-10: {worst}")
    if worst["integration"] > 1e-4:
        raise AssertionError(f"2D frequency disagrees with numerical z-integration: {worst}")
    if naive_err < 1e-2:
        raise AssertionError("probe cannot distinguish the z-corrected frequency from f_proj")
    return {**{f"max_err_{k}": v for k, v in worst.items()},
            "naive_fproj_integration_err": naive_err}


def _camera(device, W=64, H=48, f=60.0):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device),
            "width": W, "height": H}


def renderer_frequency(method, device):
    _cuda(device)
    from splatstudio.components import build
    from splatstudio.state import sample_scene

    renderer, _, _ = build(method)
    params, _ = sample_scene(method, device, n=24)
    with torch.no_grad():
        params["scales"].fill_(math.log(0.08))
        params[FREQ].copy_(torch.randn_like(params[FREQ]) * 2.0)
        params[WEIGHT].fill_(0.0)
    cam = _camera(device)
    out = renderer.render(params, cam, sh_degree=0)
    info = out["info"]
    vis = (info["radii"][0] > 0).all(-1)
    gab = info["gabs"].reshape(len(vis), -1, 3)[vis, :, :2].double()
    # independent per-Gaussian float64 recomputation with the literal Eq. 9-10
    K = cam["K"].double().cpu()
    errs = []
    for i in torch.nonzero(vis).flatten().tolist():
        mean = params["means"][i].double().cpu()
        M = quat_scale_to_rs(params["quats"][i:i + 1].double().cpu(),
                             params["scales"][i:i + 1].exp().double().cpu())[0]
        cov = M @ M.T
        J2 = gsplat_jacobian(mean[None], K, cam["width"], cam["height"])[0]
        cov2 = J2 @ cov @ J2.T
        dil = torch.linalg.inv(cov2 + 0.3 * torch.eye(2, dtype=torch.float64)) @ cov2
        for k in range(params[FREQ].shape[1]):
            fpap = paper_frequency(mean, cov, params[FREQ][i, k].double().cpu(), K,
                                   torch.tensor([0, 0, 1.0], dtype=torch.float64))
            row = int(torch.nonzero(vis).flatten().tolist().index(i))
            errs.append(((gab[row, k].cpu() - dil @ fpap).norm() /
                         (dil @ fpap).norm().clamp_min(1e-9)).item())
    worst = max(errs)
    if worst > 1e-3:
        raise AssertionError(f"renderer 2D frequencies differ from Eq. 9-10 (+dilation): {worst}")
    target = (torch.arange(cam["width"], device=device)[None, :, None] % 3 == 0).float()
    (out["rgb"] - target).square().mean().backward()
    gf, gw = params[FREQ].grad, params[WEIGHT].grad
    if gf is None or gw is None or not torch.count_nonzero(gf) or not torch.count_nonzero(gw):
        raise AssertionError("no gradient reaches gabor_freq / gabor_weight")
    if not (torch.isfinite(gf).all() and torch.isfinite(gw).all()):
        raise AssertionError("nonfinite Gabor gradients")
    return {"visible": int(vis.sum()), "max_rel_err": worst, "freq_grad_norm": gf.norm().item(),
            "weight_grad_norm": gw.norm().item()}


def _density_case(method, device, child_reset):
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene

    dspec = next(c for c in method.components if c.kind == "density")
    cfg = {**dspec.config, "refine_start_iter": 0, "refine_every": 100, "grow_grad2d": 0.0,
           "child_reset": child_reset}
    m = replace(method, components=tuple(replace(c, config=cfg) if c.kind == "density" else c
                                         for c in method.components))
    renderer, density, _ = build(m)
    params, opts = sample_scene(m, device, n=20)
    with torch.no_grad():
        params[FREQ].fill_(0.7)
        params[WEIGHT].fill_(0.0)  # w = 0.5
        params["scales"].fill_(math.log(0.05))
        params["opacities"].fill_(2.0)  # 0.88: survives the 0.01 prune, > 0.02 reset cap
    density.initialize(params, opts, scene_scale=1.0)
    cam = _camera(device)

    def step(k):
        out = renderer.render(params, cam, sh_degree=0)
        density.before_backward(params, opts, k, out)
        target = (torch.arange(cam["width"], device=device)[None, :, None] % 2).float()
        (out["rgb"] - target).square().mean().backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        before = {kk: v.detach().clone() for kk, v in params.items()}
        n0 = len(params["means"])
        density.after_backward(params, opts, k, out)
        check_state(m, params, opts)
        return before, n0

    before, n0 = step(100)
    return m, params, opts, density, step, before, n0


def frequency_adaptive_density(method, device):
    _cuda(device)
    _, params, opts, density, step, _, _ = _density_case(method, device, True)
    n_new = next(c for s, kind, c in density.events if kind == "child_reset")
    n = len(params["means"])
    if n_new == 0:
        raise AssertionError("fixture produced no densification children")
    child_f, child_w = params[FREQ][n - n_new:], params[WEIGHT][n - n_new:].sigmoid()
    if not torch.allclose(child_f, torch.full_like(child_f, 0.001)):
        raise AssertionError("children frequencies not reset to 0.001")
    if not torch.allclose(child_w, torch.full_like(child_w, 0.01), atol=1e-6):
        raise AssertionError("children weights not reset to 0.01")
    parents = params[FREQ][: n - n_new]
    if len(parents) and torch.allclose(parents, torch.full_like(parents, 0.001)):
        raise AssertionError("surviving parents/originals were reset too")
    if opts["opacities"].param_groups[0]["lr"] != 0.025:
        raise AssertionError("opacity learning rate is not 0.025")
    # ablation: without child reset the children inherit the parents' frequencies
    _, p2, _, _, _, _, _ = _density_case(method, device, False)
    n2 = len(p2["means"])
    if torch.allclose(p2[FREQ][-1], torch.full_like(p2[FREQ][-1], 0.001)):
        raise AssertionError("probe cannot distinguish child reset from no reset")
    # periodic weight reset with the opacity reset at step 3000 (< refine_stop_iter 15000)
    with torch.no_grad():
        params[WEIGHT].fill_(1.5)
    step(3000)
    w = params[WEIGHT].sigmoid()
    o = params["opacities"].sigmoid()
    if not torch.allclose(w, torch.full_like(w, 0.01), atol=1e-6):
        raise AssertionError("Gabor weights not reset to 0.01 at step 3000")
    if (o > 0.02 + 1e-6).any():
        raise AssertionError("opacity not reset to <= 0.02 at step 3000")
    for name, p in params.items():
        if not (p.requires_grad and p.is_leaf):
            raise AssertionError(f"{name} lost requires_grad after density control")
    # the reset fields still learn after the population changes
    f0 = params[FREQ].detach().clone()
    step(3001)
    if torch.equal(f0[: len(params[FREQ])], params[FREQ].detach()[: len(f0)]):
        raise AssertionError("gabor_freq does not move after density control")
    return {"children_reset": n_new, "population": n, "noreset_population": n2,
            "events": [list(e) for e in density.events]}


def recipe(method, device):
    fields = {f.name: f for f in method.fields}
    f, w = fields[FREQ], fields[WEIGHT]
    dspec = next(c for c in method.components if c.kind == "density").config
    rspec = next(c for c in method.components if c.kind == "renderer").config
    checks = {
        "num_freqs": (rspec["num_freqs"], 2), "freq_shape": (f.shape, (2, 3)),
        "weight_shape": (w.shape, (2,)), "init_freq": (f.initial_value, 0.05),
        "init_weight": (round(1 / (1 + math.exp(-w.initial_value)), 6), 0.01),
        "lr_freq": (f.learning_rate, 0.02), "lr_weight": (w.learning_rate, 0.01),
        "lr_opacity": (dspec["opacity_lr"], 0.025), "prune_opa": (dspec["prune_opa"], 0.01),
        "opacity_reset_value": (2 * dspec["prune_opa"], 0.02),
        "child_freq": (dspec["child_freq"], 0.001), "child_weight": (dspec["child_weight"], 0.01),
        "reset_weight": (dspec["reset_weight"], 0.01),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise AssertionError(f"recipe differs from supp. B: {bad}")
    return {k: v[0] for k, v in checks.items()}
