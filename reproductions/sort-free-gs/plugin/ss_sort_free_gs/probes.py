"""Mechanism probes for the Sort-free GS (WSR) plugin. Each raises on failure."""

import math

import torch

from .reference import alpha_composite, dense_wsr


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _scene(device, n=48, seed=0, W=40, H=32):
    """Random projected splats: some large/overlapping, u up to 2.5 (unclamped), varied w."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means2d = torch.rand(n, 2, generator=g) * torch.tensor([W + 8.0, H + 8.0]) - 4.0
    s = torch.rand(n, 2, generator=g) * 4.0 + 0.8
    rho = (torch.rand(n, generator=g) - 0.5) * 1.2
    cov = torch.stack((s[:, 0] ** 2, rho * s[:, 0] * s[:, 1], s[:, 1] ** 2), dim=-1)
    det = cov[:, 0] * cov[:, 2] - cov[:, 1] ** 2
    conics = torch.stack((cov[:, 2] / det, -cov[:, 1] / det, cov[:, 0] / det), dim=-1)
    feats = torch.rand(n, 4, generator=g)
    opac = torch.rand(n, generator=g) * 2.5 + 0.01
    weights = torch.rand(n, generator=g) * 0.9 + 0.05
    depths = torch.rand(n, generator=g) * 5 + 1
    radii = torch.ceil(3.33 * torch.sqrt(torch.stack((cov[:, 0], cov[:, 2]), -1))).int()
    t = [x.to(device) for x in (means2d, conics, feats, opac, weights, depths, radii)]
    return (*t, W, H)


def _isect(means2d, radii, W, H, tile=16):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    tw, th = math.ceil(W / tile), math.ceil(H / tile)
    zeros = torch.zeros(1, len(means2d), device=means2d.device)
    _, ids, flat = isect_tiles(means2d[None], radii[None], zeros, tile, tw, th, packed=False,
                               n_images=1)
    return isect_offset_encode(ids, 1, tw, th).reshape(th, tw), flat


def wsr_forward_vs_dense(method, device):
    _cuda(device)
    from .kernels import rasterize_wsr

    m2, con, f, u, w, dep, radii, W, H = _scene(device)
    off, flat = _isect(m2, radii, W, H)
    S, Sw = rasterize_wsr(m2, con, f, u, w, off, flat, W, H)
    live = torch.ones(len(m2), dtype=torch.bool, device=device)
    Sr, Swr = dense_wsr(*(x.double() for x in (m2, con, f, u, w)), live, W, H)
    err = max((S.double() - Sr).abs().max().item(), (Sw.double() - Swr).abs().max().item())
    scale = Swr.abs().max().item()
    if err > 1e-4 * max(1.0, scale):
        raise AssertionError(f"CUDA WSR sums differ from dense reference: {err:.3e}")
    # order independence: a random permutation of the Gaussians gives the same image
    perm = torch.randperm(len(m2), generator=torch.Generator().manual_seed(1)).to(device)
    off_p, flat_p = _isect(m2[perm], radii[perm], W, H)
    Sp, Swp = rasterize_wsr(m2[perm], con[perm], f[perm], u[perm], w[perm], off_p, flat_p, W, H)
    perm_err = max((Sp - S).abs().max().item(), (Swp - Sw).abs().max().item())
    if perm_err > 1e-4 * max(1.0, scale):
        raise AssertionError(f"WSR depends on Gaussian order: {perm_err:.3e}")
    # it is not alpha compositing: a depth permutation changes alpha compositing but not WSR
    rgb = (S[..., :3] / Sw.clamp_min(1e-10)).double()
    comp = alpha_composite(*(x.double() for x in (m2, con, f[:, :3], u.clamp(max=1.0))), dep,
                           live, W, H)
    comp_flip = alpha_composite(*(x.double() for x in (m2, con, f[:, :3], u.clamp(max=1.0))),
                                -dep, live, W, H)
    if (comp - comp_flip).abs().max().item() < 1e-2:
        raise AssertionError("fixture has no occlusion; cannot separate WSR from compositing")
    covered = Sw[..., 0] > 0.05
    gap = (rgb - comp).abs()[covered].mean().item()
    if gap < 1e-3:
        raise AssertionError("WSR output equals alpha compositing")
    return {"max_abs_err": err, "permutation_err": perm_err, "w_sum_max": scale,
            "mean_abs_diff_to_alpha_compositing": gap, "n_isects": int(flat.numel())}


def wsr_backward_vs_dense(method, device):
    _cuda(device)
    from .kernels import rasterize_wsr

    m2, con, f, u, w, _, radii, W, H = _scene(device, seed=3)
    off, flat = _isect(m2, radii, W, H)
    g = torch.Generator(device="cpu").manual_seed(7)
    vf = torch.randn(H, W, 4, generator=g).to(device)
    vw = torch.randn(H, W, 1, generator=g).to(device)
    xs = [x.clone().requires_grad_(True) for x in (m2, con, f, u, w)]
    S, Sw = rasterize_wsr(*xs, off, flat, W, H)
    ((S * vf).sum() + (Sw * vw).sum()).backward()
    ref = [x.detach().double().requires_grad_(True) for x in (m2, con, f, u, w)]
    live = torch.ones(len(m2), dtype=torch.bool, device=device)
    Sr, Swr = dense_wsr(*ref, live, W, H)
    ((Sr * vf.double()).sum() + (Swr * vw.double()).sum()).backward()
    out = {}
    for name, a, b in zip(("means2d", "conics", "feats", "u", "w"), xs, ref):
        err = (a.grad.double() - b.grad).abs().max().item()
        scale = b.grad.abs().max().item()
        out[name] = {"max_abs_err": err, "ref_max": scale}
        if not scale > 0:
            raise AssertionError(f"{name}: reference gradient is zero")
        if err > 2e-3 * scale + 1e-5:
            raise AssertionError(f"{name}: CUDA gradient differs from autograd ({err:.3e})")
    return out


def _camera(device, eye=(0.0, 0.0, 0.0)):
    view = torch.eye(4, device=device)
    view[:3, 3] = -torch.tensor(eye, device=device)
    return {"viewmat": view, "K": torch.tensor([[30.0, 0, 16], [0, 30.0, 12], [0, 0, 1]],
                                              device=device), "width": 32, "height": 24}


def _params(method, device, n=24):
    from splatstudio.state import sample_scene

    params, _ = sample_scene(method, device, n)
    with torch.no_grad():
        params["scales"].add_(math.log(3.0))
    return params


def lc_weight(method, device):
    _cuda(device)
    from splatstudio.components import build

    renderer, _, _ = build(method)
    params = _params(method, device)
    kind = renderer.cfg["weight"]
    with torch.no_grad():
        params["wsr_sigma"].fill_(3.2)  # sample_scene z in [2.5, 3.5]: some beyond sigma
        params["opacities"].copy_(torch.linspace(-2, 2, len(params["means"]), device=device))
    cam = _camera(device)
    out = renderer.render(params, cam, sh_degree=0)
    wsr = out["info"]["wsr"]
    z = (params["means"] @ cam["viewmat"][:3, :3].T + cam["viewmat"][:3, 3])[:, 2]
    v = torch.sigmoid(params["opacities"])
    expected = torch.clamp(1 - z / 3.2, min=0) * v if kind == "lc" else v
    err = (wsr["w"] - expected).abs().max().item()
    if err > 1e-6:
        raise AssertionError(f"weight differs from Eq. 10 ({kind}): {err:.3e}")
    beyond = int((z >= 3.2).sum())
    if kind == "lc" and (beyond == 0 or wsr["w"][z >= 3.2].abs().max() != 0):
        raise AssertionError("Gaussians beyond sigma must get zero weight")
    target = torch.rand_like(out["rgb"])
    (out["rgb"] - target).square().mean().backward()
    grads = [params[k].grad for k in ("wsr_sigma", "wsr_wbg")]
    g_sigma, g_bg = (0.0 if g is None else g.abs().sum().item() for g in grads)
    if g_bg == 0:
        raise AssertionError("background weight w_B receives no gradient")
    if kind == "lc" and g_sigma == 0:
        raise AssertionError("LC sigma receives no gradient")
    if kind == "dir" and g_sigma != 0:
        raise AssertionError("DIR-WSR weight must not depend on sigma")
    return {"weight": kind, "max_err": err, "beyond_sigma": beyond, "grad_sigma": g_sigma,
            "grad_w_bg": g_bg}


def view_dependent_opacity(method, device):
    _cuda(device)
    from gsplat.cuda._wrapper import spherical_harmonics
    from splatstudio.components import build

    renderer, _, _ = build(method)
    params = _params(method, device)
    g = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        params["opa_sh0"].copy_(torch.rand(len(params["means"]), 1, generator=g).to(device) * 4)
        params["opa_shN"].copy_(torch.randn(len(params["means"]), 15, generator=g).to(device))
    us = []
    for eye in ((0.0, 0.0, 0.0), (0.4, -0.3, 0.2)):
        cam = _camera(device, eye)
        u = renderer.render(params, cam, sh_degree=3)["info"]["wsr"]["u"]
        coeffs = torch.cat((params["opa_sh0"], params["opa_shN"]), dim=-1)
        pad = torch.zeros(len(coeffs), 16, 3, device=device)
        pad[..., 0] = coeffs
        dirs = params["means"] - torch.tensor(eye, device=device)
        ref = renderer.cfg["u_offset"] + spherical_harmonics(3, dirs, pad)[..., 0]
        err = (u - ref).abs().max().item()
        if err > 1e-4:
            raise AssertionError(f"u differs from gsplat SH reference: {err:.3e}")
        us.append(u.detach())
    change = (us[0] - us[1]).abs().max().item()
    if change < 1e-3:
        raise AssertionError("opacity is not view dependent")
    if us[0].max().item() <= 1.0:
        raise AssertionError("u appears clamped to [0,1] (Eq. 11: unclamped)")
    return {"max_view_change": change, "u_max": us[0].max().item(), "sh_ref_err": err}


def recipe(method, device):
    from gsplat.strategy.ops import duplicate
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene

    _, density, _ = build(method)
    params, opts = sample_scene(method, device, 12)
    density.initialize(params, opts, scene_scale=5.0)
    st = density.strategy
    if st.prune_opa != 0.0 or density.base.opacity_reset:
        raise AssertionError("opacity pruning / reset must be disabled")
    if abs(opts["opacities"].param_groups[0]["lr"] - 0.01) > 1e-12:
        raise AssertionError("v_i learning rate not applied")
    lrs = [density.sigma_lr(s) for s in (0, 15000, 30000)]
    expect = [0.5, 0.5 * math.sqrt(0.01), 0.005]
    if any(abs(a - b) > 1e-9 for a, b in zip(lrs, expect)):
        raise AssertionError(f"sigma lr schedule {lrs} != {expect}")
    # global rows stay synchronised through a clone with zero-initialised moments
    (params["wsr_sigma"].mean() * 2.0 + params["wsr_wbg"].mean()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    mask = torch.zeros(12, dtype=torch.bool, device=device)
    mask[:3] = True
    duplicate(params, opts, density.state, mask)
    from .density import sync_globals

    sync_globals(params, opts)
    check_state(method, params, opts)
    for name in ("wsr_sigma", "wsr_wbg"):
        p = params[name]
        m = opts[name].state[p]["exp_avg_sq"]
        if (p - p[0]).abs().max() > 0 or (m - m[0]).abs().max() > 0 or m[0] == 0:
            raise AssertionError(f"{name} rows/moments not synchronised")
    adam = global_adam_equivalence(method, device)
    return {"prune_opa": st.prune_opa, "opacity_reset": density.base.opacity_reset,
            "sigma_lr": lrs, "global_sync": True, "scalar_adam_equivalence": adam}


def global_adam_equivalence(method, device):
    """The global-row parameter follows a scalar Adam parameter exactly while N changes
    (clone x2, then prune) — fails for the plain row-mean (grad g/N) parametrisation."""
    from gsplat.strategy.ops import duplicate, remove
    from splatstudio.state import sample_scene

    from .density import sync_globals
    from .renderer import global_value

    params, opts = sample_scene(method, device, 12)
    lr = 0.05
    for name in ("wsr_sigma", "wsr_wbg"):
        opts[name].param_groups[0]["lr"] = lr
    ref = torch.nn.Parameter(torch.tensor(10.0, device=device))
    ref_opt = torch.optim.Adam([ref], lr=lr, eps=1e-15)
    g = torch.Generator(device="cpu").manual_seed(11)
    state = {}
    for step in range(1, 41):
        noise = float(torch.rand(1, generator=g)) + 0.5
        for x, o in ((global_value(params, "wsr_sigma"), None), (ref, ref_opt)):
            (noise * (x - 3.0) ** 2).backward()
        for o in (opts["wsr_sigma"], ref_opt):
            o.step()
            o.zero_grad(set_to_none=True)
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        n = len(params["means"])
        if step in (5, 15):
            duplicate(params, opts, state, torch.ones(n, dtype=torch.bool, device=device))
        if step == 25:
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            mask[: n // 2 + 1] = True
            remove(params, opts, state, mask)
        sync_globals(params, opts)
    val = global_value(params, "wsr_sigma").item()
    err = abs(val - ref.item())
    if err > 1e-4 * max(1.0, abs(ref.item() - 10.0)):
        raise AssertionError(f"global row param {val} != scalar Adam {ref.item()}")
    # float32 exactness at a realistic population (the mean*N - mean*(N-1) form fails here)
    big = {"x": torch.nn.Parameter(torch.full((2_000_000,), 10.123456, device=device))}
    bv = global_value(big, "x")
    bv.backward()
    if bv.item() != big["x"][0].item() or (big["x"].grad - 1.0).abs().max().item() != 0:
        raise AssertionError(f"global value inexact at N=2e6: {bv.item()}")
    return {"final_rows": len(params["means"]), "value": val, "scalar_adam": ref.item(),
            "abs_err": err, "value_at_2e6_rows": bv.item()}
