"""Mechanism probes for Deformable Beta Splatting. Each raises when its mechanism is removed or
wrong."""

import math

import torch
from splatstudio.components import build
from splatstudio.state import sample_scene

from .density import add_new, noise_delta, relocate
from .kernels import rasterize_beta, spherical_beta
from .losses import regularizer
from .reference import dense_render, spherical_beta_ref
from .renderer import beta_activation, beta_radii, colors_sb, project, sb_activation


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return "cuda"


def synthetic_scene(n=250, seed=0, device="cuda", m=2):
    """Primitives in front of an identity camera (z in [2.5, 4.5]) with random Beta shapes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * 1.0,
                         (torch.rand(n, generator=g) * 2 - 1) * 0.8,
                         2.5 + torch.rand(n, generator=g) * 2.0], dim=-1)
    sb = torch.randn(n, m, 6, generator=g) * 0.5
    sb[..., 3] = torch.rand(n, m, generator=g) * math.pi
    sb[..., 4] = torch.rand(n, m, generator=g) * 2 * math.pi
    params = {
        "means": means,
        "scales": -2.6 + torch.rand(n, 3, generator=g) * 1.0,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g) * 1.2,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.zeros(n, 15, 3),
        "betas": torch.rand(n, generator=g) * 2.0 - 0.5,  # beta in [2.4, 18]
        "sb_params": sb,
    }
    return {k: v.to(device) for k, v in params.items()}


def camera(width, height, device="cuda"):
    f = 1.1 * width
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": width, "height": height}


SIZES = ((64, 48), (61, 37))
LEAVES = ("means2d", "conics", "colors", "opacities", "betas")


def _inputs(params, cam):
    with torch.no_grad():
        meta = project(params, cam)
        visible = (meta["radii"][0] > 0).all(-1)
        colors = colors_sb(params, cam, visible)
    leaves = {"means2d": meta["means2d"][0], "conics": meta["conics"][0], "colors": colors,
              "opacities": params["opacities"].sigmoid(),
              "betas": beta_activation(params["betas"])}
    return {k: v.detach().clone().contiguous() for k, v in leaves.items()}, meta, visible


def _cuda_render(leaves, meta, grad=False):
    x = {k: v.clone().requires_grad_(grad) for k, v in leaves.items()}
    rgb, alpha = rasterize_beta(x["means2d"], x["conics"], x["colors"], x["opacities"],
                                x["betas"], meta["isect_offsets"][0], meta["flatten_ids"],
                                meta["width"], meta["height"], meta["tile_size"])
    return x, rgb, alpha


def _dense(leaves, meta, visible, kernel="beta", grad=False):
    x = {k: v[visible].double().clone().requires_grad_(grad) for k, v in leaves.items()}
    depths = meta["depths"][0][visible].double()
    rgb, alpha = dense_render(x["means2d"], x["conics"], x["colors"], x["opacities"],
                              x["betas"], depths, meta["width"], meta["height"], kernel=kernel)
    return x, rgb, alpha


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


# ----------------------------------------------------------------------------- C1 forward
def beta_forward_vs_dense(method, device):
    from gsplat import rasterize_to_pixels
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    dev = _cuda()
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        leaves, meta, visible = _inputs(params, camera(w, h, dev))
        _, rgb, alpha = _cuda_render(leaves, meta)
        _, rgb_ref, alpha_ref = _dense(leaves, meta, visible)
        err_rgb = float((rgb.double() - rgb_ref).abs().max())
        err_a = float((alpha.double() - alpha_ref).abs().max())
        # validate the reference's compositing against gsplat's own Gaussian rasterizer
        with torch.no_grad():
            q = leaves["conics"]
            det = (q[:, 0] * q[:, 2] - q[:, 1] ** 2)[:, None]
            r = torch.ceil(3.33 * (torch.stack((q[:, 2], q[:, 0]), -1) / det).sqrt()).int()
            r = torch.where(visible[:, None], r, torch.zeros_like(r))
            _, ids, flat = isect_tiles(leaves["means2d"][None], r[None], meta["depths"], 16,
                                       meta["tile_width"], meta["tile_height"], packed=False,
                                       n_images=1)
            offs = isect_offset_encode(ids, 1, meta["tile_width"], meta["tile_height"])
            g_rgb, _ = rasterize_to_pixels(
                leaves["means2d"][None], leaves["conics"][None], leaves["colors"][None],
                leaves["opacities"][None], w, h, 16,
                offs.reshape(1, meta["tile_height"], meta["tile_width"]), flat)
            _, g_ref, _ = _dense(leaves, meta, visible, kernel="gaussian")
        err_gauss_ref = float((g_rgb[0].double() - g_ref).abs().max())
        diff_gauss = float((g_rgb[0].double() - rgb_ref).abs().max())
        out[f"{w}x{h}"] = {"max_err_rgb": err_rgb, "max_err_alpha": err_a,
                           "reference_vs_gsplat_gaussian": err_gauss_ref,
                           "beta_vs_gaussian_maxdiff": diff_gauss,
                           "mean_alpha": float(alpha.mean()), "visible": int(visible.sum())}
        if err_gauss_ref > 2e-3:
            raise AssertionError(f"dense reference disagrees with gsplat (gaussian): {out}")
        if err_rgb > 2e-3 * max(1.0, float(rgb_ref.abs().max())) or err_a > 2e-3:
            raise AssertionError(f"Beta CUDA render != dense reference: {out}")
        if diff_gauss < 0.05:
            raise AssertionError(f"Beta render indistinguishable from Gaussian: {out}")
    return out


# ----------------------------------------------------------------------------- C1 backward
def beta_backward_vs_dense(method, device):
    dev = _cuda()
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        leaves, meta, visible = _inputs(params, camera(w, h, dev))
        g = torch.Generator(device="cpu").manual_seed(1)
        v_rgb = torch.randn(h, w, 3, generator=g).to(dev)
        v_a = torch.randn(h, w, 1, generator=g).to(dev)
        x, rgb, alpha = _cuda_render(leaves, meta, grad=True)
        ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
        xr, rgb_r, alpha_r = _dense(leaves, meta, visible, grad=True)
        ((rgb_r * v_rgb.double()).sum() + (alpha_r * v_a.double()).sum()).backward()
        errs = {k: _rel(x[k].grad[visible], xr[k].grad) for k in LEAVES}
        errs["beta_grad_norm"] = float(x["betas"].grad.norm())
        out[f"{w}x{h}"] = errs
        if max(errs[k] for k in LEAVES) > 5e-3:
            raise AssertionError(f"CUDA backward != autograd of dense reference: {out}")
        if errs["beta_grad_norm"] == 0:
            raise AssertionError("no gradient reaches the Beta shape parameter")
    return out


# ----------------------------------------------------------------------------- C1 support
def beta_support(method, device):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    dev = _cuda()
    W = H = 48
    s = 6.0  # isotropic 2D std in pixels
    means2d = torch.tensor([[24.3, 23.7]], device=dev)
    conics = torch.tensor([[1 / s**2, 0.0, 1 / s**2]], device=dev)
    colors = torch.ones(1, 3, device=dev)
    o = torch.tensor([0.6], device=dev)
    radii = beta_radii(means2d[None], conics[None], torch.ones(1, 1, 2, dtype=torch.int32,
                                                              device=dev), W, H)
    if radii[0, 0].tolist() != [math.ceil(s), math.ceil(s)]:
        raise AssertionError(f"1-sigma box wrong: {radii.tolist()}")
    _, ids, flat = isect_tiles(means2d[None], radii, torch.ones(1, 1, device=dev), 16, 3, 3,
                               packed=False, n_images=1)
    offs = isect_offset_encode(ids, 1, 3, 3).reshape(3, 3)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5,
                            indexing="ij")
    r2 = ((xs - 24.3) ** 2 + (ys - 23.7) ** 2) / s**2
    out = {}
    means = {}
    for beta in (1.0, 4.0, 16.0):
        b = torch.tensor([beta], device=dev)
        _, alpha = rasterize_beta(means2d, conics, colors, o, b, offs, flat, W, H, 16)
        want = torch.where(r2 < 1, 0.6 * (1 - r2).clamp_min(0) ** beta, torch.zeros_like(r2))
        err = float((alpha[..., 0] - want).abs().max())
        outside = float(alpha[..., 0][r2 >= 1].abs().max())
        means[beta] = float(alpha.mean())
        out[f"beta={beta}"] = {"max_err": err, "max_alpha_outside_r1": outside,
                               "mean_alpha": means[beta]}
        if err > 1e-4 or outside != 0.0:
            raise AssertionError(f"Beta footprint wrong: {out}")
    if not means[1.0] > means[4.0] > means[16.0]:
        raise AssertionError(f"shape parameter does not control the footprint: {out}")
    # Gaussian-like at b = 0 (beta = 4): paper's c = 4 matches the exp(-9x/2) integral
    integral = 1.0 / (4.0 + 1.0)
    target = (1 - math.exp(-4.5)) / 4.5
    out["b0_integral_vs_gaussian"] = {"beta_kernel": integral, "exp(-9x/2)": target}
    if abs(integral - target) > 0.03 or float(beta_activation(torch.zeros(1))) != 4.0:
        raise AssertionError(f"b = 0 initialisation is not Gaussian-like: {out}")
    return out


# ----------------------------------------------------------------------------- C2 SB
def spherical_beta_vs_reference(method, device):
    dev = _cuda()
    g = torch.Generator(device="cpu").manual_seed(3)
    n, m = 400, 2
    dirs = torch.randn(n, 3, generator=g).to(dev)
    c0 = torch.rand(n, 3, generator=g).to(dev)
    raw = torch.randn(n, m, 6, generator=g).to(dev) * 0.7
    coeffs = sb_activation(raw).detach()
    x_c0, x_co = c0.clone().requires_grad_(), coeffs.clone().requires_grad_()
    x_dirs = dirs.clone().requires_grad_()
    out_cuda = spherical_beta(x_dirs, x_c0, x_co)
    v = torch.randn(n, 3, generator=g).to(dev)
    (out_cuda * v).sum().backward()
    r_c0, r_co = c0.double().requires_grad_(), coeffs.double().requires_grad_()
    out_ref = spherical_beta_ref(dirs.double(), r_c0, r_co)
    (out_ref * v.double()).sum().backward()
    res = {
        "fwd_max_err": float((out_cuda.double() - out_ref).abs().max()),
        "grad_c0_rel": _rel(x_c0.grad, r_c0.grad),
        "grad_coeffs_rel": _rel(x_co.grad, r_co.grad),
        "lobe_contribution": float((out_ref - c0.double()).abs().mean()),
        "dirs_grad": None if x_dirs.grad is None else float(x_dirs.grad.abs().max()),
    }
    if res["fwd_max_err"] > 1e-4 or res["grad_c0_rel"] > 1e-4 or res["grad_coeffs_rel"] > 2e-3:
        raise AssertionError(f"Spherical Beta CUDA != reference: {res}")
    if res["lobe_contribution"] < 1e-2:
        raise AssertionError(f"lobes contribute nothing: {res}")
    if res["dirs_grad"] not in (None, 0.0):
        raise AssertionError(f"official SB has no view-direction gradient: {res}")
    # renderer: view-dependent colours from the lobes; equal to base colour with zero lobe colour
    params = synthetic_scene(device=dev)
    vis = torch.ones(len(params["means"]), dtype=torch.bool, device=dev)
    cam1, cam2 = camera(64, 48, dev), camera(64, 48, dev)
    cam2["viewmat"] = cam2["viewmat"].clone()
    cam2["viewmat"][0, 3] = 1.5
    with torch.no_grad():
        c1, c2 = colors_sb(params, cam1, vis), colors_sb(params, cam2, vis)
        flat = dict(params)
        flat["sb_params"] = params["sb_params"].clone()
        flat["sb_params"][..., :3] = -1e4  # softplus -> 0
        base = colors_sb(flat, cam1, vis)
    want_base = (0.28209479177387814 * params["sh0"][:, 0] + 0.5).clamp_min(0)
    res["view_dependence"] = float((c1 - c2).abs().mean())
    res["base_colour_err"] = float((base - want_base).abs().max())
    if res["view_dependence"] < 1e-3 or res["base_colour_err"] > 1e-5:
        raise AssertionError(f"renderer colour path wrong: {res}")
    return res


# ----------------------------------------------------------------------------- C3 MCMC
def _density_and_scene(method, device, n=400):
    _, density, _ = build(method)
    params, opts = sample_scene(method, device, n=n)
    params["scales"].data.normal_(-2.0, 0.3)
    params["quats"].data.copy_(torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1))
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    return density, params, opts


def mcmc_opacity_only(method, device):
    torch.manual_seed(0)
    density, params, opts = _density_and_scene(method, device)
    density.initialize(params, opts)
    out = {"init_opacity": float(params["opacities"].sigmoid().mean()),
           "init_beta_raw_max": float(params["betas"].abs().max()),
           "theta_range": [float(params["sb_params"][..., 3].min()),
                           float(params["sb_params"][..., 3].max())]}
    if abs(out["init_opacity"] - 0.5) > 1e-6 or out["init_beta_raw_max"] != 0.0:
        raise AssertionError(f"official initialisation not applied: {out}")
    n = len(params["means"])
    with torch.no_grad():
        params["opacities"][:40] = math.log(0.003 / 0.997)  # dead
        params["opacities"][40:] = torch.logit(torch.rand(n - 40, device=device) * 0.6 + 0.2)
    before = {k: v.detach().clone() for k, v in params.items()}
    torch.manual_seed(1)
    n_dead = relocate(params, opts, 0.005)
    # independent re-derivation with the same RNG stream
    torch.manual_seed(1)
    o = before["opacities"].sigmoid()
    alive = (o > 0.005).nonzero(as_tuple=True)[0]
    p = o[alive] / (o[alive].sum() + torch.finfo(torch.float32).eps)
    sampled = alive[torch.multinomial(p, 40, replacement=True)]
    k = torch.bincount(sampled)[sampled].double()
    want_o = (1 - (1 - o[sampled].double()) ** (1 / (k + 1))).clamp(0.005, 1 - 1.2e-7)
    got = params["opacities"].sigmoid()
    out["relocated"] = n_dead
    out["opacity_err"] = float((got[:40].double() - want_o).abs().max())
    out["source_opacity_err"] = float((got[sampled].double() - want_o).abs().max())
    out["scales_moved_from_source"] = float((params["scales"][:40] - before["scales"][sampled]).abs().max())
    out["means_copied"] = float((params["means"][:40] - before["means"][sampled]).abs().max())
    out["source_scales_changed"] = float((params["scales"][sampled] - before["scales"][sampled]).abs().max())
    m = opts["means"].state[params["means"]]["exp_avg"]
    out["sampled_moments"] = float(m[sampled].abs().max())
    if n_dead != 40 or out["opacity_err"] > 1e-5 or out["source_opacity_err"] > 1e-5:
        raise AssertionError(f"relocation opacity rule wrong: {out}")
    if out["scales_moved_from_source"] != 0 or out["source_scales_changed"] != 0:
        raise AssertionError(f"kernel-agnostic MCMC must not adjust scales: {out}")
    if out["means_copied"] != 0 or out["sampled_moments"] != 0:
        raise AssertionError(f"relocation copy / Adam reset wrong: {out}")
    # growth: 5% up to cap
    added = add_new(params, opts, cap_max=10_000, growth=1.05, dead_opacity=0.005)
    out["added"] = added
    if added != int(1.05 * n) - n or len(params["means"]) != n + added:
        raise AssertionError(f"5% growth wrong: {out}")
    if add_new(params, opts, cap_max=len(params["means"]), growth=1.05, dead_opacity=0.005):
        raise AssertionError("growth beyond cap_max")
    for name, prm in params.items():
        if not (prm.is_leaf and prm.requires_grad):
            raise AssertionError(f"{name} is no longer a trainable leaf after densification")
    # noise: Sigma (eps (1-o)^100 noise_lr lr); independent rotation from quaternions
    with torch.no_grad():
        torch.manual_seed(5)
        d = noise_delta(params, 1e-3, 5e4, 100.0)
        torch.manual_seed(5)
        eps = torch.randn_like(params["means"])
        q = torch.nn.functional.normalize(params["quats"], dim=-1)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                        -1).reshape(-1, 3, 3)
        S2 = torch.exp(2 * params["scales"])
        cov = (R * S2[:, None, :]) @ R.transpose(1, 2)
        scale = (1 - params["opacities"].sigmoid()) ** 100 * 5e4 * 1e-3
        want = (cov @ (eps * scale[:, None])[..., None])[..., 0]
        out["noise_rel_err"] = _rel(d, want)
        # gsplat/3DGS-MCMC: op_sigmoid(1 - o) = 1 / (1 + exp(-100 (1 - o - 0.995)))
        gate = 1 / (1 + torch.exp(-100 * (1 - params["opacities"].sigmoid() - 0.995)))
        logit_noise = (cov @ (eps * gate[:, None] * 5e4 * 1e-3)[..., None])[..., 0]
        out["noise_vs_3dgs_mcmc_logit"] = _rel(want, logit_noise)
    if out["noise_rel_err"] > 1e-4 or out["noise_vs_3dgs_mcmc_logit"] < 0.1:
        raise AssertionError(f"Beta noise wrong: {out}")
    # schedule: inactive at 500 / 501 / 25000, active at 600
    out["active_steps"] = [s for s in (100, 500, 600, 24900, 25000) if density.active(s)]
    if out["active_steps"] != [600, 24900]:
        raise AssertionError(f"densification schedule wrong: {out}")
    return out


def regularizer_schedule(method, device):
    _, _, losses = build(method)
    fn = dict(losses)["dbs-regularizer"]
    params, _ = sample_scene(method, device, n=50)
    with torch.no_grad():
        params["opacities"].normal_()
        params["scales"].normal_(-2, 0.5)
    want = 0.01 * params["opacities"].sigmoid().mean() + 0.01 * params["scales"].exp().mean()
    vals = {s: float(fn(params, {"step": s}, None)) for s in (500, 501, 12000, 24999, 25000)}
    v = fn(params, {"step": 1000}, None)
    v.backward()
    out = {"values": vals, "want": float(want),
           "opacity_grad": float(params["opacities"].grad.abs().sum())}
    if abs(vals[501] - float(want)) > 1e-7 or vals[500] != 0 or vals[25000] != 0:
        raise AssertionError(f"regulariser value/schedule wrong: {out}")
    if abs(float(regularizer(params, 1000, 0.01, 0.01, 500, 25000)) - float(want)) > 1e-7:
        raise AssertionError("regularizer() mismatch")
    if out["opacity_grad"] == 0:
        raise AssertionError("regulariser does not reach opacities")
    return out
