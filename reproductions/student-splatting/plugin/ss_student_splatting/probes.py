"""Mechanism probes for 3D Student Splatting and Scooping. Each raises when its mechanism is
removed or wrong."""

import math

import torch
from splatstudio.components import build
from splatstudio.state import check_state, sample_scene

from .density import (
    add_components,
    binomials,
    nu_lr,
    op_sigmoid,
    recycle,
    relocation_student_t,
    sghmc_lr,
    sghmc_update,
)
from .kernels import rasterize_t
from .losses import regularizer
from .projection import covariance3d, empirical_factor, inria_conic, project
from .reference import dense_render
from .renderer import colors_sh, nu_activation, signed_opacity, tile_lists


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return "cuda"


def synthetic_scene(n=220, seed=0, device="cuda"):
    """Primitives in front of an identity camera (z in [2.5, 4.5]), signed opacities, nu spread
    over the interpolated critical-value table (1..13) and the Gaussian-like regime."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * 1.0,
                         (torch.rand(n, generator=g) * 2 - 1) * 0.8,
                         2.5 + torch.rand(n, generator=g) * 2.0], dim=-1)
    nus = torch.where(torch.rand(n, generator=g) < 0.7, 1.0 + torch.rand(n, generator=g) * 14.0,
                      torch.full((n,), 100.0))
    params = {
        "means": means,
        "scales": -3.0 + torch.rand(n, 3, generator=g) * 1.0,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g) * 0.6,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
        "degrees": nus,
    }
    return {k: v.to(device) for k, v in params.items()}


def camera(width, height, device="cuda"):
    f = 1.1 * width
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": width, "height": height}


SIZES = ((64, 48), (61, 37))
LEAVES = ("means2d", "conics", "colors", "opacities", "nus")


def _inputs(params, cam):
    with torch.no_grad():
        nus = nu_activation(params["degrees"])
        proj = project(params["means"], params["quats"], params["scales"], nus, cam["viewmat"],
                       cam["K"], cam["width"], cam["height"])
        offsets, ids = tile_lists(proj, cam["width"], cam["height"], 16)
        colors = colors_sh(params, cam, 3, proj["valid"])
    leaves = {"means2d": proj["means2d"], "conics": proj["conics"], "colors": colors,
              "opacities": signed_opacity(params["opacities"]), "nus": nus}
    leaves = {k: v.detach().clone().contiguous() for k, v in leaves.items()}
    return leaves, proj, offsets, ids


def _cuda_render(leaves, cam, offsets, ids, grad=False, exact=False):
    x = {k: v.clone().requires_grad_(grad) for k, v in leaves.items()}
    rgb, alpha = rasterize_t(x["means2d"], x["conics"], x["colors"], x["opacities"], x["nus"],
                             offsets, ids, cam["width"], cam["height"], 16, None, 0.99, exact)
    return x, rgb, alpha


def _dense(leaves, proj, cam, grad=False, exact=False, kernel="t"):
    vis = proj["valid"]
    x = {k: v[vis].double().clone().requires_grad_(grad) for k, v in leaves.items()}
    rgb, alpha = dense_render(x["means2d"], x["conics"], x["colors"], x["opacities"], x["nus"],
                              proj["depths"][vis].double(), proj["radii"][vis, 0], cam["width"],
                              cam["height"], 16, kernel=kernel, exact=exact)
    return x, rgb, alpha


def t_forward_vs_dense(method, device):
    device = _cuda()
    from gsplat.cuda._wrapper import rasterize_to_pixels

    out = {}
    for W, H in SIZES:
        params = synthetic_scene(device=device)
        cam = camera(W, H, device)
        leaves, proj, offsets, ids = _inputs(params, cam)
        neg = float((leaves["opacities"][proj["valid"]] < 0).float().mean())
        if not 0.2 < neg < 0.8:
            raise AssertionError(f"fixture must mix signs, negative fraction {neg}")
        _, rgb, alpha = _cuda_render(leaves, cam, offsets, ids)
        _, rgb_d, alpha_d = _dense(leaves, proj, cam)
        e_rgb = (rgb.double() - rgb_d).abs().max().item()
        e_a = (alpha.double() - alpha_d).abs().max().item()
        if e_rgb > 2e-4 or e_a > 2e-4:
            raise AssertionError(f"{W}x{H}: CUDA t render differs from dense ({e_rgb}, {e_a})")
        # the reference scaffold reproduces gsplat's Gaussian rasterizer (|o| > 0)
        pos = dict(leaves, opacities=leaves["opacities"].abs())
        g_rgb, _ = rasterize_to_pixels(pos["means2d"][None], pos["conics"][None],
                                       pos["colors"][None], pos["opacities"][None], W, H, 16,
                                       offsets[None], ids)
        _, gd_rgb, _ = _dense(pos, proj, cam, kernel="gaussian")
        e_g = (g_rgb[0].double() - gd_rgb).abs().max().item()
        if e_g > 2e-4:
            raise AssertionError(f"dense reference disagrees with gsplat Gaussian: {e_g}")
        gap_gauss = (gd_rgb - rgb_d).abs().max().item()
        flip = dict(leaves, opacities=leaves["opacities"].abs())
        _, rgb_pos, _ = _cuda_render(flip, cam, offsets, ids)
        gap_sign = (rgb_pos - rgb).abs().max().item()
        if gap_gauss < 1e-2 or gap_sign < 1e-2:
            raise AssertionError("t kernel indistinguishable from Gaussian or sign ignored")
        out[f"{W}x{H}"] = {"max_err_rgb": e_rgb, "max_err_alpha": e_a, "gsplat_ref_err": e_g,
                           "t_vs_gaussian": gap_gauss, "signed_vs_abs": gap_sign,
                           "negative_fraction": neg, "min_T": float((1 - alpha).min()),
                           "max_T": float((1 - alpha).max())}
    return out


def t_backward_vs_dense(method, device):
    device = _cuda()
    out = {}
    for exact in (False, True):
        for W, H in SIZES:
            params = synthetic_scene(seed=1, device=device)
            cam = camera(W, H, device)
            leaves, proj, offsets, ids = _inputs(params, cam)
            g = torch.Generator(device="cpu").manual_seed(3)
            v_rgb = torch.randn(H, W, 3, generator=g).to(device)
            v_a = torch.randn(H, W, 1, generator=g).to(device)
            x, rgb, alpha = _cuda_render(leaves, cam, offsets, ids, grad=True, exact=exact)
            ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
            xd, rgb_d, alpha_d = _dense(leaves, proj, cam, grad=True, exact=exact)
            ((rgb_d * v_rgb.double()).sum() + (alpha_d * v_a.double()).sum()).backward()
            vis = proj["valid"]
            errs = {}
            for k in LEAVES:
                a, b = x[k].grad[vis].double(), xd[k].grad
                rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
                errs[k] = rel
                if not torch.isfinite(a).all() or rel > 2e-3:
                    raise AssertionError(f"exact={exact} {W}x{H}: d/d{k} relative error {rel}")
                if b.norm() == 0:
                    raise AssertionError(f"{k}: reference gradient is zero")
            out[f"{'exact' if exact else 'official'}-{W}x{H}"] = errs
    # the official partial dnu really differs from the exact derivative
    params = synthetic_scene(seed=1, device=device)
    cam = camera(*SIZES[0], device)
    leaves, proj, offsets, ids = _inputs(params, cam)
    grads = []
    for exact in (False, True):
        x, rgb, _ = _cuda_render(leaves, cam, offsets, ids, grad=True, exact=exact)
        rgb.sum().backward()
        grads.append(x["nus"].grad)
    gap = ((grads[0] - grads[1]).norm() / grads[1].norm()).item()
    if gap < 1e-2:
        raise AssertionError("official and exact nu gradients do not differ")
    out["official_vs_exact_nu_grad_gap"] = gap
    return out


def t_projection(method, device):
    device = _cuda()
    from gsplat.cuda._wrapper import fully_fused_projection

    params = synthetic_scene(n=400, seed=2, device=device)
    params["means"][:, 0] *= 3.0  # many splats beyond the 1.3 tan-fov clamp
    W, H = 64, 48
    cam = camera(W, H, device)
    nus = nu_activation(params["degrees"])
    proj = project(params["means"], params["quats"], params["scales"], nus, cam["viewmat"],
                   cam["K"], W, H)
    out = {}
    for eps2d in (0.0, 0.3):
        radii, m2, _, con, _ = fully_fused_projection(
            params["means"], None, params["quats"], params["scales"].exp(), cam["viewmat"][None],
            cam["K"][None], W, H, eps2d=eps2d, near_plane=0.2, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False)
        both = proj["valid"] & (radii[0] > 0).all(-1)
        e_m = (m2[0][both] - proj["means2d"][both]).abs().max().item()
        e_c = ((con[0][both] - proj["conics"][both]).abs() /
               con[0][both].abs().clamp_min(1e-3)).max().item()
        out[f"gsplat_eps{eps2d}"] = {"means2d_err": e_m, "conic_rel_err": e_c,
                                     "compared": int(both.sum())}
    if out["gsplat_eps0.0"]["means2d_err"] > 1e-3 or out["gsplat_eps0.0"]["conic_rel_err"] > 1e-3:
        raise AssertionError(f"projection differs from EWA without low-pass: {out}")
    if out["gsplat_eps0.3"]["conic_rel_err"] < 1e-2:
        raise AssertionError("projection is indistinguishable from the low-pass-filtered one")
    # empirical radius table (forward.cu)
    nu = torch.tensor([1.0, 1.25, 2.0, 7.25, 12.25, 13.0, 100.0], device=device)
    e = empirical_factor(nu).tolist()
    # official forward.cu: case k -> (T[k-1] - T[k]) * (nu - k) + T[k]
    want = [9.925, (63.657 - 9.925) * 0.25 + 9.925, 5.841, (3.499 - 3.355) * 0.25 + 3.355,
            (3.055 - 3.012) * 0.25 + 3.012, 3.0, 3.0]
    if max(abs(a - b) for a, b in zip(e, want)) > 1e-4:
        raise AssertionError(f"empirical factor {e} != {want}")
    with torch.no_grad():
        cov = torch.linalg.inv(torch.stack((proj["conics"][:, [0, 1]],
                                            proj["conics"][:, [1, 2]]), dim=1))
        mid = 0.5 * (cov[:, 0, 0] + cov[:, 1, 1])
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        r_want = torch.ceil(empirical_factor(nus) * lam.sqrt())
        v = proj["valid"]
        r_err = (proj["radii"][v, 0].float() - r_want[v]).abs().max().item()
    if r_err > 1.0:
        raise AssertionError(f"radius != ceil(e(nu) sqrt(lambda_max)): {r_err}")
    out["radius_err"] = r_err
    out["empirical_factor"] = e
    # Inria conic backward vs autograd of the exact inverse (well conditioned) and its damping
    a = torch.tensor([4.0, 2.0e-3], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([3.0, 2.0e-3], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[0.3, -0.7, 1.1], [0.3, -0.7, 1.1]], dtype=torch.float64)
    (inria_conic(a, b, c) * w).sum().backward()
    g_inria = torch.stack((a.grad, b.grad, c.grad))
    a2, b2, c2 = (t.detach().clone().requires_grad_(True) for t in (a, b, c))
    det2 = a2 * c2 - b2 * b2
    (torch.stack((c2 / det2, -b2 / det2, a2 / det2), -1) * w).sum().backward()
    g_exact = torch.stack((a2.grad, b2.grad, c2.grad))
    well = ((g_inria[:, 0] - g_exact[:, 0]).abs() / g_exact[:, 0].abs().clamp_min(1e-9)).max()
    damp = (g_inria[:, 1].abs().max() / g_exact[:, 1].abs().max()).item()
    if well > 1e-6 or damp > 0.5:
        raise AssertionError(f"Inria conic backward wrong: well {well}, damping {damp}")
    out["conic_backward_well_conditioned_rel_err"] = float(well)
    out["conic_backward_tiny_det_damping"] = damp
    # near-plane cull at z <= 0.2
    pz = torch.tensor([[0.0, 0.0, 0.19], [0.0, 0.0, 0.21]], device=device)
    pr = project(pz, torch.tensor([[1.0, 0, 0, 0]] * 2, device=device),
                 torch.full((2, 3), -4.0, device=device), torch.full((2,), 100.0, device=device),
                 cam["viewmat"], cam["K"], W, H)
    if pr["valid"].tolist() != [False, True]:
        raise AssertionError("near plane 0.2 cull is wrong")
    return out


def scooping_sign(method, device):
    device = _cuda()
    W, H = 32, 32
    cam = camera(W, H, device)
    # two splats on the optical axis: front (z=2) and back (z=3); colours via SH0 only
    params = {
        "means": torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], device=device),
        "scales": torch.full((2, 3), -2.0, device=device),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * 2, device=device),
        "sh0": torch.tensor([[[0.8, 0.0, 0.0]], [[0.8, 0.8, 0.8]]], device=device),
        "shN": torch.zeros(2, 15, 3, device=device),
        "degrees": torch.tensor([2.0, 5.0], device=device),
    }
    out = {}
    for raw in (0.7, -0.7):
        params["opacities"] = torch.tensor([raw, 1.2], device=device)
        leaves, _proj, offsets, ids = _inputs(params, cam)
        _, rgb, alpha = _cuda_render(leaves, cam, offsets, ids)
        px, py = 16, 16  # pixel centre (16.5, 16.5); means2d at (16, 16)
        cols = leaves["colors"]
        ga = []
        for i in range(2):
            d = leaves["means2d"][i] - torch.tensor([px + 0.5, py + 0.5], device=device)
            q = leaves["conics"][i]
            base = q[0] * d[0] ** 2 + q[2] * d[1] ** 2 + 2 * q[1] * d[0] * d[1]
            nu = leaves["nus"][i]
            ga.append((leaves["opacities"][i] * (1 + base / nu) ** (-(nu + 2) / 2)).clamp(-.99, .99))
        want = cols[0] * ga[0] + cols[1] * ga[1] * (1 - ga[0])
        err = (rgb[py, px] - want).abs().max().item()
        if err > 1e-5:
            raise AssertionError(f"o={raw}: two-splat composite {rgb[py, px]} != {want}")
        out[f"raw{raw}"] = {"rgb": rgb[py, px].tolist(), "alpha_front": float(ga[0]),
                            "T": float(1 - alpha[py, px])}
    if not (out["raw-0.7"]["rgb"][0] < out["raw0.7"]["rgb"][0] - 0.2):
        raise AssertionError("negative component does not subtract its colour")
    if not out["raw-0.7"]["T"] > out["raw0.7"]["T"]:
        raise AssertionError("negative component must raise transmittance")
    o = signed_opacity(torch.tensor([-50.0, -0.5, 0.0, 0.5, 50.0]))
    if not (o.abs() <= 1).all() or o[1] >= 0 or o[3] <= 0:
        raise AssertionError("opacity activation is not tanh")
    out["tanh"] = o.tolist()
    return out


def sghmc_step(method, device):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"betas": (0.9, 0.999), "adam_eps": 1e-15, "burnin_iterations": 10,
           "C_burnin": 5e5, "C": 120.0}
    n = 64
    g = torch.Generator(device="cpu").manual_seed(0)
    m = torch.randn(n, 3, generator=g).to(dev) * 1e-3
    v = torch.rand(n, 3, generator=g).to(dev) * 1e-6
    o = torch.cat((torch.full((n // 2,), 0.001), torch.full((n // 2,), 0.9))).to(dev)
    sig = op_sigmoid(1 - o.abs())[:, None]
    L = covariance3d(torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(dev),
                     (torch.rand(n, 3, generator=g) * -3).to(dev))
    lr = 0.01
    out = {}
    for it, label in ((5, "burnin"), (11, "reset"), (12, "post")):
        mu = torch.zeros(n, 3, device=dev)
        r0 = torch.randn(n, 3, generator=g).to(dev) * 1e-3
        r = r0.clone()
        gen = torch.Generator(device=dev).manual_seed(7)
        sghmc_update(mu, r, m, v, 20.0, it, lr, sig, L, cfg, generator=gen)
        gen = torch.Generator(device=dev).manual_seed(7)
        eps = torch.randn(n, 3, device=dev, generator=gen)
        bc1, bc2 = 1 - 0.9 ** 20, 1 - 0.999 ** 20
        grad = math.sqrt(bc2) / bc1 * m / (v.sqrt() + 1e-15)
        if label == "burnin":  # friction removed, covariance-shaped noise
            want_r = -lr * grad + torch.bmm(L, (eps * math.sqrt(2 * lr * 5e5) * sig)[..., None])[..., 0]
        else:
            rr = torch.zeros_like(r0) if label == "reset" else r0
            want_r = rr * sig * (1 - lr * 120.0) - lr * grad + eps * math.sqrt(2 * lr * 120.0) * sig
        e_r = (r - want_r).abs().max().item()
        e_mu = (mu - lr * want_r).abs().max().item()
        if e_r > 1e-5 * max(1.0, want_r.abs().max().item()) or e_mu > 1e-6:
            raise AssertionError(f"{label}: SGHMC update wrong ({e_r}, {e_mu})")
        # opaque half: sig ~ 0 -> plain -lr^2 g step (no friction, no noise)
        hi = slice(n // 2, n)
        e_hi = (mu[hi] + lr * lr * grad[hi]).abs().max().item()
        if e_hi > 1e-7:
            raise AssertionError(f"{label}: opaque components not gated: {e_hi}")
        out[label] = {"momentum_err": e_r, "means_err": e_mu, "opaque_gate_err": e_hi}
    s = 4.2
    out["lr_start_sq"] = sghmc_lr(1, s, {"lr_init_base": 0.001226442339330813,
                                         "lr_final_base": 1.226630619638022e-05,
                                         "lr_reference_scale": 7.45176315307617,
                                         "lr_max_steps": 30000}) ** 2
    if abs(out["lr_start_sq"] - 0.001226442339330813 * s / 7.45176315307617 *
           math.exp(math.log(1.226630619638022e-05 / 0.001226442339330813) / 30000)) > 1e-9:
        raise AssertionError("SGHMC lr schedule wrong")
    nl = {"nu_lr": 10.0, "nu_lr_gamma": 0.5, "nu_lr_step": 5000, "nu_lr_min": 0.01}
    if [nu_lr(k, nl) for k in (1, 4999, 5000, 10000, 200000)] != [10.0, 10.0, 5.0, 2.5, 0.01]:
        raise AssertionError("nu learning-rate schedule wrong")
    # the density component drives the means: Adam lr 0, opacity lr 0.005, inits
    _, density, _ = build(method)
    params, optimizers = sample_scene(method, dev)
    scales0 = params["scales"].detach().clone()
    density.initialize(params, optimizers, 2.0)
    if optimizers["means"].param_groups[0]["lr"] != 0.0:
        raise AssertionError("means must not be stepped by Adam")
    if abs(optimizers["opacities"].param_groups[0]["lr"] - 0.005) > 1e-12:
        raise AssertionError("opacity lr must be 0.005")
    if abs(float(torch.tanh(params["opacities"][0])) - 0.5) > 1e-6:
        raise AssertionError("initial signed opacity must be 0.5")
    if (params["scales"] - scales0 - math.log(0.1)).abs().max() > 1e-6:
        raise AssertionError("initial scales must be 0.1 x the 3-NN scale")
    out["init"] = "ok"
    return out


def _reloc_reference(o, s, nu, N):
    """float64 python port of compute_relocation_student_t's double loop."""
    o_new = 1 - (1 - o) ** (1 / N)
    beta = lambda x, y: math.exp(math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y))
    K = 0.0
    for i in range(1, N + 1):
        for k in range(i):
            K += math.comb(i - 1, k) * (-1) ** k * o_new ** (k + 1) * \
                beta(0.5, ((k + 1) * (nu + 3) - 1) / 2)
    return o_new, (o * beta(0.5, (nu + 2) / 2) / K) ** 2 * s


def t_relocation(method, device):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    binoms = binomials(51, dev)
    cases = [(0.9, 1.0, 100.0, 2), (0.6, 0.5, 3.5, 3), (-0.8, 2.0, 1.2, 4), (0.3, 1.0, 20.0, 1),
             (0.95, 1.0, 8.0, 8), (-0.5, 1.0, 50.0, 2)]
    o = torch.tensor([c[0] for c in cases], device=dev)
    s = torch.tensor([[c[1]] * 3 for c in cases], device=dev)
    nu = torch.tensor([c[2] for c in cases], device=dev)
    N = torch.tensor([c[3] for c in cases], device=dev)
    o_new, s_new = relocation_student_t(o, s, nu, N, binoms)
    errs = []
    for j, c in enumerate(cases):
        ro, rs = _reloc_reference(*c)
        errs.append(max(abs(o_new[j].item() - ro), abs(s_new[j, 0].item() - rs) / rs))
    if max(errs) > 1e-4:
        raise AssertionError(f"t relocation differs from the formula: {errs}")
    if abs(s_new[3, 0].item() - 1.0) > 1e-5 or abs(o_new[3].item() - 0.3) > 1e-6:
        raise AssertionError("N = 1 must leave the component unchanged")
    # density mechanics on a fixture with allocated Adam/SGHMC state
    _, density, _ = build(method)
    n = 400
    g = torch.Generator(device="cpu").manual_seed(0)
    points = torch.rand(n, 3, generator=g).to(dev)
    from splatstudio.state import initialize

    params, optimizers = initialize(method, points, torch.rand(n, 3, generator=g).to(dev))
    density.initialize(params, optimizers, 1.0)
    with torch.no_grad():
        sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
        params["opacities"].copy_((sign * (0.5 + torch.rand(n, generator=g))).to(dev))
        params["opacities"][:60] = 0.0  # 15 % dead -> only 5 % recycled
    loss = sum(p.sum() for p in params.values())
    loss.backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    st = optimizers["means"].state[params["means"]]
    st["momentum"] = torch.ones_like(params["means"])
    for name, p in params.items():
        optimizers[name].state[p]["exp_avg"].fill_(1.0)
    cfg = dict(density.cfg, cap_max=10_000)
    torch.manual_seed(0)
    k = recycle(params, optimizers, density.binoms, cfg)
    if k != int(0.05 * n):
        raise AssertionError(f"recycled {k}, expected the 5 % cap {int(0.05 * n)}")
    o_act = torch.tanh(params["opacities"])
    still_dead = int(((o_act.abs() < 0.005)).sum())
    mom = optimizers["means"].state[params["means"]]["momentum"]
    zero_mom = int((mom == 0).all(-1).sum())
    zero_avg = int((optimizers["means"].state[params["means"]]["exp_avg"] == 0).all(-1).sum())
    if still_dead != 60 - k or zero_mom != k or zero_avg == 0:
        raise AssertionError(f"recycling bookkeeping wrong: dead {still_dead}, momentum zeroed "
                             f"{zero_mom}, sources zeroed {zero_avg}")
    added = add_components(params, optimizers, density.binoms, cfg)
    if added != int(1.05 * n) - n or len(params["means"]) != int(1.05 * n):
        raise AssertionError(f"growth wrong: added {added}")
    check_state(method, params, optimizers)
    for name, p in params.items():
        if not (p.is_leaf and p.requires_grad):
            raise AssertionError(f"{name} lost requires_grad after add_components")
        stn = optimizers[name].state[p]
        if stn["exp_avg"][n:].abs().max() != 0:
            raise AssertionError(f"{name}: new rows must start with zero moments")
    if optimizers["means"].state[params["means"]]["momentum"].shape != params["means"].shape:
        raise AssertionError("SGHMC momentum not aligned after growth")
    capped = add_components(params, optimizers, density.binoms, dict(cfg, cap_max=len(params["means"])))
    if capped != 0:
        raise AssertionError("growth must stop at cap_max")
    return {"relocation_max_err": max(errs), "recycled": k, "added": added,
            "sources_with_zeroed_moments": zero_avg}


def regularizer_probe(method, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    params = {"opacities": torch.randn(50, generator=g, requires_grad=True),
              "scales": torch.randn(50, 3, generator=g, requires_grad=True)}
    v = regularizer(params, 0.01, 0.01)
    want = 0.01 * torch.tanh(params["opacities"]).abs().mean() + \
        0.01 * params["scales"].exp().mean()
    if abs(v.item() - want.item()) > 1e-7:
        raise AssertionError("regulariser formula wrong")
    _, _, losses = build(method)
    names = [n for n, _ in losses]
    if "mcmc-regularizer" not in names:
        raise AssertionError("regulariser not part of the method")
    fn = dict(losses)["mcmc-regularizer"]
    if abs(fn(params, {"step": 1}, None).item() - want.item()) > 1e-7:
        raise AssertionError("regulariser must be active from step 1")
    v.backward()
    if not (torch.sign(params["opacities"].grad) == torch.sign(params["opacities"].detach())).all():
        raise AssertionError("opacity regulariser must shrink |o| for both signs")
    return {"value": v.item(), "losses": names}
