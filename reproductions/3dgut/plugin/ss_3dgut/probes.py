"""Mechanism probes for the 3DGUT port. Each compares the plugin's actual CUDA path to an
independently written float64 reference and fails if the mechanism is removed or reverted."""

import math

import torch

from .cameras import project_points
from .density import (
    DEFAULTS as DENSITY_DEFAULTS,
)
from .density import (
    Density,
    densification_statistic,
    official_init_values,
    step_condition,
)
from .renderer import Renderer


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _renderer(method):
    comp = next(c for c in method.components if c.kind == "renderer")
    return Renderer(comp)


def _rotmat(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _ut_reference(mean, scale, quat, K, model, radial, tangential, alpha, beta, kappa):
    """Eq. 5-9, float64, camera frame == world frame."""
    R = _rotmat(quat.double())
    lam = alpha**2 * (3 + kappa) - 3
    pts = [mean]
    for i in range(3):
        delta = math.sqrt(3 + lam) * scale[i] * R[:, i]
        pts.append(mean + delta)
    for i in range(3):
        delta = math.sqrt(3 + lam) * scale[i] * R[:, i]
        pts.append(mean - delta)
    pts = torch.stack(pts)
    v = project_points(pts, K.double(), model, radial, tangential)
    wm = torch.full((7,), 1 / (2 * (3 + lam)), dtype=torch.float64)
    wc = wm.clone()
    wm[0] = lam / (3 + lam)
    wc[0] = lam / (3 + lam) + (1 - alpha**2 + beta)
    mu = (wm[:, None] * v).sum(0)
    d = v - mu
    cov = (wc[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    return mu, cov


def _scene(device, n=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    means = torch.stack(
        (torch.rand(n, generator=g) * 1.2 - 0.6, torch.rand(n, generator=g) * 0.8 - 0.4,
         torch.rand(n, generator=g) * 1.0 + 1.5), dim=-1)
    scales = torch.rand(n, 3, generator=g) * 0.12 + 0.03
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    return means.to(device), scales.to(device), quats.to(device)


def _params(means, scales, quats, opacity, device, color=0.6):
    n = len(means)
    return {
        "means": means, "quats": quats, "scales": scales.log(),
        "opacities": torch.logit(torch.full((n,), opacity, device=device)),
        "sh0": torch.full((n, 1, 3), (color - 0.5) / 0.28209479177387814, device=device),
        "shN": torch.zeros(n, 15, 3, device=device),
    }


CAMERAS = {
    "pinhole": {},
    "opencv-radial": {"radial_coeffs": [-0.25, 0.08, 0.0, 0.0, 0.0, 0.0],
                      "tangential_coeffs": [0.001, -0.002]},
    "fisheye": {"camera_model": "fisheye", "radial_coeffs": [0.05, -0.01, 0.0, 0.0]},
}


def _camera(device, extra, size=(160, 120), f=110.0):
    w, h = size
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device),
           "width": w, "height": h}
    cam.update(extra)
    return cam


def _ref_kwargs(extra):
    model = extra.get("camera_model", "pinhole")
    return model, extra.get("radial_coeffs"), extra.get("tangential_coeffs")


def _gsplat_cov(conics, eps2d):
    a, b, c = conics.double().unbind(-1)
    inv = torch.stack((a, b, b, c), dim=-1).reshape(*a.shape, 2, 2)
    return torch.linalg.inv(inv) - eps2d * torch.eye(2, dtype=torch.float64, device=conics.device)


def ut_moments(method, device):
    """Renderer's UT 2D mean/cov == independent Eq. 5-9 with the paper's alpha=1,beta=2,kappa=0;
    gsplat's default alpha=0.1 (what rasterization(with_ut=True) uses) differs measurably."""
    _require_cuda(device)
    r = _renderer(method)
    cfg = r.config
    if (cfg["ut_alpha"], cfg["ut_beta"], cfg["ut_kappa"]) != (1.0, 2.0, 0.0):
        raise AssertionError(f"UT parameters differ from the paper: {cfg}")
    means, scales, quats = _scene(device)
    params = _params(means, scales, quats, 0.8, device)
    evidence = {}
    for name, extra in CAMERAS.items():
        cam = _camera(device, extra)
        info = r.render(params, cam, sh_degree=0)["info"]
        mu_g = info["means2d"][0].double()
        cov_g = _gsplat_cov(info["conics"][0], cfg["eps2d"])
        model, radial, tang = _ref_kwargs(extra)
        worst_mu, worst_cov, worst_alt = 0.0, 0.0, 0.0
        for i in range(len(means)):
            if not (info["radii"][0, i] > 0).all():
                raise AssertionError(f"{name}: Gaussian {i} unexpectedly culled")
            mu, cov = _ut_reference(means[i].double().cpu(), scales[i].double().cpu(),
                                    quats[i].cpu(), cam["K"].cpu(), model, radial, tang,
                                    1.0, 2.0, 0.0)
            _, cov_alt = _ut_reference(means[i].double().cpu(), scales[i].double().cpu(),
                                       quats[i].cpu(), cam["K"].cpu(), model, radial, tang,
                                       0.1, 2.0, 0.0)
            worst_mu = max(worst_mu, (mu_g[i].cpu() - mu).abs().max().item())
            rel = ((cov_g[i].cpu() - cov).norm() / cov.norm()).item()
            worst_cov = max(worst_cov, rel)
            worst_alt = max(worst_alt, ((cov_alt - cov).norm() / cov.norm()).item())
        if worst_mu > 2e-2 or worst_cov > 2e-3:
            raise AssertionError(f"{name}: UT mismatch mean {worst_mu:.3g}px cov {worst_cov:.3g}")
        if name != "pinhole" and worst_alt < 5 * max(worst_cov, 1e-4):
            raise AssertionError(f"{name}: alpha=0.1 indistinguishable from alpha=1 ({worst_alt})")
        evidence[name] = {"max_mean_err_px": worst_mu, "max_cov_rel_err": worst_cov,
                          "alpha0.1_cov_rel_diff": worst_alt}
    return evidence


def _kl2(mu0, s0, mu1, s1):
    """KL(N0 || N1) for 2D Gaussians."""
    i1 = torch.linalg.inv(s1)
    d = mu1 - mu0
    return 0.5 * (torch.trace(i1 @ s0) + d @ i1 @ d - 2 + torch.logdet(s1) - torch.logdet(s0))


def ut_vs_ewa_mc(method, device):
    """Supp. C: under strong distortion UT's projected Gaussian is closer (KL) to a Monte-Carlo
    reference than the EWA first-order linearisation J Sigma J^T."""
    _require_cuda(device)
    r = _renderer(method)
    mean = torch.tensor([0.9, 0.5, 1.2], device=device)
    scale = torch.tensor([0.25, 0.08, 0.15], device=device)
    quat = torch.nn.functional.normalize(torch.tensor([0.9, 0.2, -0.3, 0.1], device=device), dim=0)
    extra = {"camera_model": "fisheye", "radial_coeffs": [0.0, 0.0, 0.0, 0.0]}
    cam = _camera(device, extra, size=(320, 240), f=120.0)
    info = r.render(_params(mean[None], scale[None], quat[None], 0.9, device), cam, 0)["info"]
    mu_ut = info["means2d"][0, 0].double().cpu()
    cov_ut = _gsplat_cov(info["conics"][0, 0], r.config["eps2d"]).cpu()
    K = cam["K"].double().cpu()
    R = _rotmat(quat.double().cpu())
    S = torch.diag(scale.double().cpu())
    sigma = R @ S @ S @ R.T
    g = torch.Generator().manual_seed(0)
    samples = mean.double().cpu() + torch.randn(400000, 3, generator=g, dtype=torch.float64) @ (
        R @ S).T
    proj = project_points(samples, K, "fisheye", [0.0] * 4)
    mu_mc = proj.mean(0)
    cov_mc = torch.cov(proj.T)

    def f(x):
        return project_points(x[None], K, "fisheye", [0.0] * 4)[0]

    J = torch.autograd.functional.jacobian(f, mean.double().cpu())
    mu_ewa, cov_ewa = f(mean.double().cpu()), J @ sigma @ J.T
    kl_ut = _kl2(mu_mc, cov_mc, mu_ut, cov_ut).item()
    kl_ewa = _kl2(mu_mc, cov_mc, mu_ewa, cov_ewa).item()
    if not kl_ut < kl_ewa:
        raise AssertionError(f"UT not closer to MC than EWA: {kl_ut} vs {kl_ewa}")
    return {"kl_ut_vs_mc": kl_ut, "kl_ewa_vs_mc": kl_ewa}


def _ray_alpha(mean, scale, quat, opacity, cam):
    """Eq. 10 / 3DGRT: alpha = sigma * exp(-0.5 |d_g x o_g|^2) on each pixel ray (float64)."""
    w, h = cam["width"], cam["height"]
    K = cam["K"].double().cpu()
    jj, ii = torch.meshgrid(torch.arange(w, dtype=torch.float64) + 0.5,
                            torch.arange(h, dtype=torch.float64) + 0.5, indexing="xy")
    d = torch.stack(((jj - K[0, 2]) / K[0, 0], (ii - K[1, 2]) / K[1, 1], torch.ones_like(jj)), -1)
    R = _rotmat(quat.double().cpu())
    inv_s = 1 / scale.double().cpu()
    o_g = inv_s * (R.T @ (-mean.double().cpu()))  # o = 0 (camera at the world origin)
    d_g = torch.nn.functional.normalize((d @ R) * inv_s, dim=-1)
    cr = torch.linalg.cross(d_g, o_g.expand_as(d_g))
    return (opacity * torch.exp(-0.5 * (cr * cr).sum(-1))).clamp(max=0.999)


def eval3d_response(method, device):
    """Rendered alpha of a tilted anisotropic Gaussian == ray-space max response (Eq. 10), and
    != the 2D EWA conic response of gsplat's classic rasteriser."""
    _require_cuda(device)
    from gsplat import rasterization

    r = _renderer(method)
    mean = torch.tensor([0.1, -0.05, 0.9], device=device)
    scale = torch.tensor([0.30, 0.04, 0.12], device=device)
    quat = torch.nn.functional.normalize(torch.tensor([0.8, 0.3, 0.4, -0.2], device=device), dim=0)
    cam = _camera(device, {}, size=(96, 80), f=70.0)
    out = r.render(_params(mean[None], scale[None], quat[None], 0.7, device), cam, 0)
    alpha = out["alpha"][..., 0].double().cpu()
    ref = _ray_alpha(mean, scale, quat, 0.7, cam)
    ref = torch.where(ref < 1 / 255, torch.zeros_like(ref), ref)
    inside = ref > 0.02  # away from the tile-bounding edge
    err = (alpha - ref)[inside].abs().max().item()
    _, ewa, _ = rasterization(mean[None], quat[None], scale[None], torch.tensor([0.7], device=device),
                              torch.full((1, 3), 0.6, device=device), cam["viewmat"][None],
                              cam["K"][None], cam["width"], cam["height"], packed=False)
    ewa_diff = (ewa[0, ..., 0].double().cpu() - ref)[inside].abs().max().item()
    if err > 2e-4:
        raise AssertionError(f"eval3d alpha differs from the ray-space reference by {err}")
    if ewa_diff < 20 * max(err, 1e-5):
        raise AssertionError(f"Test case does not separate 3D eval from 2D EWA ({ewa_diff})")
    return {"max_abs_err_vs_ray_reference": err, "ewa_2d_max_abs_diff": ewa_diff,
            "pixels_checked": int(inside.sum())}


def _reference_render(means, log_scales, quats, logit_op, sh0, cam):
    """Differentiable float64 front-to-back compositor of eval3d (sorted by camera z)."""
    w, h = cam["width"], cam["height"]
    K = cam["K"].double()
    jj, ii = torch.meshgrid(torch.arange(w, dtype=torch.float64) + 0.5,
                            torch.arange(h, dtype=torch.float64) + 0.5, indexing="xy")
    d = torch.stack(((jj - K[0, 2]) / K[0, 0], (ii - K[1, 2]) / K[1, 1], torch.ones_like(jj)), -1)
    order = torch.argsort(means[:, 2].detach())
    T = torch.ones(h, w, dtype=torch.float64)
    rgb = torch.zeros(h, w, 3, dtype=torch.float64)
    for i in order.tolist():
        R = _rotmat(quats[i])
        inv_s = torch.exp(-log_scales[i])
        o_g = inv_s * (R.T @ (-means[i]))
        d_g = torch.nn.functional.normalize((d @ R) * inv_s, dim=-1)
        cr = torch.linalg.cross(d_g, o_g.expand_as(d_g))
        a = (torch.sigmoid(logit_op[i]) * torch.exp(-0.5 * (cr * cr).sum(-1))).clamp(max=0.999)
        a = torch.where(a < 1 / 255, torch.zeros_like(a), a)
        color = (sh0[i, 0] * 0.28209479177387814 + 0.5).clamp_min(0)
        rgb = rgb + (a * T)[..., None] * color
        T = T * (1 - a)
    return rgb + T[..., None]  # white background


def eval3d_gradient(method, device):
    """Kernel gradients wrt means / log-scales / quats / opacity logits / sh0 == autograd of the
    independent float64 compositor (two overlapping Gaussians, pinhole)."""
    _require_cuda(device)
    r = _renderer(method)
    cam = _camera(device, {}, size=(48, 40), f=40.0)
    base = {
        "means": torch.tensor([[0.05, 0.0, 1.0], [-0.08, 0.04, 1.4]]),
        "scales": torch.tensor([[0.10, 0.05, 0.08], [0.12, 0.09, 0.05]]).log(),
        "quats": torch.nn.functional.normalize(torch.tensor([[0.9, 0.1, 0.3, 0.0],
                                                             [0.7, -0.2, 0.1, 0.4]]), dim=-1),
        "opacities": torch.tensor([0.4, -0.3]),
        "sh0": torch.tensor([[[0.5, -0.2, 0.1]], [[-0.4, 0.6, 0.2]]]),
    }
    weights = torch.rand(40, 48, 3, generator=torch.Generator().manual_seed(1))
    gp = {k: v.to(device).clone().requires_grad_(True) for k, v in base.items()}
    gp["shN"] = torch.zeros(2, 15, 3, device=device)
    out = r.render(gp, cam, sh_degree=0)["rgb"]
    (out * weights.to(device)).sum().backward()
    rp = {k: v.double().clone().requires_grad_(True) for k, v in base.items()}
    cam_cpu = {**cam, "K": cam["K"].cpu()}
    ref = _reference_render(rp["means"], rp["scales"], rp["quats"], rp["opacities"], rp["sh0"],
                            cam_cpu)
    (ref * weights.double()).sum().backward()
    img_err = (out.detach().double().cpu() - ref.detach()).abs().max().item()
    evidence = {"image_max_abs_err": img_err}
    for k in base:
        g, gr = gp[k].grad.double().cpu(), rp[k].grad
        rel = ((g - gr).norm() / gr.norm().clamp_min(1e-12)).item()
        evidence[f"{k}_grad_rel_err"] = rel
        if rel > 2e-2:
            raise AssertionError(f"{k} gradient mismatch {rel}: {g} vs {gr}")
    if img_err > 1e-3:
        raise AssertionError(f"Image mismatch {img_err}")
    return evidence


def fisheye_render(method, device):
    """A small off-axis Gaussian rendered through an equidistant fisheye peaks at the analytic
    fisheye projection of its centre, far from its pinhole projection."""
    _require_cuda(device)
    r = _renderer(method)
    mean = torch.tensor([1.1, 0.6, 1.0], device=device)
    scale = torch.full((3,), 0.01, device=device)
    quat = torch.tensor([1.0, 0, 0, 0], device=device)
    cam = _camera(device, {"camera_model": "fisheye", "radial_coeffs": [0.0, 0.0, 0.0, 0.0]},
                  size=(320, 240), f=100.0)
    alpha = r.render(_params(mean[None], scale[None], quat[None], 0.9, device), cam, 0)["alpha"]
    a = alpha[..., 0].double().cpu()
    if a.max() < 0.5:
        raise AssertionError(f"Fisheye render lost the Gaussian (max alpha {a.max():.3f})")
    jj, ii = torch.meshgrid(torch.arange(a.shape[1], dtype=torch.float64) + 0.5,
                            torch.arange(a.shape[0], dtype=torch.float64) + 0.5, indexing="xy")
    peak = torch.stack(((a * jj).sum() / a.sum(), (a * ii).sum() / a.sum()))
    K = cam["K"].double().cpu()
    fish = project_points(mean.double().cpu()[None], K, "fisheye", [0.0] * 4)[0]
    pin = project_points(mean.double().cpu()[None], K, "pinhole")[0]
    err, sep = (peak - fish).norm().item(), (pin - fish).norm().item()
    if err > 0.5 or sep < 10:
        raise AssertionError(f"Fisheye peak {peak.tolist()} vs analytic {fish.tolist()} ({err})")
    return {"peak_err_px": err, "pinhole_vs_fisheye_px": sep}


def densify_statistic(method, device):
    """3D-gradient statistic ||grad * dist|| / 2 (gs.py:130), clone/split routing and the
    300/100/3000 cadence; fails for 2D screen gradients or the 3DGS 100-step cadence."""
    comp = next(c for c in method.components if c.kind == "density")
    cfg = {**DENSITY_DEFAULTS, **comp.config}
    if (cfg["densify_every"], cfg["prune_every"], cfg["reset_every"]) != (300, 100, 3000):
        raise AssertionError(f"Cadence differs from the official config: {cfg}")
    dens = Density(comp)
    n = 4
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor([[0.0, 0, 2], [1, 0, 3], [0, 1, 4], [0, 0, 5.0]],
                                                 device=device)),
        "scales": torch.nn.Parameter(torch.tensor([[0.001] * 3, [0.5] * 3, [0.001] * 3,
                                                   [0.001] * 3], device=device).log()),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * n, device=device)),
        "opacities": torch.nn.Parameter(torch.logit(torch.tensor([0.5, 0.5, 0.5, 0.001],
                                                                 device=device))),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3, device=device)),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })
    opts = {k: torch.optim.Adam([p], lr=0.0, eps=1e-15) for k, p in params.items()}
    dens.initialize(params, opts, scene_scale=10.0)
    campos = torch.tensor([0.0, 0.0, -1.0], device=device)
    coeff = torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.002, 0.0], [0.0, 0, 0], [1e-5, 0, 0]],
                         device=device)
    expected = torch.zeros(n, device=device)
    for step in range(598, 601):
        dens.before_backward(params, opts, step, {"info": {"campos": campos}})
        (params["means"] * coeff).sum().backward()
        for o in opts.values():  # runtime order: optimizer step + zero_grad before after_backward
            o.step()
            o.zero_grad(set_to_none=True)
        if step < 600:
            dens.after_backward(params, opts, step, {})
            m, inc = densification_statistic(coeff, params["means"].detach(), campos)
            expected[m] += inc
            if not torch.allclose(dens.state["accum"], expected, rtol=1e-5):
                raise AssertionError(f"statistic {dens.state['accum']} != {expected}")
        else:
            ref = (coeff.norm(dim=1) * (params["means"] - campos).norm(dim=1) / 2).detach()
            if not torch.allclose(dens.state["accum"] / 2, ref, rtol=1e-5):
                raise AssertionError("statistic is not ||grad|| * distance / 2")
            dens.after_backward(params, opts, step, {})
    # step 600: G0 small & hot -> clone, G1 big & hot -> split, G2 cold, G3 cold + pruned.
    count = len(params["means"])
    if count != 4 + 1 + 1 - 1:
        raise AssertionError(f"Expected 5 Gaussians after clone+split+prune, got {count}")
    sched = {s: (step_condition(s, 500, 15000, 300), step_condition(s, 500, 15000, 100),
                 step_condition(s, 0, 15000, 3000))
             for s in (100, 500, 600, 700, 900, 3000, 12000, 14700, 15000)}
    want = {100: (False, False, False), 500: (False, False, False), 600: (True, True, False),
            700: (False, True, False), 900: (True, True, False), 3000: (True, True, True),
            12000: (True, True, True), 14700: (True, True, False), 15000: (False, False, False)}
    if sched != want:
        raise AssertionError(f"Schedule mismatch {sched}")
    return {"gaussians_after_densify": count, "statistic": "||dL/dmu * dist|| / 2",
            "schedule_checked": sorted(want)}


def official_init(method, device):
    """set_training_cameras applies scale = 0.01 * nearest-camera distance, unit random quats."""
    comp = next(c for c in method.components if c.kind == "density")
    dens = Density(comp)
    if not dens.cfg["official_init"]:
        raise AssertionError("official_init disabled")
    means = torch.tensor([[0.0, 0, 3], [2.0, 0, 0], [0, 0, -4.0]], device=device)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means.clone()),
        "scales": torch.nn.Parameter(torch.zeros(3, 3, device=device)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * 3, device=device)),
    })
    dens.initialize(params, {}, 1.0)
    views = []
    for c in ([0.0, 0, 0], [2.0, 0, 0.5]):
        vm = torch.eye(4, device=device)
        vm[:3, 3] = -torch.tensor(c, device=device)
        views.append({"viewmat": vm})
    dens.set_training_cameras(views)
    want = torch.tensor([3.0, 0.5, 4.0], device=device) * 0.01
    got = params["scales"].exp()
    if not torch.allclose(got, want[:, None].expand(-1, 3), rtol=1e-5):
        raise AssertionError(f"init scales {got} != {want}")
    norms = params["quats"].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms)) or torch.allclose(
        params["quats"][:, 1:], torch.zeros(3, 3, device=device)
    ):
        raise AssertionError("quaternions not random unit quaternions")
    _, q = official_init_values(means, torch.zeros(1, 3, device=device), 0.01, 42)
    if not torch.allclose(q, params["quats"]):
        raise AssertionError("init rotations are not the seeded official draw")
    return {"scales": got[:, 0].tolist(), "quat_norms": norms.tolist()}

