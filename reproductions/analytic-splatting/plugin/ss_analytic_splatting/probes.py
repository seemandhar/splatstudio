"""Mechanism probes for the Analytic-Splatting port. Each raises on failure and returns
measurements."""

import math
from dataclasses import replace

import torch

from . import reference as ref

F_PIX = 100.0  # focal length of the probe camera (px)
Z = 5.0  # depth of the probe Gaussians -> 20 px per world unit


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _camera(device, size=16):
    K = torch.tensor([[F_PIX, 0, size / 2], [0, F_PIX, size / 2], [0, 0, 1]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": size, "height": size}


def _gaussians(spec, device, size=16):
    """spec rows: (px, py, sigma_major_px, sigma_minor_px, theta, opacity, rgb, depth)."""
    pxs = Z / F_PIX
    means, scales, quats, opac, sh0 = [], [], [], [], []
    for px, py, s1, s2, th, o, rgb, dz in spec:
        z = Z + dz
        # Inria pixel coordinate p = f x / z + c - 0.5
        means.append([(px - size / 2 + 0.5) * z / F_PIX, (py - size / 2 + 0.5) * z / F_PIX, z])
        scales.append([s1 * pxs * z / Z, s2 * pxs * z / Z, 1e-3])
        quats.append([math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)])
        opac.append(math.log(o / (1 - o)))
        sh0.append([[(c - 0.5) / ref.C0 for c in rgb]])
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    n = len(spec)
    return {"means": t(means), "scales": t(scales).log(), "quats": t(quats),
            "opacities": t(opac), "sh0": t(sh0), "shN": torch.zeros(n, 15, 3, device=device)}


def _kernel_render(params, camera, bg):
    from .raster import render

    color, means2d, radii = render(params, camera, 0, bg)
    return color.permute(1, 2, 0), means2d, radii


# ---------------------------------------------------------------------------- C1/C2 forward
FORWARD_CASES = {
    # name: (sigma_major, sigma_minor, theta) in px; centre at a sub-pixel offset
    "small": (0.45, 0.35, 0.6),
    "medium": (1.5, 0.8, 0.4),
    "anisotropic": (4.0, 1.0, 0.5),
}


def forward_integral(method=None, device="cuda"):
    """Kernel pixel response vs (a) independent Eq. 13, (b) quadrature, (c) 3DGS centre sample.

    One white Gaussian, black background: rendered value = min(0.99, o * I(u)) where it is
    >= 1/255. Fails if the kernel is not Eq. 13 or not closer to the true pixel integral than
    the point sample for a sub-pixel Gaussian (i.e. if the mechanism were removed).
    """
    _cuda(device)
    cam = _camera(device)
    bg = torch.zeros(3, device=device)
    o = 0.3
    out = {}
    for name, (s1, s2, th) in FORWARD_CASES.items():
        params = _gaussians([(7.3, 8.2, s1, s2, th, o, (1.0, 1.0, 1.0), 0.0)], device)
        with torch.no_grad():
            img = _kernel_render(params, cam, bg)[0][..., 0].double().cpu()
        cpu = {k: v.cpu() for k, v in params.items()}
        mu, cov2, _ = ref.project(cpu, {k: (v.cpu() if torch.is_tensor(v) else v)
                                        for k, v in cam.items()})
        ys, xs = torch.meshgrid(torch.arange(16, dtype=torch.float64),
                                torch.arange(16, dtype=torch.float64), indexing="ij")
        d = torch.stack([xs, ys], -1).reshape(-1, 2) - mu[0]
        c2 = cov2[0].expand(len(d), 2, 2)
        eq13 = (o * ref.window_integral_2d(d, c2)).reshape(16, 16)
        true = (o * ref.quadrature_2d(d, c2, 96)).reshape(16, 16)
        point = (o * ref.point_sample_2d(d, c2)).reshape(16, 16)
        mask = eq13 > 2.0 / 255.0  # away from the 1/255 skip threshold
        err_eq13 = float((img - eq13)[mask].abs().max() / eq13.max())
        e_kernel = float((img - true)[mask].abs().sum() / true[mask].sum())
        e_point = float((point - true)[mask].abs().sum() / true[mask].sum())
        out[name] = {"max_rel_vs_eq13": err_eq13, "l1_rel_vs_true_kernel": e_kernel,
                     "l1_rel_vs_true_point_sample": e_point,
                     "peak_kernel": float(img.max()), "peak_true": float(true.max()),
                     "peak_point": float(point.max())}
        if err_eq13 > 2e-3:
            raise AssertionError(f"{name}: kernel differs from Eq. 13 ({err_eq13:.2e})")
        if e_kernel > 0.05:
            raise AssertionError(f"{name}: kernel far from the pixel integral ({e_kernel:.3f})")
    small = out["small"]
    if not small["l1_rel_vs_true_kernel"] < 0.5 * small["l1_rel_vs_true_point_sample"]:
        raise AssertionError(f"sub-pixel Gaussian: analytic response not better than the centre "
                             f"sample ({small})")
    return out


# ---------------------------------------------------------------------------- C1/C2 backward
BACKWARD_SPEC = [  # px, py, s_major, s_minor, theta, opacity, rgb, depth offset
    (6.4, 7.2, 6.0, 4.0, 0.5, 0.45, (0.9, 0.3, 0.2), 0.0),
    (9.1, 8.6, 7.0, 4.5, -0.8, 0.5, (0.2, 0.8, 0.4), 0.3),
    (7.7, 9.3, 6.5, 5.0, 1.2, 0.4, (0.3, 0.4, 0.9), 0.6),
]
SMALL = (8.3, 6.6, 0.6, 0.35, 0.3, 0.3, (0.8, 0.8, 0.1), -0.4)
FIELDS = ("means", "scales", "quats", "opacities", "sh0")


def _loss(rgb, w):
    return (rgb.double() * w).sum()


def backward(method=None, device="cuda"):
    """Kernel gradients vs float64 autograd of the independent reference, and vs central
    finite differences of the kernel's own forward (3 composited Gaussians, one tile)."""
    _cuda(device)
    cam = _camera(device)
    camc = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cam.items()}
    bg = torch.tensor([0.1, 0.2, 0.3], device=device)
    g = torch.Generator().manual_seed(0)
    w = torch.rand(16, 16, 3, generator=g, dtype=torch.float64)
    out = {"autograd": {}, "finite_difference": {}}

    # (a) autograd reference, including a sub-pixel Gaussian
    base = _gaussians(BACKWARD_SPEC + [SMALL], device)
    params = {k: v.clone().requires_grad_(True) for k, v in base.items()}
    rgb, means2d, radii = _kernel_render(params, cam, bg)
    if not bool((radii > 0).all()):
        raise AssertionError("probe Gaussians culled")
    _loss(rgb, w.to(device)).backward()
    rp = {k: v.detach().cpu().double().requires_grad_(True) for k, v in base.items()}
    off = torch.zeros(len(rp["means"]), 2, dtype=torch.float64, requires_grad=True)
    _loss(ref.composite(rp, camc, bg.cpu(), off), w).backward()
    worst = 0.0
    for k in FIELDS:
        gk, gr = params[k].grad.detach().cpu().double(), rp[k].grad
        rel = float((gk - gr).norm() / gr.norm().clamp_min(1e-12))
        out["autograd"][k] = rel
        worst = max(worst, rel)
    # Inria means2D gradient: dL/d ndc = dL/d pixel * (W/2, H/2)
    g2 = means2d.grad[:, :2].detach().cpu().double()
    g2r = off.grad * torch.tensor([8.0, 8.0], dtype=torch.float64)
    out["autograd"]["means2d"] = float((g2 - g2r).norm() / g2r.norm())
    worst = max(worst, out["autograd"]["means2d"])
    if worst > 1e-3:
        raise AssertionError(f"kernel gradients differ from the reference: {out['autograd']}")

    # (b) central finite differences of the kernel (large Gaussians: no threshold crossings)
    base = _gaussians(BACKWARD_SPEC, device)
    params = {k: v.clone().requires_grad_(True) for k, v in base.items()}
    _loss(_kernel_render(params, cam, bg)[0], w.to(device)).backward()
    checks = [("means", (0, 0), 1e-2), ("means", (1, 1), 1e-2), ("scales", (0, 0), 1e-2),
              ("scales", (2, 1), 1e-2), ("quats", (1, 3), 1e-2), ("opacities", (2,), 1e-2),
              ("sh0", (0, 0, 0), 1e-2)]
    for k, idx, eps in checks:
        vals = []
        for sgn in (1, -1):
            p = {kk: vv.clone() for kk, vv in base.items()}
            p[k][idx] += sgn * eps
            with torch.no_grad():
                vals.append(float(_loss(_kernel_render(p, cam, bg)[0], w.to(device))))
        fd = (vals[0] - vals[1]) / (2 * eps)
        an = float(params[k].grad[idx])
        rel = abs(fd - an) / max(abs(fd), 1e-6)
        out["finite_difference"][f"{k}{list(idx)}"] = {"fd": fd, "kernel": an, "rel": rel}
        if rel > 5e-3:
            raise AssertionError(f"finite difference mismatch for {k}{list(idx)}: "
                                 f"fd {fd:.5g} vs kernel {an:.5g}")
    return out


# ---------------------------------------------------------------------------- C2 (CPU)
def logistic_cdf(method=None, device="cpu"):
    """Eq. 7 approximates the Gaussian CDF (paper: errors ~1e-4); closed-form derivative;
    Eq. 13 matches quadrature where the domain rotation is exact (axis-aligned)."""
    x = torch.linspace(-6, 6, 24001, dtype=torch.float64, requires_grad=True)
    S = ref.logistic_cdf(x)
    err_cdf = float((S - ref.gaussian_cdf(x)).abs().max())
    (gx,) = torch.autograd.grad(S.sum(), x)
    err_grad = float((gx - ref.logistic_cdf_grad(x.detach())).abs().max())
    # 1D window error over sigma in [0.3, 6.6] (paper Sec. 5.1)
    worst_win = 0.0
    for sig in (0.3, 0.5, 1.0, 2.0, 6.6):
        u = torch.linspace(0, 3 * sig, 301, dtype=torch.float64)
        win_s = ref.logistic_cdf((u + 0.5) / sig) - ref.logistic_cdf((u - 0.5) / sig)
        win_g = ref.gaussian_cdf((u + 0.5) / sig) - ref.gaussian_cdf((u - 0.5) / sig)
        worst_win = max(worst_win, float((win_s - win_g).abs().max()))
    # axis-aligned 2D: Eq. 13 / (2 pi s1 s2) is a product of 1D windows -> equals quadrature
    d = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 9, dtype=torch.float64),
                                   torch.linspace(-2, 2, 9, dtype=torch.float64),
                                   indexing="xy"), -1).reshape(-1, 2) + 0.17
    cov = torch.diag(torch.tensor([1.3 ** 2, 0.6 ** 2], dtype=torch.float64)).expand(len(d), 2, 2)
    e13 = ref.window_integral_2d(d, cov)
    quad = ref.quadrature_2d(d, cov, 128)
    err_2d = float((e13 - quad).abs().max())
    out = {"max_abs_cdf_err": err_cdf, "max_abs_dSdx_err": err_grad,
           "max_abs_window_err": worst_win, "max_abs_eq13_vs_quadrature_axis_aligned": err_2d}
    if err_cdf > 5e-4 or err_grad > 1e-9 or worst_win > 1e-3 or err_2d > 5e-3:
        raise AssertionError(f"logistic CDF approximation check failed: {out}")
    # S is not the plain logistic: the cubic term matters (removal raises the error 10x)
    plain = float((torch.sigmoid(1.6 * x.detach()) - ref.gaussian_cdf(x.detach())).abs().max())
    out["plain_logistic_err"] = plain
    if plain < 10 * err_cdf:
        raise AssertionError("cubic term has no effect")
    return out


# ---------------------------------------------------------------------------- 3DGS density
def _density(method, **config):
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")
    return create_density(replace(spec, config={**spec.config, **config}))


def _params(means, scales, opac):
    n = len(means)
    vals = {"means": torch.tensor(means, dtype=torch.float32),
            "scales": torch.tensor(scales, dtype=torch.float32).log(),
            "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
            "opacities": torch.logit(torch.tensor(opac, dtype=torch.float32)),
            "sh0": torch.rand(n, 1, 3), "shN": torch.rand(n, 15, 3)}
    params = {k: torch.nn.Parameter(v) for k, v in vals.items()}
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for k, p in params.items():
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return params, opts


def _step(dens, params, opts, step, grad2d, radii=None):
    n = len(params["means"])
    m2 = torch.zeros(n, 3, requires_grad=True)
    m2.grad = torch.zeros(n, 3)
    m2.grad[:, 0] = torch.as_tensor(grad2d, dtype=torch.float32)
    r = torch.ones(n, dtype=torch.int32) if radii is None else torch.as_tensor(radii)
    out = {"info": {"means2d": m2, "radii": r}}
    dens.before_backward(params, opts, step, out)
    dens.after_backward(params, opts, step, out)


def density_schedule(method, device="cpu"):
    """3DGS ADC kept: cadence, clone vs split, no screen-size prune, filter-aware reset."""
    cam = {"K": torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]]),
           "viewmat": torch.eye(4), "width": 64, "height": 64}
    dens = _density(method)
    dens.set_training_cameras([cam])
    # 0 small high-grad -> clone; 1 big high-grad -> split; 2 low-grad; 3 huge screen radius
    params, opts = _params([[0, 0, 4.0], [0.1, 0, 4.0], [0, 0.1, 4.0], [0.1, 0.1, 4.0]],
                           [[0.005] * 3, [0.05] * 3, [0.005] * 3, [0.005] * 3],
                           [0.5, 0.5, 0.5, 0.5])
    dens.initialize(params, opts, scene_scale=1.0)
    _step(dens, params, opts, 550, [1e-3, 1e-3, 1e-5, 0.0], radii=[1, 1, 1, 500])
    n550 = len(params["means"])
    _step(dens, params, opts, 600, [1e-3, 1e-3, 1e-5, 0.0], radii=[1, 1, 1, 500])
    ev = dens.state["events"][-1]
    out = {"n_after_550": n550, "event_600": ev}
    if n550 != 4 or len(dens.state["events"]) != 1:
        raise AssertionError("densified outside the 100-step cadence")
    if (ev["clone"], ev["split"], ev["pruned"]) != (1, 1, 0) or ev["n_after"] != 6:
        raise AssertionError(f"unexpected clone/split/prune counts: {ev}")
    # opacity reset at 3000: o <- min(o, 0.01 / coef), coef from the 3D filter
    with torch.no_grad():
        params["opacities"].data[:] = torch.logit(torch.tensor(0.6))
    f = dens.filter_3d(params).double()
    s2 = params["scales"].detach().double().exp() ** 2
    coef = torch.sqrt(s2.prod(1) / (s2 + f[:, None] ** 2).prod(1))
    expect = torch.minimum(torch.full_like(coef, 0.6), 0.01 / coef)
    dens.grad_accum.zero_()
    _step(dens, params, opts, 3000, [0.0] * len(params["means"]))
    got = params["opacities"].detach().double().sigmoid()
    if len(got) != len(expect):
        raise AssertionError("population changed at the reset step")
    err = float((got - expect).abs().max())
    out.update({"reset_max_err": err, "reset_values": got.tolist(),
                "plain_3dgs_reset_would_be": 0.01})
    if err > 1e-4 or not bool((got > 0.0101).any()):
        raise AssertionError(f"filter-aware opacity reset wrong (err {err:.2e}, {got.tolist()})")
    n = len(params["means"])
    _step(dens, params, opts, 15000, [1.0] * n)
    if len(params["means"]) != n:
        raise AssertionError("densified at/after densify_until")
    return out
