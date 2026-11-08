"""Mechanism probes for Optimal Gaussian Splatting. Each raises when its mechanism is removed
or wrong."""

import torch

from .kernels import rasterize_sphere
from .projection import (
    ewa_jacobian,
    official_jacobian_coefficients,
    tangent_jacobian,
    tangent_jacobian_exact,
)
from .reference import (
    dense_render,
    exact_ray_response,
    official_jacobian,
    paper_jacobian,
    tile_mask,
)
from .renderer import OPTIONS, project


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return "cuda"


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


def random_means(n=400, seed=0, dtype=torch.float64, spread=1.2):
    g = torch.Generator().manual_seed(seed)
    z = 1.0 + 4.0 * torch.rand(n, generator=g, dtype=dtype)
    xy = (torch.rand(n, 2, generator=g, dtype=dtype) * 2 - 1) * spread * z[:, None]
    return torch.cat((xy, z[:, None]), -1)


# --------------------------------------------------------------------------- C1 Jacobian
def tangent_jacobian_probe(method, device):
    t = random_means()
    fx, fy = 900.0, 850.0
    J = tangent_jacobian_exact(t, fx, fy)
    err_official = _rel(J, official_jacobian(t, fx, fy))
    err_paper = _rel(J, paper_jacobian(t, fx, fy))
    # on the optical axis the tangent plane is the image plane: J_p == EWA J
    axis = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    err_axis = _rel(tangent_jacobian_exact(axis, fx, fy), ewa_jacobian(axis, fx, fy, 1e9, 1e9))
    # off axis (~40-50 deg) they differ
    off = torch.tensor([[2.0, 1.5, 2.0], [-1.8, 0.4, 2.0]], dtype=torch.float64)
    diff_off = _rel(tangent_jacobian_exact(off, fx, fy), ewa_jacobian(off, fx, fy, 1e9, 1e9))
    out = {"rel_err_vs_official_formula": err_official, "rel_err_vs_paper_QJp": err_paper,
           "on_axis_vs_ewa": err_axis, "off_axis_vs_ewa": diff_off}
    if err_official > 1e-6 or err_paper > 1e-9 or err_axis > 1e-12 or diff_off < 0.1:
        raise AssertionError(f"tangent-plane Jacobian check failed: {out}")
    return out


def cov_gradient_probe(method, device):
    t = random_means(200, seed=1)
    fx, fy = 900.0, 850.0
    tt = t.clone().requires_grad_(True)
    Jx = tangent_jacobian_exact(tt, fx, fy)
    auto = torch.stack([torch.stack([torch.autograd.grad(Jx[:, i, k].sum(), tt,
                                                         retain_graph=True)[0]
                                     for k in range(3)]) for i in range(2)])  # [2,3,N,3]
    auto = auto.permute(2, 0, 1, 3)  # [N,2,3,axis]
    coef = official_jacobian_coefficients(t, fx, fy)
    errs = {}
    for i in range(2):
        for k in range(3):
            for a, ax in enumerate("xyz"):
                errs[f"j{i}{k}_{ax}"] = _rel(coef[:, i, k, a], auto[:, i, k, a]) \
                    if auto[:, i, k, a].abs().max() > 0 else float(coef[:, i, k, a].abs().max())
    bad = {k: v for k, v in errs.items() if v > 1e-6}
    if set(bad) != {"j12_x"}:
        raise AssertionError(f"official coefficients: expected only j12_x (typo) to differ: {bad}")
    # the custom backward is used when official_cov_grad=True and is exact otherwise
    g = torch.randn(t.shape[0], 2, 3, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(3))
    t1 = t.clone().requires_grad_(True)
    (tangent_jacobian(t1, fx, fy, True) * g).sum().backward()
    t2 = t.clone().requires_grad_(True)
    (tangent_jacobian(t2, fx, fy, False) * g).sum().backward()
    want_official = (g[..., None] * coef).sum((1, 2))
    e1 = _rel(t1.grad, want_official)
    diff = _rel(t1.grad, t2.grad)
    out = {"j12_x_rel_diff": errs["j12_x"], "max_other_rel_err":
           max(v for k, v in errs.items() if k != "j12_x"),
           "official_backward_err": e1, "official_vs_exact_grad_rel": diff}
    if e1 > 1e-9 or diff <= 0:
        raise AssertionError(f"official covariance backward not wired: {out}")
    return out


# --------------------------------------------------------------------------- C2 rasterizer
def synthetic_scene(n=250, seed=0, device="cuda", spread=1.1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = 2.5 + torch.rand(n, generator=g) * 2.0
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * spread * z,
                         (torch.rand(n, generator=g) * 2 - 1) * 0.8 * spread * z, z], dim=-1)
    params = {
        "means": means,
        "scales": -2.6 + torch.rand(n, 3, generator=g) * 1.0,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g) * 1.5,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
    }
    return {k: v.to(device) for k, v in params.items()}


def camera(width, height, f_ratio=0.6, device="cuda"):
    """Wide field of view (f = 0.6 W, ~80 deg horizontal) so the projections differ."""
    f = f_ratio * width
    K = torch.tensor([[f, 0, width / 2 + 0.7], [0, 1.05 * f, height / 2 - 0.4], [0, 0, 1.0]],
                     device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": width, "height": height}


SIZES = ((64, 48), (61, 37))


def _inputs(params, cam, mode):
    cfg = {**OPTIONS, "projection": mode, "focal_floor": 20.0}
    with torch.no_grad():
        colors, meta = project(params, cam, 3, cfg)
    feats = torch.cat((colors, meta["depths"][0][..., None]), -1)
    leaves = {"means2d": meta["means2d"][0], "conics": meta["conics"], "colors": feats,
              "opacities": meta["opacities"]}
    leaves = {k: v.detach().clone().contiguous() for k, v in leaves.items()}
    return leaves, meta, cfg


def _cuda_render(leaves, meta, cam, cfg, grad=False):
    x = {k: v.clone().requires_grad_(grad) for k, v in leaves.items()}
    rgb, alpha = rasterize_sphere(
        x["means2d"], x["conics"], x["colors"], x["opacities"], meta["isect_offsets"][0],
        meta["flatten_ids"], meta["width"], meta["height"], cam["K"],
        focal_floor=cfg["focal_floor"], mode=cfg["projection"], alpha_max=cfg["alpha_max"])
    return x, rgb, alpha


def _dense(leaves, meta, cam, cfg, grad=False):
    x = {k: v.double().clone().requires_grad_(grad) for k, v in leaves.items()}
    mask = tile_mask(x["means2d"].detach(), meta["radii"][0], meta["width"], meta["height"])
    rgb, alpha = dense_render(x["means2d"], x["conics"], x["colors"], x["opacities"],
                              meta["depths"][0].double(), mask, cam["K"].double().cpu(),
                              meta["width"], meta["height"], focal_floor=cfg["focal_floor"],
                              mode=cfg["projection"], alpha_max=cfg["alpha_max"])
    return x, rgb, alpha


def raster_forward_probe(method, device):
    dev = _cuda()
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        cam = camera(w, h, device=dev)
        res = {}
        renders = {}
        for mode in ("optimal", "ewa"):
            leaves, meta, cfg = _inputs(params, cam, mode)
            _, rgb, alpha = _cuda_render(leaves, meta, cam, cfg)
            _, rgb_ref, alpha_ref = _dense(leaves, meta, cam, cfg)
            res[mode] = {"max_err_rgbd": float((rgb.double() - rgb_ref).abs().max()),
                         "max_err_alpha": float((alpha.double() - alpha_ref).abs().max()),
                         "mean_alpha": float(alpha.mean())}
            renders[mode] = rgb[..., :3]
            scale = max(1.0, float(rgb_ref.abs().max()))
            if res[mode]["max_err_rgbd"] > 2e-3 * scale or res[mode]["max_err_alpha"] > 2e-3:
                raise AssertionError(f"CUDA render != dense reference ({mode}, {w}x{h}): {res}")
        res["optimal_vs_ewa_maxdiff"] = float((renders["optimal"] - renders["ewa"]).abs().max())
        if res["optimal_vs_ewa_maxdiff"] < 0.05:
            raise AssertionError(f"optimal render indistinguishable from EWA: {res}")
        out[f"{w}x{h}"] = res
    return out


def raster_backward_probe(method, device):
    dev = _cuda()
    params = synthetic_scene(n=120, seed=2, device=dev)
    out = {}
    for w, h in SIZES:
        cam = camera(w, h, device=dev)
        for mode in ("optimal", "ewa"):
            leaves, meta, cfg = _inputs(params, cam, mode)
            g = torch.Generator(device="cpu").manual_seed(5)
            v_rgb = torch.randn(h, w, 4, generator=g).to(dev)
            v_a = torch.randn(h, w, 1, generator=g).to(dev)
            x, rgb, alpha = _cuda_render(leaves, meta, cam, cfg, grad=True)
            ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
            xr, rgb_r, alpha_r = _dense(leaves, meta, cam, cfg, grad=True)
            ((rgb_r * v_rgb.double()).sum() + (alpha_r * v_a.double()).sum()).backward()
            errs = {k: _rel(x[k].grad, xr[k].grad) for k in x}
            out[f"{mode}-{w}x{h}"] = errs
            if any(e > 5e-3 for e in errs.values()):
                raise AssertionError(f"CUDA backward != dense autograd ({mode}): {errs}")
    return out


# --------------------------------------------------------------------------- mechanism
def projection_error_probe(method, device):
    """Off-axis splats under a wide-FOV camera: the optimal projection's rendered alpha is
    closer to the exact ray-maximum of the 3D Gaussian than the EWA projection's (Sec. 4-5)."""
    dev = _cuda()
    W, H = 160, 120
    f = 0.35 * W  # ~132 deg horizontal FOV
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=dev)
    cam = {"viewmat": torch.eye(4, device=dev), "K": K, "width": W, "height": H}
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    cases = [(1.6, 0.9, 1.0), (-1.4, -0.8, 1.0), (1.2, -0.2, 1.0), (0.05, 0.05, 1.0)]
    g = torch.Generator().manual_seed(7)
    errs = {"optimal": [], "ewa": []}
    for i, (x, y, z) in enumerate(cases):
        mean = torch.tensor([[x, y, z]]) * 3.0
        quat = torch.nn.functional.normalize(torch.randn(1, 4, generator=g), dim=-1)
        params = {"means": mean.to(dev), "quats": quat.to(dev),
                  "scales": torch.log(torch.tensor([[0.25, 0.12, 0.18]])).to(dev),
                  "opacities": torch.logit(torch.tensor([0.98])).to(dev),
                  "sh0": torch.zeros(1, 1, 3, device=dev), "shN": torch.zeros(1, 15, 3, device=dev)}
        cov = quat_scale_to_covar_preci(params["quats"], params["scales"].exp(), True, False,
                                        triu=False)[0][0].double().cpu()
        exact = 0.98 * exact_ray_response(mean[0].double(), cov, K.double().cpu(), W, H)
        for mode in ("optimal", "ewa"):
            cfg = {**OPTIONS, "projection": mode, "eps2d": 0.0}
            with torch.no_grad():
                _, meta = project(params, cam, 0, cfg)
            ones = torch.ones(1, H, W, dtype=torch.bool)
            _, alpha = dense_render(
                meta["means2d"][0].double().cpu(), meta["conics"].double().cpu(),
                torch.ones(1, 3, dtype=torch.float64), meta["opacities"].double().cpu(),
                meta["depths"][0].double().cpu(), ones, K.double().cpu(), W, H,
                focal_floor=cfg["focal_floor"], mode=mode)
            errs[mode].append(float((alpha[..., 0] - exact).abs().sum() / exact.sum()))
    out = {"rel_l1_optimal": errs["optimal"], "rel_l1_ewa": errs["ewa"]}
    off = range(3)  # off-axis cases
    if not all(errs["optimal"][i] < 0.5 * errs["ewa"][i] for i in off):
        raise AssertionError(f"optimal projection not closer to the exact response: {out}")
    if errs["optimal"][3] > 0.02 or errs["ewa"][3] > 0.02:
        raise AssertionError(f"near the axis both projections should be accurate: {out}")
    return out
