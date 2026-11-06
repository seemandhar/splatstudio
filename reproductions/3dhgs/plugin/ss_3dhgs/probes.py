"""Mechanism probes for the 3D-HGS port. Each raises on failure and returns measurements."""

import math
from dataclasses import replace

import torch

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _camera(W=160, H=128, f=160.0, angle=0.3, t=(0.2, -0.1, 3.0)):
    R = torch.tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                      [-math.sin(angle), 0, math.cos(angle)]])
    V = torch.eye(4)
    V[:3, :3] = R
    V[:3, 3] = torch.tensor(t)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return {"viewmat": V, "K": K, "width": W, "height": H}


def _scene(n=12, seed=0, omin=0.1, omax=0.45, smin=0.06, sspan=0.15):
    g = torch.Generator().manual_seed(seed)

    def u(*shape):
        return torch.rand(*shape, generator=g)

    return {"means": (u(n, 3) - 0.5) * torch.tensor([1.2, 1.0, 0.6]),
            "scales": (u(n, 3) * sspan + smin).log(),
            "quats": torch.randn(n, 4, generator=g),
            "opacities": torch.logit(u(n) * (omax - omin) + omin),
            "opacity2": torch.logit(u(n) * (omax - omin) + omin),
            "sh0": u(n, 1, 3) - 0.2, "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
            "hgs_normal": torch.randn(n, 3, generator=g)}


def _to(d, device, dtype, grad=False):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.is_floating_point():
            v = v.to(device=device, dtype=dtype)
            out[k] = torch.nn.Parameter(v) if grad else v
        else:
            out[k] = v
    return out


def _kernel(p, cam, device, degree=3, grad=False, cull=False, opacities2=None):
    from . import raster

    P = _to(p, device, torch.float32, grad)
    c = _to(cam, device, torch.float32)
    bg = torch.zeros(3, device=device)
    with torch.set_grad_enabled(grad):
        if opacities2 is None:
            col, _, radii = raster.render(P, c, degree, bg, cull)
        else:
            sh = torch.cat((P["sh0"], P["shN"]), 1)
            col, _, radii = raster.rasterize(
                P["means"], P["hgs_normal"], sh, opacities2.to(device), P["scales"].exp(),
                torch.nn.functional.normalize(P["quats"], dim=-1), c, degree, bg, cull)
    return P, col.permute(1, 2, 0), radii


def _reference(p, cam, degree=3, grad=False, **kw):
    from . import reference

    P = _to(p, "cpu", torch.float64, grad)
    c = _to(cam, "cpu", torch.float64)
    with torch.set_grad_enabled(grad):
        return P, reference.render(P, c, degree, **kw)


# ---------------------------------------------------------------------------- C1
def equal_opacity(method, device):
    """alpha1 = alpha2 reduces Eq. 10 to alpha * G: the half-Gaussian kernel must equal a plain
    3DGS compositor and ignore the plane normal."""
    _cuda(device)
    cam = _camera()
    p = _scene(seed=2, omax=0.9)
    p["opacity2"] = p["opacities"].clone()
    _, rgb, radii = _kernel(p, cam, device)
    _, ref_full = _reference(p, cam, kernel="full")
    d_full = (rgb.cpu().double() - ref_full).abs().max().item()
    if d_full > 1e-4:
        raise AssertionError(f"equal halves differ from the 3DGS compositor: {d_full:.2e}")
    q = dict(p)
    q["hgs_normal"] = torch.randn(len(p["means"]), 3, generator=torch.Generator().manual_seed(9))
    _, rgb_n, _ = _kernel(q, cam, device)
    d_normal = (rgb - rgb_n).abs().max().item()
    if d_normal > 1e-6:
        raise AssertionError(f"equal-opacity render depends on the normal: {d_normal:.2e}")
    r = dict(p)
    r["opacity2"] = torch.full_like(p["opacities"], -6.0)
    _, rgb_h, _ = _kernel(r, cam, device)
    d_half = (rgb - rgb_h).abs().max().item()
    if d_half < 1e-2:
        raise AssertionError("making one half transparent does not change the render")
    # gsplat 3DGS (3-sigma support, no -4.8 power cutoff) for context
    import gsplat

    P = _to(p, device, torch.float32)
    c = _to(cam, device, torch.float32)
    g_rgb, _, _ = gsplat.rasterization(
        P["means"], torch.nn.functional.normalize(P["quats"], dim=-1), P["scales"].exp(),
        P["opacities"].sigmoid(), torch.cat([P["sh0"], P["shN"]], 1), c["viewmat"][None],
        c["K"][None], cam["width"], cam["height"], sh_degree=3, eps2d=0.3)
    d_gsplat = (g_rgb[0] - rgb).abs()
    if d_gsplat.mean().item() > 5e-3:
        raise AssertionError(f"equal-opacity render far from gsplat 3DGS: {d_gsplat.mean():.2e}")
    return {"max_abs_vs_float64_3dgs": d_full, "normal_invariance_max_abs": d_normal,
            "one_half_transparent_max_abs_change": d_half,
            "vs_gsplat_mean_abs": d_gsplat.mean().item(),
            "vs_gsplat_max_abs": d_gsplat.max().item(),
            "visible": int((radii > 0).sum()), "image_mean": rgb.mean().item()}


def _half_probability(cov, normal, u, v, n_w=4001):
    """P(n . (u, v, w) <= 0 | u, v) for (u, v, w) ~ N(0, cov) by trapezoid quadrature over w."""
    cov = cov.double()
    prec = torch.linalg.inv(cov)
    sw = math.sqrt(cov[2, 2].item())
    w = torch.linspace(-12 * sw, 12 * sw, n_w, dtype=torch.float64)
    U, V, Wg = u[..., None], v[..., None], w
    q = (prec[0, 0] * U * U + prec[1, 1] * V * V + prec[2, 2] * Wg * Wg
         + 2 * prec[0, 1] * U * V + 2 * prec[0, 2] * U * Wg + 2 * prec[1, 2] * V * Wg)
    dens = torch.exp(-0.5 * (q - q.min(dim=-1, keepdim=True).values))
    side = (normal[0] * U + normal[1] * V + normal[2] * Wg) <= 0
    return torch.trapezoid(dens * side, w, dim=-1) / torch.trapezoid(dens, w, dim=-1)


def half_plane(method, device):
    """Analytic discontinuity: one Gaussian, one opaque half. The kernel's erf factor must equal
    a numerical integration of the ray-space 3D Gaussian over the half-space (Eq. 7-8)."""
    _cuda(device)
    from .reference import project

    cam = _camera(W=96, H=96, f=96.0, angle=0.0, t=(0.0, 0.0, 0.0))  # identity: n_cam = n
    p = {"means": torch.tensor([[0.05, -0.03, 3.0]]),
         "scales": torch.tensor([[0.35, 0.25, 0.12]]).log(),
         "quats": torch.tensor([[0.9, 0.2, -0.3, 0.1]]),
         "opacities": torch.zeros(1), "opacity2": torch.zeros(1),
         "sh0": torch.full((1, 1, 3), 0.5 / C0), "shN": torch.zeros(1, 15, 3),
         "hgs_normal": torch.tensor([[0.8, -0.3, 0.5]])}
    a1 = 0.9
    ops = torch.tensor([[a1, 0.0]])
    _, rgb, _ = _kernel(p, cam, device, degree=0, opacities2=ops)
    _, rgb_sw, _ = _kernel(p, cam, device, degree=0, opacities2=ops.flip(-1))
    alpha, alpha_sw = rgb[..., 0].cpu().double(), rgb_sw[..., 0].cpu().double()

    P = _to(p, "cpu", torch.float64)
    xy, cov, n_cam, _ = project(P["means"], P["scales"].exp(), P["quats"], P["hgs_normal"],
                                _to(cam, "cpu", torch.float64))
    W, H = cam["width"], cam["height"]
    py, px = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    u, v = px - xy[0, 0], py - xy[0, 1]  # pixel offset from the projected centre
    S2 = cov[0, :2, :2]
    con = torch.linalg.inv(S2)
    power = -0.5 * (con[0, 0] * u * u + 2 * con[0, 1] * u * v + con[1, 1] * v * v)
    G = torch.exp(power)
    prob = _half_probability(cov[0], n_cam[0], u, v)
    expected = (a1 * G * prob).clamp_max(0.99)
    region = (power > -4.5) & (expected > 2 / 255)
    err = (alpha - expected).abs()[region].max().item()
    if err > 2e-3:
        raise AssertionError(f"half-Gaussian alpha differs from the quadrature: {err:.2e}")
    # discontinuity: inside 2 sigma both a nearly-full and a nearly-empty half are present
    inner = power > -2.0
    opaque = int((inner & (prob > 0.99)).sum())
    empty = int((inner & (prob < 0.01)).sum())
    if opaque == 0 or empty == 0:
        raise AssertionError("test case does not exercise both sides of the splitting plane")
    leak = alpha[inner & (prob < 1e-4)].max().item()
    if leak > 1e-3:
        raise AssertionError(f"transparent half renders alpha {leak:.2e}")
    unclamped = region & (a1 * G < 0.98) & (alpha > 2 / 255) & (alpha_sw > 2 / 255)
    comp = (alpha + alpha_sw - a1 * G)[unclamped].abs().max().item()
    if comp > 2e-3:
        raise AssertionError(f"swapped halves are not complementary: {comp:.2e}")
    return {"max_abs_vs_quadrature": err, "pixels_checked": int(region.sum()),
            "opaque_side_pixels": opaque, "transparent_side_pixels": empty,
            "transparent_side_max_alpha": leak, "complement_max_abs": comp,
            "convention": "alpha1 on the side n_cam . (u, v, w) <= 0 in the kernel ray space"}


# ---------------------------------------------------------------------------- C2
def _grad_compare(p, cam, device, weights):
    Pk, rgb, _ = _kernel(p, cam, device, grad=True)
    (rgb * weights.to(device)).sum().backward()
    Pr, ref = _reference(p, cam, grad=True)
    (ref * weights.double()).sum().backward()
    # float32 vs float64 can flip the 1/255 / power -4.8 / T thresholds at isolated pixels:
    # report the max, gate on the fraction of pixels off by more than 1e-4.
    d = (rgb.detach().cpu().double() - ref.detach()).abs().max(-1).values
    fwd = {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
           "frac_pixels_gt_1e-4": (d > 1e-4).double().mean().item()}
    rel = {}
    for k in Pk:
        a = Pk[k].grad.cpu().double()
        b = Pr[k].grad
        rel[k] = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    return fwd, rel


def gradients(method, device):
    """Vendored backward vs float64 autograd of the independent compositor."""
    _cuda(device)
    cam = _camera()
    weights = torch.rand(cam["height"], cam["width"], 3,
                         generator=torch.Generator().manual_seed(5))
    fwd, rel = _grad_compare(_scene(seed=0, omax=0.45), cam, device, weights)
    bad = {k: v for k, v in rel.items() if v > 1e-3}
    if fwd["frac_pixels_gt_1e-4"] > 5e-4 or fwd["mean_abs"] > 1e-5 or bad:
        raise AssertionError(f"kernel disagrees with float64 reference: fwd {fwd}, {bad}")
    fwd_hi, rel_hi = _grad_compare(_scene(seed=1, omax=0.95), cam, device, weights)
    if fwd_hi["frac_pixels_gt_1e-4"] > 5e-4 or max(rel_hi.values()) > 0.1:
        raise AssertionError(f"high-opacity case grossly off: fwd {fwd_hi}, {rel_hi}")
    return {"forward_max_abs": fwd, "relative_grad_error": rel,
            "high_opacity_forward": fwd_hi,
            "high_opacity_relative_grad_error": rel_hi,
            "note": "high-opacity gradient error comes from the official backward not applying "
                    "the forward's power < -4.8 skip (kept unmodified)"}


def test_cull(method, device):
    """Official test path (non-empty cov3D_precomp_small) switches on half-aware rectangles."""
    _cuda(device)
    from .renderer import create_renderer

    cam = _camera()
    p = _scene(seed=3)
    g = torch.Generator().manual_seed(3)
    p["scales"] = (torch.rand(len(p["means"]), 3, generator=g) * 0.5 + 0.1).log()
    p["opacities"] = torch.logit(torch.rand(len(p["means"]), generator=g) * 0.5 + 0.45)
    p["opacity2"] = torch.full_like(p["opacities"], -6.0)
    _, train_path, _ = _kernel(p, cam, device)
    _, cull_path, _ = _kernel(p, cam, device, cull=True)
    diff = (train_path - cull_path).abs()
    if diff.max().item() == 0:
        raise AssertionError("half-aware test-time rectangles have no effect on this scene")
    if diff.mean().item() > 5e-3:
        raise AssertionError(f"test-time culling changes the image too much: {diff.mean():.2e}")
    spec = next(c for c in method.components if c.kind == "renderer")
    renderer = create_renderer(spec)
    P = _to(p, device, torch.float32, grad=True)
    c = _to(cam, device, torch.float32)
    out = renderer.render(P, c, sh_degree=3)
    with torch.no_grad():
        out_eval = renderer.render(P, c, sh_degree=3)
    if out["info"]["test_time_cull"] or not out_eval["info"]["test_time_cull"]:
        raise AssertionError("test-time culling must be off in training and on in evaluation")
    if (out_eval["rgb"] - cull_path).abs().max().item() > 1e-6:
        raise AssertionError("evaluation render does not use the culling path")
    if (out["rgb"].detach() - train_path).abs().max().item() > 1e-6:
        raise AssertionError("training render does not use the training path")
    return {"cull_vs_train_max_abs": diff.max().item(), "cull_vs_train_mean_abs":
            diff.mean().item()}


# ---------------------------------------------------------------------------- C3
def _params(rows, device="cpu"):
    """rows: list of (mean, scale, alpha1, alpha2, normal)."""
    n = len(rows)
    t = torch.tensor
    vals = {"means": t([r[0] for r in rows], dtype=torch.float32),
            "scales": t([[r[1]] * 3 for r in rows], dtype=torch.float32).log(),
            "quats": t([[1.0, 0, 0, 0]] * n),
            "opacities": torch.logit(t([r[2] for r in rows])),
            "opacity2": torch.logit(t([r[3] for r in rows])),
            "hgs_normal": t([r[4] for r in rows], dtype=torch.float32),
            "sh0": torch.rand(n, 1, 3), "shN": torch.rand(n, 15, 3)}
    params = {k: torch.nn.Parameter(v.to(device)) for k, v in vals.items()}
    opts = {k: torch.optim.Adam([p], lr=0.05 if "opac" in k else 1e-3, eps=1e-15)
            for k, p in params.items()}
    for k, p in params.items():  # allocate Adam moments, then restore the values
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return params, opts


def _outputs(n, grad2d, device="cpu"):
    m2 = torch.zeros(n, 3, device=device, requires_grad=True)
    m2.grad = torch.zeros(n, 3, device=device)
    m2.grad[:, 0] = torch.as_tensor(grad2d, dtype=torch.float32, device=device)
    return {"info": {"means2d": m2, "radii": torch.ones(n, dtype=torch.int32, device=device)}}


def density(method, device):
    from .core import clone_opacity_logit
    from .density import create_density

    dev = "cpu"
    spec = next(c for c in method.components if c.kind == "density")
    dens = create_density(replace(spec, config=dict(spec.config)))
    rows = [([0, 0, 0], 0.005, 0.6, 0.3, [1.0, 0, 0]),  # A small, high grad -> clone
            ([1, 0, 0], 0.05, 0.7, 0.2, [0, 1.0, 0]),  # B big, high grad -> split
            ([2, 0, 0], 0.005, 0.5, 0.001, [0, 0, 1.0]),  # C low grad, one opaque half: keep
            ([3, 0, 0], 0.005, 0.005, 0.008, [1.0, 1, 0])]  # D both halves < 0.01: prune
    params, opts = _params(rows, dev)
    dens.initialize(params, opts, scene_scale=1.0)  # extent 1 -> percent_dense limit 0.01
    before = {k: v.detach().clone() for k, v in params.items()}
    out = _outputs(4, [1e-3, 1e-3, 0.0, 0.0], dev)
    dens.after_backward(params, opts, 600, out)
    n = len(params["means"])
    if n != 5:  # A, C, clone(A), 2 x split(B)
        raise AssertionError(f"expected 5 Gaussians after densify/prune, got {n}")
    # survivors keep order: A, C, clone A, split B x2
    torch.testing.assert_close(params["opacities"][0], before["opacities"][0])
    torch.testing.assert_close(params["opacity2"][0], before["opacity2"][0])
    for k in ("opacities", "opacity2"):
        torch.testing.assert_close(params[k][2], clone_opacity_logit(before[k][0]))
        torch.testing.assert_close(params[k][3:], before[k][1].expand(2))
    expect = 1 - math.sqrt(1 - 0.6)
    if abs(params["opacities"][2].sigmoid().item() - expect) > 1e-5:
        raise AssertionError("clone opacity is not 1 - sqrt(1 - alpha)")
    torch.testing.assert_close(params["hgs_normal"][2], before["hgs_normal"][0])
    torch.testing.assert_close(params["hgs_normal"][3:], before["hgs_normal"][1].expand(2, 3))
    torch.testing.assert_close(params["scales"][3:].exp(),
                               before["scales"][1].exp().expand(2, 3) / 1.6)
    torch.testing.assert_close(params["hgs_normal"][1], before["hgs_normal"][2])  # C kept
    for k, p in params.items():
        m = opts[k].state[p]["exp_avg"]
        if torch.count_nonzero(m[2:]):
            raise AssertionError(f"{k}: appended Gaussians must start with zero Adam moments")
    # opacity reset at 3000: both halves -> min(alpha, 0.02), moments zeroed
    out = _outputs(n, [0.0] * n, dev)
    a2_before = params["opacity2"].sigmoid().detach().clone()
    dens.after_backward(params, opts, 3000, out)
    for k in ("opacities", "opacity2"):
        if params[k].sigmoid().max().item() > 0.02 + 1e-6:
            raise AssertionError(f"{k} not reset to 0.02")
        if torch.count_nonzero(opts[k].state[params[k]]["exp_avg"]):
            raise AssertionError(f"{k} moments not zeroed at reset")
    small = a2_before < 0.02
    torch.testing.assert_close(params["opacity2"].sigmoid()[small], a2_before[small])
    # opacity LR decay at 5000 for both halves; normal LR untouched
    lr0 = {k: opts[k].param_groups[0]["lr"] for k in opts}
    dens.after_backward(params, opts, 5000, _outputs(len(params["means"]), [0.0] *
                                                     len(params["means"])))
    for k in opts:
        ratio = lr0[k] / opts[k].param_groups[0]["lr"]
        want = 1.4 if k in ("opacities", "opacity2") else 1.0
        if abs(ratio - want) > 1e-9:
            raise AssertionError(f"{k}: LR ratio {ratio} != {want} at step 5000")
    # densification window: 19900 densifies, 20000 does not touch topology/statistics
    m = len(params["means"])
    dens.after_backward(params, opts, 20000, _outputs(m, [1.0] * m))
    if len(params["means"]) != m or dens.denom.sum().item() != 0:
        raise AssertionError("densification active at step 20000")
    dens.after_backward(params, opts, 19900, _outputs(m, [1.0] * m))
    if len(params["means"]) == m:
        raise AssertionError("no densification at step 19900")
    return {"after_first_densify": n, "clone_opacity_alpha1": expect,
            "events": dens.state["events"], "opacity_lr": dens.state["opacity_lr"]}
