"""Mechanism probes for the RaDe-GS port. Each raises on failure and returns measurements.

References are written independently of the kernel: exact perspective ray / Gaussian maximum
(paper Eq. 6) in float64, plane normals from its derivatives, the 2DGS distortion sum, and
float64 central finite differences of the forward kernel for the backward.
"""

import math

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
NEAR, FAR = 0.2, 100.0  # kernel NEAR_PLANE / FAR_PLANE (auxiliary.h)


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _camera(device, w=64, h=64, f=60.0):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device),
            "width": w, "height": h}


def _quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def _params(means, scales, quats, opac, device, sh0=None):
    n = len(means)
    if sh0 is None:
        sh0 = torch.full((n, 1, 3), 0.3)
    p = {"means": means, "scales": torch.log(scales), "quats": quats,
         "opacities": torch.logit(opac), "sh0": sh0, "shN": torch.zeros(n, 15, 3),
         "filter_3d": torch.zeros(n)}
    return {k: torch.nn.Parameter(v.float().to(device)) for k, v in p.items()}


def _one_gaussian(device, flat=False):
    """A tilted anisotropic Gaussian slightly off-axis (filter_3d = 0)."""
    q = F.normalize(torch.tensor([[0.9, 0.35, -0.25, 0.15]]), dim=-1)
    s = torch.tensor([[0.30, 0.18, 1e-4 if flat else 0.06]])
    return _params(torch.tensor([[0.15, -0.1, 3.0]]), s, q, torch.tensor([0.9]), device)


def _maps(params, camera, grad=False):
    from .raster import activated, raster_settings, rasterize

    with torch.set_grad_enabled(grad):
        means2D = torch.zeros_like(params["means"], requires_grad=grad) + 0
        if grad:
            means2D.retain_grad()
        scales, opac, rots, sh = activated(params)
        s = raster_settings(camera, torch.zeros(3, device=params["means"].device), 0)
        out = rasterize(params["means"], means2D, sh, opac[:, None], scales, rots, s)
    return out, means2D


def _ray(camera, px, py, dd=torch.float64):
    K = camera["K"].to(dd)
    W, H = int(camera["width"]), int(camera["height"])
    px = torch.as_tensor(px, dtype=dd, device=K.device)
    py = torch.as_tensor(py, dtype=dd, device=K.device)
    return torch.stack([(px + 0.5 - W / 2) / K[0, 0], (py + 0.5 - H / 2) / K[1, 1],
                        torch.ones_like(px)], -1)


def exact_depth(params, camera, px, py, cosine="centre"):
    """Independent float64 reference for one Gaussian: t* = distance to the maximum of G along
    the perspective ray through pixel (px, py) (paper Eq. 6), depth d = cos(theta_c) t* with
    the centre-ray cosine z_c / t_c (paper Eq. 14). ``cosine="pixel"`` returns the true camera z
    of the intersection instead (the July-2024 code formulation). The kernel's pixel grid is the
    Inria one: pixel index i sits at x = (i + 0.5 - W/2) / f, the projected centre at
    ndc2Pix(...) = f x/z + W/2 - 0.5."""
    dd = torch.float64
    mu = params["means"].detach().to(dd)[0]
    R = _quat_to_rot(params["quats"].detach().to(dd))[0]
    S = params["scales"].detach().to(dd).exp()[0]
    sigma_inv = R @ torch.diag(1 / S ** 2) @ R.T
    r = _ray(camera, px, py)
    s_star = (r @ sigma_inv @ mu) / ((r @ sigma_inv) * r).sum(-1)  # point = s* r, z = s*
    if cosine == "pixel":
        return s_star
    t_star = s_star * r.norm(dim=-1)
    return t_star * mu[2] / mu.norm()


def rasterized_depth(method, device):
    """C1: kernel per-pixel depth = first-order (affine) model of the exact intersection depth;
    a constant centre depth (3DGS) fails the slope check."""
    _cuda(device)
    cam = _camera(device)
    p = _one_gaussian(device)
    out, _ = _maps(p, cam)
    _, _, depth_acc, middepth, alpha, _, _ = out
    W = cam["width"]
    f = float(cam["K"][0, 0])
    mu = p["means"].detach()[0]
    cx_pix = (f * mu[0] / mu[2] + W / 2 - 0.5).item()  # kernel ndc2Pix centre
    cy_pix = (f * mu[1] / mu[2] + cam["height"] / 2 - 0.5).item()
    ic, jc = round(cx_pix), round(cy_pix)
    d = (depth_acc[0] / alpha[0]).double()
    zc = float(mu[2])
    # kernel linear model: d(i,j) = z_c + p_x (cx - i) + p_y (cy - j); recover p by least squares
    ii, jj = torch.meshgrid(torch.arange(ic - 3, ic + 4, device=device),
                            torch.arange(jc - 3, jc + 4, device=device), indexing="xy")
    vals = d[jj, ii].reshape(-1)
    A = torch.stack([(cx_pix - ii).reshape(-1).double(), (cy_pix - jj).reshape(-1).double()], -1)
    p_kernel = torch.linalg.lstsq(A, (vals - zc)[:, None]).solution[:, 0]
    lin_resid = (A @ p_kernel + zc - vals).abs().max().item()
    # exact depth: value at the centre and slope by central differences (in pixels)
    h = 1e-3
    x0, y0 = cx_pix, cy_pix
    d0 = exact_depth(p, cam, x0, y0).item()
    gx = (exact_depth(p, cam, x0 + h, y0) - exact_depth(p, cam, x0 - h, y0)).item() / (2 * h)
    gy = (exact_depth(p, cam, x0, y0 + h) - exact_depth(p, cam, x0, y0 - h)).item() / (2 * h)
    p_exact = torch.tensor([-gx, -gy], dtype=torch.float64, device=device)  # Δ = centre - pixel
    hz = [(exact_depth(p, cam, x0 + h, y0, "pixel") - exact_depth(p, cam, x0 - h, y0, "pixel")
           ).item() / (2 * h), (exact_depth(p, cam, x0, y0 + h, "pixel")
                                 - exact_depth(p, cam, x0, y0 - h, "pixel")).item() / (2 * h)]
    p_pixelcos = torch.tensor([-hz[0], -hz[1]], dtype=torch.float64, device=device)
    pixelcos_err = ((p_kernel - p_pixelcos).norm() / p_exact.norm()).item()
    slope_err = ((p_kernel - p_exact).norm() / p_exact.norm()).item()
    if abs(d0 - zc) > 1e-6 * zc:
        raise AssertionError(f"Exact intersection at the centre ray must be z_c: {d0} vs {zc}")
    if lin_resid > 2e-4 * zc:
        raise AssertionError(f"Kernel depth is not affine in the pixel offset ({lin_resid})")
    if slope_err > 2e-3:
        raise AssertionError(f"Kernel depth slope {p_kernel.tolist()} != exact {p_exact.tolist()}")
    if pixelcos_err <= 2e-3:
        raise AssertionError("Fixture cannot tell the centre cosine (paper) from a per-pixel one")
    if p_exact.norm() * 3 < 1e-3:
        raise AssertionError("Probe fixture is not tilted; slope check would be vacuous")
    # ablation: 3DGS-style constant centre depth (slope 0) against the same reference
    centre_err = ((torch.zeros_like(p_exact) - p_exact).norm() / p_exact.norm()).item()
    if centre_err <= 2e-3:
        raise AssertionError("Constant centre depth would pass the slope check")
    # median depth equals the rasterized depth for a single opaque Gaussian at the centre pixel
    med_err = abs(float(middepth[0, jc, ic]) - float(d[jc, ic]))
    if med_err > 1e-4:
        raise AssertionError(f"Median depth != rasterized depth for one Gaussian ({med_err})")
    return {"p_kernel_per_px": p_kernel.tolist(), "p_exact_per_px": p_exact.tolist(),
            "slope_rel_err": slope_err, "affine_residual": lin_resid,
            "centre_depth_slope_rel_err": centre_err,
            "per_pixel_cosine_slope_rel_err": pixelcos_err, "median_err": med_err}


def rasterized_normal(method, device):
    """C2: kernel normal = normal of the surface of ray/Gaussian intersection points s*(r) r
    at the centre ray (paper Eq. 18-22); for a flat Gaussian = its short axis, oriented towards
    the camera."""
    _cuda(device)
    cam = _camera(device)
    res = {}
    for flat in (False, True):
        p = _one_gaussian(device, flat)
        out, _ = _maps(p, cam)
        normal, alpha = out[5], out[4]
        f = float(cam["K"][0, 0])
        W, H = cam["width"], cam["height"]
        mu = p["means"].detach()[0].double()
        x0 = (f * mu[0] / mu[2] + W / 2 - 0.5).item()
        y0 = (f * mu[1] / mu[2] + H / 2 - 0.5).item()
        ic, jc = round(x0), round(y0)
        n_k = F.normalize(normal[:, jc, ic].double(), dim=0)
        # plane of intersection points P(u, v) = d(u, v) (u, v, 1), u = (x + 0.5 - W/2) / f
        h = 1e-3

        def point(x, y):
            d = exact_depth(p, cam, x, y, "pixel")  # intersection point s* r
            return d * torch.tensor([(x + 0.5 - W / 2) / f, (y + 0.5 - H / 2) / f, 1.0],
                                    dtype=torch.float64, device=device)

        du = point(x0 + h, y0) - point(x0 - h, y0)
        dv = point(x0, y0 + h) - point(x0, y0 - h)
        n_ref = F.normalize(torch.cross(du, dv, dim=0), dim=0)
        if torch.dot(n_ref, mu) > 0:   # face the camera (official: n' = -(q, 1))
            n_ref = -n_ref
        err = (1 - torch.dot(n_k, n_ref)).item()
        if err > 2e-3:
            raise AssertionError(f"Kernel normal {n_k.tolist()} != plane normal {n_ref.tolist()}")
        if flat:
            axis = _quat_to_rot(p["quats"].detach().double())[0][:, 2]
            axis = -axis if torch.dot(axis, mu) > 0 else axis
            axis_err = (1 - torch.dot(n_k, axis)).item()
            if axis_err > 2e-3:
                raise AssertionError(f"Flat Gaussian normal != short axis ({axis_err})")
            res["flat_axis_err"] = axis_err
        res["flat" if flat else "tilted"] = {"kernel": n_k.tolist(), "ref": n_ref.tolist(),
                                             "one_minus_cos": err}
        if float(alpha[0, jc, ic]) <= 0:
            raise AssertionError("Fixture not rendered")
    return res


def reference_render(params, camera):
    """Independent differentiable float64 renderer of the paper's model (kernel conventions):
    EWA splat (no 2D dilation), per-pixel depth d = z_c + p.Δ with p the pixel derivative of
    cos(theta_c) t*(r) (Eq. 14) in closed form, normal of the intersection surface s*(r) r
    (Eq. 18-22), front-to-back blending (alpha <= 0.99, 1/255 and depth <= 0.2 skips,
    T < 1e-4 stop), median = d of the last Gaussian with T > 0.5, 2DGS distortion on NDC depth
    with detached weights. SH degree 0, filter_3d = 0, black background."""
    from .raster import activated

    dd = torch.float64
    scales, opac, rots, sh = activated(params)
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = float(camera["K"][0, 0]), float(camera["K"][1, 1])
    mu = params["means"].to(dd)  # identity camera: camera frame = world frame
    Rg = _quat_to_rot(rots.to(dd))
    S2 = scales.to(dd) ** 2
    Sigma = Rg @ torch.diag_embed(S2) @ Rg.transpose(1, 2)
    A = Rg @ torch.diag_embed(1 / S2) @ Rg.transpose(1, 2)
    x, y, z = mu.unbind(-1)
    zeros = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zeros, -fx * x / z ** 2], -1),
                     torch.stack([zeros, fy / z, -fy * y / z ** 2], -1)], 1)  # [N,2,3]
    cov = J @ Sigma @ J.transpose(1, 2)
    conic = torch.linalg.inv(cov)
    cpx = fx * x / z + W / 2 - 0.5
    cpy = fy * y / z + H / 2 - 0.5
    r = mu / z[:, None]                                   # centre ray (u_c, v_c, 1)
    Ar = (A @ r[..., None])[..., 0]
    b = (r * Ar).sum(-1)
    n2 = (r * r).sum(-1)
    grad_r = z[:, None] * (r / n2[:, None] - Ar / b[:, None])  # d(cos θc t*)/dr at the centre
    p_pix = torch.stack([-grad_r[:, 0] / fx, -grad_r[:, 1] / fy], -1)  # per pixel of Δ
    e = torch.eye(3, dtype=dd, device=mu.device)
    du = e[0] - Ar[:, :1] * r / b[:, None]
    dv = e[1] - Ar[:, 1:2] * r / b[:, None]
    nrm = F.normalize(torch.cross(du, dv, dim=-1), dim=-1)
    nrm = torch.where(((nrm * mu).sum(-1) > 0)[:, None], -nrm, nrm)
    color = (sh[:, 0].to(dd) * C0 + 0.5).clamp_min(0)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dd, device=mu.device),
                            torch.arange(W, dtype=dd, device=mu.device), indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    P_ = xs.numel()
    T = torch.ones(P_, dtype=dd, device=mu.device)
    alive = torch.ones(P_, dtype=torch.bool, device=mu.device)
    out = {"color": torch.zeros(P_, 3, dtype=dd, device=mu.device),
           "depth": torch.zeros(P_, dtype=dd, device=mu.device),
           "middepth": torch.zeros(P_, dtype=dd, device=mu.device),
           "alpha": torch.zeros(P_, dtype=dd, device=mu.device),
           "normal": torch.zeros(P_, 3, dtype=dd, device=mu.device),
           "dist": torch.zeros(P_, dtype=dd, device=mu.device)}
    d1 = torch.zeros(P_, dtype=dd, device=mu.device)
    d2 = torch.zeros(P_, dtype=dd, device=mu.device)
    for i in torch.argsort(z.detach()).tolist():
        dx, dy = cpx[i] - xs, cpy[i] - ys
        power = -0.5 * (conic[i, 0, 0] * dx * dx + conic[i, 1, 1] * dy * dy) \
            - conic[i, 0, 1] * dx * dy
        depth = z[i] + p_pix[i, 0] * dx + p_pix[i, 1] * dy
        a = torch.clamp(opac[i].to(dd) * torch.exp(power), max=0.99)
        ok = alive & (power <= 0) & (depth > NEAR) & (a >= 1 / 255)
        stop = ok & (T * (1 - a) < 1e-4)
        alive = alive & ~stop
        ok = ok & ~stop
        w = torch.where(ok, a * T, torch.zeros_like(a))
        out["color"] = out["color"] + w[:, None] * color[i]
        out["depth"] = out["depth"] + w * depth
        out["normal"] = out["normal"] + w[:, None] * nrm[i]
        out["middepth"] = torch.where(ok & (T > 0.5), depth, out["middepth"])
        m = (FAR * depth - FAR * NEAR) / ((FAR - NEAR) * depth)
        wd = w.detach()
        acc = out["alpha"].detach()
        out["dist"] = out["dist"] + wd * (m * m * acc + d2 - 2 * m * d1)
        d1 = d1 + m * wd
        d2 = d2 + m * m * wd
        out["alpha"] = out["alpha"] + w
        T = torch.where(ok, T * (1 - a), T)
    return {"color": out["color"].T.reshape(3, H, W), "depth": out["depth"].reshape(1, H, W),
            "middepth": out["middepth"].reshape(1, H, W), "alpha": out["alpha"].reshape(1, H, W),
            "normal": out["normal"].T.reshape(3, H, W), "dist": out["dist"].reshape(1, H, W)}


def kernel_gradients(method, device):
    """C1/C2 forward and backward: the kernel's six outputs equal the independent float64
    reference, and so do the gradients (means, scales, quats, opacity, SH) of a weighted sum of
    them. Known official quirk (kept, kernel vendored unmodified): the backward of the normal
    map w.r.t. the Gaussian *means* is inexact (a finite difference of the kernel's own forward
    agrees with the reference, not with the kernel backward); it is measured, not asserted."""
    _cuda(device)
    cam = _camera(device, 32, 32, 30.0)
    g = torch.Generator().manual_seed(1)
    n = 4
    means = torch.tensor([[0.1, -0.05, 2.5], [-0.2, 0.15, 3.0], [0.05, 0.1, 3.6],
                          [-0.1, -0.15, 2.8]])
    scales = torch.rand(n, 3, generator=g) * 0.25 + 0.1
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.tensor([0.5, 0.6, 0.7, 0.4])
    sh0 = (torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8
    p = _params(means, scales, quats, opac, device, sh0)
    names = ("color", "depth", "middepth", "alpha", "normal", "dist")
    wts = dict(zip(names, [torch.rand(c, 32, 32, generator=g).to(device)
                           for c in (3, 1, 1, 1, 3, 1)]))
    wts["dist"] = wts["dist"] * 1000
    fields = ("means", "scales", "quats", "opacities", "sh0")

    def grads(loss):
        for v in p.values():
            v.grad = None
        loss.backward()
        out = {k: (torch.zeros_like(p[k], dtype=torch.float64) if p[k].grad is None
                   else p[k].grad.detach().double().clone()) for k in fields}
        for v in p.values():
            v.grad = None
        return out

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

    def kernel_maps():
        out, _ = _maps(p, cam, grad=True)
        color, _, depth, middepth, alpha, normal, dist = out
        return dict(zip(names, (color, depth, middepth, alpha, normal, dist)))

    km = kernel_maps()
    with torch.no_grad():
        ref0 = reference_render(p, cam)
    fwd = {k: (km[k].detach().double() - ref0[k]).abs().max().item() for k in names}
    for k, err in fwd.items():
        if err > 2e-4 * max(1.0, ref0[k].abs().max().item()):
            raise AssertionError(f"Kernel {k} differs from the reference by {err}")
    no_normal = [k for k in names if k != "normal"]
    ga = grads(sum((wts[k] * km[k]).sum() for k in no_normal))
    ref = reference_render(p, cam)
    gr = grads(sum((wts[k].double() * ref[k]).sum() for k in no_normal))
    errs = {k: rel(ga[k], gr[k]) for k in fields}
    gan = grads((wts["normal"] * kernel_maps()["normal"]).sum())
    grn = grads((wts["normal"].double() * reference_render(p, cam)["normal"]).sum())
    errs_n = {k: rel(gan[k], grn[k]) for k in fields if grn[k].norm() > 0}
    for k, e in {**errs, **{f"normal/{k}": v for k, v in errs_n.items() if k != "means"}}.items():
        if e > 1e-3:
            raise AssertionError(f"Kernel gradient {k} off by {e:.2e} (relative)")
    return {"forward_max_abs": fwd, "gradient_rel_err": errs,
            "normal_map_gradient_rel_err": errs_n,
            "official_quirk": "normal -> means backward inexact (kept)"}


def distortion_formula(method, device):
    """C3: kernel distortion = Σ_{j<i} w_i w_j (m_i - m_j)^2 with NDC m(d); the backward
    detaches the weights (zero opacity gradient) but not the depths."""
    _cuda(device)
    cam = _camera(device, 16, 16, 20.0)
    zs = torch.tensor([2.0, 2.6, 3.5])
    n = len(zs)
    means = torch.stack([torch.zeros(n), torch.zeros(n), zs], -1)
    scales = torch.tensor([[0.6, 0.6, 0.6]]).repeat(n, 1)
    quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    opac = torch.tensor([0.3, 0.4, 0.5])
    p = _params(means, scales, quats, opac, device)
    out, _ = _maps(p, cam, grad=True)
    dist = out[6][0]
    # centre pixel: isotropic Gaussians are front-facing -> pixel depth = z_c + p·Δ, p=0
    W = 16
    i = j = W // 2  # pixel 8: offset 0.5 px from the centre (7.5)
    dd = torch.float64
    x = (i + 0.5 - W / 2) / 20.0
    y = (j + 0.5 - W / 2) / 20.0
    G = torch.exp(-0.5 * ((x * zs.double()) ** 2 + (y * zs.double()) ** 2) / 0.36)
    a = opac.double() * G
    T = torch.cumprod(torch.cat([torch.ones(1, dtype=dd), 1 - a[:-1]]), 0)
    w = a * T
    m = (FAR * zs.double() - FAR * NEAR) / ((FAR - NEAR) * zs.double())
    ref = sum(w[a_] * w[b] * (m[a_] - m[b]) ** 2 for a_ in range(n) for b in range(a_))
    k = float(dist[j, i])
    if abs(k - float(ref)) > 1e-3 * float(ref) + 1e-9:
        raise AssertionError(f"Distortion {k} != reference {float(ref)}")
    dist.sum().backward()
    g_op = p["opacities"].grad.abs().max().item()
    g_mu = p["means"].grad.abs().max().item()
    if g_op != 0.0 or g_mu == 0.0:
        raise AssertionError(f"Distortion gradient: opacity {g_op} (must be 0), means {g_mu}")
    return {"kernel": k, "reference": float(ref), "opacity_grad": g_op, "means_grad": g_mu}


def regularizer_schedule(method, device):
    """C3 loss: zero before step 15000, then 100 L_d + 0.05 (0.4 L_exp + 0.6 L_med) vs an
    independent re-implementation; depth_double_to_normal recovers an analytic plane."""
    from .losses import create_regularizer
    from .raster import depth_double_to_normal

    spec = next(c for c in method.components if c.name == "radegs-reg")
    cfg = spec.config
    if (cfg["lambda_distortion"], cfg["lambda_depth_normal"], cfg["depth_ratio"],
            cfg["regularization_from_iter"]) != (100.0, 0.05, 0.6, 15000):
        raise AssertionError(f"Regularizer config differs from the official code: {cfg}")
    loss = create_regularizer(spec)
    torch.manual_seed(0)
    H = W = 12
    cam = _camera("cpu", W, H, 10.0)
    # plane z = 2 + 0.3 x (camera frame): normal ∝ (0.3, 0, -1)
    us = (torch.arange(W) + 0.5 - W / 2) / 10.0
    depth = (2.0 / (1 - 0.3 * us))[None, :].repeat(H, 1)
    nrm = depth_double_to_normal(cam, depth, depth)
    expect = F.normalize(torch.tensor([0.3, 0.0, -1.0]), dim=0)
    plane_err = (1 - (nrm[0, 1:-1, 1:-1] * expect).sum(-1)).abs().max().item()
    if plane_err > 1e-5:
        raise AssertionError(f"depth_double_to_normal plane error {plane_err}")
    outputs = {
        "rgb": torch.rand(H, W, 3, requires_grad=True),
        "alpha": torch.rand(H, W, 1) * 0.9 + 0.05,
        "distortion": torch.rand(H, W, 1) * 0.01,
        "normal": F.normalize(torch.randn(H, W, 3), dim=-1),
        "surface_normal_expected": F.normalize(torch.randn(H, W, 3), dim=-1),
        "surface_normal_median": F.normalize(torch.randn(H, W, 3), dim=-1),
    }
    target = torch.rand(H, W, 3)
    vals = {}
    for step in (14999, 15000):
        vals[step] = float(loss(None, {**outputs, "step": step}, target))
    # independent formula (train.py + scene/cameras.py edge)
    gi = target.permute(2, 0, 1)
    c = gi[:, 1:-1, 1:-1]
    mg = torch.stack([(c - gi[:, 1:-1, :-2]).abs().mean(0), (c - gi[:, 1:-1, 2:]).abs().mean(0),
                      (c - gi[:, :-2, 1:-1]).abs().mean(0), (c - gi[:, 2:, 1:-1]).abs().mean(0)
                      ]).max(0).values
    edge = torch.zeros(H, W)
    edge[1:-1, 1:-1] = torch.exp(-mg)
    a = outputs["alpha"][..., 0]
    ld = (outputs["distortion"][..., 0] / a ** 2 * edge).mean()
    e1 = (1 - (outputs["normal"] * outputs["surface_normal_expected"]).sum(-1)).mean()
    e2 = (1 - (outputs["normal"] * outputs["surface_normal_median"]).sum(-1)).mean()
    ref = 100 * ld + 0.05 * (0.4 * e1 + 0.6 * e2)
    if vals[14999] != 0.0 or not math.isclose(vals[15000], float(ref), rel_tol=1e-5):
        raise AssertionError(f"Regularizer {vals} vs reference {float(ref)}")
    return {"loss": vals, "reference": float(ref), "plane_normal_err": plane_err}


def densify_metric(method, device):
    """Inherited GOF densification: kernel abs channel >= |signed| per Gaussian, and a Gaussian
    with cancelling gradients is densified only with the abs metric."""
    _cuda(device)
    from dataclasses import replace

    from .density import Density

    cam = _camera(device, 32, 32, 30.0)
    g = torch.Generator().manual_seed(3)
    n = 5
    means = torch.rand(n, 3, generator=g) * torch.tensor([1.2, 1.2, 1.5]) - torch.tensor(
        [0.6, 0.6, -2.5])
    p = _params(means, torch.rand(n, 3, generator=g) * 0.3 + 0.05,
                F.normalize(torch.randn(n, 4, generator=g), dim=-1),
                torch.rand(n, generator=g) * 0.5 + 0.3, device)
    out, means2D = _maps(p, cam, grad=True)
    target = (torch.arange(32, device=device)[None, :, None] % 2).float().expand(32, 32, 3)
    (out[0].permute(1, 2, 0) - target).square().mean().backward()
    gr = means2D.grad
    vis = out[1] > 0
    l1 = gr[vis, 0].abs() + gr[vis, 1].abs()
    if not (gr[vis, 2] + 1e-9 >= l1 * (1 - 1e-4)).all() or not (gr[vis, 2] > 0).all():
        raise AssertionError("Abs gradient channel is not the sum of per-pixel norms")

    def run(abs_gradient):
        spec = next(c for c in method.components if c.kind == "density")
        spec = replace(spec, config={**spec.config, "abs_gradient": abs_gradient})
        d = Density(spec)
        k = 4
        q = {"means": torch.zeros(k, 3), "scales": torch.full((k, 3), math.log(0.001)),
             "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(k, 1),
             "opacities": torch.full((k,), 2.0), "sh0": torch.zeros(k, 1, 3),
             "shN": torch.zeros(k, 15, 3), "filter_3d": torch.zeros(k)}
        q = torch.nn.ParameterDict({kk: torch.nn.Parameter(v) for kk, v in q.items()})
        opts = {kk: torch.optim.Adam([v], lr=1e-3) for kk, v in q.items()}
        d.initialize(q, opts, 1.0)
        d.stats["grad_accum"][:] = torch.tensor([3e-4, 0.0, 1e-5, 1e-5])
        d.stats["grad_accum_abs"][:] = torch.tensor([5e-4, 9e-4, 1e-5, 2e-5])
        d.stats["denom"][:] = 1
        d.densify_and_prune(q, opts, False)
        return d.state["last"]

    with_m, without = run(True), run(False)
    if with_m["clone"] != 2 or without["clone"] != 1:
        raise AssertionError(f"Abs-gradient densification not active: {with_m} {without}")
    return {"abs_vs_l1_min_ratio": (gr[vis, 2] / l1.clamp_min(1e-12)).min().item(),
            "clones_with_M": with_m["clone"], "clones_without_M": without["clone"]}


def density_recipe(method, device):
    """Official densify/prune/reset/3D-filter/LR recipe on a CPU fixture (as the GOF port)."""
    from .density import Density, expon_lr, is_filter_step
    from .filters import compute_filter_3d, filtered_scales_opacities

    spec = next(c for c in method.components if c.kind == "density")
    d = Density(spec)
    torch.manual_seed(0)
    n = 6
    s = torch.tensor([0.005, 0.005, 0.05, 0.05, 0.005, 0.2])  # extent 1 -> dense 0.01
    p = {"means": torch.randn(n, 3) + torch.tensor([0, 0, 4.0]),
         "scales": torch.log(s)[:, None].repeat(1, 3),
         "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
         "opacities": torch.logit(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.04, 0.5])),
         "sh0": torch.randn(n, 1, 3), "shN": torch.zeros(n, 15, 3), "filter_3d": torch.zeros(n)}
    p = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()})
    opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
    sum(v.square().sum() for v in p.values()).backward()
    for o in opts.values():
        o.step()
    cams = [{"viewmat": torch.eye(4), "K": torch.tensor([[50.0, 0, 32], [0, 50.0, 24],
                                                          [0, 0, 1]]),
             "width": 64, "height": 48}]
    d.initialize(p, opts, 1.0)
    d.set_training_cameras(cams)
    z = p["means"][:, 2].detach()
    u = p["means"][:, 0].detach() / z * 50 + 32
    v = p["means"][:, 1].detach() / z * 50 + 24
    inside = (u >= -0.15 * 64) & (u <= 1.15 * 64) & (v >= -0.15 * 48) & (v <= 1.15 * 48)
    exp_f = torch.where(inside, z, z[inside].max()) / 50 * math.sqrt(0.2)
    if not torch.allclose(p["filter_3d"].detach(), exp_f, rtol=1e-5):
        raise AssertionError("3D filter differs from the independent formula")
    torch.testing.assert_close(compute_filter_3d(p["means"], cams), exp_f)
    parent_means = p["means"].detach().clone()
    d.stats["grad_accum"][:] = torch.tensor([3e-4, 0, 3e-4, 0, 3e-4, 3e-4])
    d.stats["grad_accum_abs"][:] = torch.tensor([3e-4, 0, 3e-4, 0, 3e-4, 3e-4])
    d.stats["denom"][:] = 1
    d.densify_and_prune(p, opts, True)
    last = d.state["last"]
    if (last["clone"], last["split"]) != (2, 2):
        raise AssertionError(f"Clone/split selection wrong: {last}")
    if last["after"] != n + 2 + 2 - 4 or last["prune"] != 4:
        raise AssertionError(f"Prune wrong: {last}")
    sc = p["scales"].exp()[:, 0]
    if not torch.isclose(sc, torch.tensor(0.05 / 1.6), rtol=1e-2).sum() == 2:
        raise AssertionError("Split children must have scale s / 1.6")
    small = torch.isclose(sc, torch.tensor(0.005), rtol=1e-2)
    dist = torch.cdist(p["means"][small].detach(), parent_means[:1])[:, 0]
    if int((dist == 0).sum()) != 1 or int(((dist > 0) & (dist < 0.05)).sum()) != 1:
        raise AssertionError("Clone did not resample its position around the parent")
    d.update_filter(p)
    d.reset_opacity(p, opts)
    _, fo = filtered_scales_opacities(p["scales"], p["opacities"], p["filter_3d"])
    if not (fo <= 0.01 + 1e-6).all() or (opts["opacities"].state[p["opacities"]]["exp_avg"]
                                          != 0).any():
        raise AssertionError("Filter-aware opacity reset wrong")
    sched = {k: is_filter_step(k, 100, 500, 15000, 30000)
             for k in (500, 600, 15000, 15100, 29800, 29900)}
    if sched != {500: False, 600: True, 15000: False, 15100: True, 29800: True, 29900: False}:
        raise AssertionError(f"3D filter schedule wrong: {sched}")
    lrs = [d.xyz_lr(k) for k in (0, 15000, 30000)]
    expect = [1.6e-4, math.sqrt(1.6e-4 * 1.6e-6), 1.6e-6]
    if not all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(lrs, expect)):
        raise AssertionError(f"xyz LR schedule wrong {lrs}")
    if not math.isclose(expon_lr(10, 1.0, 0.01, 0, 0.01, 100), 0.01 ** 0.1, rel_tol=1e-9):
        raise AssertionError("expon_lr wrong")
    return {"last": last, "filter_updates": d.counts["filter_updates"], "xyz_lr": lrs}
