"""Mechanism probes. Each one raises when its mechanism is removed or wrong."""

import math

import numpy as np
import torch
from splatstudio.components import build

from .kernels import rasterize_jets
from .renderer import low_res_camera, render_gsplat, render_jets
from .upscale import bicubic_upscale, central_differences, hermite_basis, spline_upscale


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return "cuda"


def synthetic_scene(n=300, seed=0, device="cuda", scale_range=(-2.6, -1.4), sh=True):
    """Gaussians in front of an identity camera (z in [2.5, 4.5])."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * 1.1,
                         (torch.rand(n, generator=g) * 2 - 1) * 1.1,
                         2.5 + torch.rand(n, generator=g) * 2.0], dim=-1)
    lo, hi = scale_range
    params = {
        "means": means,
        "scales": lo + torch.rand(n, 3, generator=g) * (hi - lo),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g) * 1.0,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * (0.1 if sh else 0.0),
    }
    return {k: v.to(device) for k, v in params.items()}


def camera(width, height, focal=None, device="cuda"):
    f = focal if focal is not None else 1.1 * width
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]],
                     device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": width, "height": height}


def psnr(a, b):
    return float(-10 * torch.log10((a - b).square().mean().clamp_min(1e-12)))


# --------------------------------------------------------------------------- C2 forward
def jet_value_equals_gsplat(method, device):
    """The value channel of the jet rasterizer is gsplat's rendering (same numerics)."""
    dev = _cuda()
    params = synthetic_scene(device=dev)
    worst = 0.0
    with torch.no_grad():
        for w, h in ((96, 80), (61, 37)):
            cam = camera(w, h, device=dev)
            jets, alpha, _ = render_jets(params, cam, 3)
            ref, ref_alpha, _ = render_gsplat(params, cam, 3)
            worst = max(worst, float((jets[0] - ref).abs().max()),
                        float((alpha - ref_alpha[..., 0]).abs().max()))
    if worst > 2e-5:
        raise AssertionError(f"jet value differs from gsplat by {worst}")
    return {"max_abs_diff": worst}


def _err_stats(pred, ref, mask, scale):
    e = (pred - ref)[mask].abs()
    return float(e.median()) / scale, float(e.quantile(0.99)) / scale


def jet_derivs_vs_finite_diff(method, device):
    """I_x, I_y equal central differences of the rendered image under sub-pixel shifts of the
    sampling position (same projected Gaussians, same kernel); I_xy equals the y-difference of
    I_x."""
    dev = _cuda()
    params = synthetic_scene(device=dev, scale_range=(-2.3, -1.3))
    cam = camera(80, 64, device=dev)
    with torch.no_grad():
        jets, _, _ = render_jets(params, cam, 3)

        def img(ox, oy, k=0):
            return render_jets(params, cam, 3, offset=(ox, oy))[0][k].double()

        def fd_x(d, k=0):
            return (img(d, 0, k) - img(-d, 0, k)) / (2 * d)

        def fd_y(d, k=0):
            return (img(0, d, k) - img(0, -d, k)) / (2 * d)

        # I_xy: y-difference of the analytic I_x (itself verified against differences of I)
        fds = {"x": (fd_x(2e-3), fd_x(4e-3)), "y": (fd_y(2e-3), fd_y(4e-3)),
               "xy": (fd_y(2e-3, 1), fd_y(4e-3, 1))}
    out = {}
    for k, name in enumerate(("x", "y", "xy"), start=1):
        a = jets[k].double()
        b, b2 = fds[name]
        scale = float(b.abs().max())
        if scale < 1e-3:
            raise AssertionError("degenerate probe scene (no image gradients)")
        # The 1/255 alpha cutoff and early termination make the image discontinuous; finite
        # differences across a jump depend on the step. Keep pixels where two step sizes agree.
        smooth = (b - b2).abs() <= 1e-2 * scale
        frac = float(smooth.double().mean())

        med, q99 = _err_stats(a, b, smooth, scale)
        _, zero_q99 = _err_stats(torch.zeros_like(a), b, smooth, scale)  # mechanism removed
        out[name] = {"median_err_rel": med, "q99_err_rel": q99, "smooth_fraction": frac,
                     "scale": scale, "removed_q99_rel": zero_q99}
        if frac < 0.9 or med > 1e-3 or q99 > 1e-2:
            raise AssertionError(f"analytic d/d{name} disagrees with finite differences: {out}")
        if zero_q99 < 10 * max(q99, 1e-4):
            raise AssertionError(f"probe cannot detect missing d/d{name}: {out}")
    return out


def _reference_jets(means2d, conics, colors, opac, bg, W, H, order):
    """Independent dense float64 reference: alpha compositing with torch, spatial derivatives
    by autograd w.r.t. the pixel coordinates (double backward for I_xy)."""
    dev = means2d.device
    py, px = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64) + 0.5,
                            torch.arange(W, device=dev, dtype=torch.float64) + 0.5,
                            indexing="ij")
    px.requires_grad_(True)
    py.requires_grad_(True)
    T = torch.ones(H, W, dtype=torch.float64, device=dev)
    C = torch.zeros(H, W, 3, dtype=torch.float64, device=dev)
    for g in order.tolist():
        ddx, ddy = means2d[g, 0] - px, means2d[g, 1] - py
        a, b, c = conics[g]
        sigma = 0.5 * (a * ddx * ddx + c * ddy * ddy) + b * ddx * ddy
        alpha = (opac[g] * torch.exp(-sigma)).clamp(max=0.999)
        keep = ((sigma >= 0) & (alpha >= 1.0 / 255.0)).detach()
        alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
        C = C + colors[g] * (alpha * T)[..., None]
        T = T * (1 - alpha)
    out = C + T[..., None] * bg
    jx, jy, jxy = [], [], []
    for k in range(3):
        gx, = torch.autograd.grad(out[..., k].sum(), px, create_graph=True)
        gy, = torch.autograd.grad(out[..., k].sum(), py, create_graph=True)
        gxy, = torch.autograd.grad(gx.sum(), py, create_graph=True)
        jx.append(gx)
        jy.append(gy)
        jxy.append(gxy)
    return torch.stack([out, torch.stack(jx, -1), torch.stack(jy, -1), torch.stack(jxy, -1)]), T


def jet_vs_torch_autograd(method, device):
    """CUDA forward jets and backward gradients equal an independent float64 autograd reference."""
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    dev = _cuda()
    g = torch.Generator(device="cpu").manual_seed(3)
    n, W, H = 14, 40, 36
    means2d = torch.stack([torch.rand(n, generator=g) * W, torch.rand(n, generator=g) * H], -1)
    L = torch.randn(n, 2, 2, generator=g) * 0.25
    cov = L @ L.transpose(1, 2) + torch.eye(2) * torch.rand(n, 1, 1, generator=g) * 30 + 4
    inv = torch.linalg.inv(cov)
    conics = torch.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], -1)
    colors = torch.rand(n, 3, generator=g)
    opac = 0.1 + 0.4 * torch.rand(n, generator=g)
    depths = torch.rand(n, generator=g) + 1.0
    bg = torch.tensor([1.0, 0.9, 0.8])
    leaves = [t.to(dev).float().requires_grad_(True) for t in (means2d, conics, colors, opac)]
    radii = torch.full((1, n, 2), 64, dtype=torch.int32, device=dev)
    tw, th = math.ceil(W / 16), math.ceil(H / 16)
    _, isect_ids, flatten_ids = isect_tiles(leaves[0][None].detach(), radii,
                                            depths.to(dev)[None], 16, tw, th, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th)[0]
    jets, alpha, tfin, _ = rasterize_jets(*leaves, bg.to(dev), offsets, flatten_ids, W, H, 16)
    if float(tfin[..., 0].min()) <= 2e-4:
        raise AssertionError("probe scene triggers early termination")
    wts = torch.randn(4, H, W, 3, generator=g).to(dev)
    wa = torch.randn(H, W, generator=g).to(dev)
    ((jets * wts).sum() + (alpha * wa).sum()).backward()

    ref_leaves = [t.detach().double().requires_grad_(True) for t in leaves]
    order = torch.argsort(depths)
    ref, T = _reference_jets(*ref_leaves, bg.to(dev).double(), W, H, order)
    ((ref * wts.double()).sum() + ((1 - T) * wa.double()).sum()).backward()
    out = {}
    for i, name in enumerate(("I", "I_x", "I_y", "I_xy")):
        scale = float(ref[i].abs().max())
        out[f"fwd_{name}"] = float((jets[i].double() - ref[i]).abs().max()) / scale
    for name, a, b in zip(("means2d", "conics", "colors", "opacities"), leaves, ref_leaves):
        out[f"bwd_{name}"] = float((a.grad.double() - b.grad).norm() / b.grad.norm())
    bad = {k: v for k, v in out.items() if v > 1e-3}
    if bad:
        raise AssertionError(f"jet rasterizer disagrees with autograd reference: {bad}")
    return out


# --------------------------------------------------------------------------- C1 upscaler
def spline_equals_paper_matrix(method, device):
    """Separable Hermite upscaling equals the paper's A = C^-1 F C^-T patch formula."""
    rng = np.random.default_rng(0)
    f, fx, fy, fxy = (rng.normal(size=(2, 2, 1)) for _ in range(4))  # [y, x, C]
    out = spline_upscale(*(torch.tensor(a, dtype=torch.float64) for a in (f, fx, fy, fxy)), 7, 9)
    Cm = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 2, 3]], dtype=np.float64)
    Ci = np.linalg.inv(Cm)

    def at(img, x, y):  # f(x, y) with x = column, y = row
        return img[y, x, 0]

    F = np.array([
        [at(f, 0, 0), at(f, 0, 1), at(fy, 0, 0), at(fy, 0, 1)],
        [at(f, 1, 0), at(f, 1, 1), at(fy, 1, 0), at(fy, 1, 1)],
        [at(fx, 0, 0), at(fx, 0, 1), at(fxy, 0, 0), at(fxy, 0, 1)],
        [at(fx, 1, 0), at(fx, 1, 1), at(fxy, 1, 0), at(fxy, 1, 1)],
    ])
    A = Ci @ F @ Ci.T
    worst = 0.0
    for Y in range(7):
        for X in range(9):
            x = (X + 0.5) * 2 / 9 - 0.5
            y = (Y + 0.5) * 2 / 7 - 0.5
            p = np.array([1, x, x * x, x ** 3]) @ A @ np.array([1, y, y * y, y ** 3])
            worst = max(worst, abs(p - float(out[Y, X, 0])))
    if worst > 1e-10:
        raise AssertionError(f"spline differs from the paper's matrix formula by {worst}")
    h = hermite_basis(torch.tensor([0.0, 1.0], dtype=torch.float64))
    if not torch.allclose(torch.stack(h), torch.tensor([[1.0, 0], [0, 1], [0, 0], [0, 0]],
                                                       dtype=torch.float64)):
        raise AssertionError("Hermite basis does not interpolate the corner values")
    return {"max_abs_diff": worst}


def _keys(x, a=-0.5):
    x = x.abs()
    return torch.where(x <= 1, (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1,
                       torch.where(x < 2, a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a,
                                   torch.zeros_like(x)))


def _keys_upscale(img, H, W):
    """Independent separable Keys-cubic (Catmull-Rom, a=-0.5) convolution, edge replicate."""
    h, w = img.shape[:2]

    def mat(n_out, n_in):
        u = (torch.arange(n_out, dtype=torch.float64) + 0.5) * n_in / n_out - 0.5
        m = torch.zeros(n_out, n_in, dtype=torch.float64)
        base = u.floor()
        for o in range(-1, 3):
            idx = (base + o).clamp(0, n_in - 1).long()
            m.scatter_add_(1, idx[:, None], _keys(u - (base + o))[:, None])
        return m

    My, Mx = mat(H, h), mat(W, w)
    return torch.einsum("Yy,yxc,Xx->YXc", My, img, Mx)


def spline_equals_catmull_rom(method, device):
    """With finite-difference derivative estimates the spline is classical (Catmull-Rom)
    bicubic, i.e. the paper's statement that bicubic = spline with estimated gradients."""
    g = torch.Generator().manual_seed(1)
    img = torch.rand(12, 15, 3, generator=g, dtype=torch.float64)
    H, W = 48, 60
    fx, fy, fxy = central_differences(img)
    sp = spline_upscale(img, fx, fy, fxy, H, W)
    ref = _keys_upscale(img, H, W)
    # interior: cells whose 4x4 neighbourhood is inside the image
    s = 4
    inner = (slice(2 * s, H - 2 * s), slice(2 * s, W - 2 * s))
    diff = float((sp[inner] - ref[inner]).abs().max())
    if diff > 1e-10:
        raise AssertionError(f"FD-gradient spline differs from Catmull-Rom by {diff}")
    # wrong derivative scaling (e.g. derivatives in output pixels) must be detected
    wrong = float((spline_upscale(img, fx / s, fy / s, fxy / s ** 2, H, W)[inner]
                   - ref[inner]).abs().max())
    if wrong < 1e-3:
        raise AssertionError("probe cannot distinguish derivative scaling")
    return {"max_abs_diff_interior": diff, "wrong_scaling_diff": wrong}


def analytic_beats_fd(method, device):
    """End-to-end mechanism check: rendering at W/4 with analytic derivatives and spline
    upscaling reconstructs the native W render better than the same spline with finite
    difference derivatives and than PyTorch bicubic (paper Fig. 2/5 claim)."""
    dev = _cuda()
    out = {}
    for seed in (0, 1):
        params = synthetic_scene(n=400, seed=seed, device=dev, scale_range=(-2.4, -1.5))
        cam = camera(256, 192, device=dev)
        with torch.no_grad():
            gt = render_gsplat(params, cam, 3)[0]
            low = low_res_camera(cam, 4)
            jets, _, _ = render_jets(params, low, 3)
            H, W = cam["height"], cam["width"]
            spline = spline_upscale(jets[0], jets[1], jets[2], jets[3], H, W)
            fd = spline_upscale(jets[0], *central_differences(jets[0]), H, W)
            bic = bicubic_upscale(jets[0], H, W)
        res = {"spline_analytic": psnr(spline, gt), "spline_fd": psnr(fd, gt),
               "bicubic": psnr(bic, gt)}
        out[f"seed{seed}"] = res
        if not (res["spline_analytic"] > res["spline_fd"] + 0.2
                and res["spline_analytic"] > res["bicubic"] + 0.2):
            raise AssertionError(f"analytic gradients do not improve upscaling: {res}")
    return out


# --------------------------------------------------------------------------- C3 training
def renderer_end_to_end(method, device):
    """The configured renderer trains through projection, jets and upscaler: output at target
    size, finite gradients, and d(loss)/d(opacity logit) agrees with finite differences."""
    dev = _cuda()
    renderer, _, _ = build(method)
    params = synthetic_scene(n=120, seed=5, device=dev, scale_range=(-2.0, -1.4), sh=False)
    params = {k: torch.nn.Parameter(v.double().float()) for k, v in params.items()}
    cam = camera(64, 48, device=dev)
    g = torch.Generator().manual_seed(2)
    target = torch.rand(48, 64, 3, generator=g).to(dev)

    def loss_fn():
        out = renderer.render(params, cam, sh_degree=0)
        return (out["rgb"] - target).square().mean(), out

    loss, out = loss_fn()
    if tuple(out["rgb"].shape) != (48, 64, 3):
        raise AssertionError(f"renderer output {tuple(out['rgb'].shape)} is not the target size")
    out["info"]["means2d"].retain_grad()
    loss.backward()
    for k, p in params.items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise AssertionError(f"no finite gradient for {k}")
    if not torch.count_nonzero(out["info"]["means2d"].grad):
        raise AssertionError("no screen-space gradient for densification")
    grad = params["opacities"].grad
    idx = int(grad.abs().argmax())
    eps = 1e-2
    with torch.no_grad():
        params["opacities"][idx] += eps
        lp = loss_fn()[0].item()
        params["opacities"][idx] -= 2 * eps
        lm = loss_fn()[0].item()
        params["opacities"][idx] += eps
    fd = (lp - lm) / (2 * eps)
    rel = abs(fd - float(grad[idx])) / max(abs(fd), 1e-12)
    if rel > 0.05:
        raise AssertionError(f"opacity gradient {float(grad[idx])} vs finite difference {fd}")
    return {"loss": loss.item(), "opacity_grad": float(grad[idx]), "fd": fd, "rel_err": rel,
            "low_res": list(out["low_res"])}
