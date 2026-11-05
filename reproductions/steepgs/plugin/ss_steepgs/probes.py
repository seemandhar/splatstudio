"""Mechanism probes for the SteepGS port. Each raises on failure and returns measurements."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _density(method, device, **config):
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")
    return create_density(replace(spec, config={**spec.config, **config}))


def _params(means, scales, quats=None, opac=0.5, device="cpu"):
    n = len(means)
    if quats is None:
        quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    vals = {"means": torch.as_tensor(means, dtype=torch.float32),
            "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
            "quats": torch.as_tensor(quats, dtype=torch.float32),
            "opacities": torch.full((n,), math.log(opac / (1 - opac))),
            "sh0": torch.rand(n, 1, 3), "shN": torch.rand(n, 15, 3)}
    params = {k: torch.nn.Parameter(v.to(device)) for k, v in vals.items()}
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for k, p in params.items():  # allocate Adam moments, then restore the values
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return params, opts


def _outputs(n, c, grad2d, visible, device):
    """Synthetic renderer info: c = sum_pix dL/dG per Gaussian, grad2d = |screen gradient|."""
    m2 = torch.zeros(n, 3, device=device, requires_grad=True)
    m2.grad = torch.zeros(n, 3, device=device)
    m2.grad[:, 0] = torch.as_tensor(grad2d, dtype=torch.float32, device=device)
    sp = torch.zeros(n, 3, 3, device=device, requires_grad=True)
    sp.grad = torch.zeros(n, 3, 3, device=device)
    sp.grad[:, 0, 0] = torch.as_tensor(c, dtype=torch.float32, device=device)
    sp.grad[:, 1, :] = 7.0  # kernel overwrites these rows (dG/du); the density must ignore them
    radii = torch.as_tensor(visible, device=device).int()
    return {"info": {"means2d": m2, "splitting_mats": sp, "radii": radii}}


def _step(density, params, opts, step, n, c, g, vis=None, device="cpu"):
    out = _outputs(n, c, g, [1] * n if vis is None else vis, device)
    density.before_backward(params, opts, step, out)
    density.after_backward(params, opts, step, out)


def _rot_independent(q):
    """Rotation by rotating basis vectors with v' = v + 2w(u x v) + 2 u x (u x v)."""
    q = F.normalize(q.double(), dim=-1)
    w, u = q[:, :1], q[:, 1:]
    cols = []
    for e in torch.eye(3, dtype=torch.float64):
        v = e.expand_as(u)
        t = 2 * torch.cross(u, v, dim=-1)
        cols.append(v + w * t + torch.cross(u, t, dim=-1))
    return torch.stack(cols, dim=-1)


# ---------------------------------------------------------------------------- C1 (CUDA)
def _reference(params, camera, bg, s):
    """Independent float64 Inria compositor (forward.cu conventions) on a one-tile image.

    Returns (c_i = o_i * sum_pix dL/dalpha_i, dL/d ndc_xy) for L = sum_p w_p . C_p with the
    weights w provided in s['w']. alpha clamp (0.99) is ignored in the backward like the kernel.
    """
    dd = torch.float64
    params = {k: v.detach().cpu() for k, v in params.items()}
    camera = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in camera.items()}
    W, H = camera["width"], camera["height"]
    means = params["means"].detach().to(dd)
    view = camera["viewmat"].to(dd)
    t = means @ view[:3, :3].T + view[:3, 3]
    fx, fy = float(camera["K"][0, 0]), float(camera["K"][1, 1])
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tz = t[:, 2]
    tx = (t[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
    J = torch.zeros(len(t), 2, 3, dtype=dd)
    J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
    R = _rot_independent(params["quats"].detach())
    sc = params["scales"].detach().to(dd).exp()
    cov3 = R @ torch.diag_embed(sc ** 2) @ R.transpose(1, 2)
    Wr = view[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2) + 0.3 * torch.eye(2, dtype=dd)
    conic = torch.linalg.inv(cov2)
    # projected centre: ndc of the full projection, ndc2Pix
    ndc = torch.stack([fx * t[:, 0] / tz * 2 / W, fy * t[:, 1] / tz * 2 / H], -1)
    ndc = ndc + torch.tensor([2 * float(camera["K"][0, 2]) / W - 1,
                              2 * float(camera["K"][1, 2]) / H - 1], dtype=dd)
    ndc = ndc.clone().requires_grad_(True)
    xy = ((ndc + 1) * torch.tensor([W, H], dtype=dd) - 1) * 0.5
    opac = params["opacities"].detach().to(dd).sigmoid()
    col = (C0 * params["sh0"].detach().to(dd)[:, 0] + 0.5).clamp_min(0)
    order = torch.argsort(tz)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dd), torch.arange(W, dtype=dd), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    e = torch.zeros(len(pix), len(means), dtype=dd, requires_grad=True)
    T = torch.ones(len(pix), dtype=dd)
    C = torch.zeros(len(pix), 3, dtype=dd)
    alive = torch.ones(len(pix), dtype=torch.bool)
    for i in order.tolist():
        d = xy[i] - pix
        a, b, c = conic[i, 0, 0], conic[i, 0, 1], conic[i, 1, 1]
        power = -0.5 * (a * d[:, 0] ** 2 + c * d[:, 1] ** 2) - b * d[:, 0] * d[:, 1]
        raw = opac[i] * power.exp()
        alpha = raw - (raw - 0.99).clamp_min(0).detach() + e[:, i]
        valid = alive & (power.detach() <= 0) & (alpha.detach() >= 1 / 255)
        test_T = T * (1 - alpha)
        stop = valid & (test_T.detach() < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        C = C + torch.where(use[:, None], col[i] * (alpha * T)[:, None], 0)
        T = torch.where(use, test_T, T)
    C = C + T[:, None] * bg.to(dd)
    L = (s["w"].to(dd).reshape(-1, 3) * C).sum()
    L.backward()
    return opac * e.grad.sum(0), ndc.grad


def dldg_kernel(method, device):
    _cuda(device)
    from .raster import render

    g = torch.Generator().manual_seed(5)
    n = 6
    means = torch.rand(n, 3, generator=g) * torch.tensor([0.6, 0.6, 1.0]) + torch.tensor(
        [-0.3, -0.3, 2.5])
    scales = torch.rand(n, 3, generator=g) * 0.12 + 0.04
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    params, _ = _params(means, scales, quats, 0.5, device)
    with torch.no_grad():
        params["opacities"].copy_(torch.logit(torch.rand(n, generator=g) * 0.5 + 0.2).to(device))
        params["sh0"].copy_(((torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8).to(device))
        params["shN"].zero_()
    W = H = 16  # one 16x16 tile: every Gaussian is tested at every pixel
    camera = {"viewmat": torch.eye(4, device=device), "width": W, "height": H,
              "K": torch.tensor([[18.0, 0, 8], [0, 18.0, 8], [0, 0, 1]], device=device)}
    bg = torch.tensor([0.2, 0.3, 0.1], device=device)
    w = torch.randn(3, H, W, generator=g)
    color, means2d, splitting, radii = render(params, camera, 0, bg)
    (color * w.to(device)).sum().backward()
    if not bool((radii > 0).all()):
        raise AssertionError(f"probe scene not fully visible: radii {radii.tolist()}")
    ref_c, ref_ndc = _reference(params, camera, bg.cpu(), {"w": w.permute(1, 2, 0)})
    got = splitting.grad[:, 0, 0].double().cpu()
    err = (got - ref_c).abs().max().item() / ref_c.abs().max().item()
    g2 = means2d.grad[:, :2].double().cpu()
    err2 = (g2 - ref_ndc).abs().max().item() / ref_ndc.abs().max().item()
    if err > 2e-3 or err2 > 2e-3:
        raise AssertionError(f"kernel dL/dG or screen gradient differs: rel {err:.2e}, {err2:.2e}")
    if ref_c.abs().min() == 0:
        raise AssertionError("degenerate probe: a Gaussian has zero reference dL/dG")
    return {"rel_err_dLdG": err, "rel_err_screen": err2, "dLdG": got.tolist()}


# ---------------------------------------------------------------------------- C1 density side
def splitting_matrix(method, device):
    n = 3
    scales = torch.tensor([[0.02, 0.05, 0.08], [0.06, 0.03, 0.09], [0.04, 0.07, 0.01]])
    quats = F.normalize(torch.tensor([[0.9, 0.2, -0.3, 0.1], [0.5, 0.5, 0.5, 0.5],
                                      [0.7, -0.1, 0.4, 0.6]]), dim=-1)
    params, opts = _params(torch.zeros(n, 3), scales, quats, 0.5, device)
    dens = _density(method, device)
    dens.initialize(params, opts, 1.0)
    cs = [[-2e-3, 1e-3, -5e-4], [-1e-3, 3e-3, 9e-4]]
    vis = [[1, 1, 1], [1, 1, 0]]
    for step, (c, v) in enumerate(zip(cs, vis), start=1):
        out = _outputs(n, c, [1e-3] * n, v, device)
        dens.before_backward(params, opts, step, out)
        with torch.no_grad():  # optimizer step between render and density: must not leak in
            params["scales"].add_(0.3)
        dens.after_backward(params, opts, step, out)
        with torch.no_grad():
            params["scales"].sub_(0.3)
    R = _rot_independent(quats)
    inv = R @ torch.diag_embed(1.0 / scales.double() ** 2) @ R.transpose(1, 2)
    cbar = torch.tensor([(cs[0][0] + cs[1][0]) / 2, (cs[0][1] + cs[1][1]) / 2, cs[0][2]],
                        dtype=torch.float64)
    expected = cbar[:, None, None] * inv
    from . import core

    got = core.averaged(dens.S, dens.denom).double().cpu()
    err = ((got - expected).abs().max() / expected.abs().max()).item()
    if err > 1e-4:
        raise AssertionError(f"accumulated splitting matrix differs from c*Sigma^-1: {err:.2e}")
    lam, vec = core.smallest_eig(core.averaged(dens.S, dens.denom))
    lam, vec = lam.double().cpu(), vec.double().cpu()
    k = scales[0].argmin()
    exp_lam = cbar[0] / scales[0, k].double() ** 2
    align = abs(float(vec[0] @ R[0][:, k]))
    if abs(lam[0] - exp_lam) / abs(exp_lam) > 1e-4 or align < 1 - 1e-5:
        raise AssertionError(f"eigen pair wrong: {lam[0]} vs {exp_lam}, |v.axis| {align}")
    if lam[1] < 0:
        raise AssertionError("positive c must give a PSD splitting matrix")
    return {"rel_err": err, "lambda_min": lam.tolist(), "axis_alignment": align}


# ---------------------------------------------------------------------------- C2
def _selected_after(method, device, c, g, **config):
    n = len(c)
    params, opts = _params(torch.rand(n, 3), torch.full((n, 3), 0.005), None, 0.5, device)
    dens = _density(method, device, **config)
    dens.initialize(params, opts, 1.0)
    _step(dens, params, opts, 600, n, c, g, device=device)
    ev = dens.state["events"][-1]
    return ev, len(params["means"])


def saddle_criterion(method, device):
    # [neg/high, neg/low, pos/high, pos/low, tiny-neg/high]: lambda = c / s^2 with s = 0.005
    c = [-1e-3, -1e-3, 1e-3, 1e-3, -1e-12]
    g = [1e-3, 1e-5, 1e-3, 1e-5, 1e-3]
    ev, n = _selected_after(method, device, c, g)
    if ev["selected"] != 1 or ev["clone"] != 1 or n != 6:
        raise AssertionError(f"default criterion should densify exactly Gaussian 0: {ev}")
    ev2, _ = _selected_after(method, device, c, g, options=["no_eig_cond"])
    ev3, _ = _selected_after(method, device, c, g, options=["no_uncertain"])
    if ev2["selected"] != 3 or ev3["selected"] != 2:
        raise AssertionError(f"ablations do not change the selection as expected: {ev2} {ev3}")
    return {"default": ev, "no_eig_cond": ev2["selected"], "no_uncertain": ev3["selected"]}


# ---------------------------------------------------------------------------- C3
def split_direction(method, device):
    s = torch.tensor([0.05, 0.08, 0.03])  # shortest axis = z -> v_min = +-e_z for c < 0
    out = {}
    for K in (1, 2):
        n = K + 1
        means = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])[:n]
        params, opts = _params(means, s.repeat(n, 1), None, 0.4, device)
        sh0 = params["sh0"].detach().clone()
        dens = _density(method, device)
        dens.initialize(params, opts, 1.0)
        c = [-1e-3] * K + [0.0]
        _step(dens, params, opts, 600, n, c, [1e-3] * n, device=device)
        if len(params["means"]) != n + K:
            raise AssertionError(f"K={K}: expected {n + K} Gaussians, got {len(params['means'])}")
        kids = params["means"].detach().cpu()[1:]  # survivor (index K) first, then 2K kids
        rows = []
        for j in range(2 * K):
            parent = j % K
            off = kids[j] - means[parent]
            if off[:2].abs().max() > 1e-6 or off[2].abs() < 1e-6:
                raise AssertionError(f"offset {off.tolist()} not along v_min = e_z")
            rows.append(float(off[2]))
        for p in range(K):
            same = rows[p] * rows[p + K] > 0
            # official repeat(2) + [0::2]/[1::2] signs: opposite for odd K, same side for even K
            if same != (K % 2 == 0):
                raise AssertionError(f"K={K}: sign pattern {rows} differs from the official one")
        sc = params["scales"].detach().exp().cpu()[1:]
        if (sc - s / 1.6).abs().max() > 1e-6:
            raise AssertionError("split scale must be s / 1.6")
        if (params["opacities"].detach().sigmoid().cpu() - 0.4).abs().max() > 1e-6:
            raise AssertionError("offspring opacity must be copied (official default)")
        if (params["sh0"].detach()[1:1 + K] - sh0[:K]).abs().max() > 0:
            raise AssertionError("offspring colour must be copied")
        out[f"K{K}_offsets"] = rows
    # clone: copy and original both at p + lr v_min
    params, opts = _params(torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]),
                           torch.tensor([[0.005, 0.008, 0.003]] * 2), None, 0.4, device)
    opts["means"].param_groups[0]["lr"] = 0.01
    dens = _density(method, device)
    dens.initialize(params, opts, 1.0)
    _step(dens, params, opts, 600, 2, [-1e-3, 0.0], [1e-3, 1e-3], device=device)
    m = params["means"].detach().cpu()
    if len(m) != 3 or (m[0] - m[2]).abs().max() > 1e-7:
        raise AssertionError("clone copy must coincide with the shifted original")
    if m[0, :2].abs().max() > 1e-7 or abs(abs(float(m[0, 2])) - 0.01) > 1e-6:
        raise AssertionError(f"clone shift must be lr * v_min, got {m[0].tolist()}")
    out["clone_shift"] = float(m[0, 2])
    return out


# ---------------------------------------------------------------------------- schedule
def schedule(method, device):
    means = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    scales = torch.tensor([[0.005, 0.006, 0.004], [0.5, 0.5, 0.5]])  # #1 > 0.1 extent
    params, opts = _params(means, scales, None, 0.5, device)
    dens = _density(method, device)
    dens.initialize(params, opts, 1.0)
    big_alive = []
    opac_after_reset = None
    for step in range(1, 15301):
        n = len(params["means"])
        c = [0.0] * n
        c[0] = -1e-3
        g = [0.0] * n
        g[0] = 1e-3
        _step(dens, params, opts, step, n, c, g, device=device)
        if step in (3000, 3100):
            big_alive.append(bool((params["scales"].exp().max(dim=1).values > 0.4).any()))
        if step == 3000:
            opac_after_reset = float(params["opacities"].detach().sigmoid().max())
    steps = [e["step"] for e in dens.state["events"]]
    if steps != list(range(600, 15000, 100)):
        raise AssertionError(f"densification steps wrong: first {steps[:3]} last {steps[-3:]}")
    if big_alive != [True, False]:
        raise AssertionError(f"world-size prune must start after step 3000: {big_alive}")
    if opac_after_reset is None or opac_after_reset > 0.01 + 1e-6:
        raise AssertionError(f"opacity reset at 3000 missing: max opacity {opac_after_reset}")
    return {"n_densify": len(steps), "first": steps[0], "last": steps[-1],
            "big_alive_3000_3100": big_alive, "max_opacity_after_reset": opac_after_reset}
