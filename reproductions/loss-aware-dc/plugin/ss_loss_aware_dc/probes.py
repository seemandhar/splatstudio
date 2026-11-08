"""Mechanism probes for the loss-aware density-control port. Each raises on failure and returns
measurements; each compares against an independently written computation."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from . import surrogates as sg


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _density(method, **config):
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")
    return create_density(replace(spec, config={**spec.config, **config}))


def _check(name, ok, **meas):
    if not ok:
        raise AssertionError(f"{name} failed: {meas}")


def _camera(device, dtype=torch.float32, W=40, H=36):
    K = torch.tensor([[50.0, 0, W / 2], [0, 52.0, H / 2], [0, 0, 1]], device=device, dtype=dtype)
    a = 0.15
    R = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]],
                     dtype=dtype, device=device)
    V = torch.eye(4, device=device, dtype=dtype)
    V[:3, :3], V[:3, 3] = R, torch.tensor([0.05, -0.03, 3.0], device=device, dtype=dtype)
    return {"K": K, "viewmat": V, "width": W, "height": H}


def _scene(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([1.6, 1.4, 0.8])
    scales = torch.log(0.05 + 0.15 * torch.rand(n, 3, generator=g))
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = 0.2 + 0.7 * torch.rand(n, generator=g)
    colors = torch.rand(n, 3, generator=g)
    return [t.to(device) for t in (means, scales, quats, opac, colors)]


# ------------------------------------------------------------------ reference compositor
def _reference_tile_stats(info, colors, c_total, G, omega, W, H):
    """Independent float64 compositor over gsplat's own per-tile depth-sorted lists, using
    autograd for dL/dalpha, dC/dalpha (Delta) and the screen-space gradient / Hessian."""
    ts = info["tile_size"]
    th, tw = info["tile_height"], info["tile_width"]
    offs = info["isect_offsets"][0].reshape(-1).tolist() + [len(info["flatten_ids"])]
    fid = info["flatten_ids"].long()
    n = colors.shape[0]
    z = torch.cat([info["means2d"][0], info["conics"][0]], dim=1).double().detach()
    z.requires_grad_(True)
    opac = info["opacities"][0].double()
    col = colors.double()
    out = torch.zeros(n, 24, dtype=torch.float64, device=z.device)
    F_terms = []
    for t in range(th * tw):
        ids = fid[offs[t]:offs[t + 1]]
        if len(ids) == 0:
            continue
        ty, tx = divmod(t, tw)
        ys, xs = torch.meshgrid(torch.arange(ty * ts, min((ty + 1) * ts, H)),
                                torch.arange(tx * ts, min((tx + 1) * ts, W)), indexing="ij")
        ys, xs = ys.reshape(-1).to(z.device), xs.reshape(-1).to(z.device)
        px, py = xs.double() + 0.5, ys.double() + 0.5
        zz = z[ids]
        dx = zz[None, :, 0] - px[:, None]
        dy = zz[None, :, 1] - py[:, None]
        sig = 0.5 * (zz[None, :, 2] * dx * dx + zz[None, :, 4] * dy * dy) + zz[None, :, 3] * dx * dy
        alpha_raw = torch.clamp(opac[ids][None] * torch.exp(-sig), max=0.999)
        valid = (sig >= 0) & (alpha_raw >= 1 / 255)
        a_det = (alpha_raw * valid).detach()
        tnext = torch.cumprod(1 - a_det, dim=1)
        stop = torch.cummax((tnext <= 1e-4).int(), dim=1).values.bool()
        valid = valid & ~stop
        a_leaf = (alpha_raw * valid).detach().requires_grad_(True)
        T = torch.cumprod(torch.cat([torch.ones_like(a_leaf[:, :1]), 1 - a_leaf[:, :-1]], 1), 1)
        Tfin = T[:, -1] * (1 - a_leaf[:, -1])
        C = ((a_leaf * T)[..., None] * col[ids][None]).sum(1) + Tfin[:, None]  # white bg
        pix = ys * W + xs
        Gp, Op = G.reshape(-1, 3)[pix].double(), omega.reshape(-1, 3)[pix].double()
        _check("reference-compositor", torch.allclose(C.detach(), c_total.reshape(-1, 3)[pix]
                                                      .double(), atol=2e-4),
               err=float((C.detach() - c_total.reshape(-1, 3)[pix]).abs().max()))
        w = torch.autograd.grad((Gp * C).sum(), a_leaf, retain_graph=True)[0]  # dL/dalpha
        dC = torch.stack([torch.autograd.grad(C[:, c].sum(), a_leaf, retain_graph=True)[0]
                          for c in range(3)], -1)
        delta = a_leaf.detach()[..., None] * dC  # Delta_c = T alpha (c - b) = alpha dC/dalpha
        vm = valid.double()
        dot = (Gp[:, None] * delta).sum(-1) * vm
        wsq = (Op[:, None] * delta * delta).sum(-1) * vm
        a1 = 1 - a_leaf.detach()
        for k, v in ((20, a1 * dot), (21, a1 * a1 * wsq), (22, dot), (23, wsq)):
            out[:, k].index_add_(0, ids, v.sum(0))
        F_terms.append((w.detach() * vm * alpha_raw).sum())
    Fz = sum(F_terms)
    gz = torch.autograd.grad(Fz, z, create_graph=True)[0]
    out[:, :5] = gz.detach()
    iu = torch.triu_indices(5, 5)
    Hrows = []
    for a in range(5):
        Hrows.append(torch.autograd.grad(gz[:, a].sum(), z, retain_graph=True)[0])  # [n,5]
    Hm = torch.stack(Hrows, 1)  # [n,5,5] (block diagonal per Gaussian)
    out[:, 5:20] = Hm[:, iu[0], iu[1]].detach()
    return out


def stats_kernel(method, device):
    """CUDA kernel g_z, H_z, g/h_dup, g/h_dec vs the independent float64 compositor; the
    kernel's g_z also equals gsplat's own means2d / conics gradients."""
    _cuda(device)
    from gsplat import rasterization

    from .kernel import surrogate_stats

    torch.manual_seed(0)
    cam = _camera(device)
    W, H = cam["width"], cam["height"]
    means, scales, quats, opac, colors = _scene(30, device)
    means.requires_grad_(True)
    rgb_a, alpha, info = rasterization(means, quats, scales.exp(), opac, colors,
                                       cam["viewmat"][None], cam["K"][None], W, H,
                                       sh_degree=None, packed=False)
    info["means2d"].retain_grad()
    info["conics"].retain_grad()
    rgb = (rgb_a[0] + (1 - alpha[0])).detach().requires_grad_(True)
    target = torch.rand(H, W, 3, device=device)
    loss = 0.8 * (rgb - target).abs().mean() + 0.2 * (1 - __import__(
        "splatstudio.losses", fromlist=["ssim"]).ssim(rgb, target))
    loss.backward()
    G = rgb.grad
    (rgb_a[0] + (1 - alpha[0])).backward(G)  # same dL/dI through gsplat's own backward
    omega = sg.proxy_curvature(rgb.detach(), target)
    k = surrogate_stats(info["means2d"][0].detach(), info["conics"][0].detach(),
                        info["opacities"][0].detach(), colors, info["isect_offsets"][0],
                        info["flatten_ids"], rgb.detach(), G, omega, W, H, info["tile_size"])
    ref = _reference_tile_stats(info, colors, rgb.detach(), G, omega, W, H).float()
    errs = {}
    for name, sl in (("g_z", slice(0, 5)), ("H_z", slice(5, 20)), ("dup", slice(20, 22)),
                     ("dec", slice(22, 24))):
        scale = ref[:, sl].abs().max().clamp_min(1e-20)
        errs[name] = float((k[:, sl] - ref[:, sl]).abs().max() / scale)
    gs_ref = torch.cat([info["means2d"].grad[0], info["conics"].grad[0]], 1)
    errs["g_z_vs_gsplat"] = float((k[:, :5] - gs_ref).abs().max() / gs_ref.abs().max())
    _check("stats-kernel", all(v < 2e-3 for v in errs.values()), **errs)
    _check("stats-kernel-nontrivial", bool((k[:, 22] != 0).sum() > 10), nz=int((k[:, 22] != 0).sum()))
    # gsplat's projected conic equals our z(phi) (screen_variables)
    zz = torch.stack([sg.screen_variables(torch.cat([means[i], scales[i]]), quats[i],
                                          cam["viewmat"], cam["K"], W, H) for i in range(30)])
    vis = (info["radii"][0] > 0).all(-1)
    zg = torch.cat([info["means2d"][0], info["conics"][0]], 1).detach()
    errs["z_phi_vs_gsplat"] = float(((zz - zg).abs() / zg.abs().clamp_min(1e-3))[vis].max())
    _check("screen-variables", errs["z_phi_vs_gsplat"] < 1e-3, **errs)
    return errs


def joint_split_matrix(method, device):
    """S_phi from (g_z, H_z) + second-order chain rule equals the brute-force float64 Hessian
    of sum_x w_x alpha(x; phi) (mixed term kept); dropping the mixed term changes only S_ps.
    Normalised eigmin = min of the Rayleigh quotient under the Gaussian metric (Eq. app_gaussian_
    local_metric), and children are p +- L v_p, log s +- v_s / sqrt 2."""
    dt = torch.float64
    cam = _camera("cpu", dt)
    W, H = cam["width"], cam["height"]
    g = torch.Generator().manual_seed(1)
    phi = torch.tensor([0.1, -0.05, 0.2, math.log(0.12), math.log(0.07), math.log(0.2)], dtype=dt)
    quat = F.normalize(torch.randn(4, generator=g, dtype=dt), dim=0)
    o = 0.7
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dt), torch.arange(W, dtype=dt), indexing="ij")
    px, py = xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5
    wpix = torch.randn(len(px), generator=g, dtype=dt)

    def alpha_of_z(z):
        dx, dy = z[0] - px, z[1] - py
        return o * torch.exp(-(0.5 * (z[2] * dx * dx + z[4] * dy * dy) + z[3] * dx * dy))

    def F_phi(ph):
        return (wpix * alpha_of_z(sg.screen_variables(ph, quat, cam["viewmat"], cam["K"], W, H))).sum()

    S_brute = torch.autograd.functional.hessian(F_phi, phi)
    z0 = sg.screen_variables(phi, quat, cam["viewmat"], cam["K"], W, H).detach()
    zl = z0.clone().requires_grad_(True)
    Fz = (wpix * alpha_of_z(zl)).sum()
    gz = torch.autograd.grad(Fz, zl, create_graph=True)[0]
    Hz = torch.stack([torch.autograd.grad(gz[a], zl, retain_graph=True)[0] for a in range(5)])
    iu = torch.triu_indices(5, 5)
    h15 = Hz[iu[0], iu[1]][None]
    S_full = sg.joint_split_matrix(phi[None], quat[None], gz.detach()[None], h15, cam["viewmat"],
                                   cam["K"], W, H, drop_mixed=False)[0]
    S_impl = sg.joint_split_matrix(phi[None], quat[None], gz.detach()[None], h15, cam["viewmat"],
                                   cam["K"], W, H, drop_mixed=True)[0]
    err = float((S_full - S_brute).abs().max() / S_brute.abs().max())
    _check("joint-split-chain-rule", err < 1e-8, err=err)
    diff = S_impl - S_full
    _check("mixed-term-only-ps", float(diff[:3, :3].abs().max()) == 0.0
           and float(diff[3:, 3:].abs().max()) == 0.0 and float(diff[:3, 3:].abs().max()) > 0,
           pp=float(diff[:3, :3].abs().max()), ps=float(diff[:3, 3:].abs().max()))
    # normalised eigen-direction (Eqs. 6-7) under d_G^2 = dp^T Sigma^-1 dp + 2 |ds|^2
    lam, v, J = sg.normalized_eigmin(S_impl[None], phi[None, 3:], quat[None])
    Rq = sg.quat_to_rotmat(quat)
    s = phi[3:].exp()
    Sig_inv = Rq @ torch.diag(1 / s**2) @ Rq.T
    dirs = torch.randn(4000, 6, generator=g, dtype=dt)
    dG = (torch.einsum("ni,ij,nj->n", dirs[:, :3], Sig_inv, dirs[:, :3])
          + 2 * (dirs[:, 3:] ** 2).sum(-1)).sqrt()
    dirs = dirs / dG[:, None]
    Ssym = 0.5 * (S_impl + S_impl.T)
    rq = torch.einsum("ni,ij,nj->n", dirs, Ssym, dirs)
    dphi = (J[0] @ v[0])
    rq_v = float(dphi @ Ssym @ dphi)
    _check("eigmin-metric", float(rq.min()) >= float(lam[0]) - 1e-9 and abs(rq_v - float(lam[0]))
           < 1e-9 * max(1.0, abs(float(lam[0]))), lam=float(lam[0]), rq_min=float(rq.min()))
    means_c, logs_c = sg.split_children(phi[None, :3], phi[None, 3:], v, J)
    Lv = Rq @ (s * v[0, :3])
    _check("split-children", torch.allclose(means_c[0], phi[:3] + Lv)
           and torch.allclose(means_c[1], phi[:3] - Lv)
           and torch.allclose(logs_c[0], phi[3:] + v[0, 3:] / math.sqrt(2))
           and torch.allclose(logs_c[1], phi[3:] - v[0, 3:] / math.sqrt(2)))
    return {"chain_rule_rel_err": err, "lambda_min": float(lam[0]), "rq_min": float(rq.min())}


# ------------------------------------------------------------------ Alg. 1 decisions
def _toy(n, device="cpu"):
    g = torch.Generator().manual_seed(3)
    vals = {"means": torch.randn(n, 3, generator=g), "scales": torch.full((n, 3), math.log(0.05)),
            "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1),
            "opacities": torch.full((n,), 0.0), "sh0": torch.rand(n, 1, 3, generator=g),
            "shN": torch.rand(n, 15, 3, generator=g)}
    params = {k: torch.nn.Parameter(v.to(device)) for k, v in vals.items()}
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for k, p in params.items():
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return params, opts


def _fill(d, n):
    """Synthetic accumulators (one view each). Rows:
    0 split (stationary, negative curvature, uncertain)   1 like 0 but moving (|g_pos| > tau)
    2 like 0 but positive curvature                       3 clone (eta_dup = 8e-3, o = 0.9: child survives the o < 0.005 prune)
    4 clone candidate with eta_dup > 1e-2                  5 decay (eta_dec = 5e-3)
    6 decay candidate with eta_dec < tau_decay            7 split AND clone/decay-eligible
    8 like 0 but not uncertain (low screen gradient)       9 nothing
    """
    d._reset_stats(n, "cpu")
    d.denom[:] = 1
    d.g2d[:] = 1e-3
    d.g2d[8] = 1e-5
    neg = -torch.diag(torch.tensor([1.0, 2.0, 0.5, 1.0, 1.0, 1.0]))
    for i in (0, 1, 7, 8):
        d.S[i] = neg
    d.S[2] = torch.eye(6)
    d.S[[3, 4, 5, 6, 9]] = torch.eye(6)
    d.g_pos[1] = torch.tensor([1e-3, 0, 0])
    # sur = g_dup, h_dup, g_dec, h_dec
    d.sur[3] = torch.tensor([-8e-3, 1.0, 0.0, 1.0])
    d.sur[4] = torch.tensor([-1.0, 1.0, 0.0, 1.0])
    d.sur[5] = torch.tensor([0.0, 1.0, 5e-3, 1.0])
    d.sur[6] = torch.tensor([0.0, 1.0, 1e-5, 1.0])
    d.sur[7] = torch.tensor([-1e-3, 1.0, 5e-3, 1.0])


def decisions(method, device):
    """Eqs. 8, 13, 17 and Alg. 1 order on synthetic statistics: only stationary + negative
    curvature (+ uncertain) Gaussians split into 2 children with opacity o/2; clone appends
    eta*o duplicates within (tau_clone, 1e-2]; decay scales o by (1-eta) within (tau_decay, 1e-2];
    sets are disjoint with split priority; switching a mechanism off changes the outcome."""
    n = 10
    d = _density(method)
    params, opts = _toy(n)
    d.initialize(params, opts, 1.0)
    with torch.no_grad():
        params["opacities"][3] = math.log(0.9 / 0.1)
    _fill(d, n)
    dec = d.decisions(params)
    exp_split = {0, 7}
    got = {k: set(dec[k].nonzero()[:, 0].tolist()) for k in ("split", "clone", "decay")}
    _check("split-set", got["split"] == exp_split, got=got)
    _check("clone-set", got["clone"] == {3}, got=got)
    _check("decay-set", got["decay"] == {5}, got=got)
    o_before = params["opacities"].sigmoid().detach().clone()
    p_before = {k: params[k].detach().clone() for k in params}
    d._densify(params, opts, step=600)
    nn_ = len(params["means"])
    _check("population", nn_ == n - 2 + 1 + 4, n=nn_)  # -2 parents, +1 clone, +4 children
    o_after = params["opacities"].sigmoid().detach()
    # survivors keep order: [0..9 minus {0,7}] + clone + 4 children
    keep = [i for i in range(n) if i not in exp_split]
    pos5 = keep.index(5)
    _check("decay-value", abs(float(o_after[pos5]) - float(o_before[5]) * (1 - 5e-3)) < 1e-6)
    clone_o = float(o_after[len(keep)])
    _check("clone-value", abs(clone_o - 8e-3 * float(o_before[3])) < 1e-6
           and torch.allclose(params["means"][len(keep)], p_before["means"][3]), clone_o=clone_o)
    ch = params["opacities"][len(keep) + 1:].sigmoid()
    _check("split-opacity", torch.allclose(ch, torch.full_like(ch, 0.25), atol=1e-6))
    # children positions are symmetric about the parent
    c = params["means"][len(keep) + 1:].detach()
    mid = 0.5 * (c[:2] + c[2:])
    _check("split-symmetric", torch.allclose(mid, p_before["means"][[0, 7]], atol=1e-6))
    # ablations change the outcome (a probe that would pass without the mechanism is useless)
    for flag, key in (("split", "split"), ("clone", "clone"), ("decay", "decay")):
        d2 = _density(method, **{flag: False})
        p2, o2 = _toy(n)
        d2.initialize(p2, o2, 1.0)
        _fill(d2, n)
        _check(f"ablation-{flag}", int(d2.decisions(p2)[key].sum()) == 0)
    d3 = _density(method, require_uncertain=False)
    p3, o3 = _toy(n)
    d3.initialize(p3, o3, 1.0)
    _fill(d3, n)
    _check("uncertain-filter", 8 in d3.decisions(p3)["split"].nonzero()[:, 0].tolist())
    return {k: sorted(v) for k, v in got.items()} | {"n_after": nn_}


def schedule(method, device):
    """Densify only at s > 500, s % 100 == 0, s < 15000; no global opacity reset ever."""
    d = _density(method)
    params, opts = _toy(4)
    d.initialize(params, opts, 1.0)
    calls = []
    d._accumulate = lambda outputs: None
    d._densify = lambda p, o, s: calls.append(s)
    o0 = params["opacities"].detach().clone()
    for s in list(range(50, 30001, 50)) + [3000, 6000, 15000]:
        d.after_backward(params, opts, s, {})
    exp = [s for s in list(range(50, 30001, 50)) + [3000, 6000, 15000]
           if 500 < s < 15000 and s % 100 == 0]
    _check("densify-schedule", calls == exp, first=calls[:3], n=len(calls))
    _check("no-opacity-reset", torch.equal(params["opacities"].detach(), o0))
    return {"densify_events": len(calls)}


def surrogate_first_order(method, device):
    """The kernel's g_dup / g_dec equal the true first-order image effect of actually adding
    a duplicate with opacity eta*o (resp. scaling o by 1-eta), measured by re-rendering with
    gsplat: sum_x dL/dI . (C(eta) - C(0)) / eta -> g_dup (resp. -g_dec)."""
    _cuda(device)
    from gsplat import rasterization

    from .kernel import surrogate_stats

    cam = _camera(device)
    W, H = cam["width"], cam["height"]
    means, scales, quats, opac, colors = _scene(30, device, seed=5)

    def render(m, s, q, o, c):
        r, a, info = rasterization(m, q, s.exp(), o, c, cam["viewmat"][None], cam["K"][None],
                                   W, H, sh_degree=None, packed=False)
        return r[0] + (1 - a[0]), info

    rgb, info = render(means, scales, quats, opac, colors)
    rgb = rgb.detach().requires_grad_(True)
    target = torch.rand(H, W, 3, device=device)
    (0.8 * (rgb - target).abs().mean()).backward()
    G = rgb.grad
    omega = sg.proxy_curvature(rgb.detach(), target)
    k = surrogate_stats(info["means2d"][0].detach(), info["conics"][0].detach(),
                        info["opacities"][0].detach(), colors, info["isect_offsets"][0],
                        info["flatten_ids"], rgb.detach(), G, omega, W, H, info["tile_size"])
    # C is exactly linear in each Gaussian's opacity; eta is large enough that the duplicate's
    # alpha stays mostly above gsplat's 1/255 cut-off (the remaining tail sets the tolerance).
    eta = 0.5
    errs = []
    order = k[:, 22].abs().argsort(descending=True)[:5].tolist()
    for i in order:
        cat = lambda t: torch.cat([t, t[i:i + 1]])  # noqa: E731
        o_new = torch.cat([opac, eta * opac[i:i + 1]])
        c_dup, _ = render(cat(means), cat(scales), cat(quats), o_new, cat(colors))
        g_dup_fd = float((G * (c_dup - rgb.detach())).sum() / eta)
        o_dec = opac.clone()
        o_dec[i] *= 1 - 0.01
        c_dec, _ = render(means, scales, quats, o_dec, colors)
        g_dec_fd = float(-(G * (c_dec - rgb.detach())).sum() / 0.01)
        errs.append((abs(g_dup_fd - float(k[i, 20])) / abs(float(k[i, 20])),
                     abs(g_dec_fd - float(k[i, 22])) / abs(float(k[i, 22]))))
    worst = max(max(e) for e in errs)
    _check("surrogate-first-order", worst < 0.05, errs=errs)
    return {"worst_rel_err": worst}


def proxy_curvature(method, device):
    """omega vs an explicit per-pixel window sum (independent of conv2d) and Charbonnier
    rho'' vs autograd; removing the SSIM proxy or the Charbonnier term breaks the match."""
    g = torch.Generator().manual_seed(2)
    H, W = 20, 23
    rgb, tgt = torch.rand(H, W, 3, generator=g, dtype=torch.float64), torch.rand(
        H, W, 3, generator=g, dtype=torch.float64)
    om = sg.proxy_curvature(rgb, tgt)
    coords = torch.arange(11, dtype=torch.float64) - 5
    gw = torch.exp(-(coords**2) / (2 * 1.5**2))
    gw = gw / gw.sum()
    win = gw[:, None] * gw[None, :]

    def wsum(img, y, x, w):  # zero padded window sum at (y, x)
        acc = 0.0
        for a in range(11):
            for b in range(11):
                yy, xx = y + a - 5, x + b - 5
                if 0 <= yy < H and 0 <= xx < W:
                    acc = acc + w[a, b] * img[yy, xx]
        return acc

    mu = torch.zeros(H, W, 3, dtype=torch.float64)
    ex2 = torch.zeros_like(mu)
    for y in range(H):
        for x in range(W):
            mu[y, x] = wsum(tgt, y, x, win)
            ex2[y, x] = wsum(tgt * tgt, y, x, win)
    nu = ex2 - mu * mu
    dmu = 1 / (2 * mu * mu + 1e-4 + 1e-8)
    dnu = 1 / (2 * nu + 9e-4 + 1e-8)
    errs = []
    for (y, x) in ((0, 0), (7, 9), (19, 22), (10, 3)):
        ws = 2 * wsum(dmu, y, x, win * win) + 2 * (wsum(dnu, y, x, win) - wsum(dnu, y, x, win * win))
        r = (rgb[y, x] - tgt[y, x]).clone().requires_grad_(True)
        rho = (torch.sqrt(r * r + 1e-6) - 1e-3).sum()
        g1 = torch.autograd.grad(rho, r, create_graph=True)[0]
        rho2 = torch.stack([torch.autograd.grad(g1[c], r, retain_graph=True)[0][c] for c in range(3)])
        ref = 0.8 * rho2 + 0.2 * 1e-4 * ws
        errs.append(float(((om[y, x] - ref).abs() / ref.abs()).max()))
    _check("proxy-curvature", max(errs) < 1e-9, errs=errs)
    return {"max_rel_err": max(errs)}


def pipeline_step(method, device):
    """Built-in renderer-gradient probe, but through this method's loss (which hands the target
    to the density): finite nonzero screen gradient, statistics accumulate for visible Gaussians,
    a densification step at s = 600 keeps params / optimizers aligned. The built-in probe cannot
    be used because it bypasses the method's losses (the density then has no target and raises)."""
    _cuda(device)
    from splatstudio.components import build
    from splatstudio.probes import sample_scene
    from splatstudio.state import check_state

    renderer, density, losses = build(method)
    params, opts = sample_scene(method, device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
              "width": 16, "height": 16}
    target = (torch.arange(16, device=device)[None, :, None] % 2).expand(16, 16, 3).float()
    density.initialize(params, opts)
    for step in (599, 600):
        outputs = renderer.render(params, camera, sh_degree=0)
        outputs["step"] = step
        density.before_backward(params, opts, step, outputs)
        sum(loss(params, outputs, target) for _, loss in losses).backward()
        xy = outputs["info"]["means2d"]
        if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
            raise AssertionError("Renderer has no finite nonzero screen-space gradient")
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if step == 599:
            density.after_backward(params, opts, step, outputs)
            seen = int((density.denom > 0).sum())
            _check("statistics-accumulated", seen > 0 and bool(torch.isfinite(density.S).all())
                   and bool((density.sur[:, 1] >= 0).all()), seen=seen)
        else:
            density.after_backward(params, opts, step, outputs)
    check_state(method, params, opts)
    return {"visible_with_stats": seen, "events": density.state["events"][-1:]}


def joint_split_cuda(method, device):
    """CUDA hyper-dual S_phi accumulation (used in training) equals the functorch torch
    reference `surrogates.joint_split_matrix` (itself checked against a brute-force Hessian),
    with and without the dropped mixed term, including frustum-clamped Gaussians."""
    _cuda(device)
    from .kernel import joint_split_accumulate

    cam = _camera(device)
    g = torch.Generator().manual_seed(7)
    n = 500
    means = ((torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([6.0, 6.0, 2.0])).to(device)
    logs = torch.log(0.02 + 0.3 * torch.rand(n, 3, generator=g)).to(device)
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device)
    stats = torch.randn(n, 24, generator=g).to(device)
    ids = torch.arange(0, n, 2, device=device)
    errs, outs = {}, []
    for drop in (True, False):
        S_acc = torch.zeros(n, 6, 6, device=device)
        joint_split_accumulate(ids, means, logs, quats, stats, cam["viewmat"], cam["K"],
                               cam["width"], cam["height"], drop, S_acc)
        ref = sg.joint_split_matrix(torch.cat([means, logs], 1)[ids].double(),
                                    quats[ids].double(), stats[ids, :5].double(),
                                    stats[ids, 5:20].double(), cam["viewmat"].double(),
                                    cam["K"].double(), cam["width"], cam["height"],
                                    drop_mixed=drop)
        rel = ((S_acc[ids].double() - ref).abs().amax((1, 2))
               / ref.abs().amax((1, 2)).clamp_min(1e-12))
        errs[f"drop_mixed={drop}"] = float(rel.max())
        outs.append(S_acc[ids])
        _check("untouched-rows", float(S_acc[1::2].abs().max()) == 0.0)
    _check("joint-split-cuda", all(v < 1e-4 for v in errs.values()), **errs)
    d = (outs[0] - outs[1]).abs()
    _check("mixed-term-cuda", float(d[:, :3, :3].max()) == 0.0 and float(d[:, :3, 3:].max()) > 0)
    return errs
