"""Mechanism probes for the Deblurring 3DGS port. Each raises if its mechanism is removed or
changed; independent formulas are written here rather than imported from the components."""

import math

import torch
import torch.nn.functional as F
from gsplat import rasterization
from splatstudio.components import build
from splatstudio.state import check_state, initialize

from . import bank as bank_mod
from .bank import BlurBank
from .density import (
    DEFAULTS,
    add_points,
    expon_lr,
    num_extra_points,
)
from .losses import ssim_same


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _need_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# -- independent reference formulas ------------------------------------------------------------
def ref_embed(x, L):
    out = [x]
    for k in range(L):
        out += [torch.sin(x * 2.0 ** k), torch.cos(x * 2.0 ** k)]
    return torch.cat(out, -1)


def ref_rotation(q):
    """R = I + 2 w K + 2 K^2 with K = [v]_x: equals the Inria matrix for any (unnormalised) q."""
    w, v = q[..., 0], q[..., 1:]
    K = torch.zeros(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)
    K[..., 0, 1], K[..., 0, 2] = -v[..., 2], v[..., 1]
    K[..., 1, 0], K[..., 1, 2] = v[..., 2], -v[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -v[..., 1], v[..., 0]
    eye = torch.eye(3, device=q.device, dtype=q.dtype).expand_as(K)
    return eye + 2 * w[..., None, None] * K + 2 * K @ K


def ref_cov(s, q):
    R = ref_rotation(q)
    return R @ torch.diag_embed(s * s) @ R.transpose(-1, -2)


def ref_deltas(net, means, scales, quats, center, lam_s, lam_p, max_clamp):
    n = len(means)
    x = torch.cat([ref_embed(means, net.res_pos), ref_embed(center.expand(n, 3), net.res_view),
                   scales, quats], -1)
    h = x
    for layer in net.linears:
        h = layer(h)
    ds = (lam_s * net.s(h) + 1 - lam_s).clamp(1.0, max_clamp)
    dr = (lam_s * net.r(h) + 1 - lam_s).clamp(1.0, max_clamp)
    dx = lam_p * net.p(h) if net.pos_delta else None
    return ds, dr, dx


def _scene(device, n=300, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1, torch.rand(n, 1, generator=g) * 2 + 3],
                      1)
    params = {
        "means": means, "scales": torch.log(torch.rand(n, 3, generator=g) * 0.05 + 0.02),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.rand(n, generator=g) * 0.5 + 0.3),
        "sh0": torch.rand(n, 1, 3, generator=g), "shN": torch.randn(n, 15, 3, generator=g) * 0.05,
    }
    params = {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]], device=device),
              "width": 64, "height": 48}
    return params, camera


def _raster(params, camera, means, covars=None, sh=1):
    colors = torch.cat((params["sh0"], params["shN"]), 1)
    kw = dict(quats=params["quats"], scales=params["scales"].exp()) if covars is None else dict(
        quats=None, scales=None, covars=covars)
    out, alpha, _ = rasterization(means=means, opacities=params["opacities"].sigmoid(),
                                  colors=colors, viewmats=camera["viewmat"][None],
                                  Ks=camera["K"][None], width=camera["width"],
                                  height=camera["height"], sh_degree=sh, packed=False,
                                  rasterize_mode="classic", **kw)
    return out[0]


def _bank_for(method, device):
    cfg = {k[5:]: v for k, v in _component(method, "density").config.items()
           if k.startswith("bank_")}
    return BlurBank(cfg, device)


# -- probes --------------------------------------------------------------------------------------
def mlp_architecture(method, device):
    """GTnet: 91 input channels (PE pos 3 / view 10 with input, + s 3 + q 4), 3 shared
    Linear+ReLU, linear heads, xavier gains, clamp to [1, 1.1], detached inputs."""
    bank = _bank_for(method, device)
    net, c = bank.net, bank.cfg
    _check(net.in_channels == 91, f"input channels {net.in_channels} != 91")
    lin = [m for m in net.linears if isinstance(m, torch.nn.Linear)]
    _check(len(lin) == 3 and all(m.out_features == 64 for m in lin), "trunk is not 3 x 64")
    m = c["num_moments"]
    heads = {"s": 3, "r": 4} if not bank.motion else {"s": 3 * (m + 1), "r": 4 * (m + 1),
                                                       "p": 3 * m}
    for name, out in heads.items():
        layer = getattr(net, name)
        _check(layer.out_features == out, f"head {name} has {layer.out_features} outputs")
        gain = 0.1 if out in (2, 3) else 1.0
        std = gain * math.sqrt(2.0 / (64 + out))
        _check(abs(float(layer.weight.std()) / std - 1) < 0.5, f"head {name} init std")
        _check(float(layer.bias.abs().max()) == 0.0, f"head {name} bias not zero")
    x = torch.rand(5, 3, device=device)
    _check(torch.allclose(bank_mod.embed(x, 3), ref_embed(x, 3)), "positional encoding differs")
    gen = torch.Generator(device="cpu").manual_seed(11)
    means = torch.rand(256, 3, generator=gen).to(device).requires_grad_(True)
    scales = torch.rand(256, 3, generator=gen).to(device).requires_grad_(True)
    quats = F.normalize(torch.randn(256, 4, generator=gen), dim=-1).to(device)
    center = torch.rand(3, generator=gen).to(device)
    with torch.no_grad():
        for layer in heads:
            b = getattr(net, layer).bias
            b.copy_((torch.rand(b.shape, generator=gen) * 120 - 60).to(device))
    ds, dr, dx = bank.deltas(means, scales, quats, center)
    rs, rr, rx = ref_deltas(net, means, scales, quats, center, c["lambda_s"], c["lambda_p"],
                            c["max_clamp"])
    _check(torch.allclose(ds, rs) and torch.allclose(dr, rr), "deltas differ from reference")
    _check(float(ds.min()) >= 1.0 and float(ds.max()) <= c["max_clamp"] + 1e-6, "clamp range")
    _check(float(ds.max()) > 1.0 and float(ds.min()) == 1.0, "clamp never active")
    if bank.motion:
        _check(torch.allclose(dx, rx), "position deltas differ")
    (ds.sum() + dr.sum()).backward()
    _check(means.grad is None and scales.grad is None, "MLP inputs are not detached")
    return {"in_channels": net.in_channels, "heads": heads, "ds_range": [float(ds.min()),
                                                                          float(ds.max())]}


def defocus_blur(method, device):
    """Training render = gsplat with Sigma(q~ unnormalised) from the MLP deltas; eval render =
    plain gsplat; MLP receives gradient; transform enlarges (ds >= 1)."""
    _need_cuda()
    renderer, _, _ = build(method)
    params, camera = _scene(device)
    bank = _bank_for(method, device)
    if bank.motion:
        bank = BlurBank({**bank.cfg, "mode": "defocus"}, device)
    with torch.no_grad():
        bank.net.s.bias.fill_(8.0)  # ds = 1.07
        bank.net.r.bias.fill_(5.0)  # dr = 1.04
    old, bank_mod.ACTIVE = bank_mod.ACTIVE, bank
    try:
        out = renderer.render(params, camera, sh_degree=1)
        with torch.no_grad():
            sharp = renderer.render(params, camera, sh_degree=1)["rgb"]
    finally:
        bank_mod.ACTIVE = old
    c = bank.cfg
    q = F.normalize(params["quats"], dim=-1)
    center = torch.zeros(3, device=device)
    ds, dr, _ = ref_deltas(bank.net, params["means"].detach(), params["scales"].exp().detach(),
                           q.detach(), center, c["lambda_s"], c["lambda_p"], c["max_clamp"])
    with torch.no_grad():
        ref = _raster(params, camera, params["means"],
                      ref_cov(params["scales"].exp() * ds, q * dr))
        ref_norm = _raster(params, camera, params["means"],
                           ref_cov(params["scales"].exp() * ds, F.normalize(q * dr, dim=-1)))
        plain = _raster(params, camera, params["means"])
    e_blur = float((out["rgb"] - ref).abs().max())
    e_sharp = float((sharp - plain).abs().max())
    d_blur = float((out["rgb"].detach() - sharp).abs().mean())
    d_norm = float((ref - ref_norm).abs().mean())
    _check(e_blur < 1e-4, f"training render differs from reference ({e_blur})")
    _check(e_sharp < 1e-5, f"evaluation render is not the sharp 3DGS render ({e_sharp})")
    _check(d_blur > 1e-3, "blur transform has no effect")
    _check(d_norm > 1e-5, "quaternion delta was renormalised (Inria uses it unnormalised)")
    out["rgb"].sum().backward()
    g = sum(float(p.grad.abs().sum()) for p in bank.net.parameters() if p.grad is not None)
    _check(g > 0, "MLP receives no gradient")
    # with deltas clamped to 1 the training render equals the sharp render
    with torch.no_grad():
        bank.net.s.bias.fill_(-1e4)
        bank.net.r.bias.fill_(-1e4)
    bank_mod.ACTIVE = bank
    try:
        same = renderer.render(params, camera, sh_degree=1)["rgb"]
    finally:
        bank_mod.ACTIVE = old
    e_id = float((same.detach() - sharp).abs().max())
    _check(e_id < 1e-4, f"identity deltas do not reproduce the sharp render ({e_id})")
    return {"blur_vs_ref_max": e_blur, "sharp_vs_plain_max": e_sharp, "blur_vs_sharp_mean": d_blur,
            "unnormalised_vs_normalised_mean": d_norm, "identity_vs_sharp_max": e_id,
            "mlp_grad_l1": g}


def motion_blur(method, device):
    """Training render = mean of M+1 = 5 renders: (x, ds[...,M], dr[...,M]) and
    (x + lambda_p dx_i, ds_i, dr_i); differs from a single render."""
    _need_cuda()
    renderer, _, _ = build(method)
    params, camera = _scene(device, seed=1)
    bank = _bank_for(method, device)
    if not bank.motion:
        bank = BlurBank({**bank.cfg, "mode": "motion"}, device)
    with torch.no_grad():
        bank.net.p.bias.uniform_(-3.0, 3.0)  # shifts up to 0.03 world units
        bank.net.s.bias.uniform_(-5.0, 8.0)
        bank.net.r.bias.uniform_(-5.0, 5.0)
    old, bank_mod.ACTIVE = bank_mod.ACTIVE, bank
    try:
        out = renderer.render(params, camera, sh_degree=1)
    finally:
        bank_mod.ACTIVE = old
    c, M = bank.cfg, bank.cfg["num_moments"]
    q = F.normalize(params["quats"], dim=-1)
    ds, dr, dx = ref_deltas(bank.net, params["means"].detach(), params["scales"].exp().detach(),
                            q.detach(), torch.zeros(3, device=device), c["lambda_s"],
                            c["lambda_p"], c["max_clamp"])
    ds, dr, dx = ds.view(-1, 3, M + 1), dr.view(-1, 4, M + 1), dx.view(-1, 3, M)
    s = params["scales"].exp()
    with torch.no_grad():
        imgs = [_raster(params, camera, params["means"], ref_cov(s * ds[..., M], q * dr[..., M]))]
        imgs += [_raster(params, camera, params["means"] + dx[..., i],
                         ref_cov(s * ds[..., i], q * dr[..., i])) for i in range(M)]
    ref = torch.stack(imgs).mean(0)
    e = float((out["rgb"] - ref).abs().max())
    d_single = float((out["rgb"].detach() - imgs[0]).abs().mean())
    _check(out["n_renders"] == M + 1 == 5, f"{out['n_renders']} renders instead of 5")
    _check(e < 1e-4, f"motion blur differs from the mean of 5 renders ({e})")
    _check(d_single > 1e-4, "averaging over moments has no effect")
    out["rgb"].sum().backward()
    _check(bank.net.p.weight.grad is not None and float(bank.net.p.weight.grad.abs().sum()) > 0,
           "position head receives no gradient")
    _check(out["info"]["means2d"].shape[0] == 5, "per-render screen statistics missing")
    return {"renders": out["n_renders"], "mean_vs_ref_max": e, "mean_vs_render0_mean": d_single}


def extra_points(method, device):
    """Alg. 1 + add_points: N_p, distance-threshold rejection, distance-weighted SH, full
    re-initialisation and fresh Adam, against a brute-force loop of the official algorithm."""
    cfg = {**DEFAULTS, **_component(method, "density").config, "knn_chunk": 7}
    g = torch.Generator(device="cpu").manual_seed(3)
    pts = torch.cat([torch.rand(60, 3, generator=g) * 4, torch.rand(40, 3, generator=g) + 10])
    cols = torch.rand(100, 3, generator=g)
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["shN"].normal_(0, 0.1)
        params["scales"].fill_(-3.0)
        params["opacities"].fill_(2.0)
    for name, p in params.items():  # give every Adam a state
        p.grad = torch.ones_like(p)
        opts[name].step()
        opts[name].zero_grad()
    old_means = params["means"].detach().clone()
    old_feat = torch.cat((params["sh0"], params["shN"]), 1).detach().clone()
    pmin, pmax = old_means.amin(0), old_means.amax(0)
    n_req = num_extra_points(pmin, pmax, cfg)
    _check(n_req == int(min(float((pmax - pmin).prod()) / 1.1 ** 3, 200000)), "N_p formula")
    gen = torch.Generator(device=device).manual_seed(5)
    gen_ref = torch.Generator(device=device).manual_seed(5)
    added = add_points(params, opts, cfg, pmin, pmax, generator=gen)
    # brute-force reference (official loop)
    srt = old_means.sort(0)[0]
    lo, hi = srt[cfg["pts_add_bound"]], srt[-cfg["pts_add_bound"]]
    samples = torch.rand((n_req, 3), device=device, generator=gen_ref) * (hi - lo) + lo
    keep, feats = [], []
    for p in samples:
        dist = torch.norm(old_means - p, dim=1)
        idx = torch.topk(-dist, cfg["pts_N_intpl"]).indices
        w = dist[idx]
        mask = w < cfg["pts_dist"]
        if mask.sum() == 0:
            continue
        w = w[mask] / w[mask].sum()
        keep.append(p)
        feats.append((old_feat[idx][mask] * w[:, None, None]).sum(0))
    _check(added == len(keep) and 0 < added < n_req, f"kept {added} vs reference {len(keep)}")
    n = len(old_means) + added
    _check(len(params["means"]) == n, "population size")
    _check(torch.allclose(params["means"][: len(old_means)], old_means), "old positions changed")
    _check(torch.allclose(params["means"][len(old_means):], torch.stack(keep), atol=1e-6),
           "sample positions")
    feat = torch.cat((params["sh0"], params["shN"]), 1)
    _check(torch.allclose(feat[: len(old_means)], old_feat), "old SH changed")
    _check(torch.allclose(feat[len(old_means):], torch.stack(feats), atol=1e-5),
           "interpolated SH differs from the official loop")
    d2 = (params["means"][:, None] - params["means"][None]).square().sum(-1)  # exact
    d2.fill_diagonal_(float("inf"))
    ref_s = d2.topk(3, largest=False).values.mean(1).clamp_min(1e-7).sqrt().log()
    _check(torch.allclose(params["scales"], ref_s[:, None].expand(-1, 3), atol=1e-5),
           "scales not re-initialised from 3-NN")
    _check(torch.allclose(params["quats"], torch.tensor([1.0, 0, 0, 0], device=device)
                          .expand(n, 4)), "rotations not reset")
    _check(torch.allclose(params["opacities"].sigmoid(), torch.full((n,), 0.1, device=device)),
           "opacities not reset to 0.1")
    _check(all(len(o.state) == 0 for o in opts.values()), "Adam state not reset")
    check_state(method, params, opts)
    return {"requested": n_req, "added": added, "population": n}


def depth_pruning(method, device):
    """Opacity prune threshold t_p / (1 + (w_p - 1)(z - z_min)/(z_max - z_min)): a far Gaussian
    with t_p / w_p < o < t_p survives, a near one is pruned; plain 3DGS pruning removes both."""
    cfg = _component(method, "density").config
    t_p = {**DEFAULTS, **cfg}["densify_prune_threshold"]
    kept = {}
    for depth in (True, False):
        pts = torch.tensor([[0.0, 0, 1.0], [0, 0, 6.0], [0, 0, 11.0]], device=device)
        params, opts = initialize(method, pts, torch.full((3, 3), 0.5, device=device))
        o = torch.tensor([0.6 * t_p, 0.6 * t_p, 1.0 - 1e-3], device=device)  # last = anchor
        with torch.no_grad():
            params["opacities"].copy_(torch.logit(o))
        from .density import densify_and_prune
        c = {**DEFAULTS, **cfg, "densify_with_depth": depth}
        densify_and_prune(params, opts, torch.zeros(3, device=device), c, extent=1.0)
        kept[depth] = sorted(float(z) for z in params["means"][:, 2])
    # z range [1, 11]: z = 6 -> w = 2 -> threshold t_p/2 < 0.6 t_p (kept); z = 1 -> pruned
    _check(kept[True] == [6.0, 11.0], f"depth-based pruning kept {kept[True]}")
    _check(kept[False] == [11.0], f"plain pruning kept {kept[False]}")
    return {"kept_depth": kept[True], "kept_plain": kept[False]}


def schedule(method, device):
    """Paper/code schedule values; no opacity reset; xyz lr log-linear over 30000."""
    renderer_cfg = _component(method, "density").config
    c = {**DEFAULTS, **renderer_cfg}
    motion = c["bank_mode"] == "motion"
    want = (5e-4, 1e-2) if motion else (2e-4, 5e-3)
    _check((c["densify_grad_threshold"], c["densify_prune_threshold"]) == want, "thresholds")
    _check((c["densify_from_iter"], c["densify_until_iter"], c["densification_interval"],
            c["pts_iter"]) == (500, 15000, 100, 2500), "densification schedule")
    _check(c["bank_lambda_s"] == 0.01 and c["bank_lambda_p"] == 0.01, "lambda_s / lambda_p")
    ext = 2.0
    lr = [expon_lr(s, 1.6e-3 * ext, 1.6e-5 * ext, 30000) for s in (0, 20000, 30000, 40000)]
    _check(abs(lr[0] - 3.2e-3) < 1e-9 and abs(lr[2] - 3.2e-5) < 1e-11 and lr[3] == lr[2],
           "xyz lr endpoints")
    _check(abs(lr[1] - 3.2e-3 * 0.01 ** (2 / 3)) < 1e-9, "xyz lr is not log-linear over 30000")
    # density over steps 2999..3001 (3DGS would reset opacity at 3000): opacities untouched
    from .density import create_density
    spec = _component(method, "density")
    from dataclasses import replace
    dens = create_density(replace(spec, config={**spec.config, "add_points": False}))
    g = torch.Generator(device="cpu").manual_seed(0)
    pts = torch.rand(50, 3, generator=g).to(device)
    params, opts = initialize(method, pts, torch.rand(50, 3, generator=g).to(device))
    with torch.no_grad():
        params["opacities"].fill_(3.0)
    old = bank_mod.ACTIVE
    try:
        dens.initialize(params, opts, ext)
        for step in (2999, 3000, 3001):
            m2 = torch.zeros(1, len(params["means"]), 2, device=device, requires_grad=True)
            m2.sum().backward()
            info = {"means2d": m2, "radii": torch.ones(1, len(params["means"]), 2,
                                                         device=device), "width": 8,
                    "height": 8}
            dens.after_backward(params, opts, step, {"info": info, "n_renders": 1})
    finally:
        bank_mod.ACTIVE = old
    _check(torch.allclose(params["opacities"], torch.full_like(params["opacities"], 3.0)),
           "opacity was reset")
    _check(abs(opts["means"].param_groups[0]["lr"] - expon_lr(3002, 3.2e-3, 3.2e-5, 30000))
           < 1e-12, "xyz lr not set for the next step")
    return {"thresholds": want, "xyz_lr_20000": lr[1]}


def photometric(method, device):
    """0.8 L1 + 0.2 (1 - SSIM) with zero-padded 'same' SSIM (Inria loss_utils), not the
    runtime's valid-convolution SSIM."""
    _, _, losses = build(method)
    name, loss = losses[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.rand(20, 24, 3, generator=g).to(device).requires_grad_(True)
    b = torch.rand(20, 24, 3, generator=g).to(device)
    v = loss({}, {"rgb": a}, b)
    # independent: unfold-based local statistics with zero padding
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    k = torch.exp(-((torch.arange(11.0, device=device) - 5) ** 2) / 4.5)
    k = (k / k.sum())
    w = (k[:, None] * k[None]).reshape(1, 1, 121, 1)

    def stat(t):
        u = F.unfold(t.reshape(3, 1, *t.shape[-2:]), 11, padding=5)  # [3, 121, HW]
        return (u[:, None] * w).sum(2).reshape(3, *t.shape[-2:])

    mx, my = stat(x), stat(y)
    vx, vy, cxy = stat(x * x) - mx ** 2, stat(y * y) - my ** 2, stat(x * y) - mx * my
    s = (((2 * mx * my + 1e-4) * (2 * cxy + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4)
                                                       * (vx + vy + 9e-4))).mean()
    ref = 0.8 * (a - b).abs().mean() + 0.2 * (1 - s)
    _check(abs(float(v) - float(ref)) < 1e-5, f"loss {float(v)} vs reference {float(ref)}")
    _check(abs(float(ssim_same(a, b)) - float(s)) < 1e-5, "SSIM differs")
    grad = torch.autograd.grad(v, a)[0]
    _check(torch.isfinite(grad).all() and grad.abs().sum() > 0, "no gradient")
    return {"loss": float(v), "reference": float(ref)}
