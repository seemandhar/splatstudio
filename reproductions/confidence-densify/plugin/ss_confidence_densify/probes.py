"""Mechanism probes for confidence-densify. Each raises AssertionError when the mechanism is
removed or wrong; reference formulas are written independently of density.py / losses.py."""

import math
from dataclasses import replace

import torch

from . import density as D
from .losses import SSIM_ERROR_KEY, TARGET_KEY, create_confidence_loss, create_reg


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _scene(method, n, device, seed=0, depth=3.0, spread=0.6, scale=0.06):
    from splatstudio.state import initialize

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += depth
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_((torch.full((n, 3), math.log(scale))
                                + 0.3 * torch.randn(n, 3, generator=g)).to(device))
        params["quats"].copy_(torch.nn.functional.normalize(torch.randn(n, 4, generator=g),
                                                            dim=-1).to(device))
        params["opacities"].copy_(torch.logit(torch.rand(n, generator=g) * 0.8 + 0.1).to(device))
    return params, opts


def _camera(device, w=40, h=32, f=30.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": w, "height": h}


def _density(method, **cfg):
    spec = _spec(method, "cad-density")
    return D.ConfidenceDensity(replace(spec, config={**spec.config, **cfg}))


# ------------------------------------------------------------------------------------------
def confidence_loss(method, device):
    spec = _spec(method, "confidence-photometric")
    loss = create_confidence_loss(spec)
    beta = spec.config["beta_conf"]
    g = torch.Generator().manual_seed(0)
    rgb = torch.rand(24, 24, 3, generator=g).to(device)
    target = torch.rand(24, 24, 3, generator=g).to(device)
    conf = (torch.rand(24, 24, 1, generator=g) * 2 + 0.2).to(device).requires_grad_(True)
    out = {"rgb": rgb, "confidence": conf, "step": 1}
    value = loss({}, out, target)
    # independent: 3DGS-style same-padded SSIM via unfold
    from .losses import ssim_map

    pp = 0.8 * (rgb - target).abs() + 0.2 * (1 - ssim_map(rgb, target))
    weighted = spec.config.get("weight_photometric", True)
    ref = (pp * conf).mean() - beta * torch.log(conf).mean()
    torch.testing.assert_close(value, ref if weighted else ref + pp.mean())
    gc = torch.autograd.grad(value, conf, retain_graph=True)[0]
    torch.testing.assert_close(gc, (pp.mean(-1, keepdim=True) - beta / conf) / conf.numel())
    # photometric gradient: weighted by C (CoMe) or plain L_rgb (iteration 1)
    rgb_v = rgb.clone().requires_grad_(True)
    out_v = {"rgb": rgb_v, "confidence": conf.detach(), "step": 1}
    g_rgb = torch.autograd.grad(loss({}, out_v, target), rgb_v)[0]
    rgb_p = rgb.clone().requires_grad_(True)
    pp_p = 0.8 * (rgb_p - target).abs() + 0.2 * (1 - ssim_map(rgb_p, target))
    ref_obj = (pp_p * conf.detach()).mean() if weighted else pp_p.mean()
    torch.testing.assert_close(g_rgb, torch.autograd.grad(ref_obj, rgb_p)[0])
    plain = pp.mean()
    if abs(value.item() - plain.item()) < 1e-4:
        raise AssertionError("confidence loss indistinguishable from plain L_rgb")
    # minimiser of L C - beta log C over C is beta / L (per pixel)
    L = torch.tensor(0.3)
    cs = torch.linspace(0.01, 2, 20000)
    best = cs[torch.argmin(L * cs - beta * torch.log(cs))]
    assert abs(best.item() - beta / 0.3) < 1e-3, best
    # stash for the density component
    assert TARGET_KEY in out and out[SSIM_ERROR_KEY].shape == (24, 24)
    # gradient on gamma: error > beta -> confidence decreases (d loss / d gamma > 0)
    gamma = torch.zeros(1, requires_grad=True)
    big = torch.full((1,), 0.5)
    (big * gamma.exp() - beta * torch.log(gamma.exp())).sum().backward()
    assert gamma.grad.item() > 0
    # regulariser and its schedule
    reg = create_reg(_spec(method, "mcmc-reg"))
    p = {"opacities": torch.zeros(4, requires_grad=True),
         "scales": torch.zeros(4, 3, requires_grad=True)}
    r = reg(p, {"step": 10}, None)
    torch.testing.assert_close(r, torch.tensor(0.01 * 0.5 + 0.01 * 1.0))
    assert reg(p, {"step": 25000}, None).item() == 0.0
    return {"loss": value.item(), "plain_rgb": plain.item(), "argmin": best.item()}


def confidence_render(method, device):
    _cuda(device)
    from gsplat import rasterization

    from .renderer import Renderer

    params, _ = _scene(method, 30, device)
    cam = _camera(device)
    r = Renderer(_spec(method, "cad-renderer"))
    out = r.render(params, cam, sh_degree=0)
    covered = out["alpha"] > 1e-2
    partial = covered & (out["alpha"] < 0.9)
    assert partial.sum() > 20, "fixture needs partially covered pixels"
    # CoMe: opacity-normalised -> exactly 1 at gamma = 0 for any alpha (old bug: C == alpha)
    torch.testing.assert_close(out["confidence"][covered], torch.ones_like(
        out["confidence"][covered]), atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        params["confidence"].copy_(torch.randn_like(params["confidence"]) * 0.5)
    out2 = r.render(params, cam, sh_degree=0)
    # independent: blend [exp(gamma)] with plain gsplat and divide by its accumulated alpha
    ref, ra, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=params["confidence"].exp(),
        viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=cam["width"],
        height=cam["height"], sh_degree=None, packed=False)
    torch.testing.assert_close(out2["confidence"], ref[0] / ra[0].clamp_min(1e-3),
                               atol=1e-5, rtol=1e-4)
    g = params["confidence"].exp()
    lo, hi = g.min().item(), g.max().item()
    c = out2["confidence"][covered]
    assert (c >= lo - 1e-4).all() and (c <= hi + 1e-4).all(), "normalised map is a convex blend"
    ref, alpha, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat((params["sh0"], params["shN"]), 1), viewmats=cam["viewmat"][None],
        Ks=cam["K"][None], width=cam["width"], height=cam["height"], sh_degree=0, packed=False)
    torch.testing.assert_close(out2["rgb"], ref[0] + (1 - alpha[0]), atol=1e-5, rtol=1e-4)
    return {"partial_pixels": int(partial.sum()), "conf_range": [c.min().item(), c.max().item()]}


def _brute_force(info, err, n):
    """Per-pixel front-to-back compositing in Python with gsplat's rules (independent)."""
    m2 = info["means2d"][0].detach().cpu().double()
    con = info["conics"][0].detach().cpu().double()
    op = info["opacities"][0].detach().cpu().double()
    depth = info["depths"][0].detach().cpu()
    radii = info["radii"][0].cpu()
    vis = ((radii > 0).all(-1)).nonzero(as_tuple=True)[0]
    order = vis[torch.argsort(depth[vis])]
    h, w = err.shape
    maxw = torch.zeros(n, dtype=torch.float64)
    esum = torch.zeros(n, dtype=torch.float64)
    e = err.cpu().double()
    for i in range(h):
        for j in range(w):
            T = 1.0
            for g in order.tolist():
                # only Gaussians whose tile list contains this pixel's tile
                r = radii[g]
                cx, cy = m2[g]
                if abs(j + 0.5 - cx) > r[0] + 16 or abs(i + 0.5 - cy) > r[1] + 16:
                    continue
                dx, dy = cx - (j + 0.5), cy - (i + 0.5)
                a, b, c = con[g]
                sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                alpha = min(0.999, op[g] * math.exp(-sigma))
                if sigma < 0 or alpha < 1 / 255:
                    continue
                nT = T * (1 - alpha)
                if nT <= 1e-4:
                    break
                wgt = alpha * T
                maxw[g] = max(maxw[g], wgt)
                esum[g] += wgt * e[i, j]
                T = nT
    return maxw, esum


def blend_stats_oracle(method, device):
    _cuda(device)
    from gsplat import rasterization

    from .kernels import blend_stats

    n = 25
    params, _ = _scene(method, n, device, scale=0.03, spread=0.4)
    cam = _camera(device, w=32, h=32)
    _, _, info = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=params["sh0"][:, 0],
        viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=32, height=32,
        sh_degree=None, packed=False)
    err = torch.rand(32, 32, generator=torch.Generator().manual_seed(1)).to(device)
    maxw, esum = blend_stats(info, err)
    ref_max, ref_sum = _brute_force(info, err, n)
    # the tile-overlap test of the brute force is conservative (a superset of gsplat's tiles);
    # Gaussians outside a tile have alpha < 1/255 there, so results must agree.
    torch.testing.assert_close(maxw.cpu().double(), ref_max, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(esum.cpu().double(), ref_sum, atol=2e-4, rtol=2e-4)
    if torch.allclose(maxw.cpu().double(), ref_sum, atol=1e-3):
        raise AssertionError("max weight indistinguishable from an error-weighted sum")
    assert (maxw <= 0.999 + 1e-6).all() and (maxw > 0).sum() > 5
    return {"visible": int((maxw > 0).sum()), "max_w": maxw.max().item(),
            "err_sum": esum.sum().item()}


def sampling_pdf(method, device):
    g = torch.Generator().manual_seed(0)
    grad = torch.rand(200, generator=g)
    grad[:20] = 0.0  # unobserved
    conf = torch.rand(200, generator=g) * 2
    # independent ranks: number of strictly smaller values / (n-1)
    def rank(x):
        return (x[None, :] < x[:, None]).sum(1).float() / (len(x) - 1)

    for step, e in ((0, 0.0), (2500, 0.5), (5000, 1.0), (20000, 1.0)):
        ref = rank(grad) * ((1 - e) + e * (1.2 - rank(conf)))
        torch.testing.assert_close(D.sampling_pdf(grad, conf, step), ref)
    full = D.sampling_pdf(grad, conf, 6000)
    assert full[:20].eq(0).all(), "unobserved primitives must have zero probability"
    hi_c = torch.argmax(conf)
    lo_c = torch.argmin(conf)
    assert abs((1.2 - rank(conf))[hi_c].item() - 0.2) < 1e-6
    assert abs((1.2 - rank(conf))[lo_c].item() - 1.2) < 1e-6
    # 80/20 split and source exclusion
    ssim = torch.zeros(200)
    ssim[150] = 1.0  # all SSIM-term draws hit primitive 150
    allowed = torch.ones(200, dtype=torch.bool)
    allowed[100:110] = False
    src = D.draw_sources(full, ssim, 1000, 0.8, allowed, torch.Generator().manual_seed(3))
    n150 = (src == 150).sum().item()
    assert 200 <= n150 <= 215, n150  # 200 from the SSIM term + a few from Eq. pdf
    assert not ((src >= 100) & (src < 110)).any()
    # ablation: opacity-proportional sampling (3DGS-MCMC) is a different distribution
    opac = torch.rand(200, generator=g)
    p_ours, p_mcmc = full / full.sum(), opac / opac.sum()
    tv = 0.5 * (p_ours - p_mcmc).abs().sum().item()
    assert tv > 0.1, tv
    return {"ssim_term_hits": n150, "tv_vs_opacity": tv}


def relocation_schedule(method, device):
    cfg = {**D.DEFAULTS}
    vals = {t: D.relocation_share(t)[0] / 0.1 for t in (6000, 10000, 15000, 20000, 25000)}
    for t, v in ((6000, 1.0), (10000, 1.0), (15000, 0.6065), (20000, 0.1353), (25000, 0.0111)):
        assert abs(vals[t] - v) < 1e-3, (t, vals[t])
    assert D.relocation_share(4800) == (0.1, True)
    n = 1000
    w = torch.rand(n) + 0.01
    w[:30] = 0
    exempt = torch.zeros(n, dtype=torch.bool)
    exempt[:5] = True
    early = D.select_relocation(w, exempt, 2000, cfg)
    assert set(early.tolist()) == set(range(5, 30)), "before 5k only w=0, non-exempt"
    w2 = w.clone()
    w2[:300] = 0
    capped = D.select_relocation(w2, torch.zeros(n, dtype=torch.bool), 2000, cfg)
    assert len(capped) == 100 and (w2[capped] == 0).all()
    mid = D.select_relocation(w, exempt, 7000, cfg)
    ref = [i for i in torch.argsort(w).tolist() if i >= 5][:100]
    assert sorted(mid.tolist()) == sorted(ref)
    late = D.select_relocation(w, exempt, 20000, cfg)
    assert len(late) == int(0.1 * math.exp(-2.0) * n), len(late)
    return {"share_ratio": vals, "early": len(early), "mid": len(mid), "late": len(late)}


def _step_outputs(method, params, cam, step, target=None):
    from .losses import create_confidence_loss
    from .renderer import Renderer

    r = Renderer(_spec(method, "cad-renderer"))
    out = r.render(params, cam, sh_degree=1)
    out["step"] = step
    if target is None:
        target = torch.rand(cam["height"], cam["width"], 3,
                            generator=torch.Generator().manual_seed(step)).to(cam["K"].device)
    loss = create_confidence_loss(_spec(method, "confidence-photometric"))
    return out, loss({}, out, target)


def densify_step(method, device):
    _cuda(device)
    from splatstudio.state import check_state

    n = 40
    params, opts = _scene(method, n, device)
    den = _density(method, cap_max=42, refine_start=0, share_full_start=0)
    den.initialize(params, opts)
    cam = _camera(device)
    o_init = params["opacities"].sigmoid()
    assert torch.allclose(o_init, torch.full_like(o_init, 0.5)), "MCMC init opacity 0.5"
    means_before = None
    for step in range(1, 401):
        out, value = _step_outputs(method, params, cam, step)
        den.before_backward(params, opts, step, out)
        value.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if step % 100 == 0:
            means_before = params["means"].detach().clone()
        den.after_backward(params, opts, step, out)
        check_state(method, params, opts)
        if step % 100 == 0 and step % 400 != 0:
            # no noise: existing means unchanged by the density step (except none)
            torch.testing.assert_close(params["means"][: len(means_before)], means_before)
    log = den.state["log"]
    counts = [r["gaussians"] for r in log]
    assert counts[0] == 42 and counts[-1] == 42, counts  # 40 -> min(42, int(1.05*40)=42)
    assert [r["relocated"] for r in log] == [0, 0, 0, 4], log  # every 400 it, 10 % of 42
    assert int(den.state["exempt"].sum()) == 4
    # MCMC opacity rule: one source with one copy -> 1 - sqrt(1 - o)
    p2, o2 = _scene(method, 8, device)
    st = {k: torch.zeros(8, device=device) for k in ("grad_sum", "count", "max_w", "ssim_sum")}
    st["exempt"] = torch.zeros(8, dtype=torch.bool, device=device)
    st["pending_age"] = torch.full((8,), -1, dtype=torch.int32, device=device)
    o_src = p2["opacities"][3].sigmoid().item()
    s_src = p2["scales"][3].exp().clone()
    mean_src = p2["means"][3].detach().clone()
    dest = D.place(p2, o2, st, torch.tensor([3], device=device),
                   torch.tensor([6], device=device), den.binoms, 0.005)
    assert dest.tolist() == [6]
    exp_o = 1 - math.sqrt(1 - o_src)
    assert abs(p2["opacities"][3].sigmoid().item() - exp_o) < 1e-4
    assert abs(p2["opacities"][6].sigmoid().item() - exp_o) < 1e-4
    torch.testing.assert_close(p2["quats"][6], p2["quats"][3])
    torch.testing.assert_close(p2["confidence"][6], p2["confidence"][3])
    torch.testing.assert_close(p2["means"][3], mean_src)
    off = (p2["means"][6] - mean_src).norm().item()
    assert 0 < off < 10 * s_src.max().item(), "copy mean must be sampled around its source"
    assert st["pending_age"][6].item() == 0
    return {"log": log, "copy_offset": off, "opacity_after": exp_o}


def color_init(method, device):
    _cuda(device)
    params, opts = _scene(method, 12, device, scale=0.02)
    den = _density(method)
    den.initialize(params, opts)
    with torch.no_grad():
        params["shN"].normal_(0, 0.1)
    cam = _camera(device, w=48, h=40)
    target = torch.rand(40, 48, 3, generator=torch.Generator().manual_seed(5)).to(device)
    den.state["pending_age"][[2, 7]] = 0
    out, value = _step_outputs(method, params, cam, 30000, target)
    value.backward()
    den.after_backward(params, opts, 30000, out)
    from .renderer import sh_colors

    col = sh_colors(params, cam["viewmat"], 1).detach()
    uv = out["info"]["means2d"][0].detach()
    done = 0
    for i in (2, 7):
        u, v = uv[i].tolist()
        if not (0 <= u < 48 and 0 <= v < 40):
            continue
        x, y = u - 0.5, v - 0.5  # bilinear, pixel centres at +0.5 (independent formula)
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0

        def px(yy, xx):
            return target[min(max(yy, 0), 39), min(max(xx, 0), 47)]

        ref = ((1 - fx) * (1 - fy) * px(y0, x0) + fx * (1 - fy) * px(y0, x0 + 1)
               + (1 - fx) * fy * px(y0 + 1, x0) + fx * fy * px(y0 + 1, x0 + 1))
        torch.testing.assert_close(col[i], ref, atol=1e-4, rtol=1e-4)
        done += 1
    assert done == 2 and (den.state["pending_age"] < 0).all()
    # timeout
    den.state["pending_age"][0] = 400
    with torch.no_grad():
        params["means"][0, 2] = -5.0  # behind the camera: never visible
    out, value = _step_outputs(method, params, cam, 30000, target)
    value.backward()
    den.after_backward(params, opts, 30000, out)
    assert den.state["pending_age"][0].item() == -1
    return {"initialised": done}


def renderer_gradient(method, device):
    """Screen gradients exist, and the confidence term reaches only gamma (CoMe
    hierarchical_render.cuh:1513-1515, :1643): geometry/opacity/colour gradients and the
    absolute screen gradient equal those of the same loss with C_hat detached."""
    _cuda(device)
    from .losses import confidence_loss_value

    spec = _spec(method, "confidence-photometric").config
    params, opts = _scene(method, 30, device)
    with torch.no_grad():
        params["confidence"].copy_(torch.randn_like(params["confidence"]) * 0.3)
    den = _density(method)
    den.initialize(params, opts)
    cam = _camera(device)
    target = torch.rand(cam["height"], cam["width"], 3,
                        generator=torch.Generator().manual_seed(9)).to(device)
    grads = {}
    for mode in ("full", "detached"):
        for p in params.values():
            p.grad = None
        out, value = _step_outputs(method, params, cam, 1, target)
        if mode == "detached":
            value = confidence_loss_value(out["rgb"], target, out["confidence"].detach(),
                                          spec["beta_conf"], spec["ssim_weight"],
                                          spec["conf_min"], spec["conf_max"],
                                          weight_photometric=spec.get("weight_photometric",
                                                                      True))
        den.before_backward(params, opts, 1, out)
        value.backward()
        xy = out["info"]["means2d"]
        grads[mode] = {k: (p.grad.clone() if p.grad is not None else None)
                       for k, p in params.items()}
        grads[mode]["absgrad"] = xy.absgrad.clone()
        grads[mode]["grad2d"] = xy.grad.clone()
        if mode == "full":
            keep = out
    for k in ("means", "scales", "quats", "opacities", "sh0", "absgrad", "grad2d"):
        torch.testing.assert_close(grads["full"][k], grads["detached"][k], atol=1e-7, rtol=1e-5)
    gc = grads["full"]["confidence"]
    assert gc is not None and torch.count_nonzero(gc), "confidence must receive the term"
    assert grads["detached"]["confidence"] is None or not torch.count_nonzero(
        grads["detached"]["confidence"])
    xy = keep["info"]["means2d"]
    assert torch.count_nonzero(xy.grad) and torch.count_nonzero(xy.absgrad)
    assert not torch.any(xy.absgrad + 1e-6 < xy.grad.abs())
    den.after_backward(params, opts, 1, keep)
    return {"grad_norm": xy.grad.norm().item(), "absgrad_norm": xy.absgrad.norm().item(),
            "confidence_grad_norm": gc.norm().item()}
