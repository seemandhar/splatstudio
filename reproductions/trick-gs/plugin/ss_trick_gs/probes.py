"""Mechanism probes for the Trick-GS port. Each raises if its mechanism is removed or wrong."""

import math
import tempfile
from dataclasses import replace

import numpy as np
import torch
from gsplat import rasterization
from splatstudio.components import build
from splatstudio.state import check_state, initialize

from . import schedule, significance
from .density import (
    DEFAULTS,
    TrickDensity,
    lowest_error_subset,
    means_sha256,
)
from .losses import (
    SH_WEIGHTS,
    blur_image,
    gaussian_mask_term,
    sh_mask_term,
    ssim_separable,
)
from .optim import IntervalAdam
from .renderer import GS_MASK, SH_MASK, expand_sh
from .shared import STATE


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _scene(method, n, device, seed=0, spread=0.6):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += 3.0
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.05))
        params["opacities"].copy_(torch.logit(torch.rand(n, generator=g) * 0.5 + 0.3).to(device))
        params["shN"].copy_((torch.randn(n, 15, 3, generator=g) * 0.3).to(device))
    return params, opts


def _camera(device, width=64, height=48, f=60.0, angle=0.0, shift=0.0):
    c, s = math.cos(angle), math.sin(angle)
    view = torch.tensor([[c, 0, s, shift], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return {"viewmat": view.to(device), "K": K.to(device), "width": width, "height": height}


def _raw_render(params, cam, scales, opac, colors, eps2d=0.3, degree=3):
    out, alpha, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=scales, opacities=opac,
        colors=colors, viewmats=cam["viewmat"][None], Ks=cam["K"][None],
        width=int(cam["width"]), height=int(cam["height"]), sh_degree=degree, packed=False,
        render_mode="RGB", eps2d=eps2d, rasterize_mode="classic")
    return out[0] + (1 - alpha[0])


def gaussian_mask(method, device):
    _cuda(device)
    STATE.reset()
    renderer, _, _ = build(method)
    params, _ = _scene(method, 30, device)
    with torch.no_grad():
        params[GS_MASK][::3] = -5.0  # sigmoid < 0.05 -> masked
    cam = _camera(device)
    out = renderer.render(params, cam, 3)["rgb"]
    m = (torch.sigmoid(params[GS_MASK]) > 0.05).float()
    colors = torch.cat((params["sh0"], params["shN"]), 1)
    ref = _raw_render(params, cam, params["scales"].exp() * m[:, None],
                      params["opacities"].sigmoid() * m, colors)
    unmasked = _raw_render(params, cam, params["scales"].exp(), params["opacities"].sigmoid(),
                           colors)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
    if (out - unmasked).abs().max() < 1e-3:
        raise AssertionError("Gaussian mask has no effect on the render")
    # STE: forward hard, backward sigmoid'
    x = torch.randn(30, device=device)
    renderer_mod = __import__("ss_trick_gs.renderer", fromlist=["ste_mask"])
    logits = params[GS_MASK].detach().clone().requires_grad_(True)
    hard = renderer_mod.ste_mask(logits, 0.05)
    (hard * x).sum().backward()
    s = torch.sigmoid(logits.detach())
    torch.testing.assert_close(logits.grad, s * (1 - s) * x)
    torch.testing.assert_close(hard.detach(), (s > 0.05).float())
    term = gaussian_mask_term(params)
    torch.testing.assert_close(term, torch.sigmoid(params[GS_MASK]).sum() / 30)
    from .density import gs_mask_prune

    pr = gs_mask_prune(params, 0.05)
    if int(pr.sum()) != 10 or not bool(pr[::3].all()):
        raise AssertionError("mask prune rule does not remove exactly sigmoid(m) <= eps")
    return {"max_abs_vs_hand_masked": float((out - ref).abs().max()),
            "effect_vs_unmasked": float((out - unmasked).abs().max()), "pruned": int(pr.sum())}


def sh_mask(method, device):
    _cuda(device)
    STATE.reset()
    renderer, _, _ = build(method)
    params, _ = _scene(method, 30, device, seed=1)
    with torch.no_grad():
        params[SH_MASK][:15, 1] = -4.0  # degree-2 band of half the Gaussians masked
    cam = _camera(device, angle=0.2)
    out = renderer.render(params, cam, 3)["rgb"]
    shn = params["shN"].detach().clone()
    shn[:15, 3:8] = 0.0
    ref = _raw_render(params, cam, params["scales"].exp(), params["opacities"].sigmoid(),
                      torch.cat((params["sh0"], shn), 1))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
    full = _raw_render(params, cam, params["scales"].exp(), params["opacities"].sigmoid(),
                       torch.cat((params["sh0"], params["shN"]), 1))
    if (out - full).abs().max() < 1e-3:
        raise AssertionError("SH mask has no effect")
    # Before sh_mask_start the training render must not mask.
    STATE.activate({**DEFAULTS})
    STATE.next_step = 19501  # full resolution, no blur; sh_mask_start 3500 passed
    try:
        trained = renderer.render(params, cam, 3)
        STATE.next_step = 19501
        STATE.cfg = {**DEFAULTS, "sh_mask_start": 10**9}
        early = renderer.render(params, cam, 3)
    finally:
        STATE.reset()
    torch.testing.assert_close(trained["rgb"].detach(), ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(early["rgb"].detach(), full, atol=1e-5, rtol=0)
    s = torch.sigmoid(params[SH_MASK])
    ind = sum((2 * deg + 1) / 15 * s[:, deg - 1] for deg in (1, 2, 3)).mean()
    torch.testing.assert_close(sh_mask_term(params), ind)
    coeff = expand_sh(torch.tensor([[1.0, 2.0, 3.0]], device=device))[0, :, 0]
    if coeff.tolist() != [1.0] * 3 + [2.0] * 5 + [3.0] * 7 or abs(sum(SH_WEIGHTS) - 1) > 1e-9:
        raise AssertionError("SH band layout / weights wrong")
    return {"max_abs_vs_zeroed_band": float((out - ref).abs().max()), "loss": float(ind)}


def _brute(means2d, radii, depths, conics, opac, W, H):
    ys, xs = torch.meshgrid(torch.arange(H, device=opac.device),
                            torch.arange(W, device=opac.device), indexing="ij")
    px, py = xs.reshape(-1).float() + 0.5, ys.reshape(-1).float() + 0.5
    T = torch.ones_like(px)
    done = torch.zeros_like(px, dtype=torch.bool)
    score = torch.zeros(len(opac), device=opac.device, dtype=torch.float64)
    count = torch.zeros(len(opac), device=opac.device, dtype=torch.long)
    visible = radii > 0 if radii.dim() == 1 else (radii > 0).all(dim=-1)
    for j in torch.argsort(depths).tolist():
        if not visible[j]:
            continue
        dx, dy = means2d[j, 0] - px, means2d[j, 1] - py
        a, b, c = conics[j]
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        alpha = torch.clamp(opac[j] * torch.exp(power), max=0.99)
        valid = (~done) & (power <= 0) & (alpha >= 1.0 / 255.0)
        test_T = T * (1 - alpha)
        term = valid & (test_T < 1e-4)
        done = done | term
        hit = valid & ~term
        score[j] += (opac[j] * T)[hit].double().sum()
        count[j] += int(hit.sum())
        T = torch.where(hit, test_T, T)
    return count, score.float()


@torch.no_grad()
def significance_probe(method, device):
    _cuda(device)
    params, _ = _scene(method, 40, device, seed=2, spread=0.4)
    scales, opac = params["scales"].exp(), params["opacities"].sigmoid()
    cams = [_camera(device, 40, 32, 40.0, a, 0.03 * i) for i, a in enumerate((0.0, 0.15))]
    count, score = significance.hit_scores(params["means"], params["quats"], scales, opac, cams)
    rc = torch.zeros_like(count, dtype=torch.long)
    rs = torch.zeros_like(score)
    for cam in cams:
        m2, r, d, cn = significance.project(params["means"], params["quats"], scales, opac, cam)
        c, s = _brute(m2, r, d, cn, opac, cam["width"], cam["height"])
        rc += c
        rs += s
    if not torch.equal(count.long(), rc):
        raise AssertionError("CUDA hit counts differ from the brute-force reference")
    torch.testing.assert_close(score, rs, rtol=1e-4, atol=1e-4)
    if float(score.sum()) <= 0:
        raise AssertionError("significance probe scene is empty")
    vol = scales.prod(1)
    v = significance.v_importance(score, scales, 0.1)
    kth = torch.sort(vol, descending=True).values[int(0.9 * 40)]
    torch.testing.assert_close(v, (vol / kth) ** 0.1 * score)
    pm = significance.prune_mask(v, 0.6)
    thr = torch.sort(v).values[int(0.6 * 39)]
    if not torch.equal(pm, v <= thr):
        raise AssertionError("prune rule differs from LightGaussian prune_gaussians")
    rates = [schedule.sg_rate(k) for k in range(6)]
    if any(abs(r - 0.6 * 0.7**k) > 1e-12 for k, r in enumerate(rates)):
        raise AssertionError("significance prune rates are not 0.6 * 0.7^k")
    return {"hits": int(count.sum()), "pruned_at_0.6": int(pm.sum()),
            "rates": [round(r, 4) for r in rates]}


def progressive(method, device):
    _cuda(device)
    H, W = 840, 1297
    sizes = {s: schedule.ds_size(s, H, W) for s in (1, 9750, 19499, 19500)}
    w1 = round(W / 8 + (W - W / 8) * 1 / 19500)
    if sizes[1][1] != w1 or sizes[19500] != (H, W):
        raise AssertionError(f"Eq. 6 sizes wrong: {sizes}")
    w_mid = schedule.round_half_up(W / 8 + (W - W / 8) * 0.5)
    if sizes[9750][1] != w_mid:
        raise AssertionError("Eq. 6 is not linear in the resolution")
    if schedule.blur_params(1, 1.0) != (4, 2.4):
        raise AssertionError("initial blur must be 9x9, sigma 2.4")
    if schedule.blur_params(150, 1.0) != schedule.blur_params(101, 1.0) or \
            schedule.blur_params(199, 1.0) == schedule.blur_params(200, 1.0):
        raise AssertionError("blur must change exactly at multiples of k_b = 100")
    r, sg = schedule.blur_params(5000, 0.5)
    b = (1 - 5000 / 19500) * 0.5
    if r != round(4 * b) or abs(sg - 2.4 * b) > 1e-12:
        raise AssertionError("blur not scaled by the downsampling factor")
    if schedule.blur_params(19500, 1.0) is not None or schedule.blur_params(1, 0.125) is not None:
        raise AssertionError("blur must vanish at tau and when the scaled radius rounds to 0")
    # Blur numerics: independent dense 2-D Gaussian with reflect padding.
    img = torch.rand(20, 23, 3, device=device)
    out = blur_image(img, 2, 1.1)
    x = torch.arange(-2, 3, device=device, dtype=torch.float32)
    k1 = torch.exp(-x**2 / (2 * 1.1**2))
    k2 = (k1[:, None] * k1[None, :]) / (k1.sum() ** 2)
    pad = torch.nn.functional.pad(img.permute(2, 0, 1)[None], (2, 2, 2, 2), mode="reflect")
    ref = torch.nn.functional.conv2d(pad, k2[None, None].expand(3, 1, 5, 5), groups=3)[0]
    torch.testing.assert_close(out, ref.permute(1, 2, 0), atol=1e-5, rtol=1e-5)
    # Training render at the scheduled size, loss target transformed to match.
    renderer, density, losses = build(method)
    params, opts = _scene(method, 30, device, seed=3)
    density.initialize(params, opts, 1.0)
    try:
        STATE.next_step = 2000
        cam = _camera(device, 400, 240, 300.0)
        o = renderer.render(params, cam, 0)
        h, w = schedule.ds_size(2000, 240, 400)
        if tuple(o["rgb"].shape[:2]) != (h, w) or o["trick"]["blur"] is None:
            raise AssertionError("training render not at the scheduled resolution / no blur")
        o["step"] = 2000
        photometric = dict(losses)["photometric"]
        val = photometric(params, o, torch.rand(240, 400, 3, device=device))
        with torch.no_grad():
            ev = renderer.render(params, cam, 0)
        if tuple(ev["rgb"].shape[:2]) != (240, 400):
            raise AssertionError("evaluation render must be full resolution")
    finally:
        STATE.reset()
    return {"sizes": {str(k): list(v) for k, v in sizes.items()}, "train_size": [h, w],
            "blur_at_2000": list(o["trick"]["blur"]), "loss": float(val)}


def lowpass_abe(method, device):
    _cuda(device)
    if abs(schedule.lowpass_value(100, 90, 10) - 9000 / 10 / (9 * math.pi)) > 1e-9 or \
            schedule.lowpass_value(10, 10, 10**6) != 0.3:
        raise AssertionError("low-pass value formula wrong")
    due = [s for s in range(1, 12001) if schedule.lowpass_due(s)]
    if due != [1] + list(range(1000, 10000, 1000)):
        raise AssertionError(f"low-pass recompute steps wrong: {due}")
    renderer, density, _ = build(method)
    params, opts = _scene(method, 12, device, seed=4)
    density.initialize(params, opts, 2.0)
    try:
        cam = _camera(device, 400, 240, 300.0)
        STATE.next_step = 1
        o = renderer.render(params, cam, 0)
        h, w = o["trick"]["height"], o["trick"]["width"]
        s = schedule.lowpass_value(h, w, 12)
        if abs(o["trick"]["lowpass"] - s) > 1e-9 or s <= 0.3:
            raise AssertionError("low-pass not applied at step 1")
        from .renderer import scaled_camera

        sc = scaled_camera(cam, h, w)
        colors = torch.cat((params["sh0"], params["shN"]), 1)
        ref = _raw_render(params, sc, params["scales"].exp(), params["opacities"].sigmoid(),
                          colors, eps2d=s, degree=0)
        plain = _raw_render(params, sc, params["scales"].exp(), params["opacities"].sigmoid(),
                            colors, eps2d=0.3, degree=0)
        torch.testing.assert_close(o["rgb"].detach(), ref, atol=1e-5, rtol=0)
        if (ref - plain).abs().max() < 1e-3:
            raise AssertionError("low-pass value does not change the render")
        STATE.next_step = 10001
        if renderer.render(params, cam, 0)["trick"]["lowpass"] != 0.3:
            raise AssertionError("low-pass must return to 0.3 after 10000 steps")
        # ABE split: candidates with high gradient and large scale.
        res = {}
        for step in (10000, 10100):
            p2, o2 = _scene(method, 12, device, seed=4)
            d2 = TrickDensity(_component(method, "density"))
            d2.initialize(p2, o2, 2.0)
            with torch.no_grad():
                p2["scales"][:4] = math.log(0.1)  # > 0.01 * extent -> split, < 0.1 * extent
            d2.state["grad2d"] = torch.tensor([1.0] * 4 + [0.0] * 8, device=device)
            d2.state["count"] = torch.ones(12, device=device)
            before = p2["means"].detach().clone()
            ev = d2.densify(p2, o2, step)
            res[step] = ev
            if step == 10000:
                tail = p2["means"].detach()
                exp = before[:4] * 0.3 * 2.0
                hit = torch.cdist(exp, tail).min(dim=1).values.max()
                if ev["abe"] != 4 or float(hit) > 1e-6:
                    raise AssertionError("ABE split copies missing at 0.3 * extent * mu")
            elif ev["abe"] != 0:
                raise AssertionError("ABE split active after 10000 steps")
    finally:
        STATE.reset()
    # 20 % lowest-error init through the sidecar lookup.
    errs = np.linspace(2.0, 0.1, 12)
    keep = lowest_error_subset(errs, 0.2)
    if keep.tolist() != [9, 10, 11]:
        raise AssertionError(f"lowest-error subset wrong: {keep}")
    p3, o3 = _scene(method, 12, device, seed=5)
    with tempfile.TemporaryDirectory() as tmp:
        np.save(f"{tmp}/{means_sha256(p3['means'])}.npy", errs)
        spec = _component(method, "density")
        d3 = TrickDensity(replace(spec, config={**spec.config, "sfm_error_dir": tmp}))
        expected = p3["means"].detach()[torch.as_tensor(keep, device=device)].clone()
        d3.initialize(p3, o3, 1.0)
        d3.set_training_cameras([cam])
        STATE.reset()
    torch.testing.assert_close(p3["means"].detach(), expected)
    check_state(method, p3, o3)
    if not all(p.is_leaf and p.requires_grad for p in p3.values()):
        raise AssertionError("init subset froze a field")
    return {"lowpass_step1": s, "abe_events": {str(k): v["abe"] for k, v in res.items()},
            "init_kept": len(keep)}


def accelerated(method, device):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5, 3))
    q = torch.nn.Parameter(p.detach().clone())
    a = IntervalAdam([p], lr=0.1, every=16)
    b = torch.optim.Adam([q], lr=0.1, eps=1e-15)
    grads = [torch.randn(5, 3) for _ in range(32)]
    for i, g in enumerate(grads):
        p.grad = g.clone()
        a.step()
        a.zero_grad(set_to_none=True)
        if (i + 1) % 16 == 0:
            q.grad = sum(grads[i - 15 : i + 1])
            b.step()
            b.zero_grad(set_to_none=True)
        if i == 7 and not torch.equal(p.detach(), q.detach()):
            raise AssertionError("shN must not change between the 16-step updates")
    torch.testing.assert_close(p.detach(), q.detach())
    x, y = torch.rand(24, 30, 3), torch.rand(24, 30, 3)
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11)

    def conv(t):
        return torch.nn.functional.conv2d(t.permute(2, 0, 1)[None], w, padding=5, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cv = conv(x * x) - mx**2, conv(y * y) - my**2, conv(x * y) - mx * my
    ref = (((2 * mx * my + 1e-4) * (2 * cv + 9e-4)) / ((mx**2 + my**2 + 1e-4) * (vx + vy + 9e-4))).mean()
    torch.testing.assert_close(ssim_separable(x, y), ref, atol=1e-6, rtol=1e-6)
    spec = _component(method, "density")
    if int(spec.config.get("sh_update_every", DEFAULTS["sh_update_every"])) != 16:
        raise AssertionError("AT must batch SH updates over 16 steps")
    return {"interval_adam_matches": True, "ssim": float(ref)}


def redensify(method, device):
    cfg = {**DEFAULTS, **_component(method, "density").config}
    cfg["redensify"] = tuple(cfg["redensify"])
    dens = [s for s in range(1, 30001) if schedule.densify_due(s, cfg)]
    late = [s for s in dens if s >= 15000]
    if dens[0] != 600 or late != [20100, 20200, 20300, 20400, 20500] or 15000 in dens:
        raise AssertionError(f"densification schedule wrong: first {dens[0]}, late {late}")
    mp = [s for s in range(1, 30001) if schedule.mask_prune_due(s, cfg)]
    if mp[0] != 15000 or mp[-1] != 30000 or len(mp) != 31:
        raise AssertionError("mask prune every 500 after densification stops")
    resets = [s for s in range(1, 30001) if schedule.reset_due(s, cfg)]
    if resets != [3000, 6000, 9000, 12000]:
        raise AssertionError(f"opacity resets wrong: {resets}")
    if tuple(cfg["sg_steps"]) != (5000, 10000, 15000, 20000, 22000, 23000):
        raise AssertionError("significance steps differ from Fig. 3")
    win = [s for s in (14999, 15000, 19999, 20000, 20500, 20501) if schedule.in_stats_window(s, cfg)]
    if win != [14999, 20000, 20500]:
        raise AssertionError(f"gradient statistics window wrong: {win}")
    return {"densify_steps": len(dens), "late": late, "mask_prunes": len(mp)}


COMPRESSED = {
    "total_steps": 300, "densify_from": 20, "densify_until": 150, "densify_every": 10,
    "reset_every": 60, "prune_big_after": 60, "abe_until": 80, "lowpass_until": 80,
    "lowpass_every": 20, "ds_until": 200, "blur_every": 10, "blur_until": 200,
    "mask_prune_every": 20, "sh_mask_start": 60, "sg_steps": (50, 100, 150, 200, 220, 230),
    "redensify": (200, 250), "init_fraction": 0.5,
}


def cycle(method, device):
    """Native-style loop with a compressed schedule on a synthetic scene: every mechanism fires.

    The Gaussian-mask logits are frozen here (LR 1e-12): on a 400-point toy scene the per-Gaussian
    photometric gradients are noise-dominated, so Adam with the paper's LR 0.5 random-walks every
    logit below the threshold within ~30 steps and the population collapses. GM itself is
    verified by trick-gaussian-mask; on real data (garden smoke) the masks behave as expected.
    """
    _cuda(device)
    STATE.reset()
    spec = _component(method, "density")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {**spec.config, **COMPRESSED, "sfm_error_dir": tmp}
        m = replace(method, components=tuple(replace(c, config=cfg) if c.kind == "density" else c
                                             for c in method.components),
                    fields=tuple(replace(f, learning_rate=1e-12) if f.name == GS_MASK else f
                                 for f in method.fields))
        renderer, density, losses = build(m)
        params, opts = _scene(m, 400, device, seed=6, spread=1.2)
        gt_params, _ = _scene(m, 400, device, seed=6, spread=1.2)  # target = perturbed scene
        with torch.no_grad():
            gt_params["sh0"].add_(0.3 * torch.randn_like(gt_params["sh0"]))
            gt_params["means"].add_(0.01 * torch.randn_like(gt_params["means"]))
        np.save(f"{tmp}/{means_sha256(params['means'])}.npy", np.random.default_rng(0).random(400))
        cams = [_camera(device, 256, 192, 180.0, a, 0.05 * i)
                for i, a in enumerate(np.linspace(-0.3, 0.3, 8))]
        with torch.no_grad():
            targets = [_raw_render(gt_params, c, gt_params["scales"].exp(),
                                   gt_params["opacities"].sigmoid(),
                                   torch.cat((gt_params["sh0"], gt_params["shN"]), 1)) for c in cams]
        density.initialize(params, opts, 1.5)
        density.set_training_cameras(cams)
        g = torch.Generator().manual_seed(0)
        sizes = set()
        try:
            for step in range(1, 301):
                i = int(torch.randint(0, 8, (1,), generator=g))
                deg = min((step - 1) // 20, 3)
                out = renderer.render(params, cams[i], deg)
                out["step"] = step
                sizes.add(tuple(out["rgb"].shape[:2]))
                density.before_backward(params, opts, step, out)
                vals = {n: f(params, out, targets[i]) for n, f in losses}
                sum(vals.values()).backward()
                for o in opts.values():
                    o.step()
                    o.zero_grad(set_to_none=True)
                density.after_backward(params, opts, step, out)
                check_state(m, params, opts)
        finally:
            STATE.reset()
        ev = density.events
    kinds = {k for e in ev for k, v in e.items() if k != "step" and v}
    need = {"init_kept", "clone", "abe", "sg_prune", "total"}
    if not need <= kinds:
        raise AssertionError(f"mechanisms did not fire: {sorted(need - kinds)}")
    if not any(e.get("step", 0) > 200 and ("clone" in e or "split" in e) for e in ev):
        raise AssertionError("late densification did not run")
    if len([e for e in ev if "sg_round" in e]) != 6 or len(sizes) < 3:
        raise AssertionError("significance rounds / progressive sizes missing")
    for p in params.values():
        if not torch.equal(p.detach(), p.detach().half().float()):
            raise AssertionError("final parameters are not FP16-representable")
    return {"events": len(ev), "final_gaussians": len(params["means"]),
            "render_sizes": len(sizes), "sg_rounds": [e["sg_rate"] for e in ev if "sg_round" in e]}
