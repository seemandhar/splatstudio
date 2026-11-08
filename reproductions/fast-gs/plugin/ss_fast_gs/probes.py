"""Mechanism probes for the FastGS port. Each raises on failure and returns measurements.

The float64 projection / compositor helpers follow the SplatStudio Speedy-Splat port's
independent reference (reproductions/speedy-splat/plugin/ss_speedysplat/probes.py), which
re-derives the Inria forward.cu rules without tiles.
"""

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


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _density(method, **config):
    from .density import create_density

    spec = _component(method, "density")
    return create_density(replace(spec, config={**spec.config, "log": False, **config}))


def _optimizable(vals, device):
    params = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.float().to(device).clone()) for k, v in vals.items()})
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for k, p in params.items():  # allocate Adam moments, then restore the values
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return params, opts


def _toy(n, device, seed=0, log_scale=-3.0, opacity=0.5):
    g = torch.Generator().manual_seed(seed)
    vals = {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.full((n, 3), float(log_scale)),
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), math.log(opacity / (1 - opacity))),
        "sh0": torch.arange(n, dtype=torch.float32)[:, None, None].expand(n, 1, 3).clone(),
        "shN": torch.zeros(n, 15, 3),
    }
    return _optimizable(vals, device)


def _ids(params):
    return params["sh0"][:, 0, 0].round().long()


def _fixture(device, n=40, seed=3):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([2.4, 2.0, 2.0]) + torch.tensor(
        [-1.2, -1.0, 3.0])
    scales = torch.rand(n, 3, generator=g) * torch.tensor([0.25, 0.12, 0.05]) + 0.01
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.cat([torch.rand(n // 2, generator=g) * 0.09 + 0.9,
                      torch.rand(n - n // 2, generator=g) ** 3 * 0.6 + 0.005])
    sh0 = (torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8
    vals = {"means": means, "scales": scales.log(), "quats": quats,
            "opacities": torch.logit(opac), "sh0": sh0, "shN": torch.zeros(n, 15, 3)}
    W, H = 96, 80
    camera = {"viewmat": torch.eye(4, device=device), "width": W, "height": H,
              "K": torch.tensor([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]], device=device)}
    return {k: v.float().to(device) for k, v in vals.items()}, camera


def _rot(q):
    q = F.normalize(q.double(), dim=-1)
    w, u = q[:, :1], q[:, 1:]
    cols = []
    for e in torch.eye(3, dtype=torch.float64):
        v = e.expand_as(u)
        t = 2 * torch.cross(u, v, dim=-1)
        cols.append(v + w * t + torch.cross(u, t, dim=-1))
    return torch.stack(cols, dim=-1)


def _project(params, camera):
    """Independent float64 EWA projection with the Inria conventions (forward.cu)."""
    dd = torch.float64
    p = {k: v.detach().cpu().to(dd) for k, v in params.items()}
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].cpu().to(dd)
    view = camera["viewmat"].cpu().to(dd)
    t = p["means"] @ view[:3, :3].T + view[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tz = t[:, 2]
    tx = (t[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
    J = torch.zeros(len(t), 2, 3, dtype=dd)
    J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
    R = _rot(p["quats"])
    cov3 = R @ torch.diag_embed(p["scales"].exp() ** 2) @ R.transpose(1, 2)
    Wr = view[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2) + 0.3 * torch.eye(2, dtype=dd)
    conic = torch.linalg.inv(cov2)
    ndc = torch.stack([fx * t[:, 0] / tz * 2 / W + 2 * cx / W - 1,
                       fy * t[:, 1] / tz * 2 / H + 2 * cy / H - 1], -1)
    xy = ((ndc + 1) * torch.tensor([W, H], dtype=dd) - 1) * 0.5  # ndc2Pix
    opac = p["opacities"].sigmoid()
    col = (C0 * p["sh0"][:, 0] + 0.5).clamp_min(0)
    return {"xy": xy, "conic": conic, "opac": opac, "col": col, "depth": tz, "W": W, "H": H}


def _dense_reference(proj, bg, mask):
    """Unculled float64 compositor with forward.cu's rules. Returns (image HxWx3, counts [N]):
    counts[i] = #masked pixels where Gaussian i takes part in the blend (forward.cu:403)."""
    dd = torch.float64
    W, H = proj["W"], proj["H"]
    n = len(proj["xy"])
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dd), torch.arange(W, dtype=dd), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    T = torch.ones(len(pix), dtype=dd)
    C = torch.zeros(len(pix), 3, dtype=dd)
    alive = torch.ones(len(pix), dtype=torch.bool)
    counts = torch.zeros(n, dtype=torch.long)
    m = mask.reshape(-1).cpu().bool()
    for i in torch.argsort(proj["depth"]).tolist():
        if proj["depth"][i] <= 0.2:
            continue
        d = proj["xy"][i] - pix
        a, b, c = proj["conic"][i, 0, 0], proj["conic"][i, 0, 1], proj["conic"][i, 1, 1]
        power = -0.5 * (a * d[:, 0] ** 2 + c * d[:, 1] ** 2) - b * d[:, 0] * d[:, 1]
        alpha = (proj["opac"][i] * power.exp()).clamp(max=0.99)
        valid = alive & (power <= 0) & (alpha >= 1 / 255)
        test_T = T * (1 - alpha)
        stop = valid & (test_T < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        C = C + torch.where(use[:, None], proj["col"][i] * (alpha * T)[:, None], 0)
        T = torch.where(use, test_T, T)
        counts[i] = int((use & m).sum())
    C = C + T[:, None] * bg.cpu().to(dd)
    return C.reshape(H, W, 3), counts


# ---------------------------------------------------------------------------- probes
def metric_kernel(method, device):
    _cuda(device)
    from .raster import render

    params, camera = _fixture(device)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)
    W, H = camera["width"], camera["height"]
    g = torch.Generator().manual_seed(7)
    mask = (torch.rand(H, W, generator=g) > 0.5).int().to(device)
    proj = _project(params, camera)
    ref_img, ref_counts = _dense_reference(proj, bg, mask)
    # mult 1e3: the CB ellipse covers every tile -> no culling (dense reference applies)
    with torch.no_grad():
        img, _, radii, st_full = render(params, camera, 0, bg, 1e3, mask)
        _, _, _, st_cb = render(params, camera, 0, bg, 0.5, mask)
        img_cb, _, _, _ = render(params, camera, 0, bg, 0.5)
    img_err = float((img.permute(1, 2, 0).double().cpu() - ref_img).abs().max())
    cnt = st_full["counts"].long().cpu()
    cnt_err = int((cnt - ref_counts).abs().max())
    tol = max(2, int(0.002 * H * W))  # float32 vs float64 at alpha/T thresholds
    if img_err > 2e-3:
        raise AssertionError(f"rasterizer image differs from the dense reference: {img_err}")
    if cnt_err > tol or int(ref_counts.sum()) == 0:
        raise AssertionError(f"metric counts differ from brute force by {cnt_err} (tol {tol})")
    if not st_cb["num_rendered"] < st_full["num_rendered"]:
        raise AssertionError("Compact Box (mult 0.5) did not remove tile-Gaussian pairs")
    cb_diff = float((img_cb - img).abs().max())
    if cb_diff == 0.0:
        raise AssertionError("Compact Box has no effect on the image")
    # 4-column screen gradient: [:, 2:] are sums of |per-pixel grads| (AbsGS)
    p = {k: v.clone().requires_grad_(k == "means") for k, v in params.items()}
    color, means2d, _, _ = render(p, camera, 0, bg, 0.5)
    w = torch.randn(color.shape, generator=g).to(device)
    (color * w).sum().backward()
    gr = means2d.grad
    if gr is None or gr.shape[1] != 4 or not torch.count_nonzero(gr[:, 2:]):
        raise AssertionError("missing absolute screen gradient columns")
    if torch.any(gr[:, 2:] + 1e-6 < gr[:, :2].abs()):
        raise AssertionError("absolute gradient sums smaller than |signed sums|")
    return {"image_max_err": img_err, "count_max_err": cnt_err,
            "masked_contributions": int(ref_counts.sum()),
            "pairs_full": st_full["num_rendered"], "pairs_cb05": st_cb["num_rendered"],
            "cb_image_max_change": cb_diff, "visible": int((radii > 0).sum())}


def scores(method, device):
    from .density import error_mask

    d = _density(method)
    n, H, W, V = 30, 12, 14, 4
    params, opts = _toy(n, device)
    d.initialize(params, opts, 1.0)
    g = torch.Generator().manual_seed(1)
    foot = (torch.rand(n, H * W, generator=g) > 0.6).int().to(device)  # fake footprints
    views, imgs = [], []
    for v in range(V):
        gt = (torch.rand(H, W, 3, generator=g) * 255).round().to(torch.uint8)
        views.append({"image": gt, "id": v})
        imgs.append(torch.rand(3, H, W, generator=g).to(device))

    def fake_render(params, view, sh, mask):
        img = imgs[view["id"]]
        cnt = None if mask is None else (foot * mask.reshape(1, -1)).sum(dim=1).int()
        return img, cnt

    d.render_fn = fake_render
    imp, pr = d.scores(params, 5000, views)
    # independent computation
    from .losses import ssim_same

    tot = torch.zeros(n, dtype=torch.long)
    raw = torch.zeros(n, dtype=torch.float64)
    thr = d.args["loss_thresh"]
    for v in range(V):
        r = imgs[v].permute(1, 2, 0).cpu().double()
        gt = views[v]["image"].double() / 255
        e = (r - gt).abs().sum(-1) / 3
        e = (e - e.min()) / (e.max() - e.min())
        m = (e > thr).reshape(-1).long()
        c = (foot.cpu().long() * m[None]).sum(1)
        ephoto = 0.8 * (r - gt).abs().mean() + 0.2 * (1 - ssim_same(r, gt))
        tot += c
        raw += c.double() * float(ephoto)
    imp_ref = tot // V
    pr_ref = (raw - raw.min()) / (raw.max() - raw.min())
    if not torch.equal(imp.cpu().long(), imp_ref):
        raise AssertionError("importance score differs from floor(sum counts / K)")
    err = float((pr.cpu().double() - pr_ref).abs().max())
    if err > 1e-4:
        raise AssertionError(f"pruning score differs from Eq. 6: {err}")
    m0 = error_mask(imgs[0].permute(1, 2, 0), views[0]["image"].to(device).float() / 255, thr)
    return {"importance_mean": float(imp.float().mean()), "pruning_err": err,
            "mask_fraction": float(m0.float().mean())}


def densify_gate(method, device):
    n = 60
    d = _density(method)
    params, opts = _toy(n, device, log_scale=math.log(0.01), opacity=0.9)
    d.initialize(params, opts, 1.0)
    a = d.args
    g = torch.Generator().manual_seed(2)
    # half small (clone side), half large (split side) relative to dense * extent
    with torch.no_grad():
        big = torch.arange(n) % 2 == 1
        params["scales"][big.to(device)] = math.log(a["dense"] * 3)
        params["scales"][~big.to(device)] = math.log(a["dense"] * 0.5)
    d.state["denom"][:] = 1
    d.state["grad_accum"][:] = (torch.rand(n, generator=g) * 2 * d.cfg["grad_thresh"]).to(device)
    d.state["grad_accum_abs"][:] = (torch.rand(n, generator=g) * 2 * a["grad_abs_thresh"]).to(
        device)
    importance = torch.randint(0, 11, (n,), generator=g).to(device)
    pruning = torch.zeros(n, device=device)
    metric = importance > 5
    bigd = big.to(device)
    exp_clone = (d.state["grad_accum"] >= d.cfg["grad_thresh"]) & ~bigd & metric
    exp_split = (d.state["grad_accum_abs"] >= a["grad_abs_thresh"]) & bigd & metric
    # the swapped criteria (vanilla grad for split / no metric) select different sets
    alt_split = (d.state["grad_accum"] >= d.cfg["grad_thresh"]) & bigd & metric
    ids0 = _ids(params)
    row = d.densify(params, opts, 1000, importance, pruning)
    ids = _ids(params)
    counts = torch.bincount(ids.cpu(), minlength=n)
    exp_counts = torch.ones(n, dtype=torch.long)
    exp_counts[exp_clone.cpu()] = 2
    exp_counts[exp_split.cpu()] = 2  # split: original removed, two children
    if not torch.equal(counts, exp_counts):
        raise AssertionError("cloned/split set differs from the VCD gate")
    if row["clones"] != int(exp_clone.sum()) or row["splits"] != int(exp_split.sum()):
        raise AssertionError("clone/split counts mismatch")
    children = torch.isin(ids, ids0[exp_split])
    if children.any():
        sc = params["scales"][children].exp().max(dim=-1).values
        if not torch.allclose(sc, torch.full_like(sc, a["dense"] * 3 / 1.6), rtol=1e-4):
            raise AssertionError("split children do not have scale / 1.6")
    if float(params["opacities"].sigmoid().max()) > 0.8 + 1e-5:
        raise AssertionError("opacity not clamped to 0.8 after densification")
    for name, p in params.items():
        if not p.requires_grad or opts[name].param_groups[0]["params"][0] is not p:
            raise AssertionError(f"{name} not trainable / optimizer stale")
    if int(metric.sum()) == n or torch.equal(exp_split, alt_split):
        raise AssertionError("probe fixture does not discriminate the criteria")
    return {"clones": row["clones"], "splits": row["splits"], "metric_pass": int(metric.sum()),
            "count": row["count"]}


def prune(method, device):
    n = 200
    d = _density(method)
    params, opts = _toy(n, device, opacity=0.5)
    d.initialize(params, opts, 1.0)
    with torch.no_grad():
        params["opacities"][:40] = math.log(0.001 / 0.999)  # 40 candidates (o < 0.005)
    imp = torch.zeros(n, dtype=torch.long, device=device)  # no densification
    pr = torch.zeros(n, device=device)
    pr[:20] = 1.0  # weight 1e6 vs 1: candidates 0..19 are drawn first
    pr[100:105] = 1.0  # high score but not candidates: may be drawn, never pruned
    d.state["denom"][:] = 1
    row = d.densify(params, opts, 1000, imp, pr)
    left = set(_ids(params).tolist())
    removed = set(range(n)) - left
    if not removed <= set(range(40)):
        raise AssertionError("early prune removed non-candidates")
    if len(removed) > 20:
        raise AssertionError("early prune exceeded int(0.5 * #candidates)")
    hit = len(removed & set(range(20)))
    # 20 draws by 1/(1e-6+1-s_p) over 200 rows: candidates 0..19 (weight 1e6) vs 20..39 (1)
    if hit < 8:
        raise AssertionError(f"pruning sampling ignores the score ({hit} of 0..19 removed)")
    # late prune: exactly o < 0.1 or s_p > 0.9
    params2, opts2 = _toy(50, device, opacity=0.5)
    d2 = _density(method)
    d2.initialize(params2, opts2, 1.0)
    with torch.no_grad():
        params2["opacities"][:5] = math.log(0.05 / 0.95)
    pr2 = torch.linspace(0, 1, 50, device=device)
    exp = (params2["opacities"].sigmoid() < 0.1) | (pr2 > 0.9)
    exp_ids = set(_ids(params2)[exp].tolist())
    d2.late_prune(params2, opts2, 18000, pr2)
    left2 = set(_ids(params2).tolist())
    if set(range(50)) - left2 != exp_ids:
        raise AssertionError("late prune set differs from o<0.1 | s_p>0.9")
    return {"early_removed": len(removed), "early_high_score_hits": hit,
            "late_removed": len(exp_ids), "candidates": row["prune_candidates"]}


def schedule(method, device):
    from .density import densify_steps, late_prune_steps, reset_steps

    d = _density(method)
    c = d.cfg
    ds = densify_steps(c)
    interval = c["densification_interval"]
    exp = list(range(1000 if interval == 500 else 600, 15000, interval))
    if ds != exp:
        raise AssertionError("densification steps differ from the official schedule")
    if reset_steps(c) != [3000, 6000, 9000, 12000]:
        raise AssertionError("opacity reset steps differ")
    if late_prune_steps(c) != [18000, 21000, 24000, 27000]:
        raise AssertionError("late prune steps differ")
    params, opts = _toy(8, device)
    d.initialize(params, opts, 1.0)
    seen = {"densify": [], "late": [], "reset": []}
    d.densify = lambda p, o, s, *a: seen["densify"].append(s)
    d.late_prune = lambda p, o, s, *a: seen["late"].append(s)
    import ss_fast_gs.density as mod

    real_reset = mod.reset_opa

    def spy(p, o, st, value):
        seen["reset"].append(value)
        return real_reset(p, o, st, value)

    mod.reset_opa = spy
    try:
        for s in range(1, 30001):
            out = {"info": {"means2d": torch.zeros(8, 4), "radii": torch.zeros(8)}}
            d.after_backward(params, opts, s, out)
    finally:
        mod.reset_opa = real_reset
    if seen["densify"] != exp or seen["late"] != [18000, 21000, 24000, 27000]:
        raise AssertionError("after_backward does not follow the schedule")
    if seen["reset"] != [0.01] * 4:
        raise AssertionError("opacity resets not applied at 3000..12000")
    return {"densify_calls": len(exp), "first": exp[0], "last": exp[-1]}


def optimizer_schedule(method, device):
    from .optim import ScheduledAdam, main_steps_at, sh_steps_at

    mains = [s for s in range(1, 30001) if main_steps_at(s)]
    shs = [s for s in range(1, 30001) if sh_steps_at(s)]
    if len(mains) != 15000 + 157 + 156 or len(shs) != 937 + 157 + 156:
        raise AssertionError(f"schedule counts {len(mains)}, {len(shs)}")
    if 15008 not in shs or 15008 not in mains or 15001 in mains or 29952 not in mains:
        raise AssertionError("schedule boundaries differ")
    # summed gradients: one Adam step on g1+...+g32 vs per-step Adam
    p = torch.nn.Parameter(torch.zeros(3))
    opt = ScheduledAdam([p], lr=0.1, schedule="main")
    q = torch.nn.Parameter(torch.zeros(3))
    ref = torch.optim.Adam([q], lr=0.1, eps=1e-15)
    gs = [torch.tensor([1.0, -2.0, 0.5]) * (k % 5 - 2) for k in range(15009, 15041)]
    for k, gk in zip(range(15009, 15041), gs):
        opt.current_step = k
        p.grad = gk.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
    q.grad = sum(gs)
    ref.step()
    if opt.adam_steps != 1 or not torch.allclose(p, q):
        raise AssertionError("ScheduledAdam does not apply one step on the summed gradient")
    d = _density(method)
    params, opts = _toy(6, device)
    d.initialize(params, opts, 1.0)
    if not all(isinstance(o, ScheduledAdam) for o in opts.values()):
        raise AssertionError("optimizers not replaced by ScheduledAdam")
    lr = {k: o.param_groups[0]["lr"] for k, o in opts.items()}
    a = d.args
    if abs(lr["opacities"] - 0.025) > 1e-12 or abs(lr["shN"] - a["highfeature_lr"] / 20) > 1e-12:
        raise AssertionError("learning-rate overrides not applied")
    return {"main_steps": len(mains), "sh_steps": len(shs), "lr": lr}


def scene_table(method, device):
    from . import scenes

    sg, ag = scenes.resolve(138766, "base")
    sc, ac = scenes.resolve(155767, "big")
    su, au = scenes.resolve(12345, "base")
    if sg != "garden" or ag["loss_thresh"] != 0.06 or ag["grad_abs_thresh"] != 0.0008 \
            or ag["highfeature_lr"] != 0.02 or ag["dense"] != 0.001:
        raise AssertionError("garden base arguments wrong")
    if sc != "counter" or ac["grad_abs_thresh"] != 0.0004 or ac["highfeature_lr"] != 0.02:
        raise AssertionError("counter big arguments wrong")
    if su is not None or au != scenes.ARG_DEFAULTS:
        raise AssertionError("unknown scene should use the argument defaults")
    d = _density(method)
    mult = d.cfg["mult"]
    params, opts = _toy(136029, device)  # truck count: official mult 0.7
    try:
        d.initialize(params, opts, 1.0)
        refused = False
    except ValueError:
        refused = True
    if refused != (abs(mult - 0.7) > 1e-9):
        raise AssertionError("CB mult mismatch handling wrong")
    renderer = _component(method, "renderer")
    if abs(renderer.config["mult"] - mult) > 1e-12:
        raise AssertionError("renderer and density mult differ")
    return {"garden": ag, "counter_big": ac, "mult": mult, "truck_refused": refused}
