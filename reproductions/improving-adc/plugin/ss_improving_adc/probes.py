"""Mechanism probes for the Improving-ADC port. Each raises AssertionError when the mechanism is
removed or wrong; the references are written independently of density.py / core.py."""

import math
from dataclasses import replace

import torch

from . import density as density_module
from .density import ImprovedADC


def _density(method, **config):
    spec = next(c for c in method.components if c.kind == "density")
    return ImprovedADC(replace(spec, config={**spec.config, "log": False, **config}))


def _scene(n, device, seed=0, depth=3.0, spread=0.6, scale=0.08, opacity=None):
    from splatstudio.state import initialize

    from . import method as make

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += depth
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(make(), pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_(torch.full((n, 3), math.log(scale)) + 0.3 * torch.randn(
            n, 3, generator=g))
        q = torch.randn(n, 4, generator=g)
        params["quats"].copy_(torch.nn.functional.normalize(q, dim=-1))
        o = torch.rand(n, generator=g) * 0.8 + 0.1 if opacity is None else opacity
        params["opacities"].copy_(torch.logit(torch.as_tensor(o, dtype=torch.float32)))
    return params, opts


def _camera(device, w=40, h=32, f=30.0, center=(0.0, 0.0, 0.0)):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    viewmat = torch.eye(4, device=device)
    viewmat[:3, 3] = -torch.tensor(center, device=device)  # identity rotation, centre C
    return {"K": K, "viewmat": viewmat, "width": w, "height": h}


def _render(params, cam, colors=None):
    from gsplat import rasterization

    kw = dict(sh_degree=0, colors=torch.cat((params["sh0"], params["shN"]), dim=1))
    if colors is not None:
        kw = dict(sh_degree=None, colors=colors)
    rgb, alpha, meta = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), viewmats=cam["viewmat"][None],
        Ks=cam["K"][None], width=cam["width"], height=cam["height"], packed=False, **kw)
    return rgb[0], alpha[0], meta


def _brute(meta):
    """Per-pixel front-to-back loop (gsplat numerics) -> count, sum alpha*T, alpha image."""
    m2d, con, op = (meta[k][0].detach().cpu() for k in ("means2d", "conics", "opacities"))
    offs, ids = meta["isect_offsets"][0].flatten().tolist(), meta["flatten_ids"].tolist()
    ts, W, H, tw = meta["tile_size"], meta["width"], meta["height"], meta["tile_width"]
    n = m2d.shape[0]
    count, blend = [0.0] * n, [0.0] * n
    alpha_img = torch.zeros(H, W)
    for i in range(H):
        for j in range(W):
            tile = (i // ts) * tw + (j // ts)
            end = offs[tile + 1] if tile + 1 < len(offs) else len(ids)
            T, px, py = 1.0, j + 0.5, i + 0.5
            for k in range(offs[tile], end):
                gid = ids[k]
                dx, dy = float(m2d[gid, 0]) - px, float(m2d[gid, 1]) - py
                c = con[gid].tolist()
                s = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
                a = min(0.999, float(op[gid]) * math.exp(-s))
                if s < 0 or a < 1 / 255:
                    continue
                if T * (1 - a) <= 1e-4:
                    break
                count[gid] += 1
                blend[gid] += a * T
                T *= 1 - a
            alpha_img[i, j] = 1 - T
    return torch.tensor(count), torch.tensor(blend), alpha_img


# ----------------------------------------------------------------------------- probes
def stats_kernel(method, device):
    from .kernels import pixel_stats

    params, _ = _scene(30, device, seed=1)
    cam = _camera(device)
    with torch.no_grad():
        _, alpha, meta = _render(params, cam)
    cnt, blend = pixel_stats(meta)
    ref_cnt, ref_blend, alpha_ref = _brute(meta)
    torch.testing.assert_close(cnt.cpu(), ref_cnt, rtol=0, atol=0)
    torch.testing.assert_close(blend.cpu(), ref_blend, rtol=1e-4, atol=1e-3)
    if float(ref_cnt.sum()) < 200:
        raise AssertionError("Probe scene too sparse")
    torch.testing.assert_close(alpha_ref, alpha[..., 0].cpu(), rtol=1e-3, atol=1e-3)
    # official route (train.py:127-147): gradient of sum(render(fake zero colour)) w.r.t. colour
    fake = torch.zeros(len(params["means"]), 3, device=device, requires_grad=True)
    img, _, _ = _render({k: v.detach() for k, v in params.items()}, cam, colors=fake)
    img.sum().backward()
    autograd_sigma = fake.grad.mean(dim=1)
    torch.testing.assert_close(blend, autograd_sigma, rtol=1e-3, atol=1e-3)
    return {"pairs": float(ref_cnt.sum()), "sigma_total": float(blend.sum()),
            "max_sigma_err_vs_autograd": float((blend - autograd_sigma).abs().max())}


def _cameras_clustered(device, radius=0.5, n=8):
    cams = []
    for i in range(n):
        a = 2 * math.pi * i / n
        cams.append(_camera(device, center=(radius * math.cos(a), radius * math.sin(a), 0.0)))
    return cams


def scene_extent(method, device):
    g = torch.Generator().manual_seed(3)
    pts = torch.randn(500, 3, generator=g) * 6.0 + torch.tensor([0.0, 0.0, 8.0])
    cams = _cameras_clustered(device)
    centers = torch.stack([torch.tensor([0.5 * math.cos(2 * math.pi * i / 8),
                                         0.5 * math.sin(2 * math.pi * i / 8), 0.0])
                           for i in range(8)]).double()
    cbar = centers.mean(0)
    ref_new = float(sum(math.dist(p.tolist(), cbar.tolist()) for p in pts.double()) / len(pts))
    ref_cam = 1.1 * max(math.dist(c.tolist(), cbar.tolist()) for c in centers)
    out = {}
    for mode in ("sfm_mean", "camera"):
        params, opts = _scene(500, device)
        with torch.no_grad():
            params["means"].copy_(pts.to(device))
        opts["means"].param_groups[0]["lr"] = 1.6e-4 * ref_cam  # runtime: lr *= scene_scale
        d = _density(method, extent=mode)
        d.initialize(params, opts, ref_cam)
        d.set_training_cameras(cams)
        out[mode] = (d.state["extent"], opts["means"].param_groups[0]["lr"])
    e_new, lr_new = out["sfm_mean"]
    e_cam, lr_cam = out["camera"]
    if abs(e_new - ref_new) > 1e-4 * ref_new or abs(e_cam - ref_cam) > 1e-5:
        raise AssertionError(f"extent {e_new} vs Eq.13 {ref_new}; camera {e_cam} vs {ref_cam}")
    if not e_new > 5 * e_cam:
        raise AssertionError("Fixture should make the corrected extent much larger")
    if abs(lr_new - 1.6e-4 * ref_new) > 1e-9 or abs(lr_cam - 1.6e-4 * ref_cam) > 1e-9:
        raise AssertionError(f"means LR not scaled by the used extent: {lr_new}, {lr_cam}")
    # functional: a Gaussian of size 0.02 < 0.01*e' clones under e', splits under e_cam
    res = {}
    for mode, e in (("sfm_mean", e_new), ("camera", e_cam)):
        params, opts = _scene(4, device)
        with torch.no_grad():
            params["scales"].fill_(math.log(0.02))
        d = _density(method, extent=mode)
        d.initialize(params, opts, e_cam)
        d.state["extent"] = e
        d.state["grad_accum"][:] = 1.0
        d.state["pix_count"][:] = 1.0
        res[mode] = d.densify_and_prune(params, opts, 600)
    if not (res["sfm_mean"]["clone"] == 4 and res["sfm_mean"]["split"] == 0
            and res["camera"]["split"] == 4 and res["camera"]["clone"] == 0):
        raise AssertionError(f"extent does not drive clone/split: {res}")
    return {"corrected_extent": e_new, "camera_extent": e_cam, "means_lr": lr_new}


def threshold_schedule(method, device):
    d = _density(method)
    if d.intervals != 145:
        raise AssertionError(f"dense_iters should be 145, got {d.intervals}")
    ts = [d.current_threshold(k) for k in range(1, 145)]
    for k, t in enumerate(ts, start=1):
        i_over = k / 145  # Eq. 14 at i/i_max
        ref = math.exp(math.log(1e-4) * (1 - i_over) + math.log(4e-4) * i_over)
        if abs(t - ref) > 1e-12:
            raise AssertionError(f"call {k}: {t} vs Eq. 14 {ref}")
    if not (all(a < b for a, b in zip(ts, ts[1:])) and 1e-4 < ts[0] and ts[-1] < 4e-4):
        raise AssertionError("Threshold must rise strictly inside (1e-4, 4e-4)")
    # functional: tau' = 2.5e-4 is above the threshold at call 1, below at call 144
    res = {}
    for mode, call in (("exponential", 0), ("exponential", 143), ("constant", 143)):
        params, opts = _scene(3, device)
        d = _density(method, threshold=mode)
        d.initialize(params, opts, 1.0)
        d.state["calls"] = call
        d.state["grad_accum"][:] = 2.5e-4
        d.state["pix_count"][:] = 1.0
        res[(mode, call + 1)] = d.densify_and_prune(params, opts, 600)
    grown = {k: v["clone"] + v["split"] for k, v in res.items()}
    if grown[("exponential", 1)] != 3 or grown[("exponential", 144)] != 0:
        raise AssertionError(f"Ascending threshold not applied: {grown}")
    if grown[("constant", 144)] != 3:
        raise AssertionError("Constant-threshold ablation should densify at 2.5e-4 >= 2e-4")
    return {"first": ts[0], "last": ts[-1]}


def _independent_prune(sigma_old, keep_old, opacity, min_o):
    """Bottom-N_prune by explicit sort + linear interpolation (numpy-style quantile)."""
    scores = sorted(float(s) for s, k in zip(sigma_old, keep_old) if k)
    n_prune = sum(1 for o in opacity if o < min_o)
    out = [False] * len(opacity)
    if not scores or n_prune == 0:
        return out
    p = min(n_prune / len(sigma_old), 1.0) * (len(scores) - 1)
    lo, hi = math.floor(p), math.ceil(p)
    q = scores[lo] + (scores[hi] - scores[lo]) * (p - lo)
    surv = [float(s) for s, k in zip(sigma_old, keep_old) if k]
    for i, s in enumerate(surv):
        out[i] = s < q and float(opacity[i]) < min_o
    return out


def significance_prune(method, device):
    # Scene: opaque foreground wall in front, small low-opacity Gaussians hidden behind it
    # (sigma ~ 0), a large low-opacity background Gaussian visible around the wall (high sigma),
    # a huge opaque Gaussian (would be size-pruned by 3DGS), and ordinary Gaussians.
    n_fg, n_hidden, n_vis = 6, 8, 10
    n = n_fg + n_hidden + n_vis + 2
    params, opts = _scene(n, device, seed=4)
    g = torch.Generator().manual_seed(5)
    means = torch.zeros(n, 3)
    scales = torch.zeros(n, 3)
    opac = torch.zeros(n)
    i = 0
    for k in range(n_fg):  # opaque wall left of the image centre at z=2
        means[i] = torch.tensor([-0.5 + (k % 3 - 1) * 0.1, (k // 3 - 0.5) * 0.1, 2.0])
        scales[i] = math.log(0.15)
        opac[i] = 0.99
        i += 1
    hidden = list(range(i, i + n_hidden))
    for _ in range(n_hidden):  # directly behind the wall, low opacity
        xy = (torch.rand(2, generator=g) - 0.5) * 0.2 + torch.tensor([-0.5, 0.0])
        means[i] = torch.cat([xy, torch.tensor([4.0])])
        scales[i] = math.log(0.03)
        opac[i] = 0.003
        i += 1
    for _ in range(n_vis):  # ordinary visible Gaussians, opaque
        means[i] = torch.cat([(torch.rand(2, generator=g) - 0.5) * 2.4, torch.tensor([3.0])])
        scales[i] = math.log(0.06)
        opac[i] = 0.6
        i += 1
    background = i  # large far low-opacity Gaussian behind everything
    means[i] = torch.tensor([1.2, 0.0, 8.0])  # alpha >= 1/255 needs o*G >= 0.0039
    scales[i] = math.log(3.0)
    opac[i] = 0.0049
    i += 1
    huge = i  # huge opaque Gaussian far away (max scale > 0.1 * extent)
    means[i] = torch.tensor([0.0, 0.0, 30.0])
    scales[i] = math.log(10.0)
    opac[i] = 0.5
    with torch.no_grad():
        params["means"].copy_(means.to(device))
        params["scales"].copy_(scales.to(device))
        params["quats"].copy_(torch.tensor([1.0, 0, 0, 0]).expand(n, 4).to(device))
        params["opacities"].copy_(torch.logit(opac).to(device))
    cam = _camera(device, w=64, h=48, f=40.0)
    results = {}
    for mode in ("significance", "opacity"):
        p = {k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()}
        o = {k: torch.optim.Adam([v], lr=1e-3, eps=1e-15) for k, v in p.items()}
        d = _density(method, pruning=mode, grad_start=1.0, grad_end=2.0)  # no densification
        d.initialize(p, o, 1.0)
        d.state["extent"] = d.state["camera_extent"] = 2.0
        for _ in range(3):
            rgb, alpha, meta = _render(p, cam)
            meta["means2d"].retain_grad()
            (rgb + (1 - alpha)).square().mean().backward()
            d.accumulate(meta)
        sigma = d.state["sigma"].clone().cpu()
        opa_before = p["opacities"].sigmoid().detach().cpu()
        ids = torch.arange(n, dtype=torch.float32, device=device)
        p["sh0"].data[:, 0, 0] = ids  # tag Gaussians to identify survivors
        row = d.densify_and_prune(p, o, 600)
        survivors = set(int(round(float(v))) for v in p["sh0"][:, 0, 0])
        results[mode] = (row, survivors, sigma, opa_before)
    row, surv, sigma, opa = results["significance"]
    if not float(sigma[background]) > 20 * max(float(sigma[h]) for h in hidden):
        raise AssertionError(f"Fixture: background sigma {sigma[background]} not dominant")
    expected = _independent_prune(sigma.tolist(), [True] * n, opa.tolist(), 0.005)
    pruned = {k for k in range(n) if k not in surv}
    if pruned != {k for k, e in enumerate(expected) if e}:
        raise AssertionError(f"Prune set {sorted(pruned)} differs from reference")
    if background not in surv or huge not in surv:
        raise AssertionError("Background / huge Gaussian was pruned (background not preserved)")
    if not pruned or not pruned <= set(hidden):
        raise AssertionError(f"Expected only hidden low-significance Gaussians pruned: {pruned}")
    _, surv_o, _, _ = results["opacity"]
    if background in surv_o:
        raise AssertionError("Opacity-only ablation should remove the background Gaussian")
    return {"pruned": len(pruned), "low_opacity": row["low_opacity"],
            "background_sigma": float(sigma[background]),
            "hidden_sigma_max": max(float(sigma[h]) for h in hidden)}


def pixel_gradient(method, device):
    params, opts = _scene(40, device, seed=6, depth=2.0, spread=1.5)
    cam = _camera(device)
    e_cam = 10.0  # gamma * e = 3.7 -> depths 1.25..2.75 get factor < 1
    out = {}
    for flag in (True, False):
        d = _density(method, pixel_gradient=flag)
        d.initialize(params, opts, e_cam)
        rgb, alpha, meta = _render(params, cam)
        meta["means2d"].retain_grad()
        (rgb + (1 - alpha) - 0.3).abs().mean().backward()
        for v in params.values():
            v.grad = None
        d.accumulate(meta)
        out[flag] = (d.grads().cpu(), meta)
    tau, meta = out[True]
    # independent: depth from the means and the view matrix, counts from the brute-force loop
    cnt, _, _ = _brute(meta)
    m = params["means"].detach().cpu()
    vm = cam["viewmat"].cpu()
    z = (m @ vm[:3, :3].T + vm[:3, 3])[:, 2]
    f = torch.clamp((z / (0.37 * e_cam)) ** 2, max=1.0)
    gr = meta["means2d"].grad[0].cpu() * torch.tensor([cam["width"] / 2, cam["height"] / 2])
    vis = (meta["radii"][0].cpu() > 0).all(-1)
    num = torch.where(vis, f * cnt * gr.norm(dim=-1), 0.0)
    den = torch.where(vis, cnt, 0.0)
    ref = torch.nan_to_num(num / den, nan=0.0)
    torch.testing.assert_close(tau, ref, rtol=1e-4, atol=1e-9)
    if not ((f < 0.99) & (cnt > 0)).any():
        raise AssertionError("Fixture should include depth-attenuated Gaussians")
    plain = torch.where(vis, gr.norm(dim=-1), 0.0)
    torch.testing.assert_close(out[False][0], plain, rtol=1e-4, atol=1e-9)
    if torch.allclose(tau, plain, rtol=1e-2, atol=1e-9):
        raise AssertionError("PixelGS gradient should differ from the 3DGS per-view mean")
    return {"visible": int(vis.sum()), "min_depth_factor": float(f.min())}


def revised_opacity(method, device):
    params, opts = _scene(4, device, opacity=torch.tensor([0.3, 0.6, 0.8, 0.5]))
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    with torch.no_grad():
        params["scales"].copy_(torch.log(torch.tensor([[0.001] * 3, [0.5] * 3, [0.001] * 3,
                                                       [0.001] * 3])).to(device))
    o0 = params["opacities"].sigmoid().detach().clone()
    m0 = opts["means"].state[params["means"]]["exp_avg"].clone()
    d = _density(method)
    d.initialize(params, opts, 1.0)
    d.state["grad_accum"][:] = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device)
    d.state["pix_count"][:] = 1.0
    row = d.densify_and_prune(params, opts, 600)
    if (row["clone"], row["split"]) != (1, 1):
        raise AssertionError(f"Fixture should clone #0 and split #1: {row}")
    # layout after ops: [0, 2, 3, clone of 0, two children of 1]
    o = params["opacities"].sigmoid().detach()
    r = lambda x: 1 - math.sqrt(1 - float(x))  # noqa: E731  (Eq. 8)
    want = [r(o0[0]), float(o0[2]), float(o0[3]), r(o0[0]), r(o0[1]), r(o0[1])]
    torch.testing.assert_close(o.cpu(), torch.tensor(want), rtol=1e-5, atol=1e-6)
    m = opts["means"].state[params["means"]]["exp_avg"]
    torch.testing.assert_close(m[:3], m0[[0, 2, 3]])
    if torch.count_nonzero(m[3:]):
        raise AssertionError("New Gaussians must start with zero Adam moments")
    d2 = _density(method, revised_opacity=False)
    p2, o2 = _scene(4, device, opacity=torch.tensor([0.3, 0.6, 0.8, 0.5]))
    with torch.no_grad():
        p2["scales"].copy_(params["scales"][:4].detach() * 0 + math.log(0.001))
    d2.initialize(p2, o2, 1.0)
    d2.state["grad_accum"][:] = torch.tensor([1.0, 0, 0, 0], device=device)
    d2.state["pix_count"][:] = 1.0
    d2.densify_and_prune(p2, o2, 600)
    if abs(float(p2["opacities"][-1].sigmoid()) - 0.3) > 1e-5:
        raise AssertionError("Ablation without revised opacity should keep o")
    return {"clone_opacity": want[3], "split_opacity": want[4]}


def schedule(method, device):
    d = _density(method)
    params, opts = _scene(3, device)
    d.initialize(params, opts, 1.0)
    calls = {"acc": [], "dens": [], "reset": [], "rs": []}
    now = [0]
    d.accumulate = lambda info: calls["acc"].append(info["step"])
    d.densify_and_prune = lambda p, o, step: calls["dens"].append(step)
    original = density_module.reset_opa
    def fake_reset(p, o, s, value):
        calls["reset"].append(value)
        calls["rs"].append(now[0])

    density_module.reset_opa = fake_reset
    try:
        for step in range(1, 30001):
            now[0] = step
            outputs = {"info": {"step": step}, "step": step}
            if step < 15000:
                outputs["info"]["means2d"] = torch.zeros(1, requires_grad=True) * 1.0
            d.before_backward(params, opts, step, outputs)
            d.after_backward(params, opts, step, outputs)
    finally:
        density_module.reset_opa = original
    if calls["dens"] != list(range(600, 15000, 100)) or len(calls["dens"]) != 144:
        raise AssertionError(f"Densify steps wrong: {calls['dens'][:3]}..{calls['dens'][-3:]}")
    if calls["rs"] != [3000, 6000, 9000, 12000] or set(calls["reset"]) != {0.01}:
        raise AssertionError(f"Opacity reset steps wrong: {calls['rs']}")
    if calls["acc"] != list(range(1, 15000)):
        raise AssertionError("Statistics must accumulate at every step < 15000 only")
    return {"densify_calls": len(calls["dens"]), "resets": calls["rs"]}
