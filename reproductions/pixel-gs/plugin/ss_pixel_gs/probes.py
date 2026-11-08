"""Mechanism probes for the Pixel-GS port. Each raises AssertionError when the mechanism is
removed or wrong; references are written independently of density.py / core.py.

Fixture helpers (_scene, _camera, _render, _brute) follow the validated Improving-ADC port
(reproductions/improving-adc/plugin/ss_improving_adc/probes.py)."""

import math
from dataclasses import replace

import torch


def _density(method, **config):
    from .density import PixelGSDensity

    spec = next(c for c in method.components if c.kind == "density")
    return PixelGSDensity(replace(spec, config={**spec.config, **config}))


def _scene(n, device, seed=0, depth=3.0, spread=0.6, scale=0.08):
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
        params["quats"].copy_(torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1))
        o = torch.rand(n, generator=g) * 0.8 + 0.1
        params["opacities"].copy_(torch.logit(o))
    return params, opts


def _camera(device, w=40, h=32, f=30.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": w, "height": h}


def _render(params, cam):
    from gsplat import rasterization

    rgb, alpha, meta = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat((params["sh0"], params["shN"]), dim=1), sh_degree=0,
        viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=cam["width"],
        height=cam["height"], packed=False)
    return rgb[0], alpha[0], meta


def _brute(meta):
    """Per-pixel front-to-back loop with gsplat numerics -> per-Gaussian count, alpha image."""
    m2d, con, op = (meta[k][0].detach().cpu() for k in ("means2d", "conics", "opacities"))
    offs, ids = meta["isect_offsets"][0].flatten().tolist(), meta["flatten_ids"].tolist()
    ts, W, H, tw = meta["tile_size"], meta["width"], meta["height"], meta["tile_width"]
    count = [0.0] * m2d.shape[0]
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
                T *= 1 - a
            alpha_img[i, j] = 1 - T
    return torch.tensor(count), alpha_img


def pixel_count_kernel(method, device):
    from .kernels import pixel_counts

    params, _ = _scene(30, device, seed=1)
    with torch.no_grad():
        _, alpha, meta = _render(params, _camera(device))
    cnt = pixel_counts(meta).cpu()
    ref, alpha_ref = _brute(meta)
    torch.testing.assert_close(cnt, ref, rtol=0, atol=0)
    torch.testing.assert_close(alpha_ref, alpha[..., 0].cpu(), rtol=1e-3, atol=1e-3)
    if float(ref.sum()) < 200 or float(ref.max()) < 20:
        raise AssertionError("Probe scene too sparse")
    return {"pairs": float(ref.sum()), "max_count": float(ref.max())}


def pixel_aware_accumulation(method, device):
    """Two views (one view cancels the weights in Eq. 7): accumulate real render/backward
    statistics and compare with sum_k m_k f_k |g_k| / sum_k m_k from brute-force counts."""
    params, opts = _scene(40, device, seed=6, depth=2.0, spread=1.5)
    cams = [_camera(device), _camera(device, f=18.0)]
    cams[1]["viewmat"][:3, 3] = torch.tensor([0.4, -0.2, 0.3], device=device)
    radius = 10.0  # gamma * radius = 3.7 > all depths -> every factor < 1
    out, views = {}, []
    for flags in ((True, True), (True, False), (False, True), (False, False)):
        d = _density(method, pixel_aware=flags[0], depth_scaling=flags[1])
        d.initialize(params, opts, radius)
        for k, cam in enumerate(cams):
            rgb, alpha, meta = _render(params, cam)
            d.before_backward(params, opts, 1, {"info": meta})
            (rgb + (1 - alpha) - 0.3).abs().mean().backward()
            for v in params.values():
                v.grad = None
            d.strategy._update_state(params, d.state, meta)
            if flags == (True, True):
                views.append((cam, meta))
        out[flags] = (d.state["grad2d"] / d.state["count"].clamp_min(1)).cpu()  # _grow_gs
    m = params["means"].detach().cpu()
    num = {key: 0.0 for key in out}
    den = {key: 0.0 for key in out}
    counts = []
    for cam, meta in views:
        cnt, _ = _brute(meta)
        counts.append(cnt)
        vm = cam["viewmat"].cpu()
        z = (m @ vm[:3, :3].T + vm[:3, 3])[:, 2]
        f = torch.clamp((z / (0.37 * radius)) ** 2, max=1.0)
        gn = (meta["means2d"].grad[0].cpu() * torch.tensor([cam["width"] / 2,
                                                             cam["height"] / 2])).norm(dim=-1)
        vis = (meta["radii"][0].cpu() > 0).all(-1).float()
        for key in out:
            w = cnt if key[0] else torch.ones_like(gn)
            fac = f if key[1] else torch.ones_like(gn)
            num[key] = num[key] + vis * w * fac * gn
            den[key] = den[key] + vis * w
    for key in out:
        ref = torch.nan_to_num(num[key] / den[key], nan=0.0)
        torch.testing.assert_close(out[key], ref, rtol=1e-3, atol=1e-8)
    plain = out[(False, False)]
    if torch.allclose(out[(True, True)], plain, rtol=1e-2, atol=1e-9):
        raise AssertionError("Pixel-GS statistic should differ from the 3DGS per-view mean")
    if torch.allclose(out[(True, False)], plain, rtol=1e-2, atol=1e-9):
        raise AssertionError("pixel weighting had no effect on the fixture")
    if torch.allclose(out[(False, True)], plain, rtol=1e-2, atol=1e-9):
        raise AssertionError("depth scaling had no effect on the fixture")
    both = (counts[0] > 0) & (counts[1] > 0)
    if int(both.sum()) < 5 or torch.allclose(counts[0][both], counts[1][both]):
        raise AssertionError("fixture should have Gaussians with different counts per view")
    rel = ((out[(True, False)] - plain).abs() / plain.clamp_min(1e-12))[both]
    return {"seen_in_both_views": int(both.sum()), "max_rel_diff_pixel_vs_3dgs": float(rel.max())}


def depth_scaling(method, device):
    from .core import depth_factor

    gamma, R = 0.37, 4.0
    z = torch.tensor([0.37, 0.74, 1.48, 2.0, 5.0])
    f = depth_factor(z, gamma, R)
    ref = torch.tensor([(0.37 / 1.48) ** 2, (0.74 / 1.48) ** 2, 1.0, 1.0, 1.0])
    torch.testing.assert_close(f, ref)
    # scale-invariance of the threshold in radius units (Eq. 10)
    torch.testing.assert_close(depth_factor(z * 3, gamma, R * 3), f)
    # functional: a near Gaussian with |g| = 4e-4 densifies without, not with the scaling
    from .core import view_contribution

    g, m, zz = torch.tensor([4e-4, 4e-4]), torch.tensor([10.0, 10.0]), torch.tensor([0.3, 3.0])
    kw = dict(pixel_aware=True, gamma_depth=gamma, radius=R)
    n1, d1 = view_contribution(g, m, zz, depth_scaling=True, **kw)
    n0, d0 = view_contribution(g, m, zz, depth_scaling=False, **kw)
    t1, t0 = n1 / d1, n0 / d0
    if not (t0[0] > 2e-4 and t1[0] < 2e-4 and t1[1] > 2e-4):
        raise AssertionError(f"depth scaling should suppress only the near Gaussian: {t1}, {t0}")
    return {"f": f.tolist(), "near_tau_scaled": float(t1[0])}
