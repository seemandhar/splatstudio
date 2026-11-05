"""Mechanism probes for the StopThePop port. Each raises on failure and returns measurements."""

import math
import time
from dataclasses import replace

import torch
import torch.nn.functional as F

from . import reference as ref

C0 = ref.C0


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _params(means, scales, quats, opac, colors, device):
    """colors in [0,1] -> sh0 such that C0*sh0 + 0.5 = color (degree 0)."""
    n = len(means)
    vals = {"means": torch.as_tensor(means, dtype=torch.float32),
            "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
            "quats": F.normalize(torch.as_tensor(quats, dtype=torch.float32), dim=-1),
            "opacities": torch.logit(torch.as_tensor(opac, dtype=torch.float32)),
            "sh0": ((torch.as_tensor(colors, dtype=torch.float32) - 0.5) / C0)[:, None, :],
            "shN": torch.zeros(n, 15, 3)}
    return {k: torch.nn.Parameter(v.to(device)) for k, v in vals.items()}


def _camera(W, H, f, device, R=None):
    view = torch.eye(4)
    if R is not None:
        view[:3, :3] = torch.as_tensor(R, dtype=torch.float32)
    return {"viewmat": view.to(device), "width": W, "height": H,
            "K": torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)}


def _quat_axis(axis, deg):
    a = math.radians(deg) / 2
    ax = torch.tensor(axis, dtype=torch.float32)
    ax = ax / ax.norm()
    return [math.cos(a), *(ax * math.sin(a)).tolist()]


def _settings(**kw):
    from .raster import make_settings

    return make_settings(**kw)


def _render(params, camera, settings, bg):
    from .raster import render

    color, means2d, radii = render(params, camera, 0, bg, settings)
    return color, means2d, radii


def _crossing_scene(device, n_bg=18, seed=3):
    """Two flat elongated splats tilted in opposite directions (X in depth) whose per-pixel
    order flips across the image, plus small random background Gaussians (<= 20 per tile,
    so the hierarchical window always holds the full list)."""
    g = torch.Generator().manual_seed(seed)
    means = [[0.0, 0.0, 3.0], [0.0, 0.0, 3.02]]
    scales = [[0.28, 0.14, 0.01], [0.28, 0.14, 0.01]]
    quats = [_quat_axis([0, 1, 0], 35.0), _quat_axis([0, 1, 0], -35.0)]
    opac = [0.85, 0.8]
    colors = [[0.9, 0.1, 0.1], [0.1, 0.2, 0.9]]
    bm = torch.rand(n_bg, 3, generator=g) * torch.tensor([1.6, 1.6, 2.0]) + torch.tensor(
        [-0.8, -0.8, 3.6])
    means += bm.tolist()
    scales += (torch.rand(n_bg, 3, generator=g) * 0.12 + 0.05).tolist()
    quats += F.normalize(torch.randn(n_bg, 4, generator=g), dim=-1).tolist()
    opac += (torch.rand(n_bg, generator=g) * 0.6 + 0.2).tolist()
    colors += torch.rand(n_bg, 3, generator=g).tolist()
    return _params(means, scales, quats, opac, colors, device)


# ---------------------------------------------------------------------------- C1
def topt(method, device):
    """Per-ray depth is Eq. (1): a tilted flat splat in front of / behind a fronto-parallel one
    depending on the pixel. The full per-pixel sort (PPX_FULL) and HIER images equal the
    reference blended by Eq. (1) t_opt and not the one blended by the projection of the mean
    onto the ray d.(mu - o) (= Eq. (1) with Sigma = I) or by the global view z.
    (The kernel's Depth debug output is min/max-normalised and colour-mapped, so t_opt is
    checked through the blend order it induces.)"""
    _cuda(device)
    params = _params([[0.0, 0.0, 3.0], [0.0, 0.0, 3.04]], [[0.3, 0.25, 0.004], [0.3, 0.3, 0.004]],
                     [_quat_axis([0, 1, 0], 50.0), [1.0, 0, 0, 0]], [0.85, 0.85],
                     [[0.9, 0.2, 0.1], [0.1, 0.3, 0.9]], device)
    cam = _camera(48, 48, 80.0, device)
    bg = torch.zeros(3, device=device)
    with torch.no_grad():
        refs = {o: ref.composite(params, cam, bg.cpu(), o)[0] for o in ("perpixel", "proj",
                                                                         "global")}
        res = {}
        for name, s in (("ppx_full", _settings(sort_mode="PPX_FULL", sort_order="Z_DEPTH")),
                        ("hier", _settings())):
            img = _render(params, cam, s, bg)[0].permute(1, 2, 0).double().cpu()
            res[name] = {o: (img - r).abs().max().item() for o, r in refs.items()}
    gap = min((refs["perpixel"] - refs[o]).abs().max().item() for o in ("proj", "global"))
    if gap < 0.1:
        raise AssertionError(f"fixture does not separate Eq. (1) from the ablations ({gap:.3f})")
    for name, r in res.items():
        if r["perpixel"] > 2e-3:
            raise AssertionError(f"{name} blend order is not Eq. (1) t_opt: {res}")
        if min(r["proj"], r["global"]) < 0.05:
            raise AssertionError(f"{name} matches an ablated depth: {res}")
    return {"fixture_gap": gap, **res}


# ---------------------------------------------------------------------------- C2
def perpixel_order(method, device):
    """HIER (and PPX_FULL) image = independent per-pixel t_opt-sorted compositor; GLOBAL is not."""
    _cuda(device)
    params = _crossing_scene(device)
    cam = _camera(48, 48, 120.0, device)
    bg = torch.tensor([0.05, 0.1, 0.05], device=device)
    with torch.no_grad():
        want, _ = ref.composite(params, cam, bg.cpu(), "perpixel")
        want_glob, _ = ref.composite(params, cam, bg.cpu(), "global")
        res = {}
        for name, s in (("hier", _settings()),
                        ("ppx_full", _settings(sort_mode="PPX_FULL", sort_order="Z_DEPTH")),
                        ("global", _settings(sort_mode="GLOBAL", sort_order="Z_DEPTH"))):
            img = _render(params, cam, s, bg)[0].permute(1, 2, 0).double().cpu()
            res[name] = {"vs_perpixel_ref": (img - want).abs().max().item(),
                         "vs_global_ref": (img - want_glob).abs().max().item()}
    gap = (want - want_glob).abs().max().item()
    if gap < 0.1:
        raise AssertionError(f"fixture: global and per-pixel orders barely differ ({gap:.3f})")
    for name in ("hier", "ppx_full"):
        if res[name]["vs_perpixel_ref"] > 2e-3:
            raise AssertionError(f"{name} differs from per-pixel sorted reference: {res[name]}")
    if res["global"]["vs_global_ref"] > 2e-3:
        raise AssertionError(f"GLOBAL mode differs from 3DGS global-order reference: {res}")
    if res["global"]["vs_perpixel_ref"] < 0.05:
        raise AssertionError("GLOBAL mode matches per-pixel reference: probe cannot discriminate")
    return {"order_gap": gap, **res}


# ---------------------------------------------------------------------------- C4
def backward(method, device):
    """Kernel gradients (opacity logits, sh0, screen means) = autograd of the per-pixel sorted
    float64 reference for L = sum w * C; a global-order reference gives different gradients."""
    _cuda(device)
    params = _crossing_scene(device)
    cam = _camera(48, 48, 120.0, device)
    bg = torch.tensor([0.05, 0.1, 0.05], device=device)
    g = torch.Generator().manual_seed(11)
    w = torch.randn(48, 48, 3, generator=g, dtype=torch.float64)
    color, means2d, _ = _render(params, cam, _settings(), bg)
    (color.permute(1, 2, 0) * w.float().to(device)).sum().backward()
    got = {"opac_logit": params["opacities"].grad.double().cpu(),
           "sh0": params["sh0"].grad.double().cpu(),
           "ndc": means2d.grad[:, :2].double().cpu()}
    out = {}
    for order in ("perpixel", "global"):
        C, leaves = ref.composite(params, cam, bg.cpu(), order)
        (C * w).sum().backward()
        out[order] = {k: ((got[k] - leaves[k].grad).abs().max()
                          / leaves[k].grad.abs().max()).item() for k in got}
    worst = max(out["perpixel"].values())
    if worst > 5e-3:
        raise AssertionError(f"sorted backward differs from per-pixel reference: {out}")
    if max(out["global"].values()) < 0.05:
        raise AssertionError(f"fixture cannot distinguish global-order gradients: {out}")
    return {"rel_err": out}


# ---------------------------------------------------------------------------- C2 / C5 proxy
def _yaw(deg):
    a = math.radians(deg)
    return [[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]]


def _warp_errors(frames, Rs, K, border=4):
    """L1 error between frame i+1 and frame i warped by the exact rotation homography."""
    H, W = frames[0].shape[1:]
    Kd = K.double().cpu()
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3)
    errs = []
    for i in range(len(frames) - 1):
        Hm = Kd @ Rs[i] @ Rs[i + 1].T @ torch.linalg.inv(Kd)
        q = p @ Hm.T
        q = q[:, :2] / q[:, 2:]
        grid = torch.stack([2 * q[:, 0] / (W - 1) - 1, 2 * q[:, 1] / (H - 1) - 1], -1)
        warped = F.grid_sample(frames[i][None].double().cpu(), grid.reshape(1, H, W, 2),
                               align_corners=True)[0]
        valid = (q[:, 0] >= border) & (q[:, 0] <= W - 1 - border) & (q[:, 1] >= border) & (
            q[:, 1] <= H - 1 - border)
        valid = valid.reshape(H, W) & (xs >= border) & (xs <= W - 1 - border) & (
            ys >= border) & (ys <= H - 1 - border)
        e = (warped - frames[i + 1].double().cpu()).abs().sum(0)[valid]
        errs.append(e.mean().item())
    return torch.tensor(errs)


def popping(method, device):
    """Yaw sweep over two overlapping splats whose mean view-z order flips at ~2.5 deg.
    Popping = a frame whose rotation-warped consistency error jumps (paper's FLIP_t idea with
    exact flow and L1 instead of RAFT + FLIP). HIER must stay smooth; GLOBAL must pop."""
    _cuda(device)
    d = math.radians(5.0)
    params = _params([[0.0, 0.0, 3.0], [3.01 * math.sin(d), 0.0, 3.01 * math.cos(d)]],
                     [[0.35] * 3, [0.35] * 3], [[1.0, 0, 0, 0]] * 2, [0.8, 0.8],
                     [[0.95, 0.1, 0.1], [0.1, 0.9, 0.1]], device)
    angles = [-1.0 + 0.25 * i for i in range(33)]
    bg = torch.zeros(3, device=device)
    res = {}
    with torch.no_grad():
        for name, s in (("hier", _settings()),
                        ("global", _settings(sort_mode="GLOBAL", sort_order="Z_DEPTH"))):
            frames, Rs = [], []
            for a in angles:
                R = torch.tensor(_yaw(a), dtype=torch.float64)
                cam = _camera(64, 64, 60.0, device, R)
                frames.append(_render(params, cam, s, bg)[0])
                Rs.append(R)
            e = _warp_errors(frames, Rs, cam["K"])
            res[name] = {"max": e.max().item(), "median": e.median().item(),
                         "argmax_deg": angles[int(e.argmax()) + 1]}
    if res["global"]["max"] < 5 * res["hier"]["max"]:
        raise AssertionError(f"no popping contrast between GLOBAL and HIER: {res}")
    if res["hier"]["max"] > 4 * res["hier"]["median"] + 1e-4:
        raise AssertionError(f"HIER sequence has a consistency jump (popping): {res}")
    return res


# ---------------------------------------------------------------------------- C3
def culling(method, device):
    """Bounding/culling are exact (images unchanged) and reduce the number of sort entries;
    reports forward timings of HIER vs GLOBAL on a random scene."""
    _cuda(device)
    from .raster import VANILLA, forward_raw, make_settings

    g = torch.Generator().manual_seed(7)
    n = 20000
    means = torch.rand(n, 3, generator=g) * torch.tensor([4.0, 4.0, 4.0]) + torch.tensor(
        [-2.0, -2.0, 3.0])
    scales = torch.rand(n, 3, generator=g) ** 3 * 0.25 + 0.005
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.9 + 0.02
    params = _params(means, scales, quats, opac, torch.rand(n, 3, generator=g), device)
    cam = _camera(512, 384, 400.0, device)
    bg = torch.zeros(3, device=device)
    vanilla = make_settings(VANILLA)
    bounded = make_settings(VANILLA, rect_bounding=True, tight_opacity_bounding=True,
                            tile_based_culling=True, load_balancing=True)
    hier = _settings()
    hier_tc = _settings(tile_based_culling=True)
    hier_nocull = _settings(hierarchical_4x4_culling=False, rect_bounding=False,
                            tight_opacity_bounding=False, load_balancing=False)
    r = {k: forward_raw(params, cam, 0, bg, s) for k, s in
         (("vanilla", vanilla), ("bounded", bounded), ("hier", hier), ("hier_tc", hier_tc),
          ("hier_nocull", hier_nocull))}
    entries = {k: v[0] for k, v in r.items()}
    d_glob = (r["vanilla"][1] - r["bounded"][1]).abs().max().item()
    d_hier_tc = (r["hier"][1] - r["hier_tc"][1]).abs().mean().item()
    d_hier_nc = (r["hier"][1] - r["hier_nocull"][1]).abs().mean().item()
    if d_glob > 1e-4:
        raise AssertionError(f"GLOBAL bounding/tile culling changed the image: {d_glob:.2e}")
    if max(d_hier_tc, d_hier_nc) > 2e-3:
        raise AssertionError(f"HIER culling changed the image: {d_hier_tc:.2e} {d_hier_nc:.2e}")
    if not (entries["bounded"] < entries["vanilla"] and entries["hier_tc"] < entries["hier"]):
        raise AssertionError(f"culling did not reduce sort entries: {entries}")

    def timed(s, reps=20):
        forward_raw(params, cam, 0, bg, s)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            forward_raw(params, cam, 0, bg, s)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    return {"entries": entries, "max_diff_global_bounded": d_glob,
            "mean_diff_hier_tilecull": d_hier_tc, "mean_diff_hier_nocull": d_hier_nc,
            "ms_vanilla": timed(vanilla), "ms_hier": timed(hier)}


# ---------------------------------------------------------------------------- density
def schedule(method, device):
    """3DGS cadence of the density used by StopThePop (train.py): densify at s > 500 and
    s % 100 == 0 below 15000, reset at s % 3000, world-size prune only after 3000, no
    screen-size prune; opacity decay option multiplies o by 0.9995 every 50 steps."""
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")

    def make(**cfg):
        return create_density(replace(spec, config={**spec.config, **cfg}))

    def setup(n, big=False):
        params = _params(torch.zeros(n, 3), torch.full((n, 3), 0.001), [[1.0, 0, 0, 0]] * n,
                         [0.5] * n, torch.full((n, 3), 0.5), "cpu")
        if big:
            with torch.no_grad():
                params["scales"][0] = math.log(0.5)  # > 0.1 * extent (=1)
        opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
        for k, p in params.items():
            before = p.detach().clone()
            p.grad = torch.ones_like(p)
            opts[k].step()
            opts[k].zero_grad(set_to_none=True)
            p.data.copy_(before)
        return params, opts

    def out(n, grad):
        m2 = torch.zeros(n, 3, requires_grad=True)
        m2.grad = torch.zeros(n, 3)
        m2.grad[:, 0] = grad
        return {"info": {"means2d": m2, "radii": torch.ones(n, dtype=torch.int32) * 5}}

    def run(d, params, opts, step, grad):
        o = out(len(params["means"]), grad)
        d.before_backward(params, opts, step, o)
        d.after_backward(params, opts, step, o)

    res = {}
    d = make()
    params, opts = setup(4)
    d.initialize(params, opts, 1.0)
    for step, expect in ((500, 4), (550, 4), (600, 8)):
        run(d, params, opts, step, 1e-3)
        if len(params["means"]) != expect:
            raise AssertionError(f"step {step}: {len(params['means'])} Gaussians, want {expect}")
    run(d, params, opts, 15000, 1e-3)
    if len(params["means"]) != 8:
        raise AssertionError("densified at/after densify_until")
    # reset at 3000
    d = make()
    params, opts = setup(2)
    d.initialize(params, opts, 1.0)
    run(d, params, opts, 3000, 0.0)
    o = params["opacities"].sigmoid()
    if not torch.allclose(o, torch.full_like(o, 0.01), atol=1e-5):
        raise AssertionError(f"opacity reset at 3000 missing: {o.tolist()}")
    # world prune only after 3000 (big Gaussian, no gradient)
    for step, keep in ((2000, 3), (3100, 2)):
        d = make()
        params, opts = setup(3, big=True)
        d.initialize(params, opts, 1.0)
        run(d, params, opts, step, 0.0)
        if len(params["means"]) != keep:
            raise AssertionError(f"world-size prune at step {step}: {len(params['means'])}")
    res["prune_ok"] = True
    # opacity decay
    d = make(opacity_decay=0.9995)
    params, opts = setup(2)
    d.initialize(params, opts, 1.0)
    run(d, params, opts, 3000, 0.0)  # decay replaces reset; 3000 % 50 == 0
    o = params["opacities"].sigmoid()
    if not torch.allclose(o, torch.full_like(o, 0.5 * 0.9995), atol=1e-6):
        raise AssertionError(f"opacity decay wrong: {o.tolist()}")
    run(d, params, opts, 3001, 0.0)
    if not torch.allclose(params["opacities"].sigmoid(), o, atol=1e-7):
        raise AssertionError("opacity decay applied off-cadence")
    res["decay_ok"] = True
    return res
