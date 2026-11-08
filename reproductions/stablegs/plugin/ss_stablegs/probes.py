"""Mechanism probes for the StableGS port. Each raises AssertionError when the mechanism is
removed or wrong; references are written independently of renderer.py / losses.py / density.py."""

import math
from dataclasses import replace

import numpy as np
import torch

from . import pairs as P


def _scene(method, n, device, seed=0, depth=3.0, spread=0.6, scale=0.08, opacity=None):
    from splatstudio.state import initialize

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += depth
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_(torch.full((n, 3), math.log(scale), device=device))
        o = torch.full((n,), 0.5) if opacity is None else torch.as_tensor(opacity).float()
        params["opacities"].copy_(torch.logit(o).to(device))
    return params, opts


def _camera(device, w=48, h=40, f=40.0, shift=0.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    viewmat = torch.eye(4, device=device)
    viewmat[0, 3] = shift
    return {"K": K, "viewmat": viewmat, "width": w, "height": h}


def _component(method, name):
    return next(c for c in method.components if c.name == name)


def _density(method, **config):
    from .density import StableGSDensity

    spec = _component(method, "stablegs-density")
    return StableGSDensity(replace(spec, config={**spec.config, **config}))


# ------------------------------------------------------------------------------- C1
def dual_opacity_paths(method, device):
    if not str(device).startswith("cuda"):
        from splatstudio.probes import ProbeUnavailable

        raise ProbeUnavailable("needs CUDA")
    from .renderer import activations, render_path

    bg = torch.zeros(3, device=device)
    cam = _camera(device)
    params, _ = _scene(method, 6, device)
    with torch.no_grad():
        params["opacity_aux"].fill_(30.0)  # α' -> 1
    act = activations(params)
    app = render_path(act, cam, 0, bg, 0.0, True, True)
    geo = render_path(act, cam, 0, bg, 0.0, False, True)
    d1 = float((app["rgb"] - geo["rgb"]).abs().max())
    assert d1 < 1e-4, f"α'=1 appearance render differs from geometric render ({d1})"
    # Lone splat, α = 0.3: alpha map scales by α' (no clamp involved).
    one, _ = _scene(method, 2, device, spread=0.0, opacity=[0.3, 1e-4])
    with torch.no_grad():
        one["opacity_aux"].fill_(0.0)  # α' = 0.5
    a1 = render_path(activations(one), cam, 0, bg, 0.0, True, True)["alpha"].max()
    a0 = render_path(activations(one), cam, 0, bg, 0.0, False, True)["alpha"].max()
    ratio = float(a1 / a0)
    assert abs(ratio - 0.5) < 0.02, f"rendering opacity is not α'·α (ratio {ratio})"
    # Gradient routing: geometric depth loss -> no α' gradient; appearance colour loss -> α'.
    params, _ = _scene(method, 6, device)
    act = activations(params)
    geo = render_path(act, cam, 0, bg, 0.0, False, True)
    geo["depth"].mean().backward()
    g_aux = params["opacity_aux"].grad
    assert g_aux is None or float(g_aux.abs().max()) == 0.0, "geometric path leaks into α'"
    assert params["opacities"].grad is not None and float(params["opacities"].grad.abs().max()) > 0
    params, _ = _scene(method, 6, device)
    app = render_path(activations(params), cam, 0, bg, 0.0, True, False)
    (app["rgb"] - 0.3).abs().mean().backward()
    g_aux = params["opacity_aux"].grad
    assert g_aux is not None and float(g_aux.abs().max()) > 0, "appearance path ignores α'"
    return {"alpha_one_maxdiff": d1, "alpha_ratio_half": ratio,
            "aux_grad_appearance": float(g_aux.abs().max())}


# ------------------------------------------------------------------------------- C2
def _bilinear_zero(img, x, y):
    """Independent bilinear sample (align_corners=False, zero padding); x, y pixel-centre coords
    in [0, W) / [0, H) with pixel k centred at k (i.e. continuous index - 0.5)."""
    H, W = img.shape
    x0, y0 = math.floor(x), math.floor(y)
    val = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (1 - abs(x - xi)) * (1 - abs(y - yi))
            if 0 <= xi < W and 0 <= yi < H:
                val += w * img[yi, xi]
    return val


def _reference_official(d0, d1, f, shift):
    """Per-pixel loop of train_pair.py:157-176 for two identity-rotation cameras, camera 1 at
    world x = −shift (viewmat translation +shift), same intrinsics, centred principal point."""
    H, W = d0.shape
    cx, cy = (W - 1) / 2, (H - 1) / 2

    def term(da, db, sign):
        # points of view a (depth da) seen from view b (translation sign*shift), sample db.
        tot = 0.0
        for r in range(H):
            for c in range(W):
                z = da[r, c]
                X = (c - cx) / f * z + sign * shift
                Y = (r - cy) / f * z
                u, v = X / z * f + cx, Y / z * f + cy
                tot += max(_bilinear_zero(db, u, v) - z, 0.0)
        return tot / (H * W)

    return term(d0, d1, +1) + term(d1, d0, -1)


def depth_consistency_formula(method, device):
    from .losses import consistency_official, consistency_paper

    W, H, f, shift = 24, 18, 20.0, 0.3
    cam0 = _camera("cpu", W, H, f)
    cam1 = _camera("cpu", W, H, f, shift=shift)
    plane = torch.full((H, W), 5.0)
    zero_off = float(consistency_official(plane, plane, cam0, cam1))
    zero_pap = float(consistency_paper(plane, plane, cam0, cam1))
    assert zero_off < 1e-5 and zero_pap < 1e-4, f"consistent plane not zero ({zero_off}, {zero_pap})"
    floater = plane.clone()
    floater[6:11, 8:14] = 2.0  # floater in view 0 only
    off = float(consistency_official(floater, plane, cam0, cam1))
    ref = _reference_official(floater.numpy(), plane.numpy(), f, shift)
    assert off > 1e-3, "official consistency loss blind to a floater"
    assert abs(off - ref) < 1e-4 * max(1.0, ref), f"official form {off} != reference {ref}"
    pap = float(consistency_paper(floater, plane, cam0, cam1))
    pap_rev = float(consistency_paper(plane, floater, cam1, cam0))
    assert pap > 1e-3 and abs(pap - pap_rev) < 1e-5, "paper form not two-sided / symmetric"
    return {"plane_official": zero_off, "plane_paper": zero_pap, "floater_official": off,
            "floater_reference": float(ref), "floater_paper": pap}


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def pair_selection(method, device):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, -2, 4], [2, 2, 12], size=(200, 3))  # non-planar scene
    f = 500.0

    def observe(R, t, ids):
        cam = (R @ pts[ids].T).T + t
        return {"xys": f * cam[:, :2] / cam[:, 2:], "point3D_ids": np.asarray(ids)}

    all_ids = np.arange(200)
    cams = {
        "a": (_rot_y(0), np.zeros(3), all_ids),
        "b": (_rot_y(30), np.array([1.5, 0, 0]), all_ids),  # good: 30 deg, parallax
        "c": (_rot_y(5), np.array([0.2, 0, 0]), all_ids),  # rotation too small
        "d": (_rot_y(30), np.array([1.5, 0, 0]), all_ids[:25]),  # too few covisible
        "e": (_rot_y(30), np.zeros(3), all_ids),  # pure rotation -> homography fits all
    }
    names = list(cams)
    obs = {k: observe(*v) for k, v in cams.items()}
    rots = np.stack([cams[k][0] for k in names])
    pairs = P.select_pairs(names, rots, obs)
    got = {(names[i], names[j]) for i, j in pairs}
    h_good = P.homography_inlier_ratio(*P.covisible(obs["a"], obs["b"]))
    h_rot = P.homography_inlier_ratio(*P.covisible(obs["a"], obs["e"]))
    assert ("a", "b") in got, f"good pair rejected (H ratio {h_good})"
    for bad in (("a", "c"), ("a", "d"), ("a", "e")):
        assert bad not in got, f"pair {bad} should be rejected"
    assert h_rot >= 0.8 > h_good
    return {"pairs": sorted(got), "h_ratio_parallax": h_good, "h_ratio_rotation": h_rot}


def _frames(device, n=4, w=48, h=40):
    cams = [_camera(device, w, h, shift=0.05 * k) for k in range(n)]
    g = torch.Generator(device="cpu").manual_seed(1)
    ims = [torch.rand(h, w, 3, generator=g).to(device) for _ in range(n)]
    return cams, ims


def pair_stack(method, device):
    dens = _density(method)
    cams, ims = _frames("cpu")
    pairs = [(0, 1), (1, 2)]
    dens.set_training_frames(cams, ims, pairs)
    stack = P.refill_stack(pairs, 4, np.random.default_rng(0))
    assert len(stack) == 2 * len(pairs) and stack[:2] == [[0, 1], [1, 2]], "stack != P + |P|"
    p = P.inverse_frequency(pairs, 4)
    expect = np.array([1, 0.5, 1, 1]) / 3.5
    assert np.allclose(p, expect), f"inverse-frequency weights {p}"
    seen, in_p = 0, 0
    for step in range(1, 41):
        b = dens._make_batch(step)
        seen += 1
        in_p += b["in_P"]
        assert b["in_P"] == (tuple(b["indices"]) in set(pairs))
    assert 0 < in_p < seen
    assert not dens._make_batch(15000)["dual"] and dens._make_batch(15001)["geo"]
    return {"stack_len": len(stack), "in_P_fraction": in_p / seen, "weights": p.tolist()}


def loss_schedule(method, device):
    if not str(device).startswith("cuda"):
        from splatstudio.probes import ProbeUnavailable

        raise ProbeUnavailable("needs CUDA")
    from ss_taming_3dgs.core import ssim_same

    from .link import LINK
    from .losses import (
        create_depth_consistency,
        create_depth_normal,
        create_geometric_color,
        create_pair_color,
    )
    from .renderer import create_renderer

    dens = _density(method)
    cams, ims = _frames(device)
    dens.set_training_frames(cams, ims, [(0, 1), (1, 2), (2, 3), (0, 3)])
    renderer = create_renderer(_component(method, "stablegs-raster"))
    params, _ = _scene(method, 30, device, spread=1.5)
    losses = {n: fn(_component(method, n)) for n, fn in (
        ("pair-color", create_pair_color), ("geometric-color", create_geometric_color),
        ("depth-normal", create_depth_normal), ("depth-consistency", create_depth_consistency))}
    out = {}
    try:
        for step in (15000, 15001):
            dens.batch = dens._make_batch(step)
            while not dens.batch["in_P"]:
                dens.batch = dens._make_batch(step)
            o = renderer.render(params, cams[0], 0)
            o["step"] = step
            vals = {k: float(f(params, o, None)) for k, f in losses.items()}
            v = o["pair"]["views"]
            ref = 0.8 * sum((x["app"]["rgb"] - x["target"]).abs().mean() for x in v) + \
                0.2 * (2 - 2 * ssim_same(v[0]["app"]["rgb"], v[0]["target"]))
            assert abs(vals["pair-color"] - float(ref)) < 1e-5, "colour loss != official formula"
            out[step] = vals
    finally:
        dens.images = None
        LINK.attach(None)
    for k in ("geometric-color", "depth-normal", "depth-consistency"):
        assert out[15000][k] == 0.0, f"{k} active before 15000"
    assert out[15001]["geometric-color"] > 0 and out[15001]["depth-normal"] > 0
    return {str(k): v for k, v in out.items()}


def gof_qualifier(method, device):
    from .density import gof_qualifiers

    g = torch.tensor([3e-4, 1e-4, 1e-5, 1e-5, 5e-5, 0.0])
    a = torch.tensor([5e-4, 1e-4, 9e-4, 1e-5, 2e-5, 0.0])  # index 2: cancelling signed grads
    on = gof_qualifiers(g, a, 2e-4, True)
    off = gof_qualifiers(g, a, 2e-4, False)
    assert off.tolist() == [True, False, False, False, False, False]
    assert bool(on[2]) and bool(on[0]) and int(on.sum()) <= 3, f"GOF qualifier wrong {on}"
    return {"with_gof": on.tolist(), "without": off.tolist()}
