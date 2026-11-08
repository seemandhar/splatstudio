"""Mechanism probes for the faithful 3DGS-MCMC config. Each raises if its mechanism is removed."""

import sys
from unittest import mock

import numpy as np
import torch
from splatstudio import ComponentSpec

from .density import OfficialMCMCDensity
from .losses import create_regularizers
from .scenes import CAPS


def _cuda(device):
    if not str(device).startswith("cuda"):
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("needs CUDA (gsplat kernels)")


def _density(config=None):
    return OfficialMCMCDensity(ComponentSpec("d", "density", "x", (), (), config or {}))


def _scene(n, device, seed=0):
    from splatstudio.methods import make_method
    from splatstudio.state import initialize

    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) - 0.5
    pts[:, 2] += 3
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    return initialize(make_method("x"), pts.to(device), cols.to(device))


def cap_lookup(method, device="cpu"):
    out = {}
    for path, want in (("/x/native/bicycle", 5_900_000), ("/x/pilr/room", 1_500_000),
                       ("/x/native/truck", 2_600_000)):
        with mock.patch.object(sys, "argv", ["ss-train", "--data", path]):
            cap, src = _density().resolve_cap()
        if cap != want or not src.startswith("official-config"):
            raise AssertionError(f"{path}: cap {cap} ({src}) != {want}")
        out[path] = cap
    with mock.patch.object(sys, "argv", ["ss-train", "--data", "/x/flowers"]):
        try:
            _density().resolve_cap()
        except RuntimeError:
            pass
        else:
            raise AssertionError("unknown scene must not silently get a default cap")
    if len(CAPS) != 11:
        raise AssertionError("official per-scene cap table changed")
    return out


def official_init(method, device="cpu"):
    params, opts = _scene(400, device)
    d = _density({"cap_max": 1000})
    d.initialize(params, opts)
    o = params["opacities"].detach().sigmoid()
    if (o - 0.5).abs().max() > 1e-6:
        raise AssertionError("initial opacity is not 0.5")
    p = params["means"].detach().cpu().double().numpy()
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    ref = np.log(0.1 * np.sqrt(np.sort(d2, 1)[:, :3].mean(1)))
    err = np.abs(params["scales"].detach().cpu().double().numpy() - ref[:, None]).max()
    if err > 1e-3:  # float32 cdist; a missing x0.1 factor would be 2.3
        raise AssertionError(f"initial scales differ from 0.1*sqrt(mean 3-NN d^2): {err}")
    if d.strategy.cap_max != 1000:
        raise AssertionError("cap not applied to the strategy")
    return {"scale_err": float(err)}


def regularizers(method, device="cpu"):
    fn = create_regularizers(ComponentSpec("r", "loss", "x", (), (), {"opacity_weight": 0.01}))
    params, _ = _scene(50, "cpu")
    with torch.no_grad():
        params["opacities"].normal_()
        params["scales"].uniform_(-5, -2)
    fn(params, {}, None).backward()
    o, s = params["opacities"].detach(), params["scales"].detach()
    go = 0.01 * torch.sigmoid(o) * (1 - torch.sigmoid(o)) / 50
    gs = 0.01 * s.exp() / 150
    eo = (params["opacities"].grad - go).abs().max().item()
    es = (params["scales"].grad - gs).abs().max().item()
    if eo > 1e-9 or es > 1e-9:
        raise AssertionError(f"regulariser gradients differ {eo} {es}")
    with mock.patch.object(sys, "argv", ["ss-train", "--data", "/x/playroom"]):
        from .losses import resolve_weights

        if resolve_weights({}) != (0.001, 0.01):
            raise AssertionError("Deep Blending opacity_reg must be 0.001")
    return {"opacity_grad_err": eo, "scale_grad_err": es}


def black_background(method, device="cuda"):
    _cuda(device)
    from .renderer import Renderer

    params, _ = _scene(20, device)
    with torch.no_grad():
        params["means"][:, 2] = -5.0  # all behind the camera
    cam = {"K": torch.tensor([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]], device=device),
           "viewmat": torch.eye(4, device=device), "width": 64, "height": 48}
    rgb = Renderer(ComponentSpec("r", "renderer", "x", (), (), {})).render(params, cam, 0)["rgb"]
    if rgb.abs().max() > 1e-6:
        raise AssertionError("empty render is not black")
    return {"max": float(rgb.abs().max())}


def mcmc_schedule(method, device="cuda"):
    """Births (+5 %, capped) and relocation happen only on the official grid; dead Gaussians are
    relocated; population is otherwise unchanged; noise moves only near-transparent Gaussians."""
    _cuda(device)
    params, opts = _scene(1000, device)
    d = _density({"cap_max": 1080})
    d.initialize(params, opts)
    lr = opts["means"].param_groups[0]["lr"]
    counts = {}
    for step in (500, 550, 600, 700, 800, 25000):
        with torch.no_grad():
            params["opacities"][:10] = -10.0  # dead
        d.after_backward(params, opts, step, {"info": {}})
        counts[step] = len(params["means"])
        dead = int((params["opacities"].sigmoid() <= 0.005).sum())
        on_grid = 500 < step < 25000 and step % 100 == 0
        if on_grid and dead:
            raise AssertionError(f"step {step}: dead Gaussians not relocated")
        if not on_grid and dead < 10:
            raise AssertionError(f"step {step}: relocation off the grid")
    want = {500: 1000, 550: 1000, 600: 1050, 700: 1080, 800: 1080, 25000: 1080}
    if counts != want:
        raise AssertionError(f"population schedule {counts} != {want}")
    # noise: opaque Gaussians (o~0.5 -> sigmoid(100(0.5-0.995)) ~ 0) stay put, dead ones move
    with torch.no_grad():
        params["opacities"][:10] = -10.0
        params["opacities"][10:] = 0.0
    before = params["means"].detach().clone()
    d.after_backward(params, opts, 501, {"info": {}})
    moved = (params["means"].detach() - before).norm(dim=1)
    if not (moved[:10] > 0).all() or moved[10:].max() > 1e-12:
        raise AssertionError("SGLD noise not gated by opacity")
    return {"counts": counts, "lr": lr}
