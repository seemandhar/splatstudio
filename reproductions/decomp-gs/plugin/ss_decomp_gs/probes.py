"""Mechanism probes for the DecompGS port. Each raises AssertionError when its mechanism is
removed or wrong. References are written independently of schedule.py / density.py
(numpy FFT, PIL resampling, numpy sort)."""

import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import schedule as S
from .density import DecompDensity
from .link import LINK
from .losses import create_photometric
from .renderer import Renderer


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _density(method, **config):
    spec = _spec(method, "density")
    return DecompDensity(replace(spec, config={**spec.config, **config}))


def _smooth_images(n=3, h=192, w=256, seed=0, device="cpu"):
    """Low-pass noise images (Gaussian spectral envelope): energy falls monotonically with r."""
    g = torch.Generator().manual_seed(seed)
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[None, :]
    envelope = torch.exp(-(fx**2 + fy**2) / (2 * 0.1**2))
    out = []
    for _ in range(n):
        im = torch.randn(3, h, w, generator=g)
        out.append((0.5 + torch.fft.ifft2(torch.fft.fft2(im) * envelope).real).to(device))
    return out


def _pil_down(image_chw, r):
    from PIL import Image

    c, h, w = image_chw.shape
    size = (w // r, h // r)
    return np.stack([np.asarray(Image.fromarray(image_chw[k].numpy().astype(np.float32), mode="F")
                                .resize(size, Image.Resampling.BILINEAR)) for k in range(c)])


def c2f_schedule(method, device):
    """C2: Eq. 2-3 against a numpy-FFT / PIL-resize reference; schedule boundaries."""
    images = _smooth_images()
    K, T = 8, 25000
    ref_e = {}
    for r in range(1, K + 1):
        tot = 0.0
        for im in images:
            arr = im.numpy() if r == 1 else _pil_down(im, r)
            tot += float(np.abs(np.fft.fft2(arr.astype(np.float64))).sum()) * r * r
        ref_e[r] = tot
    got_e = S.scale_energies(images, K)
    for r in ref_e:
        assert abs(got_e[r] / got_e[1] - ref_e[r] / ref_e[1]) < 0.01, (r, got_e[r], ref_e[r])
    ref_t = {r: int(round(ref_e[r] / ref_e[1] * T)) for r in ref_e}
    ends = S.end_iterations(got_e, T)
    assert ends[1] == T
    assert all(abs(ends[r] - ref_t[r]) <= 0.01 * T for r in ends), (ends, ref_t)
    assert all(ends[r] > ends[r + 1] for r in range(1, K)), ends
    # Scale r trains on [T_{r+1}, T_r): coarsest first, r = 1 from T_2.
    assert S.scale_at(ends, 0) == K and S.scale_at(ends, ends[K] - 1) == K
    for r in range(2, K):
        assert S.scale_at(ends, ends[r + 1]) == r and S.scale_at(ends, ends[r] - 1) == r
    assert S.scale_at(ends, ends[2]) == 1 and S.scale_at(ends, 29999) == 1
    # Flat (constant) image: all energy at DC, E_r = E_1 -> no low-resolution phase at all.
    flat = [torch.full((1, 840, 840), 0.3)]  # 840 = lcm(1..8): exact r^2 normalisation
    flat_ends = S.end_iterations(S.scale_energies(flat, K), T)
    assert all(abs(v - T) <= 2 for v in flat_ends.values()), flat_ends
    # The density follows the schedule; with c2f off it renders at r = 1.
    d = _density(method)
    d.set_training_images(images)
    off = _density(method, c2f=False)
    off.set_training_images(images)
    assert d.current_scale == K and off.current_scale == 1
    assert d.boundary == off.boundary == ends[2]
    return {"ratios": {r: round(got_e[r] / got_e[1], 4) for r in got_e}, "ends": ends,
            "g2l_boundary": d.boundary}


def _scene(device, n=64, seed=0):
    from splatstudio.state import initialize

    from . import method as make

    g = torch.Generator().manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([1.2, 0.9, 0.4])
    pts[:, 2] += 3.0
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(make(), pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.04))
        params["opacities"].fill_(2.0)
    return params, opts


def render_scale(method, device):
    """C2: training renders at floor(H/r) with scaled K; evaluation (no_grad) at full size."""
    from gsplat import rasterization

    params, opts = _scene(device)
    W, H, f = 75, 61, 60.0
    cam = {"K": torch.tensor([[f, 0, 37.1], [0, f, 30.4], [0, 0, 1.0]], device=device),
           "viewmat": torch.eye(4, device=device), "width": W, "height": H}
    renderer = Renderer(_spec(method, "renderer"))
    d = _density(method)
    d.initialize(params, opts)
    d.set_training_images(_smooth_images(device=device))
    d.current_scale = 3
    out = renderer.render(params, cam, sh_degree=0)
    h, w = out["rgb"].shape[:2]
    assert (h, w) == (20, 25), (h, w)
    K = cam["K"].clone()
    K[0] *= 25 / W
    K[1] *= 20 / H
    ref, alpha, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                  params["opacities"].sigmoid(),
                                  torch.cat((params["sh0"], params["shN"]), 1),
                                  torch.eye(4, device=device)[None], K[None], 25, 20, sh_degree=0)
    err = float((out["rgb"] - (ref[0] + 1 - alpha[0])).abs().max())
    assert err < 1e-5, err
    with torch.no_grad():
        full = renderer.render(params, cam, sh_degree=0)["rgb"]
    assert full.shape[:2] == (H, W), "evaluation must render at full resolution"
    d.schedule_active = False
    assert renderer.render(params, cam, sh_degree=0)["rgb"].shape[:2] == (H, W)
    LINK.attach(None)
    return {"low_res": [h, w], "max_abs_vs_scaled_K": err}


def target_downsample(method, device):
    """C2: loss target = antialiased bilinear D_r (PIL reference); full-size path unchanged."""
    g = torch.Generator().manual_seed(1)
    target = torch.rand(61, 83, 3, generator=g)
    h, w = 61 // 3, 83 // 3
    ref = torch.from_numpy(_pil_down(target.permute(2, 0, 1), 3)).permute(1, 2, 0).float()
    loss = create_photometric(_spec(method, "loss"))
    rgb = ref.clone().requires_grad_()
    assert rgb.shape[:2] == (h, w)
    same = float(loss(None, {"rgb": rgb}, target))
    assert same < 1e-3, f"loss against the PIL-downsampled target must vanish, got {same}"
    alias = target[1::3, 1::3][:h, :w]
    assert float(loss(None, {"rgb": alias}, target)) > 0.02, "aliased subsampling must differ"
    full = float(loss(None, {"rgb": target.clone()}, target))
    assert full < 1e-6
    return {"loss_vs_pil_target": same, "full_res_loss": full}


def _fixture(method, device, **config):
    """Gaussians 0-3 transparent, 4-23 large (split), 24-63 small (clone); 4-39 qualified."""
    params, opts = _scene(device)
    n = len(params["means"])
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad()
    d = _density(method, g2l_boundary=1000, c2f=False, **config)
    d.initialize(params, opts, scene_scale=1.0)
    with torch.no_grad():
        params["opacities"][:4] = -8.0
        params["scales"][4:24] = math.log(0.05)
        params["scales"][24:] = math.log(0.005)
    grads = torch.full((n,), 1e-4)
    grads[4:40] = 1e-3
    d.stats["grad2d"] = grads.to(device) * 5
    d.stats["count"] = torch.full((n,), 5.0, device=device)
    return d, params, opts


def g2l_phases(method, device):
    """C1: split only before T_2 (clone candidates untouched), split + clone after."""
    out = {}
    for name, step, cfg in (("global", 900, {}), ("local", 1000, {}),
                            ("no_g2l", 900, {"g2l": False}),
                            ("split_all", 900, {"global_mode": "split_all"})):
        d, params, opts = _fixture(method, device, adaptive_prune=False, **cfg)
        n0 = len(params["means"])
        d.densify_and_prune(params, opts, step)
        info = d.last_densify
        out[name] = {k: info[k] for k in ("clone", "split", "pruned", "n")}
        expected_clone = 0 if name in ("global", "split_all") else 16
        expected_split = {"split_all": 36}.get(name, 20)
        assert info["clone"] == expected_clone, (name, info)
        assert info["split"] == expected_split, (name, info)
        assert info["pruned"] == 4, (name, info)
        assert info["n"] == n0 + expected_clone + expected_split - 4, (name, info)
        assert all(p.requires_grad and p.is_leaf for p in params.values()), "field frozen"
        for k, p in params.items():
            assert opts[k].param_groups[0]["params"][0] is p, f"optimizer stale for {k}"
    return out


def adaptive_prune(method, device):
    """C3: tau = min(sorted(alpha)[floor(N p)], tau_u) (numpy reference), prune alpha < tau."""
    g = torch.Generator().manual_seed(5)
    alpha = torch.rand(1000, generator=g) * 0.5
    ref = min(float(np.sort(alpha.numpy())[int(math.floor(1000 * 0.03))]), 0.05)
    got = S.adaptive_threshold(alpha, 0.03, 0.05)
    assert abs(got - ref) < 1e-7, (got, ref)
    assert ref < 0.05 and ref > 0.005
    high = torch.rand(1000, generator=g) * 0.5 + 0.4
    assert S.adaptive_threshold(high, 0.03, 0.05) == 0.05, "upper limit tau_u must cap tau"
    # In the density: with adaptive pruning, ~3 % of low-opacity Gaussians go; fixed 0.005 keeps them.
    res = {}
    # p = 0.2 on this 64-Gaussian fixture so that tau lands inside the 0.01-0.04 ramp.
    for name, cfg in (("adaptive", {"prune_ratio": 0.2}), ("fixed", {"adaptive_prune": False})):
        d, params, opts = _fixture(method, device, **cfg)
        with torch.no_grad():
            params["opacities"][4:] = torch.logit(torch.linspace(0.01, 0.04, 60, device=device))
        d.stats["grad2d"].zero_()  # no growth, prune only
        d.densify_and_prune(params, opts, 1000)
        res[name] = d.last_densify
    alphas = np.concatenate([np.full(4, 1 / (1 + math.exp(8.0))), np.linspace(0.01, 0.04, 60)])
    tau_ref = min(float(np.sort(alphas)[int(math.floor(64 * 0.2))]), 0.05)
    assert abs(res["adaptive"]["tau"] - tau_ref) < 1e-6
    assert res["adaptive"]["pruned"] == int((alphas < tau_ref).sum()) > 4, res
    assert res["fixed"]["pruned"] == 4 and res["fixed"]["tau"] == 0.005, res
    return {"tau": got, "tau_ref": ref, "pruned": {k: v["pruned"] for k, v in res.items()}}


def lr_schedule(method, device):
    """C2: position LR constant while r > 1, then log-linear to 1 % at 30000."""
    params, opts = _scene(device)
    lr0 = 0.00016 * 4.2
    opts["means"].param_groups[0]["lr"] = lr0
    d = _density(method)
    d.initialize(params, opts, scene_scale=4.2)
    d.set_training_images(_smooth_images())
    t2 = d.boundary
    assert 0 < t2 < 25000
    vals = {}
    for step in (1, t2 - 2, t2 + 999, 29999):
        d.update_lr(opts, step)
        vals[step + 1] = opts["means"].param_groups[0]["lr"]
    assert abs(vals[2] - lr0) < 1e-15 and abs(vals[t2 - 1] - lr0) < 1e-15
    t = 1000 / (30000 - t2)
    ref = math.exp(math.log(lr0) * (1 - t) + math.log(lr0 * 0.01) * t)
    assert abs(vals[t2 + 1000] - ref) < 1e-12
    assert abs(vals[30000] - lr0 * 0.01) < 1e-12
    assert lr0 * 0.01 ** (t2 / 30000) < 0.99 * lr0, "must differ from the 3DGS decay"
    return {"t2": t2, "lr": {str(k): v for k, v in vals.items()}}


def cycle(method, device):
    """Integration: runtime trains coarse->fine, split-only then split+clone, adaptive pruning."""
    from splatstudio.demo import create_dataset
    from splatstudio.runtime import train

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        create_dataset(data, size=128)
        spec = _spec(method, "density")
        compressed = {**spec.config, "data": str(data), "densify_until": 250, "total_steps": 300,
                      "refine_start": 20, "refine_every": 10, "reset_every": 100,
                      "max_scale_factor": 4}
        m = replace(method, components=tuple(
            replace(c, config=compressed) if c.kind == "density" else c for c in method.components))
        result = train(m, data, Path(tmp) / "run", steps=300, num_gaussians=2000,
                       max_gaussians=200000, device=device, log_every=100)
        dens = LINK.density
        hist, t2 = dens.history, dens.boundary
        LINK.attach(None)
    scales = [h["scale"] for h in hist]
    assert scales[0] > 1 and scales[-1] == 1 and all(a >= b for a, b in zip(scales, scales[1:]))
    assert all(h["clone"] == 0 for h in hist if h["step"] < t2), "clone before T_2"
    assert all(h["scale"] == 1 for h in hist if h["step"] >= t2)
    assert sum(h["pruned"] for h in hist) > 0
    assert math.isfinite(result["metrics"]["psnr"])
    return {"scales": sorted(set(scales), reverse=True), "t2": t2,
            "clone_after_t2": sum(h["clone"] for h in hist if h["step"] >= t2),
            "split_before_t2": sum(h["split"] for h in hist if h["step"] < t2),
            "psnr": result["metrics"]["psnr"], "gaussians": result["resources"]["gaussians"]}


def renderer_gradient(method, device):
    """splatstudio.probes.renderer_gradient with the C2F schedule active (r = 2)."""
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene

    renderer, density, losses = build(method)
    params, opts = sample_scene(method, device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=device),
              "width": 32, "height": 32}
    density.initialize(params, opts)
    density.set_training_images(_smooth_images(device=device))
    density.current_scale = 2
    density.ends = {1: 25000, 2: 10}  # r = 2 at step 1
    outputs = renderer.render(params, camera, sh_degree=0)
    assert outputs["rgb"].shape[:2] == (16, 16)
    outputs["step"] = 1
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(32, device=device)[None, :, None] % 2).expand(32, 32, 3).float()
    dict(losses)["photometric"](params, outputs, target).backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    assert float(density.stats["count"].sum()) > 0, "statistics were not accumulated"
    LINK.attach(None)
    return {"screen_gradient_norm": xy.grad.norm().item(),
            "accumulated": float(density.stats["count"].sum())}
