"""Mechanism probes for the DashGaussian port. Each raises AssertionError when its mechanism is
removed or wrong. References below are written independently of schedule.py/density.py."""

import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import schedule as S
from .density import DashDensity
from .link import LINK
from .losses import create_photometric, downsample_target
from .renderer import Renderer, scaled_size


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _density(method, **config):
    spec = _spec(method, "density")
    return DashDensity(replace(spec, config={**spec.config, **config}))


def _noise_images(n=4, h=96, w=128, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3, h, w, generator=g).to(device) for _ in range(n)]


def _smooth_images(n=4, h=96, w=128, seed=0, device="cpu"):
    """Low-pass noise (Gaussian spectrum envelope): most energy at low frequencies, so r_m > 2."""
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[None, :]
    envelope = torch.exp(-(fx**2 + fy**2) / (2 * 0.05**2))
    out = []
    for im in _noise_images(n, h, w, seed):
        out.append((0.5 + torch.fft.ifft2(torch.fft.fft2(im) * envelope).real).to(device))
    return out


def _ref_energy(spectrum, r):
    # Independent window: explicit index ranges of the official centred crop.
    h, w = spectrum.shape[-2:]
    hh, hw = int(h / r) // 2, int(w / r) // 2
    rows = torch.arange(h)
    cols = torch.arange(w)
    mrow = (rows >= (h + 1) // 2 - hh) & (rows < (h + 1) // 2 + hh)
    mcol = (cols >= (w + 1) // 2 - hw) & (cols < (w + 1) // 2 + hw)
    return float((spectrum * (mrow[:, None] & mcol[None, :])).sum())


def _ref_solve(spectrum, target, grid):
    # Grid search (not bisection): largest r whose window still holds >= target.
    best = grid[0]
    for r in grid:
        if _ref_energy(spectrum, r) >= target:
            best = r
    return best


def reso_schedule(method, device):
    """C1: analytic white-noise case + independent grid-search reference + boundaries."""
    images = _noise_images()
    sched = S.build_schedule(images, increase_until=27000)
    # Flat spectrum: window holds (1/r)^2 of the energy -> X = X/4 at r = 2 (a = 4).
    assert abs(sched["max_scale"] - 2.0) < 0.1, sched["max_scale"]
    mags = [torch.fft.fftshift(torch.fft.fft2(im), dim=(-2, -1)).abs() for im in images]
    mean = sum(mags) / len(mags)
    e_tot = float(mean.sum())
    grid = np.arange(1.0, 8.0, 0.002)
    r_ref = min(_ref_solve(m, float(m.sum()) / 4, grid) for m in mags)
    assert abs(r_ref - sched["max_scale"]) < 0.01, (r_ref, sched["max_scale"])
    e_min = _ref_energy(mean, sched["max_scale"])
    checked = []
    for i in (1, 8, 15, 22, 30):
        e_i = e_min + (e_tot - e_min) * i / 31
        r_i = _ref_solve(mean, e_i, grid)
        assert abs(r_i - sched["scales"][i]) < 0.01, (i, r_i, sched["scales"][i])
        # Analytic flat spectrum: r_i ~ (E_i / E_total)^(-1/2).
        assert abs(sched["scales"][i] - (e_i / e_tot) ** -0.5) < 0.06 * sched["scales"][i]
        # supp. Eq. 7 begin step of level i+1 (log modulation).
        begin = int(27000 * math.log(e_i / e_min) / math.log(e_tot / e_min))
        assert sched["begins"][i + 1] == begin, (i, begin, sched["begins"][i + 1])
        checked.append(round(sched["scales"][i], 3))
    assert sched["begins"][0] == 0 and sched["begins"][1] == 0 and sched["begins"][-1] == 27000
    scales = [S.res_scale(sched, k, 27000) for k in range(1, 27001, 50)]
    assert all(isinstance(s, int) for s in scales)
    assert all(a >= b for a, b in zip(scales, scales[1:])), "resolution must not decrease"
    assert scales[0] == int(sched["max_scale"]) and scales[-1] == 1
    # Independent inverse-square interpolation at a mid-level step.
    b, sc = sched["begins"], sched["scales"]
    k = (b[3] + b[4]) // 2
    ref = int((((k - b[3]) / (b[4] - b[3])) * (sc[3] ** -2 - sc[2] ** -2) + sc[2] ** -2) ** -0.5)
    assert S.res_scale(sched, k, 27000) == ref
    # The density follows the schedule; 'const' (mechanism removed) would render at r = 1.
    d = _density(method)
    d.set_training_images(_smooth_images())
    assert d.current_scale == S.res_scale(d.schedule, 1, 27000) > 1, d.current_scale
    const = _density(method, resolution_mode="const")
    assert const.scale_at(1) == 1 and d.scale_at(1) != const.scale_at(1)
    return {"max_scale": sched["max_scale"], "level_scales": checked,
            "begins": sched["begins"][:6], "decay_from": d.decay_from,
            "smooth_scale_at_1": d.current_scale, "smooth_max_scale": d.schedule["max_scale"]}


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
    """C1: training renders at floor(H/r) with scaled K; evaluation (no_grad) at full size."""
    params, _ = _scene(device)
    W, H, f = 75, 61, 60.0
    cam = {"K": torch.tensor([[f, 0, 37.1], [0, f, 30.4], [0, 0, 1.0]], device=device),
           "viewmat": torch.eye(4, device=device), "width": W, "height": H}
    renderer = Renderer(_spec(method, "renderer"))
    d = _density(method)
    d.initialize(params, {k: torch.optim.Adam([p]) for k, p in params.items()})
    d.set_training_images(_smooth_images(device=device))
    d.current_scale = 3
    out = renderer.render(params, cam, sh_degree=0)
    h, w = out["rgb"].shape[:2]
    assert (h, w) == (math.floor(H / 3), math.floor(W / 3)) == (20, 25), (h, w)
    # Independent reference: gsplat at hand-scaled intrinsics.
    from gsplat import rasterization

    K = cam["K"].clone()
    K[0] *= 25 / W
    K[1] *= 20 / H
    ref, alpha, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                  params["opacities"].sigmoid(),
                                  torch.cat((params["sh0"], params["shN"]), 1),
                                  torch.eye(4, device=device)[None], K[None], 25, 20, sh_degree=0)
    ref_rgb = ref[0] + (1 - alpha[0])
    err = float((out["rgb"] - ref_rgb).abs().max())
    assert err < 1e-5, err
    # The low-res render is a blurred version of the full one, not a crop.
    with torch.no_grad():
        full = renderer.render(params, cam, sh_degree=0)["rgb"]
    assert full.shape[:2] == (H, W), "evaluation must render at full resolution"
    pooled = torch.nn.functional.interpolate(full.permute(2, 0, 1)[None], size=(20, 25),
                                             mode="area")[0].permute(1, 2, 0)
    blur_err = float((pooled - out["rgb"].detach()).abs().mean())
    assert blur_err < 0.05, blur_err
    d.schedule_active = False
    assert renderer.render(params, cam, sh_degree=0)["rgb"].shape[:2] == (H, W)
    LINK.attach(None)
    return {"low_res": [h, w], "max_abs_vs_scaled_K": err, "mean_abs_vs_area_pooled": blur_err}


def target_downsample(method, device):
    """C1: loss target = antialiased bilinear reduce (PIL reference); loss uses it."""
    from PIL import Image

    g = torch.Generator().manual_seed(1)
    target = torch.rand(61, 83, 3, generator=g)
    h, w = scaled_size(61, 83, 3)
    got = downsample_target(target, h, w)
    ref = np.stack([np.asarray(Image.fromarray(target[..., c].numpy(), mode="F").resize(
        (w, h), Image.Resampling.BILINEAR)) for c in range(3)], axis=-1)
    err = float(np.abs(got.numpy() - ref).max())
    assert err < 1e-4, err
    # A plain (aliased) subsample would differ markedly.
    alias = float((target[1::3, 1::3][:h, :w] - got).abs().mean())
    assert alias > 0.05, alias
    loss = create_photometric(_spec(method, "loss"))
    rgb = torch.rand(h, w, 3, generator=g).requires_grad_()
    value = loss(None, {"rgb": rgb}, target)
    from .losses import SSIM

    ssim = SSIM[_spec(method, "loss").config.get("ssim_padding", "same")]
    gt = torch.from_numpy(ref).float()
    manual = 0.8 * (rgb - gt).abs().mean() + 0.2 * (1 - ssim(rgb, gt))
    assert abs(float(value) - float(manual)) < 1e-5
    value.backward()
    assert torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
    same = loss(None, {"rgb": target.clone().requires_grad_()}, target)
    assert float(same) < 1e-6, "full-resolution path must compare against the unchanged target"
    return {"low_res": [h, w], "max_abs_vs_pil": err, "aliased_gap": alias}


def primitive_schedule(method, device):
    """C2/C3: Eq. 4 (code form), 0.2 cap, P_fin near the end; Eq. 5 momentum."""
    init_n, max_n = 1000, 6000
    rows = []
    for it, cur, r in ((600, 1000, 8), (3000, 1100, 4), (9000, 1500, 2), (20000, 3000, 1),
                       (26800, 5000, 1), (26900, 5000, 3)):
        got = S.densify_rate(it, cur, r, init_n, max_n, 100, 27000, 27000, 0.2)
        target = max_n if it + 100 >= 27000 else init_n + int((max_n - init_n) / r ** (2 - it / 27000))
        ref = min(max((target - cur) / cur, 0.0), 0.2)
        assert abs(got - ref) < 1e-12, (it, got, ref)
        rows.append(round(got, 4))
    assert rows[-1] == 0.2 and rows[0] < 0.2, rows  # 26900: target = P_fin regardless of r
    # Momentum: M <- max(M, int(0.98 M + min(cap, P_add))).
    m = 5000
    seq = []
    for add in (10, 300, 2_000_000, 0):
        m_new = S.update_momentum(m, add, 0.98, 1_000_000)
        ref = max(m, int(0.98 * m + min(1_000_000, add)))
        assert m_new == ref
        m = m_new
        seq.append(m)
    assert seq[0] == 5000 and seq[1] > 5000 and seq[2] < 2_000_000 + 5000
    d = _density(method)
    params, opts = _scene(device)
    d.initialize(params, opts)
    assert d.max_n == 6 * len(params["means"]), "P_fin^(0) = P_init + 5 P_init (official code)"
    return {"rates": rows, "momentum": seq, "p_fin0": d.max_n}


def topk_densify(method, device):
    """C2: prune first, exactly the top-n qualified Gaussians cloned/split, no size pruning."""
    params, opts = _scene(device)
    n = len(params["means"])
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad()
    d = _density(method)
    d.initialize(params, opts, scene_scale=1.0)
    with torch.no_grad():
        params["opacities"][:4] = -8.0  # sigmoid < 0.005 -> pruned
        params["scales"][4:24] = math.log(0.05)  # > 0.01 * extent -> split
        params["scales"][24:] = math.log(0.005)  # <= 0.01 * extent -> clone
        params["scales"][5] = math.log(0.5)  # > 0.1 * extent: 3DGS would prune, Dash keeps
    g = torch.Generator().manual_seed(3)
    grads = torch.rand(n, generator=g) * 1e-3 + 5e-4  # all qualified (>= 2e-4)
    grads[40:] = 1e-4  # not qualified
    grads[5] = 1e-4  # the large Gaussian is not densified either
    d.stats["grad2d"] = grads.to(device) * 7
    d.stats["count"] = torch.full((n,), 7.0, device=device)
    keep = torch.arange(4, n)
    rate = 10 / n  # n_densify = int(n * (1 + rate) - (n - 4)) = 14
    opa_before = params["opacities"].detach().clone()
    added = d.prune_and_densify(params, opts, rate)
    info = d.last_densify
    assert info["pruned"] == 4 and info["n_densify"] == 14, info
    assert info["clone"] + info["split"] == 14, info
    assert added == 35, added  # qualified after pruning (indices 4..39 except 5)
    order = grads[keep].argsort(descending=True)[:14] + 4
    exp_split = int((order < 24).sum())
    assert info["split"] == exp_split, (info, exp_split)
    assert len(params["means"]) == n - 4 + 14
    big = torch.exp(params["scales"]).max(-1).values
    big_left = bool((big > 0.45).any())
    assert big_left, "large Gaussian must not be pruned by size (no size pruning in DashGaussian)"
    new_m = opts["means"].state[params["means"]]["exp_avg"][-14:]
    assert float(new_m.abs().sum()) == 0.0, "new Gaussians start with zero Adam moments"
    assert float(d.stats["count"].sum()) == 0.0 and len(d.stats["count"]) == len(params["means"])
    assert bool((torch.sigmoid(opa_before[:4]) < 0.005).all())
    # densify_mode='free' (scheduler removed) densifies every qualified candidate.
    params2, opts2 = _scene(device)
    free = _density(method, densify_mode="free")
    free.initialize(params2, opts2, scene_scale=1.0)
    assert free.densify_rate(1000, 100) == 1.0 and d.densify_rate(1000, 100) <= 0.2
    return {"densify": info, "p_add": added, "big_gaussian_kept": big_left}


def lr_schedule(method, device):
    """C4: position LR held until the first level with r < 2, then log-linear to 1 %."""
    params, opts = _scene(device)
    lr0 = 0.00016 * 4.2
    opts["means"].param_groups[0]["lr"] = lr0
    d = _density(method)
    d.initialize(params, opts, scene_scale=4.2)
    d.set_training_images(_smooth_images())
    dfrom = d.decay_from
    ref_first = next(b for b, s in zip(d.schedule["begins"], d.schedule["scales"]) if s < 2)
    assert dfrom == ref_first and 0 < dfrom < 27000
    values = {}
    for step in (1, dfrom - 2, dfrom + 1000, 29999):
        d.update_lr(opts, step)
        values[step + 1] = opts["means"].param_groups[0]["lr"]
    assert abs(values[2] - lr0) < 1e-15 and abs(values[dfrom - 1] - lr0) < 1e-15
    t = 1001 / (30000 - dfrom)
    ref = math.exp(math.log(lr0) * (1 - t) + math.log(lr0 * 0.01) * t)
    assert abs(values[dfrom + 1001] - ref) < 1e-12 * lr0 * 1e3
    assert abs(values[30000] - lr0 * 0.01) < 1e-12
    runtime_value = lr0 * 0.01 ** (dfrom / 30000)
    assert runtime_value < 0.99 * values[dfrom - 1], "must differ from the 3DGS decay"
    return {"decay_from": dfrom, "lr": {str(k): v for k, v in values.items()}}


def cycle(method, device):
    """Integration: native runtime trains low->full resolution, evaluates at full resolution."""
    from splatstudio.demo import create_dataset
    from splatstudio.runtime import train

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        create_dataset(data, size=128)
        spec = _spec(method, "density")
        compressed = {**spec.config, "data": str(data), "densify_until": 270, "total_steps": 300,
                      "refine_start": 50, "refine_every": 10, "reset_every": 100}
        m = replace(method, components=tuple(
            replace(c, config=compressed) if c.kind == "density" else c for c in method.components))
        result = train(m, data, Path(tmp) / "run", steps=300, num_gaussians=2000,
                       max_gaussians=200000, device=device, log_every=100)
        hist = LINK.density.history
        LINK.attach(None)
    scales = [s for _, s, _, _ in hist]
    assert scales[0] > 1 and scales[-1] == 1, scales
    assert all(a >= b for a, b in zip(scales, scales[1:]))
    assert math.isfinite(result["metrics"]["psnr"])
    return {"scales": sorted(set(scales), reverse=True), "psnr": result["metrics"]["psnr"],
            "gaussians": result["resources"]["gaussians"]}


def renderer_gradient(method, device):
    """splatstudio.probes.renderer_gradient with the DashGaussian schedule active (r = 2)."""
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene

    renderer, density, _ = build(method)
    params, opts = sample_scene(method, device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=device),
              "width": 32, "height": 32}
    density.initialize(params, opts)
    density.set_training_images(_smooth_images(device=device))
    density.current_scale = 2
    outputs = renderer.render(params, camera, sh_degree=0)
    assert outputs["rgb"].shape[:2] == (16, 16)
    outputs["step"] = 1
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(32, device=device)[None, :, None] % 2).expand(32, 32, 3).float()
    losses = dict(build(method)[2])
    losses["photometric"](params, outputs, target).backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    assert float(density.stats["count"].sum()) > 0, "statistics were not accumulated"
    LINK.attach(None)
    return {"rgb_shape": list(outputs["rgb"].shape), "screen_gradient_norm": xy.grad.norm().item(),
            "accumulated": float(density.stats["count"].sum())}
