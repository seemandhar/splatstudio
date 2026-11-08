"""Mechanism probes for the Turbo-GS port. Each raises AssertionError when its mechanism is
removed or wrong. Reference computations are written independently of the plugin modules."""

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from splatstudio.losses import ssim

from .budget import BudgetScheduler
from .density import TurboDensity
from .losses import create_photometric
from .renderer import Renderer


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _scene(device, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * torch.tensor([0.6, 0.45, 0.3])
    means[:, 2] += 3.0
    params = {
        "means": means,
        "scales": torch.log(torch.rand(n, 3, generator=g) * 0.04 + 0.01),
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.logit(torch.rand(n, generator=g) * 0.8 + 0.1),
        "sh0": torch.rand(n, 1, 3, generator=g) * 2 - 1,
        "shN": torch.zeros(n, 15, 3),
    }
    return {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}


def _camera(device, w=96, h=64):
    K = torch.tensor([[80.0, 0, w / 2 - 0.3], [0, 80.0, h / 2 + 0.2], [0, 0, 1]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": w, "height": h}


def _psnr(a, b):
    return float(-10 * torch.log10((a - b).square().mean().clamp_min(1e-20)))


def dilated_merge(method, device):
    """Interleaving the 4 dilated renders reproduces the full render (paper: > 70 dB)."""
    renderer = Renderer(_spec(method, "renderer"))
    if renderer.d != 2:
        raise AssertionError("paper dilation d = 2 not configured")
    params, cam = _scene(device), _camera(device)
    with torch.no_grad():
        full = renderer.render(params, cam, sh_degree=0)["rgb"]
    if tuple(full.shape[:2]) != (64, 96):
        raise AssertionError("evaluation render is not full resolution")
    merged = torch.zeros_like(full)
    naive = torch.zeros_like(full)
    for ox in range(2):
        for oy in range(2):
            with torch.enable_grad():
                out = renderer.render(params, cam, sh_degree=0, offset=(ox, oy))
            if tuple(out["rgb"].shape[:2]) != (32, 48) or out["dilation"] != (2, ox, oy):
                raise AssertionError("training render is not the d=2 sub-lattice")
            merged[oy::2, ox::2] = out["rgb"].detach()
            # Mechanism removed: plain half-resolution intrinsics with the default filter.
            K = cam["K"].clone()
            K[0, 0] /= 2
            K[1, 1] /= 2
            K[0, 2] = (K[0, 2] - ox - 0.5) / 2 + 0.5
            K[1, 2] = (K[1, 2] - oy - 0.5) / 2 + 0.5
            with torch.no_grad():
                r, a, _ = rasterization(
                    params["means"], params["quats"], params["scales"].exp(),
                    params["opacities"].sigmoid(),
                    torch.cat((params["sh0"], params["shN"]), 1), cam["viewmat"][None], K[None],
                    48, 32, sh_degree=0, eps2d=0.3)
            naive[oy::2, ox::2] = r[0] + (1 - a[0])
    p_merge, p_naive = _psnr(merged, full), _psnr(naive, full)
    if p_merge < 50 or p_merge < p_naive + 10:
        raise AssertionError(f"dilated merge PSNR {p_merge:.1f} dB (naive {p_naive:.1f} dB)")
    seen = {renderer.render(params, cam, sh_degree=0)["dilation"] for _ in range(64)}
    if seen != {(2, ox, oy) for ox in range(2) for oy in range(2)}:
        raise AssertionError(f"random offsets do not cover the 4 sub-lattices: {seen}")
    return {"merge_psnr_db": p_merge, "naive_psnr_db": p_naive}


def dilated_target(method, device):
    loss = create_photometric(_spec(method, "loss"))
    g = torch.Generator().manual_seed(1)
    target = torch.rand(40, 30, 3, generator=g)
    out = {"rgb": torch.rand(20, 15, 3, generator=g).requires_grad_(), "dilation": (2, 1, 0)}
    value = loss({}, out, target)
    gt = torch.stack([torch.stack([target[2 * i, 2 * j + 1] for j in range(15)])
                      for i in range(20)])
    ref = 0.8 * (out["rgb"] - gt).abs().mean() + 0.2 * (1 - ssim(out["rgb"], gt))
    if abs(float(value) - float(ref)) > 1e-6:
        raise AssertionError(f"loss {float(value)} != reference {float(ref)} on the subset")
    wrong = 0.8 * (out["rgb"] - target[::2, ::2]).abs().mean()
    if abs(float(value) - float(ref)) > abs(float(ref) - float(wrong)):
        raise AssertionError("offset not applied to the target")
    if "training_loss" not in out or out["training_loss"].requires_grad:
        raise AssertionError("detached loss value not exported for the budget scheduler")
    return {"loss": float(value)}


def _power_history(sched, exponent, steps, scale=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for t in steps:
        sched.record(t, scale * t ** (-exponent) * (1 + noise * rng.standard_normal()))


def budget_schedule(method, device):
    cfg = _spec(method, "density").config
    lam, m = float(cfg["lam"]), 1_000_000
    # Endpoints and the warm-up default (alpha = 1 -> linear in t).
    s = BudgetScheduler(1000, m, lam=lam)
    b1, b100, b50 = s.budget(1.0), s.budget(100.0), s.budget(50.5)
    if abs(b1 - 1000) > 1e-6 or abs(b100 - m) > 1e-3 or abs(b50 - (1000 + 0.5 * (m - 1000))) > 1:
        raise AssertionError("B(t) endpoints / linear warm-up schedule wrong")
    # Pure power law L = t^-0.3: alpha_hist = rate = 0.3 -> deviation 0 -> alpha = 0.3.
    s = BudgetScheduler(1000, m, lam=lam)
    _power_history(s, 0.3, range(101, 3001))
    alpha = s.update()
    if abs(alpha - 0.3) > 0.02 or abs(s.rate - 0.3) > 0.02:
        raise AssertionError(f"power-law fit alpha={alpha}, rate={s.rate} (expected 0.3)")
    if not s.m_adaptive > m:  # rate 0.3 > 0.05 -> grow by 1.1
        raise AssertionError("M_adaptive did not grow for a decreasing loss")
    ref = 1000 + (50.0**alpha - 1) / (100.0**alpha - 1) * (s.m_adaptive - 1000)
    if abs(s.budget(50.0) - ref) > 1:
        raise AssertionError("Eq. 6 budget mismatch")
    # Plateau in the recent window: slower convergence -> larger alpha (slower densification).
    p = BudgetScheduler(1000, m, lam=lam)
    _power_history(p, 0.3, range(101, 2501))
    last = 2500 ** -0.3
    for t in range(2501, 3001):
        p.record(t, last)
    alpha_p = p.update()
    expected = p.alpha_hist[-1] + lam * math.tanh(p.alpha_hist[-1] - p.rate)
    if not alpha_p > alpha + 0.05 or abs(alpha_p - min(max(expected, 0.1), 2.0)) > 1e-9:
        raise AssertionError(f"plateau alpha {alpha_p} not above power-law alpha {alpha}")
    # Rising loss -> M_adaptive shrinks, clipped at 0.5 M.
    r = BudgetScheduler(1000, m, lam=lam)
    for t in range(101, 3001):
        r.record(t, 1e-3 * t)
    for _ in range(20):
        r.update()
    if abs(r.m_adaptive - 0.5 * m) > 1e-6:
        raise AssertionError(f"M_adaptive floor wrong: {r.m_adaptive}")
    for _ in range(3):
        s.update()
    return {"alpha_powerlaw": alpha, "alpha_plateau": alpha_p, "m_floor": r.m_adaptive}


def color_densify(method, device):
    spec = _spec(method, "density")
    dens = TurboDensity(spec)
    n = 1000
    g = torch.Generator().manual_seed(0)
    g_pos = torch.rand(n, generator=g) * 1e-4  # all below tau_pos = 2e-4
    g_col = torch.rand(n, generator=g) * 1e-6  # all below tau_col = 2e-6
    g_pos[:50] = 5e-4  # position qualifiers
    g_col[50:120] = 5e-6  # colour-only qualifiers
    m_off, _ = dens.select(g_pos, g_col, n, None, color_active=False)
    m_on, st = dens.select(g_pos, g_col, n, None, color_active=True)
    if int(m_off.sum()) != 50 or int(m_on.sum()) != 120 or st["col_only"] != 70:
        raise AssertionError(f"colour branch wrong: off={int(m_off.sum())} on={int(m_on.sum())}")
    if abs(dens.tau_col - 0.01 * dens.tau_pos) > 1e-15:
        raise AssertionError("tau_color != 0.01 tau_position")
    g_pos[:50] = torch.linspace(3e-4, 1e-3, 50)
    capped, st = dens.select(g_pos, g_col, n, 30, color_active=True)
    if int(capped.sum()) != 30 or not bool(capped[20:50].all()):
        raise AssertionError("budget cap does not keep the 30 highest-score candidates")
    zero, _ = dens.select(g_pos, g_col, n, 0, color_active=True)
    if int(zero.sum()) != 0:
        raise AssertionError("zero allowance still densified")
    fired = sum(dens.rng.random() < float(spec.config["color_probability"]) for _ in range(5000))
    if abs(fired / 5000 - 0.2) > 0.03:
        raise AssertionError(f"colour branch probability {fired / 5000}")
    return {"selected_capped": int(capped.sum()), "color_fire_rate": fired / 5000}


def color_grad(method, device):
    """Accumulated colour statistic equals an independent autograd ||dL/d sh0|| per Gaussian."""
    if not torch.cuda.is_available():
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("CUDA is unavailable")
    renderer = Renderer(_spec(method, "renderer"))
    spec = _spec(method, "density")
    dens = TurboDensity(replace(spec, config={**spec.config, "verbose": False}))
    params, cam = _scene(device), _camera(device)
    opts = {k: torch.optim.Adam([v], lr=0.0) for k, v in params.items()}
    dens.initialize(params, opts, 1.0)
    target = torch.rand(64, 96, 3, generator=torch.Generator().manual_seed(3)).to(device)
    loss = create_photometric(_spec(method, "loss"))
    out = renderer.render(params, cam, sh_degree=0, offset=(1, 1))
    dens.before_backward(params, opts, 1, out)
    loss(params, out, target).backward()
    reference = params["sh0"].grad[:, 0, :].norm(dim=-1)
    dens.after_backward(params, opts, 1, out)
    visible = (out["info"]["radii"] > 0).all(-1)[0]
    got = dens.state["colgrad"]
    if not torch.allclose(got[visible], reference[visible], rtol=1e-5, atol=1e-12):
        raise AssertionError("colgrad differs from dL/dsh0 norm")
    if bool((~visible).any()) and float(got[~visible].abs().max()) > 0:
        raise AssertionError("invisible Gaussians accumulated colour gradient")
    if not float(reference.max()) > 0:
        raise AssertionError("zero colour gradient")
    return {"visible": int(visible.sum()), "colgrad_max": float(got.max())}


def renderer_gradient(method, device):
    """Built-in renderer-gradient check with the dilated target: finite nonzero screen and
    colour gradients reach the density statistics (replaces splatstudio's version, whose
    full-size target does not match a dilated training render)."""
    if not torch.cuda.is_available():
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("CUDA is unavailable")
    from splatstudio.components import build
    from splatstudio.probes import sample_scene
    from splatstudio.state import check_state

    renderer, density, _ = build(method)
    density.c["verbose"] = False
    params, opts = sample_scene(method, device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
              "width": 16, "height": 16}
    outputs = renderer.render(params, camera, sh_degree=0)
    d, ox, oy = outputs["dilation"]
    density.initialize(params, opts)
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(16, device=device)[None, :, None] % 2).expand(16, 16, 3).float()
    target = target[oy::d, ox::d][: outputs["rgb"].shape[0], : outputs["rgb"].shape[1]]
    (outputs["rgb"] - target).square().mean().backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    density.after_backward(params, opts, 1, outputs)
    if not float(density.state["count"].sum()) > 0 or not float(density.state["colgrad"].sum()) > 0:
        raise AssertionError("density statistics not accumulated")
    check_state(method, params, opts)
    return {"rgb_shape": list(outputs["rgb"].shape), "dilation": list(outputs["dilation"]),
            "screen_gradient_norm": xy.grad.norm().item()}


def refine_allowance(method, device):
    """refine() enforces allowance = B(t) - n on a real event, with t = 100 * step / total_steps
    (fails if the cap or the scheduler wiring is removed)."""
    spec = _spec(method, "density")
    params = _scene(device)
    n = len(params["means"])
    m = n + 990

    def event(**over):
        dens = TurboDensity(replace(spec, config={**spec.config, "budget": m, "verbose": False,
                                                  "use_color": False, **over}))
        p = {k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()}
        opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
        dens.initialize(p, opts, 1.0)
        dens.state["grad2d"] = torch.full((n,), 1e-3, device=device)  # all qualify
        dens.state["count"] = torch.ones(n, device=device)
        dens.state["colgrad"] = torch.zeros(n, device=device)
        dens.refine(p, opts, 3000)
        return dens.log[-1]

    row = event()
    t = 100.0 * 3000 / float(spec.config["total_steps"])  # alpha = 1 before warm-up records
    expected = int(n + (t - 1) / 99.0 * (m - n))
    if row["candidates"] != n or row["target"] != expected or row["selected"] != expected - n:
        raise AssertionError(f"allowance not enforced: {row}")
    if row["peak"] != expected:
        raise AssertionError(f"peak #G {row['peak']} != B(t) {expected}")
    free = event(use_budget=False)
    if free["selected"] != n:
        raise AssertionError("ablation without budget should densify every candidate")
    window = event(t_mode="densify_window")
    if window["target"] == expected:
        raise AssertionError("t mapping does not depend on total training steps")
    return {"target": expected, "selected": row["selected"], "uncapped": free["selected"],
            "densify_window_target": window["target"]}
