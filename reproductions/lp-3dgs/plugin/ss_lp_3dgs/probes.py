"""LP-3DGS mechanism probes. Each one fails if its mechanism is removed or wrong.

References are written independently of the plugin functions (explicit Gumbel samples, a
float64 per-pixel compositor, hand-built autograd graphs).
"""

import math
from dataclasses import replace

import torch
from splatstudio.registry import resolve
from torch import nn

from . import core, kernels

SH_C0 = 0.28209479177387814


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _spec(method, kind, name=None):
    return next(c for c in method.components if c.kind == kind and (name is None or c.name == name))


def _scene(n, device, seed=3, mask_value=1.0):
    g = torch.Generator().manual_seed(seed)
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1, torch.rand(n, 1, generator=g) * 3 + 2],
                      1)
    means[:, :2] *= means[:, 2:3] * 0.45
    values = {
        "means": means,
        "scales": (torch.rand(n, 3, generator=g) * 0.25 + 0.05).log(),
        "quats": nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.logit(torch.rand(n, generator=g) * 0.8 + 0.15),
        "sh0": ((torch.rand(n, 3, generator=g) * 0.6 + 0.2 - 0.5) / SH_C0)[:, None, :],
        "shN": torch.zeros(n, 15, 3),
        core.FIELD: torch.full((n,), mask_value),
    }
    params = nn.ParameterDict({k: nn.Parameter(v.to(device)) for k, v in values.items()})
    rates = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 0.05, "sh0": 2.5e-3,
             "shN": 1.25e-4, core.FIELD: 0.05}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    return params, opts


def _camera(width, height, f, c2w=None, device="cuda"):
    if c2w is None:
        c2w = torch.eye(4)
    return {
        "viewmat": torch.linalg.inv(c2w).to(device),
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device),
        "width": width,
        "height": height,
    }


def _logistic_reference(u1, u2, eps=1e-10):
    g0 = -math.log(-math.log(u1 + eps))  # Gumbel(0, 1) samples
    g1 = -math.log(-math.log(u2 + eps))
    return g0 - g1


# ---------------------------------------------------------------------------- CPU probes
def gumbel(method, device):
    tau = float(_spec(method, "renderer").config["tau"])
    if tau != 0.5:
        raise AssertionError(f"tau {tau} != paper/code 0.5")
    g = torch.Generator().manual_seed(0)
    u1, u2 = torch.rand(64, generator=g), torch.rand(64, generator=g)
    official = -torch.log(torch.log(u1 + 1e-10) / torch.log(u2 + 1e-10) + 1e-10)
    ref = torch.tensor([_logistic_reference(a, b) for a, b in zip(u1.tolist(), u2.tolist())])
    err_noise = (official - ref).abs().max().item()
    x = torch.linspace(-3, 3, 64, requires_grad=True)
    y = core.gumbel_sigmoid(x, tau, official)
    want = 1 / (1 + torch.exp(-(x.detach() + ref) / tau))
    err_value = (y.detach() - want).abs().max().item()
    y.sum().backward()
    want_grad = want * (1 - want) / tau
    err_grad = (x.grad - want_grad).abs().max().item()
    # stochastic: P(gs(x) > 0.5) = P(n > -x) = sigmoid(x) for logistic n
    gen = torch.Generator().manual_seed(1)
    xs = torch.tensor([-2.0, -0.5, 0.0, 0.7, 2.0])
    noise = core.logistic_noise((200000, 5), "cpu", gen)
    freq = (core.gumbel_sigmoid(xs.expand(200000, 5), tau, noise) > 0.5).float().mean(0)
    err_freq = (freq - torch.sigmoid(xs)).abs().max().item()
    spread = core.gumbel_sigmoid(torch.zeros(200000), tau, noise[:, 2])
    binary = ((spread < 0.1) | (spread > 0.9)).float().mean().item()
    if err_noise > 1e-4 or err_value > 1e-5 or err_grad > 1e-5:
        raise AssertionError(f"gumbel-sigmoid formula wrong: {err_noise} {err_value} {err_grad}")
    if err_freq > 0.01 or binary < 0.4:
        raise AssertionError(f"gumbel-sigmoid distribution wrong: freq {err_freq}, bin {binary}")
    return {"noise_error": err_noise, "value_error": err_value, "grad_error": err_grad,
            "frequency_error": err_freq, "fraction_near_binary_at_0": binary}


def mask_loss_probe(method, device):
    spec = _spec(method, "loss", "lp-mask-loss")
    cfg = spec.config
    weight, start, iters = cfg["weight"], cfg["mask_start"], cfg["mask_iters"]
    if (weight, start, iters) != (5e-4, 19500, 500):
        raise AssertionError(f"mask loss config {cfg} differs from the official code")
    loss = resolve(spec.factory)(spec)
    m = nn.Parameter(torch.randn(50))
    rgb = torch.rand(4, 4, 3, requires_grad=True)
    params = {core.FIELD: m}
    rows = {}
    for step in (start, start + 1, start + iters // 2, start + iters - 1, start + iters, 10):
        m.grad = None
        value = loss(params, {"step": step, "rgb": rgb}, None)
        value.backward()
        inside = start < step < start + iters
        want = weight * torch.sigmoid(m.detach()).mean().item() if inside else 0.0
        if abs(value.item() - want) > 1e-9:
            raise AssertionError(f"step {step}: loss {value.item()} != {want}")
        if inside:
            s = torch.sigmoid(m.detach())
            if (m.grad - weight * s * (1 - s) / len(m)).abs().max() > 1e-10:
                raise AssertionError("mask loss gradient wrong")
        elif m.grad is not None:
            raise AssertionError(f"step {step}: mask received a gradient outside the window")
        rows[str(step)] = value.item()
    return {"loss_by_step": rows}


# ---------------------------------------------------------------------------- CUDA probes
def _composite_max(means2d, conics, opacities, depths, width, height):
    """Float64 per-pixel front-to-back compositor with gsplat numerics (vectorised over pixels)."""
    order = torch.argsort(depths)
    ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.float64) + 0.5,
                            torch.arange(width, dtype=torch.float64) + 0.5, indexing="ij")
    px, py = xs.reshape(-1), ys.reshape(-1)
    T = torch.ones_like(px)
    alive = torch.ones_like(px, dtype=torch.bool)
    best = torch.zeros(len(means2d), dtype=torch.float64)
    total = torch.zeros(len(means2d), dtype=torch.float64)
    for g in order.tolist():
        dx, dy = means2d[g, 0] - px, means2d[g, 1] - py
        a, b, c = conics[g]
        sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        alpha = torch.clamp(opacities[g] * torch.exp(-sigma), max=0.999)
        use = alive & (sigma >= 0) & (alpha >= 1 / 255)
        nxt = T * (1 - alpha)
        stop = use & (nxt <= 1e-4)
        alive = alive & ~stop
        use = use & ~stop
        w = torch.where(use, alpha * T, torch.zeros_like(T))
        best[g] = w.max()
        total[g] = w.sum()
        T = torch.where(use, nxt, T)
    return best, total


def score_kernel(method, device):
    _require_cuda()
    params, _ = _scene(40, device)
    worst, gap = 0.0, 0.0
    score = torch.zeros(40, device=device)
    ref = torch.zeros(40, dtype=torch.float64)
    for k, angle in enumerate((0.0, 0.12, -0.1)):
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                                    [-math.sin(angle), 0, math.cos(angle)]])
        c2w[0, 3] = 0.2 * k
        cam = _camera(48, 36, 40.0, c2w, device)
        meta = kernels.project(params, cam, params["opacities"].sigmoid())
        view = torch.zeros(40, device=device)
        kernels.max_contribution_(meta, view)
        kernels.max_contribution_(meta, score)
        best, total = _composite_max(meta["means2d"][0].cpu().double(),
                                     meta["conics"][0].cpu().double(),
                                     meta["opacities"][0].cpu().double(),
                                     meta["depths"][0].cpu().double(), 48, 36)
        best = torch.where(meta["radii"][0].cpu().reshape(40, -1).amax(-1) > 0, best,
                           torch.zeros_like(best))
        worst = max(worst, (view.cpu().double() - best).abs().max().item())
        gap = max(gap, (total - best).abs().max().item())
        ref = torch.maximum(ref, best)
    err_total = (score.cpu().double() - ref).abs().max().item()
    if worst > 2e-3 or err_total > 2e-3:
        raise AssertionError(f"score kernel disagrees with the reference: {worst}, {err_total}")
    if gap < 0.1 or float(score.max()) < 0.3:
        raise AssertionError("synthetic scene does not distinguish max from sum")
    return {"max_view_error": worst, "max_over_views_error": err_total, "sum_minus_max": gap,
            "covered_gaussians": int((score > 0).sum())}


def mask_gradient(method, device):
    _require_cuda()
    spec = _spec(method, "renderer")
    renderer = resolve(spec.factory)(spec)
    params, _ = _scene(40, device, mask_value=0.3)
    cam = _camera(48, 36, 40.0, device=device)
    n = len(params["means"])
    g = torch.Generator(device=device).manual_seed(5)
    score = torch.rand(n, device=device, generator=g)
    noise = core.logistic_noise((n,), device, g)
    setattr(params, core.SCORE_ATTR, score)
    out = {}
    for phase in ("off", "train", "frozen"):
        setattr(params, core.PHASE_ATTR, phase)
        for p in params.values():
            p.grad = None
        renderer.render(params, cam, 0, noise=noise)["rgb"].square().mean().backward()
        out[phase] = params[core.FIELD].grad
    if out["off"] is not None or out["frozen"] is not None:
        raise AssertionError("mask received a gradient while the mask is not trained")
    if out["train"] is None or out["train"].abs().max() == 0:
        raise AssertionError("mask received no gradient in the training phase")
    # independent graph: o * 1/(1+exp(-(m S + n)/tau)) fed to gsplat directly
    from gsplat import rasterization

    m = params[core.FIELD].detach().clone().requires_grad_(True)
    tau = spec.config["tau"]
    opac = params["opacities"].detach().sigmoid() / (1 + torch.exp(-(m * score + noise) / tau))
    colors = torch.cat((params["sh0"], params["shN"]), 1).detach()
    img, alpha, _ = rasterization(params["means"].detach(), params["quats"].detach(),
                                  params["scales"].detach().exp(), opac, colors,
                                  cam["viewmat"][None], cam["K"][None], 48, 36, sh_degree=0,
                                  packed=False)
    (img[0] + (1 - alpha[0])).square().mean().backward()
    err = (m.grad - out["train"]).abs().max().item()
    scale = m.grad.abs().max().item()
    if err > 1e-5 * max(scale, 1e-6) + 1e-9:
        raise AssertionError(f"mask gradient differs from the independent graph: {err} / {scale}")
    delattr(params, core.SCORE_ATTR)
    setattr(params, core.PHASE_ATTR, "off")
    return {"grad_max": scale, "grad_error": err,
            "nonzero_grads": int((out["train"] != 0).sum())}


def schedule(method, device):
    _require_cuda()
    spec = _spec(method, "density")
    cfg = spec.config
    if (cfg["mask_start"], cfg["mask_iters"], cfg["score_every"], cfg["tau"]) != (19500, 500, 20,
                                                                                    0.5):
        raise AssertionError(f"density schedule {cfg} differs from the official code")
    start, iters, every = 40, 30, 7
    small = replace(spec, config={**cfg, "mask_start": start, "mask_iters": iters,
                                  "score_every": every, "refine_stop_iter": 10,
                                  "refine_start_iter": 1000, "reset_every": 100000})
    density = resolve(small.factory)(small)
    params, opts = _scene(60, device, mask_value=-7.0)
    cams = [_camera(48, 36, 40.0, device=device)]
    density.initialize(params, opts, 1.0)
    density.set_training_cameras(cams)
    rspec = _spec(method, "renderer")
    renderer = resolve(rspec.factory)(rspec)
    lspec = replace(_spec(method, "loss", "lp-mask-loss"),
                    config={"weight": 5e-4, "mask_start": start, "mask_iters": iters})
    loss = resolve(lspec.factory)(lspec)
    phases, score_steps, mask_at_start = {}, [], None
    keep = torch.zeros(60, dtype=torch.bool)
    for step in range(start - 2, start + iters + 1):
        out = renderer.render(params, cams[0], 0)
        out["step"] = step
        density.before_backward(params, opts, step, out)
        (out["rgb"].mean() + loss(params, out, None)).backward()
        mask_grad = params[core.FIELD].grad
        if step <= start and mask_grad is not None:
            raise AssertionError(f"mask has a gradient at step {step} before the window")
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        before = len(params["means"])
        if step == start + iters:
            # pre-compute the expected prune selection with the same RNG stream
            state = torch.cuda.get_rng_state(device)
            expect = core.prune_selection(params[core.FIELD], cfg["tau"])
            torch.cuda.set_rng_state(state, device)
            keep = ~expect.cpu()
        density.after_backward(params, opts, step, out)
        phases[step] = core.phase(params)
        if step == start:
            mask_at_start = params[core.FIELD].detach().clone()
        if density.events and density.events[-1].get("event") == "score" and \
                density.events[-1]["step"] == step + 1:
            score_steps.append(step + 1)
        if step < start + iters and len(params["means"]) != before:
            raise AssertionError("population changed inside the mask window")
    want_scores = [s for s in range(start + 1, start + iters) if (s - start) % every == 0]
    if score_steps != want_scores:
        raise AssertionError(f"score refreshed at {score_steps}, expected {want_scores}")
    lr, wd = 0.05, cfg["weight_decay"]
    if mask_at_start is None or not torch.allclose(mask_at_start,
                                                   torch.full_like(mask_at_start, 1.0)):
        raise AssertionError("mask not reset to 1 at mask_start")
    if phases[start - 1] != "off" or phases[want_scores[0] - 2] != "off" or \
            phases[want_scores[0] - 1] != "train" or phases[start + iters - 1] != "frozen" or \
            phases[start + iters] != "off":
        raise AssertionError(f"phase schedule wrong: {phases}")
    prune = density.events[-1]
    if not 0 < int(keep.sum()) < 60:
        raise AssertionError("toy scene prune selection is degenerate")
    if prune.get("event") != "prune" or prune["after"] != int(keep.sum()):
        raise AssertionError(f"prune removed the wrong set: {prune}, expected {int(keep.sum())}")
    if hasattr(params, core.SCORE_ATTR):
        raise AssertionError("score not dropped after pruning")
    from splatstudio.state import check_state

    check_state(method, params, opts)
    for name, p in params.items():
        if not (p.is_leaf and p.requires_grad):
            raise AssertionError(f"{name} lost requires_grad")
    return {"score_steps": score_steps, "kept": int(keep.sum()), "removed": 60 - int(keep.sum()),
            "decay_factor": 1 - lr * wd}
