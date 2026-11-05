"""Executable mechanism checks for PUP 3D-GS (each fails if its mechanism is removed/wrong)."""

import math
import warnings
from dataclasses import replace

import numpy as np
import torch


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("The PUP Fisher kernel needs CUDA")


def _scene(n, device, seed, sh_rest=0.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-0.6, 0.6, generator=g)
    means[:, 2] = torch.empty(n).uniform_(-0.3, 0.3, generator=g)
    params = {
        "means": means,
        "scales": torch.empty(n, 3).uniform_(math.log(0.06), math.log(0.2), generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.empty(n).uniform_(0.2, 0.6, generator=g)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * sh_rest,
    }
    return {k: v.to(device) for k, v in params.items()}


def _camera(device, angle=0.15, width=96, height=64, cx=47.3, cy=33.1, f=80.0):
    """OpenCV world-to-camera looking at the origin from distance 3, off-centre principal point."""
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = torch.tensor([0.05, -0.03, 3.0])
    K = torch.tensor([[f, 0.0, cx], [0.0, f * 1.02, cy], [0.0, 0.0, 1.0]])
    return {"K": K.to(device), "viewmat": viewmat.to(device), "width": width, "height": height}


def _render(params, camera, degree=3):
    from gsplat import rasterization

    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
        height=camera["height"], sh_degree=degree, packed=False,
    )
    return render[0] + (1 - alpha[0])  # white background, the SplatStudio convention


def _oracle_fisher(params, camera):
    """sum_{p,c} g g^T with g = d rgb[p,c] / d(means, log scales), by per-pixel autograd."""
    leaf = {k: v.detach().clone().requires_grad_(k in ("means", "scales")) for k, v in params.items()}
    rgb = _render(leaf, camera)
    n = len(leaf["means"])
    F = torch.zeros(n, 6, 6, dtype=torch.float64, device=rgb.device)
    H, W, C = rgb.shape
    for y in range(H):
        for x in range(W):
            for ch in range(C):
                gm, gs = torch.autograd.grad(rgb[y, x, ch], (leaf["means"], leaf["scales"]),
                                             retain_graph=True)
                g = torch.cat((gm, gs), dim=1).double()
                F += g[:, :, None] * g[:, None, :]
    return F


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def fisher_oracle(method, device):
    _require_cuda(device)
    from . import fisher

    params = _scene(24, device, seed=3)
    # One view constrains only the 2D mean/covariance (rank <= 5); several views make F full rank.
    cameras = [_camera(device, angle=a) for a in (-0.35, 0.15, 0.5)]
    sym = fisher.scene_fisher(params, cameras, resolution=4, background=(1.0, 1.0, 1.0))
    kernel = fisher.expand_sym(sym).double()
    # Independent low-res cameras: ceil(96/4) x ceil(64/4) = 24 x 16, K scaled by exactly 1/4.
    quarter = torch.tensor([[0.25], [0.25], [1.0]], device=device)
    oracle = sum(_oracle_fisher(params, dict(c, width=24, height=16, K=c["K"] * quarter))
                 for c in cameras)
    err = _rel(kernel, oracle)
    if err > 0.02:
        raise AssertionError(f"CUDA Fisher differs from the autograd oracle: rel err {err:.4f}")
    # Discrimination: Fisher w.r.t. activated (linear) scales, i.e. without the kernel's x s.
    s = torch.cat((torch.ones(len(oracle), 3, device=device, dtype=torch.float64),
                   params["scales"].exp().double()), dim=1)
    linear = oracle / (s[:, :, None] * s[:, None, :])
    err_linear = _rel(kernel, linear)
    full_res = sum(_oracle_fisher(params, c) for c in cameras)
    err_full = _rel(kernel, full_res)
    if err_linear < 0.2 or err_full < 0.2:
        raise AssertionError("Probe cannot distinguish scale parameterisation or patch size")
    ld_kernel = torch.linalg.slogdet(kernel)[1]
    ld_oracle = torch.linalg.slogdet(oracle)[1]
    finite = torch.isfinite(ld_oracle) & torch.isfinite(ld_kernel)
    corr = float(np.corrcoef(ld_kernel[finite].cpu(), ld_oracle[finite].cpu())[0, 1])
    if finite.sum() < 20 or corr < 0.999:
        raise AssertionError(f"log-det scores disagree (corr {corr:.5f})")
    # The kernel's score must equal log|det F| (svdvals form used by the official code).
    score = fisher.log_det_scores(sym).double()
    score_err = float((score[finite] - ld_kernel[finite]).abs().max())
    if score_err > 1e-2:
        raise AssertionError(f"log_det_scores != slogdet: {score_err}")
    return {
        "relative_error_vs_oracle": err,
        "relative_error_if_linear_scale": err_linear,
        "relative_error_if_full_resolution": err_full,
        "logdet_correlation": corr,
        "max_score_error": score_err,
        "gaussians": len(oracle),
    }


def _official_lrs(lr0, lr_iter_target, lr_start=30000, max_steps=35000, gamma=0.95, every=400):
    """Independent replay of prune_finetune.py: get_expon_lr_func + torch ExponentialLR."""
    names = list(lr0)
    dummies = [torch.nn.Parameter(torch.zeros(1)) for _ in names]
    opt = torch.optim.Adam([{"params": [p], "lr": lr0[n], "name": n} for p, n in zip(dummies, names)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scheduler stepped without optimizer.step (LR replay)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    init, final = lr0["means"], lr0["means"] * 0.01
    for lr_iter in range(lr_start + 1, lr_iter_target + 1):
        t = np.clip(lr_iter / max_steps, 0, 1)
        opt.param_groups[0]["lr"] = float(np.exp(np.log(init) * (1 - t) + np.log(final) * t))
        if lr_iter % every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sched.step()
    return {g["name"]: g["lr"] for g in opt.param_groups}


def prune_schedule(method, device):
    _require_cuda(device)
    from splatstudio.losses import create_photometric
    from splatstudio.state import check_state, initialize

    from . import fisher
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")
    ratios = list(spec.config["prune_ratios"])
    config = dict(spec.config, base_steps=4, prune_steps=[4, 7])
    density = create_density(replace(spec, config=config))
    loss_fn = create_photometric(next(c for c in method.components if c.name == "photometric"))

    torch.manual_seed(0)
    scene = _scene(400, device, seed=5, sh_rest=0.05)
    scene["means"][:10, 2] = -6.0  # behind every camera -> zero Fisher, must be pruned first
    points = scene["means"].clone()
    colors = torch.rand(400, 3, device=device)
    params, optimizers = initialize(method, points, colors)
    with torch.no_grad():
        for k in ("scales", "quats", "opacities", "sh0", "shN"):
            params[k].copy_(scene[k])
    cameras = [_camera(device, angle=a) for a in (-0.3, -0.1, 0.1, 0.3)]
    density.initialize(params, optimizers, 1.0)
    density.set_training_cameras(cameras)
    lr0 = {k: o.param_groups[0]["lr"] for k, o in optimizers.items()}
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.specs import ComponentSpec

    renderer = Renderer(ComponentSpec("gsplat", "renderer", "", config={}))
    target = torch.rand(64, 96, 3, device=device)
    measurements = {}
    for step in range(1, 11):
        out = renderer.render(params, cameras[step % 4], sh_degree=3)
        out["step"] = step
        density.before_backward(params, optimizers, step, out)
        loss_fn(params, out, target).backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] = lr0["means"] * 0.01 ** (step / 10)
        if step in (4, 7):
            ref = fisher.expand_sym(fisher.scene_fisher(params, cameras)).double()
            ref_score = torch.linalg.slogdet(ref)[1]
            ref_score[torch.linalg.det(ref) <= 0] = -math.inf
            n = len(ref_score)
            ratio = ratios[[4, 7].index(step)]
            k = int(ratio * (n - 1))
            thr = torch.sort(ref_score).values[k]
            keep_ref = ref_score > thr
            before_means = params["means"].detach().clone()
        density.after_backward(params, optimizers, step, out)
        check_state(method, params, optimizers)
        lr_means = optimizers["means"].param_groups[0]["lr"]
        if step < 4:
            expected = lr0["means"] * 0.01 ** (step / 4)
            if not math.isclose(lr_means, expected, rel_tol=1e-9):
                raise AssertionError(f"pretrain means LR {lr_means} != {expected}")
        if step in (4, 7):
            # Kept Gaussians = highest log|F| (ties within float noise of the threshold allowed).
            kept = before_means[keep_ref]
            near = (ref_score - thr).abs() <= 1e-3 * max(1.0, abs(float(thr)))
            if abs(len(params["means"]) - int(keep_ref.sum())) > int(near.sum()):
                raise AssertionError("pruned count differs from the official percentile rule")
            survivors = {tuple(r) for r in params["means"].detach().cpu().numpy().round(6)}
            missing = [i for i, r in enumerate(kept.cpu().numpy().round(6))
                       if tuple(r) not in survivors]
            if len(missing) > int(near.sum()):
                raise AssertionError(f"{len(missing)} high-sensitivity Gaussians were pruned")
            if step == 4:
                zero_left = sum(tuple(r) in survivors for r in before_means[:10].cpu().numpy().round(6))
                if zero_left:
                    raise AssertionError("zero-Fisher (unseen) Gaussians survived pruning")
            if any(len(o.state) for o in optimizers.values()):
                raise AssertionError("Adam state was not reset at the prune step")
            measurements[f"step{step}"] = {"before": n, "after": len(params["means"]),
                                           "expected_after": int(keep_ref.sum())}
        if step >= 4:
            round_start = 4 if step < 7 else 7
            official = _official_lrs(lr0, 30000 + (step + 1 - round_start))
            for name, opt in optimizers.items():
                got = opt.param_groups[0]["lr"]
                if not math.isclose(got, official[name], rel_tol=1e-6):
                    raise AssertionError(f"{name} fine-tune LR {got} != official {official[name]}")
    # ExponentialLR boundaries far into a round (lr_iter 30399/30400/30401/34800/35000).
    boundary = {}
    for lr_iter in (30399, 30400, 30401, 34800, 35000):
        density.set_learning_rates(optimizers, lr_iter - 30000 + 7 - 1)
        official = _official_lrs(lr0, lr_iter)
        for name, opt in optimizers.items():
            if not math.isclose(opt.param_groups[0]["lr"], official[name], rel_tol=1e-6):
                raise AssertionError(f"{name} LR at lr_iter {lr_iter} differs from official")
        boundary[lr_iter] = {"means": optimizers["means"].param_groups[0]["lr"],
                             "opacities": optimizers["opacities"].param_groups[0]["lr"]}
    total = 1 - len(params["means"]) / 400
    expected_total = 1 - (1 - ratios[0]) * (1 - ratios[1])
    if abs(total - expected_total) > 0.02:
        raise AssertionError(f"cumulative pruning {total:.3f} != {expected_total:.3f}")
    measurements.update(cumulative_pruned=total, lr_boundaries=boundary)
    return measurements
