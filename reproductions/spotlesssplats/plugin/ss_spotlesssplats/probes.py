"""Mechanism probes for SpotLessSplats. Each one compares an implemented mechanism against an
independently written reference, so it fails if the mechanism is removed or altered."""

import math

import numpy as np
import torch

from .context import FrameTable, Shared
from .losses import (
    RobustLoss,
    SpotLessMLP,
    cluster_vote,
    create_mlp_loss,
    hinge,
    robust_mask,
    schedule_alpha,
)


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _np_robust_mask(error, thr):
    e = error.mean(-1)
    inl = (e < thr).astype(np.float64)
    pad = np.pad(inl, 1)
    box = sum(pad[1 + di: 1 + di + e.shape[0], 1 + dj: 1 + dj + e.shape[1]]
              for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    return np.logical_or(box > 0.5, inl > 0).astype(np.float32)


def robust_mask_probe(method, device):
    g = torch.Generator().manual_seed(0)
    err = torch.rand(40, 36, 3, generator=g) * 0.1
    err[10:22, 8:20] += 0.6                            # an outlier blob (distractor)
    err[30, 30] += 0.9                                 # an isolated outlier pixel
    thr = 0.3
    ours = robust_mask(err, thr).numpy()
    ref = _np_robust_mask(err.numpy(), thr)
    if not np.array_equal(ours, ref):
        raise AssertionError("robust_mask differs from the NumPy reference")
    if ours[12:20, 10:18].any() or not ours[30, 30]:
        raise AssertionError("blob must be masked; isolated outlier kept by its inlier neighbours")
    return {"masked_fraction": float(1 - ours.mean()), "blob_masked": True}


def histogram_probe(method, device):
    spec = _spec(method, "sls-robust")
    loss = RobustLoss(spec)
    loss._ensure(torch.device("cpu"))
    cfg = loss.cfg
    rng = np.random.default_rng(1)
    hist = np.zeros(cfg["bins"])
    for _ in range(5):
        e = rng.beta(1.2, 8.0, size=(30, 20)).astype(np.float32)
        loss.update_histogram(torch.from_numpy(e))
        hist = cfg["hist_decay"] * hist + np.histogram(e, bins=cfg["bins"], range=(0, 1))[0]
    edges = np.linspace(0, 1, cfg["bins"] + 1)
    out = {}
    for name, value in zip(("robust_percentile", "lower_bound", "upper_bound"), loss.thresholds):
        k = int(np.nonzero(np.cumsum(hist) >= hist.sum() * cfg[name])[0][0])
        if abs(float(value) - edges[k]) > 1.5 / cfg["bins"]:
            raise AssertionError(f"{name}: {float(value)} != reference {edges[k]}")
        out[name] = float(value)
    if not out["lower_bound"] < out["robust_percentile"] < out["upper_bound"]:
        raise AssertionError("quantile thresholds must be ordered")
    return out


def mlp_probe(method, device):
    torch.manual_seed(0)
    spec = _spec(method, "sls-robust")
    pe_freqs = int(spec.config.get("pe_freqs", 20))
    mlp = SpotLessMLP(32 + 4 * pe_freqs, int(spec.config.get("mlp_hidden", 16))).double()
    from .losses import positional_encoding

    feat = torch.randn(32, 7, 9, dtype=torch.float64)
    pe = positional_encoding(23, 31, pe_freqs, "cpu").double()
    fast, naive = mlp(feat, pe), mlp.forward_naive(feat, pe)
    gap = float((fast - naive).abs().max())
    if gap > 1e-10:
        raise AssertionError(f"low-res first layer differs from the official order ({gap})")
    p = torch.tensor([0.9, 0.1, 0.5], requires_grad=True)
    lower = torch.tensor([0.0, 1.0, 0.0])
    upper = torch.tensor([0.0, 1.0, 1.0])
    hinge(p, lower, upper).backward()
    gp = p.grad.tolist()
    if not (gp[0] > 0 and gp[1] < 0 and gp[2] == 0):
        raise AssertionError(f"hinge gradient signs wrong: {gp}")
    return {"max_equivalence_gap": gap, "hinge_grad": gp}


def warmup_probe(method, device):
    spec = _spec(method, "sls-robust")
    beta = float(spec.config.get("schedule_beta", -3e-3))
    div = float(spec.config.get("schedule_div", 1.5))
    vals = {t: schedule_alpha(t, beta, div) for t in (0, 1, 1000, 3000)}
    ref = {t: math.exp(beta * ((t + 1) // div)) for t in vals}
    if any(abs(vals[t] - ref[t]) > 1e-12 for t in vals) or vals[0] != 1.0:
        raise AssertionError(f"schedule {vals} != {ref}")
    # Through the loss: at t=0 (alpha=1) every pixel is used -> loss == mean L1.
    from dataclasses import replace

    filt = replace(spec, config={**spec.config, "mask_mode": "filter"})
    loss = RobustLoss(filt)
    g = torch.Generator().manual_seed(3)
    target = torch.rand(24, 24, 3, generator=g)
    rgb = (target + 0.05 * torch.randn(24, 24, 3, generator=g)).requires_grad_()
    loss.thresholds = None
    loss._ensure(torch.device("cpu"))
    loss.thresholds = tuple(torch.tensor(v) for v in (0.02, 0.0, 1.0))
    v0 = loss(None, {"rgb": rgb, "step": 1, "camera": None}, target)
    l1 = (rgb.clamp(0, 1) - target).abs().mean()
    if abs(float(v0 - l1)) > 1e-7:
        raise AssertionError("alpha = 1 must use every pixel")
    loss.thresholds = tuple(torch.tensor(v) for v in (0.02, 0.0, 1.0))
    v_late = loss(None, {"rgb": rgb, "step": 20001, "camera": None}, target)
    if not float(v_late) < float(l1):
        raise AssertionError("late in training the robust mask must drop high-residual pixels")
    return {"alpha": vals, "loss_t0": float(v0), "loss_late": float(v_late), "l1": float(l1)}


def agg_probe(method, device):
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 6, size=(12, 10))
    onehot = torch.from_numpy((labels[None] == np.arange(6)[:, None, None]).astype(np.float32))
    inl = torch.from_numpy((rng.random((12, 10)) > 0.45).astype(np.float32))
    ours = cluster_vote(inl, onehot).numpy()
    ref = np.zeros((12, 10), np.float32)
    for c in range(6):
        m = labels == c
        if m.sum() and inl.numpy()[m].mean() > 0.5:
            ref[m] = 1
    if not np.array_equal(ours, ref):
        raise AssertionError("cluster vote differs from the loop reference (Eq. 3-4)")
    return {"inlier_clusters": int(sum(ref[labels == c].any() for c in range(6)))}


def _synthetic_scene(device, n=60, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * 0.4
    means[:, 2] += 3.0
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means.to(device)),
        "scales": torch.nn.Parameter((torch.rand(n, 3, generator=g) * 0.8 - 3.0).to(device)),
        "quats": torch.nn.Parameter(torch.nn.functional.normalize(
            torch.randn(n, 4, generator=g), dim=-1).to(device)),
        "opacities": torch.nn.Parameter((torch.randn(n, generator=g) * 2 + 1).to(device)),
        "sh0": torch.nn.Parameter((torch.randn(n, 1, 3, generator=g) * 0.5).to(device)),
        "shN": torch.nn.Parameter((torch.randn(n, 15, 3, generator=g) * 0.05).to(device)),
    })
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[30.0, 0, 13.0], [0, 30.0, 10.5], [0, 0, 1]], device=device),
              "width": 27, "height": 21}
    return params, camera


def ubp_kernel_probe(method, device):
    from splatstudio.verification import ProbeUnavailable

    from .renderer import Renderer
    from .ubp import utilization

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer = Renderer(_spec(method, "sls-renderer"))
    params, camera = _synthetic_scene(device)
    g = torch.Generator().manual_seed(5)
    v_render = (torch.randn(21, 27, 3, generator=g) * 1e-3).to(device)
    out = renderer.render(params, camera, sh_degree=1)
    out["info"]["means2d"].retain_grad()
    (out["rgb"] * v_render).sum().backward()
    if not torch.allclose(out["grads"]["rgb"], v_render):
        raise AssertionError("renderer hook must store dL/dI")
    ours = utilization(out["info"], out["sh_colors"], out["grads"]["rgb"])
    ref = torch.zeros_like(ours)
    for i in range(21):
        for j in range(27):
            o = renderer.render(params, camera, sh_degree=1)
            o["info"]["means2d"].retain_grad()
            (o["rgb"][i, j] * v_render[i, j]).sum().backward()
            ref += o["info"]["means2d"].grad[0].square().sum(-1)
    scale = float(ref.abs().max())
    err = float((ours - ref).abs().max())
    if scale == 0 or err > 1e-4 * scale:
        raise AssertionError(f"UBP kernel != per-pixel autograd (err {err}, scale {scale})")
    # The squared-then-summed statistic differs from the square of the summed gradient.
    summed = out["info"]["means2d"].grad[0].square().sum(-1)
    if torch.allclose(summed, ours, rtol=1e-3):
        raise AssertionError("statistic must square per pixel before summing")
    return {"max_abs_err": err, "max_ref": scale, "visible": int((ref > 0).sum())}


def density_probe(method, device):
    from .density import Density

    dspec = _spec(method, "sls-density")
    dens = Density(dspec)
    params, _ = _synthetic_scene("cpu", n=8)
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for p in params.values():
        p.grad = torch.ones_like(p)
    for o in opts.values():
        o.step()
    dens.initialize(params, opts, scene_scale=100.0)       # all Gaussians count as "small"
    with torch.no_grad():
        params["opacities"].fill_(2.0)
    dens.state["grad2d"][:] = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0.0])
    dens.state["count"][:] = 1
    dens.state["sqrgrad"][:] = torch.tensor([0, 1, 1, 0, 1, 1, 1, 1.0])
    dens.refine(params, opts)
    n = len(params["means"])
    expect_n = 8 + 2 - (2 if dens.ubp else 0)
    if n != expect_n:
        raise AssertionError(f"population {n} != {expect_n} (dup 2, UBP prune of 2 unused)")
    for k, p in params.items():
        st = opts[k].state[p]
        if st["exp_avg"].shape != p.shape:
            raise AssertionError("optimizer state misaligned")
    # SH reset (code reset_sh): clamp(max=0.001) + zero moments.
    from .density import reset_sh

    with torch.no_grad():
        params["shN"][0, 0, 0] = 0.5
        params["shN"][0, 0, 1] = -0.5
    reset_sh(params, opts, dspec.config.get("reset_sh_value", 0.001))
    sh = params["shN"]
    if float(sh.max()) > 0.001 + 1e-9 or float(sh[0, 0, 1]) != -0.5:
        raise AssertionError("SH reset must clamp from above only (official clamp(max=0.001))")
    if float(opts["shN"].state[sh]["exp_avg"].abs().max()) != 0:
        raise AssertionError("SH reset must zero Adam moments")
    return {"population": n, "last": dens.last}


def pipeline_probe(method, device):
    """One full training step (renderer -> losses -> density) on a synthetic scene with an
    injected feature / cluster table: the MLP receives gradient, 3DGS parameters do not get
    gradient through the mask, UBP statistics accumulate."""
    from splatstudio.components import build
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, losses = build(method)
    params, camera = _synthetic_scene(device)
    rob = next(c for c in method.components if c.name == "sls-robust")
    mode = rob.config.get("mask_mode", "mlp")
    dim = int(rob.config.get("feature_dim", 1280))
    saved = (Shared.features, Shared.clusters)
    try:
        vm = camera["viewmat"].cpu()
        Shared.features = FrameTable([vm], [torch.randn(dim, 5, 6)], ["synthetic"])
        labels = torch.arange(30).reshape(5, 6) % 4
        Shared.clusters = FrameTable([vm], [(labels[None] == torch.arange(4)[:, None, None])
                                            .float()], ["synthetic"])
        opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
        density.initialize(params, opts, scene_scale=1.0)
        target = torch.rand(21, 27, 3, device=device)
        evidence = {}
        for step in (1, 2):
            out = renderer.render(params, camera, sh_degree=0)
            out["step"] = step
            density.before_backward(params, opts, step, out)
            values = {name: fn(params, out, target) for name, fn in losses}
            sum(values.values()).backward()
            evidence[f"losses_{step}"] = {k: float(v) for k, v in values.items()}
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            density.after_backward(params, opts, step, out)
        rl = next(fn for name, fn in losses if name == "sls-robust")
        if mode == "mlp":
            if "sls-mlp" not in dict(losses):
                raise AssertionError("SLS-mlp needs the sls-mlp loss")
            w = rl.mlp.mlp[0].weight
            if w.grad is None or not torch.isfinite(w.grad).all():
                raise AssertionError("MLP did not receive the hinge gradient")
            evidence["mlp_grad_norm"] = float(w.grad.norm())
        if density.ubp:
            if not float(density.state["sqrgrad"].sum()) > 0:
                raise AssertionError("UBP statistic did not accumulate")
            evidence["sqrgrad_sum"] = float(density.state["sqrgrad"].sum())
        if float(rl.hist.sum()) <= 0:
            raise AssertionError("residual histogram not updated")
        evidence["mode"] = mode
        return evidence
    finally:
        Shared.features, Shared.clusters = saved


def mlp_loss_order_probe(method, device):
    """sls-mlp refuses to run without the step-local values of sls-robust."""
    loss = create_mlp_loss(_spec(method, "sls-mlp"))
    try:
        loss(None, {"step": 3}, None)
    except RuntimeError:
        return {"guard": True}
    raise AssertionError("sls-mlp must fail when sls-robust did not run first")
