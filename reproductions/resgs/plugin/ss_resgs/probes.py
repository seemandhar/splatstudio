"""Mechanism probes for the ResGS port. Each raises when its mechanism is removed or wrong."""

import math
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from .density import DEFAULTS, ResGSDensity, reduce_opacity, residual_split, thresholds
from .link import LINK
from .pyramid import pyr_down, pyr_size, pyramid_target, scaled_camera


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _with_density_config(method, **overrides):
    spec = _spec(method, "density")
    new = replace(spec, config={**spec.config, **overrides})
    return replace(method, components=tuple(new if c.kind == "density" else c
                                            for c in method.components))


def _scene(n, device, seed=0):
    from torch import nn
    g = torch.Generator(device="cpu").manual_seed(seed)
    params = nn.ParameterDict({
        "means": nn.Parameter(torch.randn(n, 3, generator=g).to(device)),
        "scales": nn.Parameter((torch.rand(n, 3, generator=g) * 2 - 3).to(device)),
        "quats": nn.Parameter(torch.randn(n, 4, generator=g).to(device)),
        "opacities": nn.Parameter(torch.randn(n, generator=g).to(device)),
        "sh0": nn.Parameter(torch.randn(n, 1, 3, generator=g).to(device)),
        "shN": nn.Parameter(torch.randn(n, 15, 3, generator=g).to(device)),
    })
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    sum((p * torch.randn_like(p)).sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    return params, opts


def _quat_to_rot(q):
    """Independent wxyz quaternion -> rotation matrix (3DGS utils.general_utils.build_rotation)."""
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def _assert_trainable(params, opts, what):
    """Every field must stay a trainable leaf owned by its optimizer and actually move on a step."""
    for k, p in params.items():
        if not (p.requires_grad and p.is_leaf):
            raise AssertionError(f"{what}: {k} is no longer trainable (requires_grad=False)")
    before = {k: p.detach().clone() for k, p in params.items()}
    sum((p * torch.randn_like(p)).sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    for k, p in params.items():
        moved = (p.detach() != before[k]).reshape(len(p), -1).any(dim=1).float().mean().item()
        if moved < 0.99:
            raise AssertionError(f"{what}: only {moved:.2%} of {k} moved on an optimizer step")


# ---- C1 residual split -------------------------------------------------------------------
def residual_split_probe(method, device):
    cfg = _spec(method, "density").config
    lam, beta = float(cfg["residual_split_scale_div"]), float(cfg["opacity_reduce_weight"])
    n = 4000
    params, opts = _scene(n, device)
    before = {k: v.detach().clone() for k, v in params.items()}
    moments = {k: {m: opts[k].state[p][m].clone() for m in ("exp_avg", "exp_avg_sq")}
               for k, p in params.items()}
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[::2] = True
    sel = torch.where(mask)[0]
    state = {"levels": torch.randint(0, 5, (n,), device=device).float(),
             "grad2d": torch.rand(n, device=device)}
    levels0 = state["levels"].clone()
    torch.manual_seed(0)
    added = residual_split(params, opts, state, mask, lam, beta)
    m = len(sel)
    if added != m or len(params["means"]) != n + m:
        raise AssertionError("residual split must add exactly one child per selected Gaussian")
    o_parent = torch.sigmoid(params["opacities"][sel])
    torch.testing.assert_close(o_parent, beta * torch.sigmoid(before["opacities"][sel]))  # Eq. 5
    torch.testing.assert_close(params["opacities"][:n][~mask], before["opacities"][~mask])
    for k in ("means", "scales", "quats", "sh0", "shN"):  # parent otherwise unchanged
        torch.testing.assert_close(params[k][:n], before[k])
    child = slice(n, n + m)
    torch.testing.assert_close(params["scales"][child].exp(), before["scales"][sel].exp() / lam)
    torch.testing.assert_close(params["opacities"][child], before["opacities"][sel])  # o_j = o_i
    for k in ("quats", "sh0", "shN"):
        torch.testing.assert_close(params[k][child], before[k][sel])
    # mu_j ~ N(mu_i, Sigma_i): whitened offsets S^-1 R^T (mu_j - mu_i) are standard normal.
    R = _quat_to_rot(before["quats"][sel])
    off = params["means"][child] - before["means"][sel]
    z = torch.einsum("nji,nj->ni", R, off) / before["scales"][sel].exp()
    mean, std = z.mean().item(), z.std().item()
    if abs(mean) > 0.05 or abs(std - 1) > 0.05:
        raise AssertionError(f"child offsets are not N(0, Sigma): mean {mean:.3f} std {std:.3f}")
    # Optimizer state: parents keep moments except opacity; children start at zero.
    for k, p in params.items():
        for mom in ("exp_avg", "exp_avg_sq"):
            v = opts[k].state[p][mom]
            if torch.count_nonzero(v[child]):
                raise AssertionError(f"{k}.{mom}: child moments must be zero")
            if k == "opacities":
                if torch.count_nonzero(v[sel]):
                    raise AssertionError("parent opacity moments must be zeroed")
                torch.testing.assert_close(v[:n][~mask], moments[k][mom][~mask])
            else:
                torch.testing.assert_close(v[:n], moments[k][mom])
    torch.testing.assert_close(state["levels"][child], levels0[sel] + 1)
    torch.testing.assert_close(state["levels"][:n], levels0)
    from splatstudio.state import check_state
    check_state(method, params, opts)
    _assert_trainable(params, opts, "after residual split")
    return {"selected": m, "gaussians_after": len(params["means"]), "lambda_s": lam, "beta": beta,
            "whitened_mean": mean, "whitened_std": std}


# ---- C3 varying threshold ----------------------------------------------------------------
def selection_probe(method, device):
    cfg = {**DEFAULTS, **_spec(method, "density").config}
    tau, K = cfg["grad_threshold"], cfg["stage_split"]
    alpha = cfg["stage_hr_factor"] ** (1.0 / K)
    if abs(alpha - 2 ** (1 / 3)) > 1e-12:
        raise AssertionError("alpha must be 2^(1/3) (paper Sec. 4.1)")
    levels = torch.arange(0, 12, device=device).float()
    for k in range(1, 10):
        ref = torch.tensor([tau if lv >= k else tau / alpha ** (k - lv) for lv in range(12)],
                           device=device)  # Eq. 8 as written in the paper
        got = thresholds(levels, k, tau, alpha, True)
        torch.testing.assert_close(got, ref.float())
    # Density selection end to end: grads exactly at the paper threshold pass, slightly below fail.
    d = ResGSDensity(_spec(method, "density"))
    k = 5
    d.schedule.cur_level = k
    lv = torch.tensor([0.0, 2, 4, 5, 7], device=device)
    thr = torch.tensor([tau if v >= k else tau / alpha ** (k - v) for v in lv.tolist()], device=device)
    d.stats = {"levels": torch.cat([lv, lv]), "count": torch.ones(10, device=device),
               "grad2d": torch.cat([thr * 1.0001, thr * 0.999])}
    mask = d.select()
    expect = torch.tensor([True] * 5 + [False] * 5, device=device)
    if not torch.equal(mask, expect):
        raise AssertionError(f"selection mask {mask.tolist()} != {expect.tolist()}")
    # Ablation: the constant threshold selects fewer coarse Gaussians.
    d.c = {**d.c, "varying_threshold": False}
    d.stats["grad2d"] = torch.full((10,), tau * 0.9, device=device)
    if bool(d.select().any()):
        raise AssertionError("constant threshold should reject grads below tau")
    d.c = {**d.c, "varying_threshold": True}
    if not bool(d.select()[:3].all()):
        raise AssertionError("varying threshold must lower tau for coarse levels")
    LINK.attach(None)
    return {"alpha": alpha, "tau": tau, "levels_checked": 12, "substages_checked": 9}


# ---- schedule (C2 stages, C3 substages) --------------------------------------------------
def official_loop(cfg, iterations=30000):
    """Independent transcription of the counters in official train.py:training.

    Returns per-iteration tuples (iteration, render_res_index, stats, densify, cur_level used by
    adjust_gaussian, reduce_opacity, prune_before_reduce).
    """
    change_iter = list(cfg["change_iter"])
    max_level = len(cfg["resolution_scales"])
    warm_up_iter = 0
    update_value_split = cfg["stage_split"]
    cur_level = 1
    last_change_iter = cfg["update_from"]
    next_change_iter = change_iter[0]
    change_split_level_iter = (next_change_iter - last_change_iter) // update_value_split
    cur_resolution = 0
    rows = []
    for iteration in range(1, iterations + 1):
        render_res = cur_resolution
        is_densify_iter = iteration > cfg["update_from"] and iteration % cfg["update_interval"] == 0
        stats = densify = False
        level_used = None
        if iteration < cfg["update_until"] and iteration > cfg["start_stat"] and warm_up_iter == 0:
            stats = True
            if is_densify_iter:
                densify, level_used = True, cur_level
        if iteration in change_iter:
            if cur_resolution + 1 < max_level:
                cur_resolution += 1
            warm_up_iter = cfg["warm_up_iter"]
            if cur_resolution != max_level - 1:
                next_change_iter = change_iter[cur_resolution]
                last_change_iter = change_iter[cur_resolution - 1] + warm_up_iter
            else:
                next_change_iter = cfg["update_until"]
                last_change_iter = change_iter[cur_resolution - 1] + warm_up_iter
            cur_level = update_value_split * cur_resolution + 1
            change_split_level_iter = (next_change_iter - last_change_iter) // update_value_split
        reduce = prune = False
        if cur_resolution != 0 and cfg["use_opacity_reduce"] and iteration < cfg["prune_until"]:
            if iteration % cfg["opacity_reduce_interval"] == 0:
                prune = iteration >= cfg["update_until"]
                reduce = True
        if warm_up_iter != 0:
            warm_up_iter -= 1
        if (change_split_level_iter != 0 and warm_up_iter == 0
                and iteration > cfg["update_from"] and iteration < cfg["update_until"]):
            change_split_level_iter -= 1
            if change_split_level_iter == 0:
                cur_level += 1
                change_split_level_iter = (next_change_iter - last_change_iter) // update_value_split
        rows.append((iteration, render_res, stats, densify, level_used, reduce, prune))
    return rows


def plugin_loop(cfg, iterations=30000):
    from .schedule import ResGSSchedule
    s = ResGSSchedule(cfg)
    rows = []
    for t in range(1, iterations + 1):
        render_res = s.res_index
        stats, densify = s.stats_active(t), s.densify_step(t)
        level_used = s.cur_level if densify else None
        s.change_stage(t)
        reduce, prune = s.opacity_reduce_step(t)
        s.end_iteration(t)
        rows.append((t, render_res, stats, densify, level_used, reduce, prune))
    return rows


def schedule_probe(method, device):
    cfg = {**DEFAULTS, **_spec(method, "density").config}
    ref, got = official_loop(cfg), plugin_loop(cfg)
    for a, b in zip(ref, got):
        if a != b:
            raise AssertionError(f"schedule differs from official loop at {a} vs {b}")
    # Hand-checked anchors for the released defaults.
    res = {t: r for t, r, *_ in got}
    if not (res[2500] == 0 and res[2501] == 1 and res[6000] == 1 and res[6001] == 2):
        raise AssertionError("stage boundaries must be 2500 / 6000 (paper Sec. 4.1)")
    dens = [(t, lv) for t, _, _, d, lv, _, _ in got if d]
    steps = [t for t, _ in dens]
    if steps[0] != 800 or steps[-1] != 11900:
        raise AssertionError("densification must run from 800 to 11900")
    if any(2500 < t < 3000 or 6000 < t < 6500 for t in steps):
        raise AssertionError("no densification during the 500-iteration warm-ups")
    levels = {}
    for t, lv in dens:
        levels.setdefault(lv, []).append(t)
    if sorted(levels) != list(range(1, 10)):
        raise AssertionError(f"substage levels used must be 1..9, got {sorted(levels)}")
    reduces = [t for t, *_, r, _ in got if r]
    if reduces[0] != 2500 or reduces[-1] != 24500 or len(reduces) != 45:
        raise AssertionError("opacity reduction must run every 500 from 2500 to 24500")
    prunes = [t for t, *_, p in got if p]
    if prunes[0] != 12000:
        raise AssertionError("post-densification pruning starts at 12000")
    # Ablation: no warm-up changes the densify steps -> the anchors above would fail.
    return {"densify_steps": len(steps),
            "level_first_step": {lv: ts[0] for lv, ts in sorted(levels.items())},
            "stage_starts": [1, 2501, 6001], "opacity_reductions": len(reduces)}


# ---- C2 pyramid --------------------------------------------------------------------------
def pyramid_probe(method, device):
    from splatstudio.verification import ProbeUnavailable
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ProbeUnavailable("cv2 is required for the pyrDown reference") from exc
    g = torch.Generator().manual_seed(1)
    out = {}
    for h, w in ((37, 53), (64, 48), (839, 1297)):
        img = torch.rand(h, w, 3, generator=g)
        ref = cv2.pyrDown(img.numpy())
        got = pyr_down(img.to(device)).cpu()
        if tuple(got.shape) != ref.shape or tuple(got.shape[:2]) != pyr_size(h, w, 1):
            raise AssertionError(f"pyrDown size {tuple(got.shape)} != {ref.shape}")
        err = float(np.abs(got.numpy() - ref).max())
        if err > 1e-5:
            raise AssertionError(f"pyrDown differs from cv2 by {err}")
        ref2 = cv2.pyrDown(ref)
        err2 = float(np.abs(pyramid_target(img.to(device), 2).cpu().numpy() - ref2).max())
        if err2 > 1e-5:
            raise AssertionError("two-level pyramid differs from cv2")
        out[f"{h}x{w}"] = max(err, err2)
    # The loss supervises against the pyramid image at the rendered level.
    from splatstudio.components import build
    _, _, losses = build(method)
    loss = dict(losses)["photometric"]
    target = torch.rand(64, 48, 3, generator=g).to(device)
    small = pyramid_target(target, 2).clone().requires_grad_(True)
    v = loss({}, {"rgb": small, "pyramid_level": 2}, target)
    if v.item() > 1e-5:
        raise AssertionError("loss at the exact pyramid target must be ~0")
    try:
        loss({}, {"rgb": small, "pyramid_level": 1}, target)
    except RuntimeError:
        pass
    else:
        raise AssertionError("loss must reject a render/pyramid size mismatch")
    LINK.attach(None)
    return {"max_abs_error_vs_cv2": out}


def render_scale_probe(method, device):
    from gsplat import rasterization
    from splatstudio.components import build
    from splatstudio.state import sample_scene
    from splatstudio.verification import ProbeUnavailable
    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    params, opts = sample_scene(method, device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[60.0, 0, 26.5], [0, 60.0, 18.5], [0, 0, 1]], device=device),
              "width": 53, "height": 37}
    density.initialize(params, opts)
    sizes = {}
    for level in (2, 1, 0):
        density.schedule.res_index = density.schedule.resolution_scales.index(level)
        out = renderer.render(params, camera, sh_degree=0)
        h, w = pyr_size(37, 53, level)
        if tuple(out["rgb"].shape[:2]) != (h, w) or out["pyramid_level"] != level:
            raise AssertionError(f"training render at level {level} has size {out['rgb'].shape}")
        cam = scaled_camera(camera, level)
        K = camera["K"].clone()
        K[0] *= w / 53
        K[1] *= h / 37
        ref, alpha, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                      params["opacities"].sigmoid(),
                                      torch.cat((params["sh0"], params["shN"]), 1),
                                      camera["viewmat"][None], K[None], w, h, sh_degree=0)
        torch.testing.assert_close(cam["K"], K)
        torch.testing.assert_close(out["rgb"], ref[0] + (1 - alpha[0]))
        sizes[level] = [h, w]
    density.schedule.res_index = 0
    with torch.no_grad():
        full = renderer.render(params, camera, sh_degree=0)
    if tuple(full["rgb"].shape[:2]) != (37, 53):
        raise AssertionError("evaluation (no_grad) render must be full resolution")
    LINK.attach(None)
    return {"training_sizes": sizes, "eval_size": [37, 53]}


# ---- AbsGS opacity reduction -------------------------------------------------------------
def opacity_reduce_probe(method, device):
    cfg = {**DEFAULTS, **_spec(method, "density").config}
    params, opts = _scene(50, device)
    before = params["opacities"].detach().clone()
    reduce_opacity(params, opts, cfg["opacity_reduce_factor"])
    torch.testing.assert_close(torch.sigmoid(params["opacities"]),
                               torch.sigmoid(before) * cfg["opacity_reduce_factor"])
    st = opts["opacities"].state[params["opacities"]]
    if torch.count_nonzero(st["exp_avg"]) or torch.count_nonzero(st["exp_avg_sq"]):
        raise AssertionError("opacity moments must be zeroed by reduce_opacity")
    _assert_trainable(params, opts, "after reduce_opacity")
    # Controller: at 12000 prune (< min_opacity) happens before the x0.8 reduction.
    from .schedule import ResGSSchedule
    d = ResGSDensity(_spec(method, "density"))
    p2, o2 = _scene(3, device, seed=3)
    with torch.no_grad():
        p2["opacities"].copy_(torch.logit(torch.tensor([0.004, 0.006, 0.5], device=device)))
    d.initialize(p2, o2)
    s = ResGSSchedule(d.c)
    for t in range(1, 12000):
        s.change_stage(t)
        s.end_iteration(t)
    d.schedule = s
    d.after_backward(p2, o2, 12000, {"info": {}})
    got = torch.sigmoid(p2["opacities"]).tolist()
    want = [0.006 * 0.8, 0.5 * 0.8]
    if len(got) != 2 or any(abs(a - b) > 1e-6 for a, b in zip(got, want)):
        raise AssertionError(f"prune-then-reduce at 12000 gave {got}, expected {want}")
    LINK.attach(None)
    return {"factor": cfg["opacity_reduce_factor"], "after_12000": got}


# ---- integration -------------------------------------------------------------------------
def cycle_probe(method, device):
    """Native runtime with a compressed schedule: pyramid 2->1->0, levels grow, eval full-res."""
    from splatstudio.demo import create_dataset
    from splatstudio.runtime import train
    from splatstudio.verification import ProbeUnavailable
    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    m = _with_density_config(method, change_iter=[150, 300], start_stat=20, update_from=40,
                             update_interval=10, update_until=600, warm_up_iter=20,
                             opacity_reduce_interval=50, prune_until=700,
                             grad_threshold=0.0002)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        create_dataset(data, size=128)
        result = train(m, data, Path(tmp) / "run", steps=800, num_gaussians=2000,
                       max_gaussians=500000, device=device, log_every=100)
        hist = LINK.density.history
        top = LINK.density.max_gaussian_level
        LINK.attach(None)
    pyr = [h["pyramid_level"] for h in hist]
    if sorted(set(pyr), reverse=True) != [2, 1, 0] or any(a < b for a, b in zip(pyr, pyr[1:])):
        raise AssertionError(f"pyramid levels during densification {sorted(set(pyr))}")
    if sum(h["split"] for h in hist) == 0 or top < 2:
        raise AssertionError("residual split never produced finer Gaussians")
    if not math.isfinite(result["metrics"]["psnr"]):
        raise AssertionError("non-finite PSNR")
    return {"densify_events": len(hist), "residual_splits": sum(h["split"] for h in hist),
            "max_level": top, "psnr": result["metrics"]["psnr"],
            "gaussians": result["resources"]["gaussians"]}
