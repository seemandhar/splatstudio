"""GaussianSpa mechanism probes. Each fails if its mechanism is removed or wrong.

References are written independently of `spa.py` (numpy argsort ranks instead of torch sort
thresholds, hand-derived gradients).
"""

from dataclasses import replace

import numpy as np
import torch
from splatstudio.registry import resolve
from torch import nn

from .ms_probes import _camera, _require_cuda, _scene


def _component(method, kind, name=None, **overrides):
    spec = next(c for c in method.components if c.kind == kind and (name is None or
                                                                       c.name == name))
    if overrides:
        spec = replace(spec, config={**spec.config, **overrides})
    return spec, resolve(spec.factory)(spec)


def _ref_keep(values, ratio, ties_keep_threshold):
    """numpy reference of the official sort-threshold rules (no ties in random inputs):
    opacity (`v > thr`): keep ranks >= int(rN); importance (`imp < thr` zeroed): keep ranks
    >= int(rN) - 1."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    index = int(ratio * n)
    ranks = np.empty(n, dtype=np.int64)
    ranks[np.argsort(v, kind="stable")] = np.arange(n)
    lo = index - 1 if ties_keep_threshold else index
    return ranks >= lo


def _random_scene(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, 1, generator=g) * 1.5 + 2.5
    xy = (torch.rand(n, 2, generator=g) * 2 - 1) * 0.6 * z
    means = torch.cat([xy * 0.5, z], 1)
    scales = torch.rand(n, 3, generator=g) * 0.1 + 0.05
    opac = torch.rand(n, generator=g) * 0.9 + 0.05
    return _scene(means, scales, opac, device=device)


def _cams(device):
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.1, 0.05, 0.0])
    return [_camera(64, 48, 40.0, device=device), _camera(64, 48, 40.0, c2w, device=device)]


def penalty_probe(method, device):
    spec, loss = _component(method, "loss", "spa-penalty")
    cfg = spec.config
    g = torch.Generator().manual_seed(1)
    n = 64
    logit = nn.Parameter(torch.randn(n, generator=g))
    z = torch.rand(n, generator=g) * (torch.rand(n, generator=g) > 0.5)
    u = torch.randn(n, generator=g) * 0.1
    params = {"opacities": logit, "spa_z": z, "spa_u": u}
    start, stop, every = cfg["spa_start"], cfg["spa_stop"], cfg["spa_interval"]
    on = start + every
    value = loss(params, {"step": on}, None)
    (grad,) = torch.autograd.grad(value, logit)
    a = torch.sigmoid(logit.detach()).double()
    r = a - z.double() + u.double()
    ref_value = 0.5 * cfg["delta"] * float((r * r).sum())
    ref_grad = cfg["delta"] * r * a * (1 - a)  # d/dlogit of delta/2 r^2, da/dlogit = a(1-a)
    if abs(float(value) - ref_value) > 1e-6 * max(ref_value, 1e-12) + 1e-12:
        raise AssertionError(f"Penalty value {float(value)} vs reference {ref_value}")
    if (grad.double() - ref_grad).abs().max() > 1e-6 * ref_grad.abs().max() + 1e-12:
        raise AssertionError("Penalty gradient differs from delta (a - z + u) a (1 - a)")
    if ref_value <= 0 or grad.abs().max() == 0:
        raise AssertionError("Penalty inactive at an ADMM step")
    off_steps = [start, on + 1, on + every // 2, stop + every, 1000]
    if not cfg["stop_inclusive"]:
        off_steps.append(stop)
    for s in off_steps:
        v = loss(params, {"step": s}, None)
        if float(v) != 0.0 or not v.requires_grad:
            raise AssertionError(f"Penalty must be zero and attached at step {s}")
    at_stop = float(loss(params, {"step": stop}, None))
    if cfg["stop_inclusive"] and at_stop == 0:
        raise AssertionError("Opacity script applies the penalty at the stop iteration")
    return {"value": float(value), "reference": ref_value,
            "max_grad_error": float((grad.double() - ref_grad).abs().max()),
            "off_steps_zero": off_steps, "active_at_stop": at_stop != 0}


def projection_probe(method, device):
    from . import spa

    dspec, _ = _component(method, "density")
    ratio = dspec.config["prune_ratio2"]
    g = torch.Generator().manual_seed(2)
    n = 1000
    a = torch.rand(n, generator=g)
    u = torch.randn(n, generator=g) * 0.2
    imp = torch.rand(n, generator=g) * 10
    results = {}
    for crit, importance in (("opacity", None), ("importance", imp)):
        z, u_new = spa.admm_step(a, u, ratio, importance)
        v = (a + u).numpy()
        if importance is None:
            keep = _ref_keep(v, ratio, ties_keep_threshold=False)
        else:
            keep = _ref_keep(imp.numpy(), ratio, ties_keep_threshold=True)
        ref_z = np.where(keep, v, 0.0)
        err = float(np.abs(z.numpy() - ref_z).max())
        err_u = float(np.abs(u_new.numpy() - (u.numpy() + a.numpy() - ref_z)).max())
        if err > 1e-6 or err_u > 1e-6:
            raise AssertionError(f"{crit} projection/multiplier differ: z {err}, u {err_u}")
        nnz = int((z != 0).sum())
        expected = n - int(ratio * n) + (1 if importance is not None else 0)
        if nnz != expected:
            raise AssertionError(f"{crit}: {nnz} non-zeros, expected {expected}")
        results[f"{crit}_nonzeros"] = nnz
        results[f"{crit}_max_error"] = err
    # the opacity projection ranks a + u, not a: the kept sets must differ for this u
    keep_a = spa.project_opacity(a, ratio) != 0
    keep_au = spa.project_opacity(a + u, ratio) != 0
    if bool((keep_a == keep_au).all()):
        raise AssertionError("Probe cannot distinguish prox(a + u) from prox(a)")
    results["ratio"] = ratio
    results["kept_set_changes_with_u"] = int((keep_a != keep_au).sum())
    return results


def _ref_project(a, u, ratio, criterion, imp):
    v = (a + u).cpu().numpy().astype(np.float64)
    if criterion == "opacity":
        keep = _ref_keep(v, ratio, ties_keep_threshold=False)
    else:
        keep = _ref_keep(imp.cpu().numpy(), ratio, ties_keep_threshold=True)
    return np.where(keep, v, 0.0)


def admm_cycle_probe(method, device):
    _require_cuda()
    _, density = _component(method, "density")
    cfg = density.cfg
    params, opts = _random_scene(60, device, seed=3)
    density.initialize(params, opts, scene_scale=1.0)
    density.set_training_cameras(_cams(device))
    start, every, stop = cfg["spa_start"], cfg["spa_interval"], cfg["spa_stop"]
    crit, ratio = cfg["criterion"], cfg["prune_ratio2"]
    g = torch.Generator(device="cpu").manual_seed(4)

    def perturb():
        params["opacities"].data += (torch.randn(60, generator=g) * 0.3).to(device)

    def imp():
        return density._importance(params) if crit == "importance" else None

    def state():
        return params["spa_z"].detach().clone(), params["spa_u"].detach().clone()

    a = params["opacities"].detach().sigmoid()
    ref = _ref_project(a, torch.zeros_like(a), ratio, crit, imp())
    density.after_backward(params, opts, start, {})
    z0, u0 = state()
    if np.abs(z0.cpu().numpy() - ref).max() > 1e-6 or u0.abs().max() != 0:
        raise AssertionError("Init step must set z = prox(a) and keep u = 0")
    perturb()
    density.after_backward(params, opts, start + every // 2, {})
    if not (torch.equal(state()[0], z0) and torch.equal(state()[1], u0)):
        raise AssertionError("z/u changed off the sparsifying interval")
    errs = []
    for k in (1, 2):
        perturb()
        a = params["opacities"].detach().sigmoid()
        z_prev, u_prev = state()
        ref_z = _ref_project(a, u_prev, ratio, crit, imp())
        ref_u = u_prev.cpu().numpy() + a.cpu().numpy() - ref_z
        density.after_backward(params, opts, start + k * every, {})
        z1, u1 = state()
        errs.append(max(np.abs(z1.cpu().numpy() - ref_z).max(), np.abs(u1.cpu().numpy() - ref_u)
                        .max()))
    if max(errs) > 1e-5 or u1.abs().max() == 0:
        raise AssertionError(f"ADMM update differs from z=prox(a+u), u+=a-z ({errs})")
    # resume: a fresh density on a copy of the checkpointed params gives the same next update
    perturb()
    saved = {k: v.detach().clone() for k, v in params.items()}
    _, fresh = _component(method, "density")
    p2 = nn.ParameterDict({k: nn.Parameter(v.clone()) for k, v in saved.items()})
    o2 = {k: torch.optim.Adam([p], lr=0.0) for k, p in p2.items()}
    fresh.initialize(p2, o2, scene_scale=1.0)
    fresh.set_training_cameras(_cams(device))
    step = start + 3 * every
    density.after_backward(params, opts, step, {})
    fresh.after_backward(p2, o2, step, {})
    for f in ("spa_z", "spa_u"):
        if not torch.equal(params[f], p2[f]):
            raise AssertionError(f"Resumed {f} differs from the uninterrupted run")
    z_last, u_last = state()
    density.after_backward(params, opts, stop + every, {})
    if not (torch.equal(state()[0], z_last) and torch.equal(state()[1], u_last)):
        raise AssertionError("ADMM must be idle after the stop iteration")
    return {"criterion": crit, "max_update_error": float(max(errs)),
            "z_nonzero": int((z_last != 0).sum()), "n": 60, "resume_bit_exact": True}


def hard_prune_probe(method, device):
    _require_cuda()
    _, density = _component(method, "density")
    cfg = density.cfg
    n = 80
    params, opts = _random_scene(n, device, seed=5)
    density.initialize(params, opts, scene_scale=1.0)
    density.set_training_cameras(_cams(device))
    for p in params.values():  # allocate Adam moments
        p.grad = torch.randn_like(p)
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    ratio = cfg["prune_ratio2"]
    if cfg["criterion"] == "opacity":
        rank = params["opacities"].detach().sigmoid()
        keep = ~torch.zeros(n, dtype=torch.bool)
        order = np.argsort(-rank.cpu().numpy(), kind="stable")
        keep[:] = False
        keep[order[: int((1 - ratio) * n)]] = True
    else:
        rank = density._importance(params)
        r = rank.cpu().numpy()
        thr = np.sort(r)[int(ratio * n) - 1]
        keep = torch.as_tensor(r > thr)
    keep = keep.to(device)
    ref_means = params["means"].detach()[keep]
    p_op = params["opacities"]
    ref_moment = opts["opacities"].state[p_op]["exp_avg"][keep].clone()
    density.after_backward(params, opts, cfg["spa_stop"], {})
    got = params["means"].detach()
    if len(got) != int(keep.sum()) or not torch.equal(got, ref_means):
        raise AssertionError(f"Hard prune kept {len(got)}, expected {int(keep.sum())} or a "
                             "different set")
    moment = opts["opacities"].state[params["opacities"]]["exp_avg"]
    if not torch.equal(moment, ref_moment):
        raise AssertionError("Survivors must keep their Adam moments (official prune_points)")
    if len(got) >= n:
        raise AssertionError("Nothing was pruned")
    return {"criterion": cfg["criterion"], "n_before": n, "n_after": len(got),
            "expected": int(keep.sum())}


def sampling_probe(method, device):
    _require_cuda()
    _, density = _component(method, "density", prune_ratio1=0.5)
    gx, gy = torch.meshgrid(torch.tensor([-0.6, -0.2, 0.2, 0.6]), torch.tensor([-0.4, 0.0, 0.4]),
                            indexing="ij")
    vis = torch.stack([gx.flatten(), gy.flatten(), 3.0 + 0.01 * torch.arange(12.0)], 1)
    wall = torch.tensor([[0.0, 0.0, 6.0]])
    g = torch.Generator().manual_seed(6)
    hidden = torch.cat([(torch.rand(27, 2, generator=g) - 0.5) * 0.3,
                        torch.full((27, 1), 8.0)], 1)
    means = torch.cat([vis, wall, hidden])
    scales = torch.cat([torch.full((12, 3), 0.25), torch.full((1, 3), 0.6),
                        torch.full((27, 3), 0.03)])
    opac = torch.cat([torch.full((12,), 0.7), torch.tensor([0.99]), torch.full((27,), 0.9)])
    params, opts = _scene(means, scales, opac, device=device)
    params["spa_z"].data.fill_(0.3)
    params["spa_u"].data.fill_(0.2)
    density.initialize(params, opts, scene_scale=1.0)
    density.set_training_cameras(_cams(device))
    imp = density._importance(params)
    nnz = int((imp != 0).sum())
    n = len(means)
    if (imp[13:] != 0).any() or (imp[:12] == 0).any():
        raise AssertionError("Probe scene: only the 12 front Gaussians may have importance")
    density.after_backward(params, opts, density.cfg["restart_step"], {})
    kept = params["means"].detach()
    expected = int(n * (1 - 0.5))
    hidden_kept = int((kept[:, 2] > 5.0).sum())  # wall + hidden: all zero importance
    ms_count = int(n * 0.5 * (nnz / n))
    if len(kept) != expected or expected == ms_count:
        raise AssertionError(f"Kept {len(kept)}, expected int(N(1-r1)) = {expected}")
    if hidden_kept < expected - nnz or hidden_kept == 0:
        raise AssertionError("Zero-importance Gaussians must be sampleable (p ∝ imp + 1)")
    if (params["opacities"].sigmoid() - 0.1).abs().max() > 1e-5:
        raise AssertionError("Sampled Gaussians must be reinitialized (opacity 0.1)")
    if params["spa_z"].abs().max() != 0 or params["spa_u"].abs().max() != 0:
        raise AssertionError("Reinit must reset the ADMM fields")
    return {"n": n, "intersected": nnz, "kept": len(kept), "mini_splatting_count": ms_count,
            "zero_importance_kept": hidden_kept}
