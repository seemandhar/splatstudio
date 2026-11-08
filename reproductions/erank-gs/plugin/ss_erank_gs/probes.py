"""Mechanism probes; each fails when its mechanism is removed or altered."""

import math

import numpy as np
import torch

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import Density, screen_stats
from .losses import create_erank_loss, effective_rank


def _component(method, name):
    return next(c for c in method.components if c.name == name)


def _np_loss(s, lam, thin, eps):
    """Independent numpy form of the official erank + thin loss (float64)."""
    s = np.asarray(s, np.float64)
    d = s ** 2
    q = d / d.sum(1, keepdims=True)
    er = np.exp(-(q * np.log(q)).sum(1))
    return lam * np.maximum(-np.log(er - 1 + eps), 0).mean() + thin * np.sort(s, 1)[:, 0].mean()


def erank_formula(method, device):
    spec = _component(method, "erank")
    cfg = spec.config
    fn = create_erank_loss(spec)
    s = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1e-4], [1.0, 1e-3, 1e-3], [0.5, 0.2, 0.05]],
                     dtype=torch.float64)
    er = effective_rank(s)
    if abs(er[0] - 3) > 1e-9 or abs(er[1] - 2) > 1e-4 or not 1 < er[2] < 1.001:
        raise AssertionError(f"effective rank wrong: {er.tolist()}")
    log_s = s.log().clone().requires_grad_(True)
    value = fn({"scales": log_s}, {"step": cfg["start"]}, None)
    expected = _np_loss(s.numpy(), cfg["erank_lambda"], cfg["thin_lambda"], cfg["eps"])
    if abs(value.item() - expected) > 1e-9 * max(1, abs(expected)):
        raise AssertionError(f"loss {value.item()} != independent {expected}")
    before = fn({"scales": log_s}, {"step": cfg["start"] - 1}, None)
    if before.item() != 0.0:
        raise AssertionError("erank loss active before its start step")
    (grad,) = torch.autograd.grad(value, log_s)
    # needle (row 2): descending the loss must enlarge its two minor axes
    if not (grad[2, 1] < 0 and grad[2, 2] < 0):
        raise AssertionError(f"needle gradient does not widen minor axes: {grad[2].tolist()}")
    # disk (row 1): erank ~ 2 -> barrier inactive except the thin term on s3
    barrier_disk = cfg["erank_lambda"] * max(-math.log(er[1].item() - 1 + cfg["eps"]), 0)
    if barrier_disk > 1e-3:
        raise AssertionError("disk is penalised by the barrier")
    return {"erank": er.tolist(), "loss": value.item(), "independent": expected,
            "needle_grad": grad[2].tolist()}


def _fake_state(n, scales, opac, device="cpu"):
    params = {
        "means": torch.nn.Parameter(torch.zeros(n, 3, device=device)),
        "scales": torch.nn.Parameter(torch.log(torch.as_tensor(scales, dtype=torch.float32,
                                                               device=device))),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * n, device=device)),
        "opacities": torch.nn.Parameter(torch.logit(torch.as_tensor(opac, dtype=torch.float32,
                                                                    device=device))),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3, device=device)),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    }
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    for v in params.values():
        v.grad = torch.zeros_like(v)
    for o in optimizers.values():
        o.step()
    return params, optimizers


class _Spec:
    def __init__(self, config):
        self.config = config


def _check_leaves(params, optimizers):
    for k, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.requires_grad and p.is_leaf):
            raise AssertionError(f"field {k} is no longer a trainable leaf")
        if optimizers[k].param_groups[0]["params"][0] is not p:
            raise AssertionError(f"optimizer for {k} references a stale parameter")


def _run_densify(cfg, grads, grads_abs, scales, opac):
    d = Density(_Spec(cfg))
    params, optimizers = _fake_state(len(grads), scales, opac)
    d.initialize(params, optimizers, scene_scale=1.0)
    d.stats = {"grad_accum": torch.tensor(grads), "grad_accum_abs": torch.tensor(grads_abs),
               "denom": torch.ones(len(grads))}
    d.densify_and_prune(params, optimizers, world_prune=True)
    _check_leaves(params, optimizers)
    return d, params


def adc_abs_metric(method, device):
    cfg = dict(_component(method, "erank-density").config)
    small = [[0.001] * 3] * 3
    grads = [1e-5, 1e-3, 0.0]          # #0: cancelling signed gradient, #1: classic densify
    grads_abs = [10.0, 5.0, 0.0]       # ratio = 1/3 -> Q = quantile(abs, 2/3) = 6.67
    _, p_fixed = _run_densify(cfg, grads, grads_abs, small, [0.5] * 3)
    _, p_plain = _run_densify({**cfg, "abs_gradient": False}, grads, grads_abs, small, [0.5] * 3)
    n_fixed, n_plain = len(p_fixed["means"]), len(p_plain["means"])
    if n_fixed != 5 or n_plain != 4:
        raise AssertionError(f"abs criterion not effective: fixed {n_fixed}, plain {n_plain}")
    return {"fixed_adc_count": n_fixed, "inria_adc_count": n_plain}


def density_recipe(method, device):
    cfg = dict(_component(method, "erank-density").config)
    out = {}
    # split: big Gaussian (0.5 > 0.01 extent) with high gradient -> 2 children at s/1.6
    _, p = _run_densify(cfg, [1e-3, 0.0], [1.0, 0.0], [[0.05, 0.02, 0.01], [0.001] * 3],
                        [0.5, 0.5])
    if len(p["means"]) != 3:
        raise AssertionError(f"split count {len(p['means'])} != 3")
    child = p["scales"].exp()[-2:]
    torch.testing.assert_close(child, torch.tensor([[0.05, 0.02, 0.01]] * 2) / 1.6)
    # clone resampling moves the copy iff clone_resample
    _, p = _run_densify(cfg, [1e-3], [1.0], [[0.001] * 3], [0.5])
    moved = bool((p["means"][1] != p["means"][0]).any())
    if moved != bool(cfg["clone_resample"]):
        raise AssertionError("clone resampling does not follow config")
    # prune: opacity below min_opacity and world-size (> 0.1 extent) Gaussians removed
    mo = cfg["min_opacity"]
    # (ratio = 0 -> Q = max(abs): only #0 qualifies; its clone inherits the low opacity)
    _, p = _run_densify(cfg, [0.0] * 3, [3.0, 1.0, 2.0],
                        [[0.001] * 3, [0.001] * 3, [0.2, 0.001, 0.001]], [mo * 0.5, 0.5, 0.5])
    if len(p["means"]) != 1:
        raise AssertionError(f"prune kept {len(p['means'])} != 1")
    # reset schedule: at step 3000 opacities clamp to 0.01
    d = Density(_Spec(cfg))
    params, optimizers = _fake_state(2, [[0.001] * 3] * 2, [0.9, 0.005])
    d.initialize(params, optimizers, 1.0)
    info = _fake_info(2)
    d.after_backward(params, optimizers, cfg["opacity_reset_interval"] + 1, {"info": info})
    if params["opacities"].sigmoid().max() < 0.5:
        raise AssertionError("opacity reset off-schedule")
    d.cfg["densify_from_iter"] = 10 ** 9  # isolate the reset
    d.after_backward(params, optimizers, cfg["opacity_reset_interval"], {"info": info})
    op = params["opacities"].sigmoid()
    if abs(op[0].item() - 0.01) > 1e-5 or abs(op[1].item() - 0.005) > 1e-6:
        raise AssertionError(f"opacity reset wrong: {op.tolist()}")
    _check_leaves(params, optimizers)
    out.update(split_child_scales=child.tolist(), clone_moved=moved, reset=op.tolist(),
               min_opacity=mo, defaults_match=cfg == {**DENSITY_DEFAULTS, **cfg})
    return out


def _fake_info(n):
    m2d = torch.zeros(1, n, 2, requires_grad=True)
    m2d.grad = torch.zeros(1, n, 2)
    m2d.absgrad = torch.zeros(1, n, 2)
    return {"means2d": m2d, "radii": torch.ones(1, n, 2), "width": 10, "height": 10}


def absgrad_units(method, device):
    from splatstudio.registry import resolve

    spec = _component(method, "erank-renderer")
    renderer = resolve(spec.factory)(spec)
    torch.manual_seed(0)
    n, W, H = 3, 16, 12
    params = {
        "means": torch.tensor([[0.0, 0.0, 3.0], [0.3, 0.1, 3.5], [-0.2, -0.1, 2.5]], device=device),
        "scales": torch.log(torch.tensor([[0.3, 0.1, 0.1], [0.2, 0.2, 0.2], [0.1, 0.25, 0.1]],
                                         device=device)),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1),
        "opacities": torch.tensor([0.5, 1.0, 0.0], device=device),
        "sh0": torch.rand(n, 1, 3, device=device),
        "shN": torch.zeros(n, 15, 3, device=device),
    }
    params = {k: v.requires_grad_(True) for k, v in params.items()}
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]], device=device),
           "width": W, "height": H}
    w = torch.rand(H, W, 3, device=device)
    out = renderer.render(params, cam, sh_degree=0)
    out["info"]["means2d"].retain_grad()
    (out["rgb"] * w).sum().backward()
    g_l2, a_l1, vis = screen_stats(out["info"])
    # independent: one backward per pixel, sum |gx|+|gy| of each pixel's contribution (NDC)
    ref = torch.zeros(n, device=device)
    scale = torch.tensor([W / 2, H / 2], device=device)
    for y in range(H):
        for x in range(W):
            o = renderer.render(params, cam, sh_degree=0)
            m2d = o["info"]["means2d"]
            m2d.retain_grad()
            (o["rgb"][y, x] * w[y, x]).sum().backward()
            ref += (m2d.grad[0] * scale).abs().sum(-1)
    torch.testing.assert_close(a_l1, ref, rtol=1e-3, atol=1e-6)
    if not (a_l1[vis] + 1e-6 >= g_l2[vis]).all() or a_l1.max() <= 0:
        raise AssertionError("abs statistic smaller than signed norm or inactive")
    return {"abs_l1": a_l1.tolist(), "reference": ref.tolist(), "signed_l2": g_l2.tolist()}
