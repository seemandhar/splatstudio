"""Mechanism probes for the Group Training port. Each raises AssertionError when the mechanism
is removed or wrong. References are written independently of density.py."""

import math

import numpy as np
import torch
from splatstudio.components import build
from splatstudio.state import check_state, initialize

from .density import GroupTrainingDensity, under_training_mask


def _density_spec(method):
    return next(c for c in method.components if c.kind == "density")


def _density(method, **overrides):
    from dataclasses import replace

    spec = _density_spec(method)
    return GroupTrainingDensity(replace(spec, config={**spec.config, **overrides}))


def _scene(method, n, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    points = (torch.rand(n, 3, generator=g) - 0.5) * 2
    points[:, 2] += 4
    colors = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, optimizers = initialize(method, points.to(device), colors.to(device))
    with torch.no_grad():
        params["opacities"].copy_(torch.randn(n, generator=g).to(device) * 2)
    return params, optimizers


def _adam_steps(params, optimizers, k, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(k):
        loss = sum(
            (p * torch.randn(p.shape, generator=g).to(p)).sum()
            for p in params.values()
            if p.requires_grad
        )
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)


def ops_sampling(method, device):
    n, trials = 400, 300
    utr = float(_density_spec(method).config["utr"])
    k = int(n * utr)
    o = torch.full((n,), 0.05)
    o[: n // 2] = 0.95
    logits = torch.logit(o).to(device)
    freq = torch.zeros(n, device=device)
    for _ in range(trials):
        mask = under_training_mask(logits, utr, _density_spec(method).config["grouping_method"])
        if int(mask.sum()) != k:
            raise AssertionError(f"OPS must select exactly int(N*UTR)={k} distinct Gaussians")
        freq += mask.float()
    freq = (freq / trials).cpu()
    # independent reference: numpy successive sampling without replacement, p = o / sum o
    rng = np.random.default_rng(0)
    p = o.double().numpy()
    p = p / p.sum()
    ref = np.zeros(n)
    for _ in range(trials):
        ref[rng.choice(n, size=k, replace=False, p=p)] += 1
    ref /= trials
    hi, lo = float(freq[: n // 2].mean()), float(freq[n // 2 :].mean())
    rhi, rlo = float(ref[: n // 2].mean()), float(ref[n // 2 :].mean())
    if abs(hi - rhi) > 0.02 or abs(lo - rlo) > 0.02:
        raise AssertionError(f"OPS inclusion ({hi:.3f},{lo:.3f}) != reference ({rhi:.3f},{rlo:.3f})")
    if hi - lo < 0.5:
        raise AssertionError("OPS does not prioritise high-opacity Gaussians (random sampling?)")
    return {"k": k, "inclusion_high": hi, "inclusion_low": lo,
            "reference_high": rhi, "reference_low": rlo, "random_would_be": utr}


def schedule(method, device):
    params, optimizers = _scene(method, 40, device)
    d = _density(method)
    d.initialize(params, optimizers, 1.0)
    frozen_at_render, events = [], {}
    total = int(d.config["total_steps"])
    for step in range(1, total + 1):
        if not params["shN"].requires_grad:
            frozen_at_render.append(step)
        before = len(d.log)
        d.group_step(params, optimizers, step)
        for it, kind, _, _ in d.log[before:]:
            events[it] = kind
    c = d.config
    expect = {g: "regroup" for g in range(c["grouping_from_iter"], c["grouping_until_iter"] + 1,
                                          c["grouping_interval"])}
    expect[c["global_densify_iter"]] = "merge"
    expect[c["grouping_until_iter"]] = "merge"
    if total < c["grouping_until_iter"]:
        expect = {g: v for g, v in expect.items() if g <= total}
        expect[total] = "final-merge"
    if events != expect:
        diff = sorted(set(events.items()) ^ set(expect.items()))[:6]
        raise AssertionError(f"Grouping events differ from the official schedule: {diff}")
    # official: f_rest frozen for iteration 14500 (unfrozen by the densify at its end) and
    # from 29000 to the end (no densification after 15000)
    want = []
    if total >= c["global_densify_iter"]:
        want.append(c["global_densify_iter"])
    if total >= c["grouping_until_iter"]:
        want += list(range(c["grouping_until_iter"], total + 1))
    if frozen_at_render != want:
        raise AssertionError(f"SH freeze windows wrong: {frozen_at_render[:5]}..")
    if d.cache is not None or len(params["means"]) != 40:
        raise AssertionError("Population not fully merged at the end of training")
    return {"groupings": len(events), "merge_only": [k for k, v in events.items() if v != "regroup"],
            "sh_frozen_steps": len(frozen_at_render)}


def subset_state(method, device):
    n = 50
    params, optimizers = _scene(method, n, device)
    _adam_steps(params, optimizers, 3)
    d = _density(method)
    d.initialize(params, optimizers, 1.0)
    d.state["grad2d"] = torch.rand(n, device=device)
    d.state["count"] = torch.ones(n, device=device)
    full = {k: v.detach().clone() for k, v in params.items()}
    full_m = {k: {m: optimizers[k].state[p][m].clone() for m in ("exp_avg", "exp_avg_sq")}
              for k, p in params.items()}
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[torch.randperm(n)[: int(n * 0.6)].to(device)] = True
    d.regroup(params, optimizers, 1000, mask=mask)
    check_state(method, params, optimizers)
    k = int(mask.sum())
    if len(params["means"]) != k or len(d.state["grad2d"]) != k:
        raise AssertionError("Regroup must keep only the under-training rows (params and stats)")
    for name in params:
        torch.testing.assert_close(params[name].detach(), full[name][mask])
        torch.testing.assert_close(d.cache["rows"][name], full[name][~mask])
    _adam_steps(params, optimizers, 4, seed=2)
    active_m = {k_: {m: optimizers[k_].state[p][m].clone() for m in ("exp_avg", "exp_avg_sq")}
                for k_, p in params.items()}
    active = {k_: v.detach().clone() for k_, v in params.items()}
    # contrast: runtime per-field Adam on the full population with zero grad on cached rows
    shadow = {k_: torch.nn.Parameter(v.clone()) for k_, v in full.items()}
    sopt = {k_: torch.optim.Adam([p], lr=optimizers[k_].param_groups[0]["lr"], eps=1e-15)
            for k_, p in shadow.items()}
    for k_, p in shadow.items():
        sopt[k_].state[p] = {"step": torch.tensor(3.0), **{m: full_m[k_][m].clone()
                                                           for m in ("exp_avg", "exp_avg_sq")}}
        p.grad = torch.zeros_like(p)
        sopt[k_].step()
    moved = float((shadow["means"].detach()[~mask] - full["means"][~mask]).abs().max())
    if moved == 0:
        raise AssertionError("Contrast failed: zero-gradient Adam rows should drift by momentum")
    d.merge(params, optimizers)
    check_state(method, params, optimizers)
    if len(params["means"]) != n or d.cache is not None:
        raise AssertionError("Merge must restore the whole population")
    for name, p in params.items():
        torch.testing.assert_close(p.detach()[:k], active[name])
        torch.testing.assert_close(p.detach()[k:], full[name][~mask], rtol=0, atol=0)
        st = optimizers[name].state[p]
        for m in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(st[m][:k], active_m[name][m])
            if torch.count_nonzero(st[m][k:]):
                raise AssertionError("Official merge appends cached rows with zero Adam moments")
    if torch.count_nonzero(d.state["grad2d"]) or torch.count_nonzero(d.state["count"]):
        raise AssertionError("Official merge zeroes the densification accumulators")
    # 'keep' alternative restores the cached moments unchanged
    params2, opt2 = _scene(method, n, device)
    _adam_steps(params2, opt2, 2)
    m0 = {k_: opt2[k_].state[p]["exp_avg"].clone() for k_, p in params2.items()}
    d2 = _density(method, cached_moments="keep")
    d2.initialize(params2, opt2, 1.0)
    d2.regroup(params2, opt2, 1000, mask=mask)
    d2.regroup(params2, opt2, 14500)
    for k_, p in params2.items():
        torch.testing.assert_close(opt2[k_].state[p]["exp_avg"][k:], m0[k_][~mask])
    if params2["shN"].requires_grad:
        raise AssertionError("SH rest must be frozen at the I_D merge")
    return {"population": n, "active": k, "cached": n - k,
            "zero_grad_adam_drift_avoided": moved}


def cuda_training(method, device):
    if not str(device).startswith("cuda"):
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("requires CUDA")
    torch.manual_seed(0)
    from dataclasses import replace

    spec = _density_spec(method)
    small = {**spec.config, "grouping_from_iter": 20, "grouping_interval": 20,
             "grouping_until_iter": 200, "global_densify_iter": 60, "refine_start_iter": 10,
             "refine_every": 10, "refine_stop_iter": 80, "reset_every": 1000, "total_steps": 110}
    method = replace(method, components=tuple(
        replace(c, config=small) if c.kind == "density" else c for c in method.components))
    renderer, density, losses = build(method)
    n = 600
    params, optimizers = _scene(method, n, device)
    with torch.no_grad():
        params["scales"].fill_(math.log(0.05))
    density.initialize(params, optimizers, 1.0)
    w = h = 64
    cam = {"viewmat": torch.eye(4, device=device), "width": w, "height": h,
           "K": torch.tensor([[64.0, 0, 32], [0, 64.0, 32], [0, 0, 1]], device=device)}
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    target = torch.stack([xx, yy, 0.5 * (xx + yy)], -1).to(device)
    rendered_rows, snapshot, checks = [], None, 0
    for step in range(1, 111):
        out = renderer.render(params, cam, sh_degree=0)
        out["step"] = step
        rows = out["info"]["means2d"].shape[1]
        if rows != len(params["means"]):
            raise AssertionError("Renderer did not render the current (sub)population")
        rendered_rows.append(rows)
        density.before_backward(params, optimizers, step, out)
        sum(f(params, out, target) for _, f in losses).backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step + 1 == 60:
            snapshot = {k: v.clone() for k, v in density.cache["rows"].items()}
            n_active = len(params["means"])
        density.after_backward(params, optimizers, step, out)
        if step + 1 == 60:
            for k, v in snapshot.items():
                torch.testing.assert_close(params[k].detach()[n_active:], v, rtol=0, atol=0)
            if params["shN"].requires_grad:
                raise AssertionError("SH rest must be frozen at I_D")
            checks += 1
        g2d = density.state["grad2d"]
        if g2d is not None and len(g2d) != len(params["means"]):
            raise AssertionError("Density statistics left subset indexing")
        check_state(method, params, optimizers)
    if density.cache is not None or checks != 1:
        raise AssertionError("Final merge or I_D merge did not happen")
    events = [e[1] for e in density.log]
    if "regroup" not in events or "final-merge" not in events:
        raise AssertionError(f"Unexpected event sequence {events}")
    return {"rendered_rows_min": min(rendered_rows), "rendered_rows_max": max(rendered_rows),
            "final_gaussians": len(params["means"]), "events": events}
