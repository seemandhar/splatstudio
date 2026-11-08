"""Executable mechanism checks for CompGS (each fails if its mechanism is removed or wrong).

Scene / camera helpers follow the c3dgs plugin probes (reproductions/c3dgs/plugin/ss_c3dgs/probes.py).
"""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from . import codec
from . import schedule as sched
from . import vq as vq_mod


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("compgs render probes need CUDA (gsplat)")


def _component(method, kind, name=None):
    return next(c for c in method.components if c.kind == kind and (name is None or c.name == name))


def _scene(n, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-0.6, 0.6, generator=g)
    means[:, 2] = torch.empty(n).uniform_(-0.3, 0.3, generator=g)
    params = {
        "means": means,
        "scales": torch.empty(n, 3).uniform_(math.log(0.04), math.log(0.15), generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.empty(n).uniform_(0.2, 0.7, generator=g)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
    }
    return {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}


def _camera(device, width=80, height=60, f=70.0, dist=3.0):
    viewmat = torch.eye(4)
    viewmat[:3, 3] = torch.tensor([0.02, -0.01, dist])
    K = torch.tensor([[f, 0.0, width / 2 - 0.3], [0.0, f, height / 2 + 0.2], [0.0, 0.0, 1.0]])
    return {"K": K.to(device), "viewmat": viewmat.to(device), "width": width, "height": height}


def _reference_kmeans(x, centers, iters):
    """Independent re-implementation of the official cluster_assign (one-hot matmul update)."""
    k = len(centers)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
        onehot = F.one_hot(d.argmin(-1), k).to(x.dtype)
        counts = onehot.sum(0) + 1e-6
        centers = (onehot.T @ x) / counts[:, None]
    d = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
    return centers, d.argmin(-1)


def kmeans_official(method, device):
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.cat([torch.randn(150, 4, generator=g) * 0.3 + c
                   for c in (torch.tensor([2.0, 0, 0, 0]), torch.tensor([0, -2.0, 1, 0]),
                             torch.tensor([0, 0, 0, 3.0]))]).to(device)
    q = vq_mod.KMeansQuantizer(8, 3, chunk=64)
    gen = torch.Generator(device=x.device).manual_seed(0)
    init = x[torch.randperm(len(x), generator=torch.Generator(device=x.device).manual_seed(0),
                            device=x.device)[:8]].clone()
    q.assign(x, gen)
    ref_c, ref_i = _reference_kmeans(x, init, 3)
    torch.testing.assert_close(q.centers, ref_c, rtol=1e-5, atol=1e-5)
    if not torch.equal(q.idx, ref_i):
        raise AssertionError("final assignment differs from the reference")
    # warm start: second assignment continues from the current centres
    x2 = x + 0.05
    prev = q.centers.clone()
    q.assign(x2, gen)
    ref_c2, ref_i2 = _reference_kmeans(x2, prev, 3)
    torch.testing.assert_close(q.centers, ref_c2, rtol=1e-5, atol=1e-5)
    # update: per-cluster mean with cached indices; empty cluster -> 0
    x3 = x2 * 1.1
    q.idx = q.idx.clone()
    q.idx[q.idx == 7] = 6
    q.update(x3)
    for c in range(8):
        sel = x3[q.idx == c]
        want = sel.sum(0) / (len(sel) + 1e-6)
        torch.testing.assert_close(q.centers[c], want, rtol=1e-5, atol=1e-6)
    if float(q.centers[7].abs().max()) != 0.0:
        raise AssertionError("empty cluster centre should be 0")
    # STE: forward = centroid, backward = identity
    p = x3.clone().requires_grad_(True)
    out = q.quantize(p)
    torch.testing.assert_close(out, q.centers[q.idx])
    (out * torch.arange(4.0, device=x.device)).sum().backward()
    torch.testing.assert_close(p.grad, torch.arange(4.0, device=x.device).expand_as(p))
    return {"clusters": 8, "points": len(x), "assignments": q.assignments}


def ste_render(method, device):
    _require_cuda(device)
    from splatstudio.backends.gsplat import Renderer

    from .renderer import create_renderer

    spec = _component(method, "renderer")
    r = create_renderer(spec)
    plain = Renderer(spec)
    params = _scene(300, device, 1)
    vq = vq_mod.VQState({"dc": 16, "sh": 16, "scale": 8, "rot": 8}, 3)
    vq.assign(params, torch.Generator(device=device).manual_seed(0))
    r.vq = vq
    cam = _camera(device)
    out = r.render(params, cam, 3)
    loss = (out["rgb"] * torch.linspace(0.5, 1.5, out["rgb"].shape[1], device=device)[None, :, None]).sum()
    loss.backward()
    dec = {k: (v.detach().clone() if k not in vq.quantizers
               else vq.quantizers[k].decoded(v.shape).clone()).requires_grad_(True)
           for k, v in params.items()}
    ref = plain.render(dec, cam, 3)
    torch.testing.assert_close(out["rgb"], ref["rgb"], rtol=1e-5, atol=1e-5)
    (ref["rgb"] * torch.linspace(0.5, 1.5, ref["rgb"].shape[1], device=device)[None, :, None]).sum().backward()
    for f in vq.quantizers:
        torch.testing.assert_close(params[f].grad, dec[f].grad, rtol=1e-4, atol=1e-6)
    raw = plain.render({k: v.detach() for k, v in params.items()}, cam, 3)
    diff = float((raw["rgb"] - out["rgb"]).abs().max())
    if diff < 1e-3:
        raise AssertionError("quantised render is identical to the unquantised one")
    return {"quantised_vs_raw_max_abs": diff}


def schedule_check(cfg):
    assigns = [it for it in range(1, cfg["total_steps"] + 1) if sched.vq_event(it, cfg) == "assign"]
    updates = sum(sched.vq_event(it, cfg) == "update" for it in range(1, cfg["total_steps"] + 1))
    prunes = [s for s in range(1, cfg["total_steps"] + 1) if sched.prunes(s, cfg)]
    return assigns, updates, prunes


def schedule(method, device):
    from .density import DEFAULTS

    cfg = {**DEFAULTS, **_component(method, "density").config}
    if method.name != "compgs":
        cfg = DEFAULTS
    assigns, updates, prunes = schedule_check(cfg)
    want = list(range(20001, 25000, 100)) + [25001]
    if assigns != want:
        raise AssertionError(f"assignment iterations {assigns[:3]}..{assigns[-3:]} != official")
    if updates != 10000 - len(want):
        raise AssertionError(f"{updates} centroid updates, expected {10000 - len(want)}")
    if prunes != [16000, 17000, 18000, 19000, 20000]:
        raise AssertionError(f"prune steps {prunes}")
    from .losses import DEFAULTS as R

    window = [s for s in (14999, 15000, 20000, 20001) if sched.reg_active(s, R["start"], R["stop"])]
    if window != [15000, 20000]:
        raise AssertionError(f"regulariser window {window}")
    return {"assignments": len(assigns), "updates": updates, "prunes": prunes}


def opacity_reg(method, device):
    from .losses import create_opacity_reg

    spec = _component(method, "loss", "compgs-opacity-reg")
    fn = create_opacity_reg(spec)
    lam, start, stop = spec.config["lambda_reg"], spec.config["start"], spec.config["stop"]
    o = torch.nn.Parameter(torch.linspace(-3, 3, 50, device=device))
    params = {"opacities": o}
    v = fn(params, {"step": (start + stop) // 2}, None)
    torch.testing.assert_close(v, lam * o.sigmoid().sum())
    v.backward()
    s = o.sigmoid().detach()
    torch.testing.assert_close(o.grad, lam * s * (1 - s))
    for step in (start - 1, stop + 1):
        if float(fn(params, {"step": step}, None)) != 0.0:
            raise AssertionError(f"regulariser active at step {step}")
    return {"lambda": lam, "window": [start, stop]}


def pipeline(method, device):
    _require_cuda(device)
    from .density import create_density

    spec = _component(method, "density")
    cfg = {**spec.config, "total_steps": 40, "vq_start": 20, "assign_freq": 5, "final_freq": 30,
           "freeze_window": 10, "prune_start": 10, "prune_stop": 20, "prune_every": 5,
           "codebooks": {"dc": 16, "sh": 16, "scale": 32, "rot": 32}, "kmeans_iters": 2}
    d = create_density(replace(spec, config=cfg))
    params = torch.nn.ParameterDict(_scene(400, device, 2))
    with torch.no_grad():
        params["opacities"][:37] = torch.logit(torch.tensor(0.002))
    optimizers = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    d.initialize(params, optimizers, 1.0)
    sum(p.square().sum() for p in params.values()).backward()
    for o in optimizers.values():
        o.step()
        o.zero_grad(set_to_none=True)
    seen = []
    for step in range(1, 41):
        n0 = d.stats["assignments"], d.stats["updates"]
        d.compgs_step(params, optimizers, step)
        if d.stats["assignments"] != n0[0]:
            seen.append(step + 1)
        if 20 <= step < 40:
            with torch.no_grad():
                for f in ("sh0", "shN", "scales", "quats"):
                    params[f].add_(0.01 * torch.randn_like(params[f]))
    if len(params["means"]) != 400 - 37:
        raise AssertionError(f"prune kept {len(params['means'])} Gaussians, expected 363")
    for k, p in params.items():
        if optimizers[k].state[p]["exp_avg"].shape != p.shape:
            raise AssertionError(f"{k}: optimizer moments misaligned after prune")
    if seen != [21, 26, 31]:
        raise AssertionError(f"assignment iterations {seen}")
    for f, q in d.vq.quantizers.items():
        rows = params[f].detach().reshape(len(params[f]), -1)
        torch.testing.assert_close(rows, q.centers[q.idx])
        if torch.unique(rows, dim=0).shape[0] > q.k:
            raise AssertionError(f"{f}: more distinct rows than codes")
    if vq_mod.ACTIVE["vq"] is not None:
        raise AssertionError("VQ state still active after bake")
    if not all(p.requires_grad and p.is_leaf for p in params.values()):
        raise AssertionError("bake/prune froze a field")
    size = d.encoded["paper_size"]
    n = 363
    want = n * 16 + (16 * 3 + 16 * 45 + 32 * 3 + 32 * 4) * 4 + math.ceil(n * (4 + 4 + 5) / 8) + 32 * 4
    if size["bytes"] != want:
        raise AssertionError(f"paper size {size['bytes']} != {want}")
    if codec.official_disk_size(n, {"a": (16, 3)})["bytes"] != n * 16 + math.ceil(n * 9 / 8) + 16 * 12:
        raise AssertionError("official disk size formula")
    return {"gaussians": n, "assignments_at": seen, "paper_bytes": size["bytes"]}
