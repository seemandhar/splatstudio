"""Mechanism probes for the FisherRF port. Each one fails if its mechanism is removed."""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import fisher, selection


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _look_at(eye, device):
    """OpenCV w2c looking at the origin from `eye` (y down)."""
    eye = torch.tensor(eye, dtype=torch.float32)
    fwd = F.normalize(-eye, dim=0)
    up = torch.tensor([0.0, 0.0, 1.0])
    right = F.normalize(torch.linalg.cross(fwd, up), dim=0)
    down = torch.linalg.cross(fwd, right)
    R = torch.stack([right, down, fwd])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = -R @ eye
    return viewmat.to(device)


def _camera(eye, device, W=40, H=32, f=44.0):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device)
    return {"viewmat": _look_at(eye, device), "K": K, "width": W, "height": H}


def _scene(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)

    def u(*shape, lo=0.0, hi=1.0):
        return (torch.rand(*shape, generator=g) * (hi - lo) + lo).to(device)

    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device)
    return {
        "means": u(n, 3, lo=-0.5, hi=0.5),
        "scales": u(n, 3, lo=math.log(0.08), hi=math.log(0.25)),
        "quats": quats,
        "opacities": torch.logit(u(n, lo=0.2, hi=0.6)),
        "sh0": u(n, 1, 3, lo=-1.0, hi=1.0),
        "shN": u(n, 15, 3, lo=-0.3, hi=0.3),
    }


def _reference_hessian(p, camera, degree):
    """Independent: gsplat forward + per-pixel autograd, sum_p (d(R+G+B)_p / d theta)^2 in the
    activated space, then the official single (unsquared) chain to the raw parameters."""
    from gsplat import rasterization

    act = {
        "means": p["means"].clone().requires_grad_(True),
        "sh": torch.cat((p["sh0"], p["shN"]), 1).clone().requires_grad_(True),
        "opac": p["opacities"].sigmoid().clone().requires_grad_(True),
        "scale": p["scales"].exp().clone().requires_grad_(True),
    }
    render, _, _ = rasterization(
        means=act["means"], quats=F.normalize(p["quats"], dim=-1), scales=act["scale"],
        opacities=act["opac"], colors=act["sh"], viewmats=camera["viewmat"][None],
        Ks=camera["K"][None], width=camera["width"], height=camera["height"],
        sh_degree=degree, packed=False, backgrounds=torch.ones(1, 3, device=p["means"].device),
        rasterize_mode="classic")
    s = render[0].sum(-1)  # [H, W]: dL/dpixel = 1 on every channel
    sq = {k: torch.zeros_like(v) for k, v in act.items()}
    plain = torch.autograd.grad(s.sum(), list(act.values()), retain_graph=True)
    for y in range(s.shape[0]):
        for x in range(s.shape[1]):
            grads = torch.autograd.grad(s[y, x], list(act.values()), retain_graph=True,
                                        allow_unused=True)
            for k, gr in zip(act, grads):
                if gr is not None:
                    sq[k] += gr.square()
    sig = p["opacities"].sigmoid()
    ref = {"means": sq["means"], "sh0": sq["sh"][:, :1], "shN": sq["sh"][:, 1:],
           "scales": sq["scale"] * p["scales"].exp(), "opacities": sq["opac"] * sig * (1 - sig)}
    pl = dict(zip(act, plain))
    unsquared = {"means": pl["means"], "sh0": pl["sh"][:, :1], "shN": pl["sh"][:, 1:],
                 "scales": pl["scale"] * p["scales"].exp(),
                 "opacities": pl["opac"] * sig * (1 - sig)}
    return ref, unsquared, render[0]


def fisher_kernel(method, device):
    _cuda(device)
    p = _scene(10, device)
    cam = _camera((0.4, -3.0, 0.8), device)
    degree = 1
    bg = torch.ones(3, device=device)
    with torch.no_grad():
        image = fisher.render_image(p, cam, degree, bg)
    h = fisher.diag_hessian(p, cam, degree, bg)
    if "quats" in h:
        raise AssertionError("rotation must be filtered out (official --filter_out_grad)")
    ref, unsquared, ref_image = _reference_hessian(p, cam, degree)
    image_err = float((image.permute(1, 2, 0) - ref_image).abs().max())
    if image_err > 2e-3:
        raise AssertionError(f"Fisher kernel forward differs from gsplat: {image_err}")
    evidence = {"image_max_abs_err": image_err}
    for k in ("means", "sh0", "shN", "scales", "opacities"):
        num = float((h[k] - ref[k]).abs().sum())
        den = float(ref[k].abs().sum())
        rel = num / max(den, 1e-12)
        evidence[f"{k}_rel_err"] = rel
        if den <= 0 or rel > 2e-2:
            raise AssertionError(f"{k}: kernel diag Hessian != per-pixel squared gradients ({rel})")
        if (h[k] < -1e-12).any():
            raise AssertionError(f"{k}: diagonal Fisher must be non-negative")
    # removing the squared accumulation (plain atomicAdd backward) gives a different statistic
    rel_plain = float((ref["means"] - unsquared["means"]).abs().sum() /
                      ref["means"].abs().sum())
    evidence["unsquared_means_rel_diff"] = rel_plain
    if rel_plain < 0.2:
        raise AssertionError("test case cannot distinguish squared from plain gradients")
    if degree < 3 and float(h["shN"][:, 3:].abs().max()) != 0.0:
        raise AssertionError("inactive SH bands must have zero information")
    return evidence


def eig_selection(method, device):
    rng = np.random.default_rng(0)
    h_train = torch.tensor(rng.random(50) * 10, dtype=torch.float64)
    cands = [torch.tensor(rng.random(50), dtype=torch.float64) for _ in range(7)]
    lam = 1e-6
    scores = selection.eig_scores(h_train, cands, lam).numpy()
    ref = np.array([np.sum(c.numpy() / (h_train.numpy() + lam)) for c in cands])
    if not np.allclose(scores, ref, rtol=1e-12):
        raise AssertionError("EIG score != sum H_c / (H_train + lambda)")
    # information where the model is already certain is worth little (Eq. 5)
    design = selection.eig_scores(torch.tensor([100.0, 1e-3]),
                                  [torch.tensor([10.0, 0.0]), torch.tensor([0.0, 0.01])], lam)
    if int(design.argmax()) != 1:
        raise AssertionError("EIG must prefer information on poorly constrained parameters")
    # greedy batch (Alg. 1) == independent sequential re-scoring
    pool = list(range(10, 17))
    picked = selection.greedy_batch(h_train, cands, pool, 3, lam)
    H, left, ref_pick = h_train.numpy().copy(), list(range(7)), []
    for _ in range(3):
        s = [np.sum(cands[i].numpy() / (H + lam)) for i in left]
        j = left.pop(int(np.argmax(s)))
        ref_pick.append(pool[j])
        H = H + cands[j].numpy()
    if picked != ref_pick:
        raise AssertionError(f"greedy batch {picked} != reference {ref_pick}")
    if len(set(picked)) != 3:
        raise AssertionError("batch picked a view twice")
    pool2 = [3, 5, 8, 9, 11]
    rnd = selection.random_select(pool2, 2, seed=0)
    expect = list(pool2)
    random.Random(0).shuffle(expect)
    if rnd != expect[:2]:
        raise AssertionError("random selector != official RandSelector")
    return {"scores": scores.tolist(), "greedy": picked, "random": rnd}


def active_schedule(method, device):
    seq1 = selection.schedule(20, 1, 4, 100)
    exp1 = {100 * sum(range(4, 4 + j + 1)): 1 for j in range(16)}
    if seq1 != exp1 or max(seq1) != 18400:
        raise AssertionError(f"seq1 schedule wrong: {seq1}")
    seq4 = selection.schedule(20, 4, 4, 300)
    if seq4 != {1200: 4, 3600: 4, 7200: 4, 12000: 4}:
        raise AssertionError(f"seq4 schedule wrong: {seq4}")
    v10 = selection.schedule(10, 1, 2, 100)
    if sorted(v10) != [200, 500, 900, 1400, 2000, 2700, 3500, 4400]:
        raise AssertionError(f"v10 schedule wrong: {v10}")
    ang = torch.arange(8) * (2 * math.pi / 8)
    centers = torch.stack([ang.cos(), ang.sin(), torch.zeros(8)], 1)
    init = selection.initial_views(centers, 4)
    if init != [0, 4, 2, 6]:
        raise AssertionError(f"farthest-point init wrong: {init}")
    return {"seq1_last": max(seq1), "seq4": sorted(seq4), "v10": sorted(v10), "init": init}


def _loop(selector, device, steps=3000):
    from splatstudio.components import build as build_components
    from splatstudio.state import check_state, initialize

    from . import build

    probe = build("fisherrf-probe", selector=selector, total_views=4, views_per_round=1,
                  num_init_views=2, interval_epochs=3, total_iterations=60,
                  position_lr_max_steps=60, sh_up_every=20)
    renderer, density, losses = build_components(probe)
    ctrl = renderer.controller
    if density.controller is not ctrl:
        raise AssertionError("density and renderer do not share the controller")
    g = torch.Generator().manual_seed(1)
    points = (torch.rand(60, 3, generator=g) - 0.5).to(device)
    colors = torch.rand(60, 3, generator=g).to(device)
    params, opts = initialize(probe, points, colors)
    density.initialize(params, opts, 1.0)
    angles = [2 * math.pi * i / 8 for i in range(8)]
    cams = [_camera((3 * math.cos(a), 3 * math.sin(a), 0.5), device) for a in angles]
    held_out = _camera((0.3, 0.2, 3.2), device)  # never registered: not a candidate
    density.set_training_cameras(cams)
    targets = [torch.rand(32, 40, 3, generator=g).to(device) for _ in cams]
    rng = random.Random(0)
    n_skip = n_train = 0
    for step in range(1, steps + 1):
        if ctrl.done:
            break
        i = rng.randrange(8)
        before = {k: v.detach().clone() for k, v in params.items()}
        restores = ctrl.restores
        out = renderer.render(params, cams[i], sh_degree=3)
        if ctrl.restores != restores:
            for k, v in ctrl.initial.items():
                if not torch.equal(params[k], v):
                    raise AssertionError("acquisition did not restore the initial Gaussians")
                if opts[k].state:
                    raise AssertionError("acquisition did not reset the optimizer")
        out["step"] = step
        density.before_backward(params, opts, step, out)
        total = sum(fn(params, out, targets[i]) for _, fn in losses)
        total.backward()
        has_grad = any(p.grad is not None for p in params.values())
        lr = opts["means"].param_groups[0]["lr"]
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opts["means"].param_groups[0]["lr"] = 123.0  # the runtime's own schedule (overridden)
        density.after_backward(params, opts, step, out)
        check_state(probe, params, opts)
        if out.get("fisherrf_skip"):
            n_skip += 1
            if has_grad:
                raise AssertionError("skipped step produced parameter gradients")
            if any(not torch.equal(before[k], params[k]) for k in params
                   if params[k].shape == before[k].shape):
                raise AssertionError("skipped step changed the Gaussians")
        else:
            n_train += 1
            if i not in ctrl.selected or not has_grad:
                raise AssertionError("training step on a non-selected view or without gradient")
            expect_lr = ctrl.lr0 * 0.01 ** (ctrl.cur_iter / 60)
            if abs(lr - expect_lr) > 1e-12 * max(1.0, expect_lr):
                raise AssertionError("xyz LR does not follow the per-round official schedule")
    if not ctrl.done:
        raise AssertionError("effective iteration budget not reached")
    if len(ctrl.selected) != 4 or len(set(ctrl.selected)) != 4 or ctrl.restores != 2:
        raise AssertionError(f"wrong acquisitions: {ctrl.history}")
    if not set(ctrl.selected) <= set(range(8)):
        raise AssertionError("selected a view outside the training pool")
    return ctrl, renderer, params, held_out, n_skip, n_train


def active_loop(method, device):
    _cuda(device)
    ctrl, renderer, params, held_out, n_skip, n_train = _loop("fisher", device)
    out = renderer.render(params, held_out, sh_degree=3)
    if "rgb" not in out:
        raise AssertionError("evaluation camera not rendered after the budget")
    # an evaluation camera before the budget is spent must stop the run
    ctrl.it = 10
    try:
        renderer.render(params, held_out, sh_degree=3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("early evaluation was not refused")
    rctrl = _loop("random", device)[0]
    pool = [i for i in range(8) if i not in rctrl.selected[:2]]
    expect = selection.random_select(pool, 1, 0)
    if rctrl.selected[2:3] != expect:
        raise AssertionError("random baseline did not use the official selector")
    return {"fisher_selected": ctrl.selected, "random_selected": rctrl.selected,
            "skipped_steps": n_skip, "train_steps": n_train, "restores": ctrl.restores}
