"""Mechanism probes for the CRiM-GS port. Each raises if its mechanism is removed or wrong."""

import math
from dataclasses import replace

import torch
from gsplat import rasterization

from . import kernel as kernel_mod
from .filters import filtered_scales_opacities, is_filter_step
from .kernel import CRiMKernel
from .losses import orthogonality
from .ode import odeint_dopri5, odeint_fixed, rk_step
from .se3 import normalize_axis, screw_exp, skew


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, name):
    return next(c for c in method.components if c.name == name)


def _cameras(device, n=2, width=64, height=48):
    cams = []
    for i in range(n):
        vm = torch.eye(4, device=device)
        vm[0, 3] = 0.1 * i
        K = torch.tensor([[60.0, 0, width / 2], [0, 60.0, height / 2], [0, 0, 1]], device=device)
        cams.append({"viewmat": vm, "K": K, "width": width, "height": height})
    return cams


def _scene(method, device, n=400, seed=3):
    from splatstudio.state import initialize

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) * torch.tensor([2.0, 1.5, 1.0]) - torch.tensor(
        [1.0, 0.75, -2.5])
    cols = torch.rand(n, 3, generator=g) * 0.8 + 0.1
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.04))
        params["opacities"].fill_(2.0)
    return params, opts


def _spread_decoders(kern, scale=0.05, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in (kern.dec_rigid_w, kern.dec_rigid_theta, kern.dec_rigid_v,
                      kern.dec_adaptive_A, kern.dec_adaptive_t):
            w = torch.randn(layer.weight.shape, generator=g) * scale
            layer.weight.copy_(w.to(layer.weight.device))


# -- C1: solver -------------------------------------------------------------------------------
def dopri5(method, device):
    torch.manual_seed(0)
    A = (torch.randn(4, 4, dtype=torch.float64) * 0.4)
    y0 = torch.randn(4, dtype=torch.float64)
    ts = [float(t) for t in range(9)]  # tau = 0..8 as in the kernel

    def f(t, y):
        return A @ y

    stats = {}
    sol = odeint_dopri5(f, y0, ts, rtol=1e-4, atol=1e-5, stats=stats)
    exact = torch.stack([torch.linalg.matrix_exp(A * t) @ y0 for t in ts])
    rel = float(((sol - exact).norm(dim=1) / exact.norm(dim=1).clamp_min(1e-12)).max())
    _check(rel < 1e-3, f"dopri5 relative error {rel:.2e} too large")
    tight = odeint_dopri5(f, y0, ts, rtol=1e-9, atol=1e-11)
    rel_tight = float(((tight - exact).norm(dim=1) / exact.norm(dim=1)).max())
    _check(rel_tight < 1e-7, f"tight-tolerance dopri5 error {rel_tight:.2e}")
    _check(stats["accepted"] > len(ts) - 1, "adaptive solver took no intermediate steps")
    # 5th-order convergence of the fixed-step tableau on [0, 2].
    e = []
    for sub in (4, 8):
        y = odeint_fixed(f, y0, [0.0, 2.0], sub)[-1]
        e.append(float((y - torch.linalg.matrix_exp(A * 2.0) @ y0).norm()))
    order = math.log2(e[0] / e[1])
    _check(4.5 < order < 6.5, f"observed order {order:.2f}, expected ~5")
    # Ablation: a single Euler step per interval must fail the same accuracy test.
    y, euler = y0, [y0]
    for _ in ts[1:]:
        y = y + f(0.0, y)
        euler.append(y)
    rel_euler = float(((torch.stack(euler) - exact).norm(dim=1)
                       / exact.norm(dim=1)).max())
    _check(rel_euler > 1e-2, "Euler ablation unexpectedly accurate: probe is not discriminative")
    y5, err, _ = rk_step(f, 0.0, y0, 0.5)
    _check(float(err.norm()) > 0, "embedded error estimate is zero")
    return {"rel_err": rel, "rel_err_tight": rel_tight, "accepted": stats["accepted"],
            "rejected": stats.get("rejected", 0), "order": order, "euler_rel_err": rel_euler}


# -- C1: screw motion ------------------------------------------------------------------------
def screw_se3(method, device):
    g = torch.Generator().manual_seed(1)
    w = torch.randn(16, 3, generator=g, dtype=torch.float64)
    w_hat = normalize_axis(w)
    theta = torch.rand(16, 1, generator=g, dtype=torch.float64) * 2.5
    v = torch.randn(16, 3, generator=g, dtype=torch.float64)
    T = screw_exp(w_hat, theta, v)
    twist = torch.zeros(16, 4, 4, dtype=torch.float64)
    twist[:, :3, :3] = skew(w_hat) * theta[..., None]
    twist[:, :3, 3] = v * theta
    ref = torch.linalg.matrix_exp(twist)
    err = float((T - ref).abs().max())
    _check(err < 1e-8, f"e^[S]theta mismatch {err:.2e}")
    norms = torch.linalg.vector_norm(w_hat, dim=-1)
    _check(float((norms - 1).abs().max()) < 1e-8, "decoded rotation axis is not unit norm")
    _check(torch.allclose(normalize_axis(3.7 * w), w_hat), "axis depends on ||w||")
    R = T[:, :3, :3]
    orth = float((R.transpose(1, 2) @ R - torch.eye(3, dtype=torch.float64)).abs().max())
    _check(orth < 1e-8, "rotation not orthogonal")
    # Ablation: dropping G(theta) (t = v theta) must be detected.
    wrong = float((ref[:, :3, 3] - v * theta).abs().max())
    _check(wrong > 1e-3, "probe cannot distinguish G(theta) v from v theta")
    return {"max_err": err, "orthogonality_err": orth, "g_ablation_gap": wrong}


# -- C1/C2: kernel poses -----------------------------------------------------------------------
def kernel_poses(method, device):
    _require_cuda(device)
    dens = _component(method, "crimgs-density").config
    cfg = {k[len("kernel_"):]: v for k, v in dens.items() if k.startswith("kernel_")
           and k != "kernel_export_every"}
    cams = _cameras(device, 3)
    kern = CRiMKernel(cams, cfg, device)
    vm0, R_a0 = kern.trajectory(1)
    init_err = float((vm0 - cams[1]["viewmat"][None]).abs().max())
    _check(init_err < 5e-3, f"initial trajectory far from input pose ({init_err:.2e})")
    _check(float((R_a0 - torch.eye(3, device=device)).abs().max()) < 1e-3, "R_a init not ~I")
    _spread_decoders(kern)
    z_r, z_a = kern.latents(1)
    e = kern.embedding.weight[1]
    _check(torch.allclose(z_r[0], kern.encoder_rigid(e)), "z_r(tau_0) != E_r(embed)")
    _check(torch.allclose(z_a[0], kern.encoder_adaptive(e)), "z_a(tau_0) != E_a(embed)")
    _check(float((z_r[-1] - z_r[0]).abs().max()) > 1e-4, "latent does not evolve along tau")
    vm, R_a = kern.trajectory(1)
    # Independent recomposition: matrix_exp of the decoded twist, explicit A + I.
    half = z_r.shape[1] // 2
    with torch.no_grad():
        w = kern.dec_rigid_w(z_r[:, :half])
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-10)
        th = kern.dec_rigid_theta(z_r[:, :half])
        v = kern.dec_rigid_v(z_r[:, half:])
        tw = torch.zeros(len(w), 4, 4, device=device)
        tw[:, 0, 1], tw[:, 0, 2], tw[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
        tw[:, 1, 0], tw[:, 2, 0], tw[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
        tw[:, :3, :3] *= th[:, :, None]
        tw[:, :3, 3] = v * th
        T_r = torch.linalg.matrix_exp(tw)
        T_a = torch.eye(4, device=device).repeat(len(w), 1, 1)
        T_a[:, :3, :3] += kern.dec_adaptive_A(z_a[:, :half]).reshape(-1, 3, 3)
        T_a[:, :3, 3] = kern.dec_adaptive_t(z_a[:, half:])
        c2w = torch.linalg.inv(cams[1]["viewmat"])[None] @ T_r @ T_a
        ref = torch.linalg.inv(c2w)
    err = float((vm - ref).abs().max())
    _check(err < 1e-4, f"trajectory != inv(c2w T_r T_a) ({err:.2e})")
    spread = float((vm[0] - vm[-1]).abs().max())
    _check(spread > 1e-3, "trajectory does not move")
    # Order matters: T_a T_r (swapped) must give a different result when both are non-trivial.
    swapped = torch.linalg.inv(torch.linalg.inv(cams[1]["viewmat"])[None] @ T_a @ T_r)
    _check(float((swapped - ref).abs().max()) > 1e-6, "composition order not observable")
    return {"init_err": init_err, "recompose_err": err, "trajectory_spread": spread,
            "ode_accepted": kern.counts["ode_accepted"]}


# -- C3: blur synthesis -----------------------------------------------------------------------
def _plain(params, viewmat, cam, kernel_size=0.1):
    sc, op = filtered_scales_opacities(params["scales"], params["opacities"], params["filter_3d"])
    r, a, _ = rasterization(params["means"], params["quats"], sc, op,
                            torch.cat((params["sh0"], params["shN"]), 1), viewmat[None],
                            cam["K"][None], cam["width"], cam["height"], sh_degree=0,
                            eps2d=kernel_size, rasterize_mode="antialiased")
    return r[0]  # black background


def blur_synthesis(method, device):
    _require_cuda(device)
    from splatstudio.components import build

    renderer, density, _ = build(method)
    params, opts = _scene(method, device)
    cams = _cameras(device, 2)
    density.initialize(params, opts, 1.0)
    density.set_training_cameras(cams)
    kern = kernel_mod.ACTIVE
    try:
        _check(kern is density.kernel, "density did not publish its kernel")
        _spread_decoders(kern, 0.03)
        cam = cams[0]
        res = {}
        with torch.no_grad():
            sharp = _plain(params, cam["viewmat"], cam)
        kern.step = 999  # rendering step 1000
        out = renderer.render(params, cam, 0)
        _check(torch.allclose(out["rgb"], sharp, atol=1e-5), "step 1000 is not the sharp render")
        _check(out.get("R_a") is None, "trajectory active at step 1000")
        kern.step = 1999
        out = renderer.render(params, cam, 0)
        with torch.no_grad():
            subs = torch.stack([_plain(params, vm, cam) for vm in out["subframe_viewmats"]])
            ref = torch.cat((subs, sharp[None])).mean(0)
        _check(torch.allclose(out["rgb"], ref, atol=1e-4), "step 2000 != mean(N renders, sharp)")
        res["warp_vs_sharp"] = float((out["rgb"] - sharp).abs().mean())
        _check(res["warp_vs_sharp"] > 1e-4, "averaged trajectory render equals the sharp render")
        _check(out["subframes"].shape[0] == kern.cfg["num_poses"], "wrong number of poses")
        kern.step = 3999
        out = renderer.render(params, cam, 0)
        P, M = out["pixel_weights"], out["mask"]
        with torch.no_grad():
            subs = torch.stack([_plain(params, vm, cam) for vm in out["subframe_viewmats"]])
            P2, M2 = kern.cnn(subs)
            ref = (1 - M2) * sharp + M2 * (P2 * subs).sum(0)
        _check(torch.allclose(P.sum(0), torch.ones_like(P[0]), atol=1e-5), "weights not softmax")
        _check(bool(((M > 0) & (M < 1)).all()), "mask outside (0, 1)")
        _check(torch.allclose(out["rgb"], ref, atol=1e-4), "step 4000 != (1-M) I0 + M sum P I")
        kern.optimizer.zero_grad(set_to_none=True)
        out["rgb"].square().mean().backward()
        grads = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
                 for n, p in (("embedding", kern.embedding.weight),
                              ("dec_rigid_v", kern.dec_rigid_v.weight),
                              ("dec_adaptive_A", kern.dec_adaptive_A.weight),
                              ("f", kern.f.linear_rot.weight),
                              ("cnn", kern.cnn.weight_head.weight),
                              ("mask", kern.cnn.mask_head.weight))}
        for n, gval in grads.items():
            _check(gval > 0 and math.isfinite(gval), f"kernel part {n} gets no gradient")
        with torch.no_grad():
            ev = renderer.render(params, cam, 0)
        _check(torch.allclose(ev["rgb"], sharp, atol=1e-5), "eval render is not sharp")
        kern.zero_grad(set_to_none=True)
        res.update(grads)
        return res
    finally:
        kernel_mod.ACTIVE = None


# -- C2/C4: objective --------------------------------------------------------------------------
def objective(method, device):
    from splatstudio.components import build
    from splatstudio.losses import ssim

    _check(_component(method, "photometric").config.get("ssim_weight") == 0.3, "lambda_c != 0.3")
    _, _, losses = build(method)
    reg = dict(losses)["crimgs-regularizers"]
    photo = dict(losses)["photometric"]
    g = torch.Generator().manual_seed(2)
    R = (torch.eye(3) + 0.05 * torch.randn(9, 3, 3, generator=g)).requires_grad_(True)
    M = torch.rand(20, 30, 1, generator=g)
    rgb = torch.rand(20, 30, 3, generator=g, requires_grad=True)
    target = torch.rand(20, 30, 3, generator=g)
    val = reg({}, {"rgb": rgb, "R_a": R, "mask": M}, target)
    ref = 1e-4 * torch.stack([((r.T @ r - torch.eye(3)) ** 2).sum().sqrt() for r in R]).mean() \
        + 1e-3 * M.mean()
    _check(abs(float(val) - float(ref)) < 1e-9, f"regulariser {float(val)} != {float(ref)}")
    _check(abs(float(orthogonality(torch.eye(3)[None]))) < 1e-12, "L_o(I) != 0")
    off = reg({}, {"rgb": rgb}, target)
    _check(float(off) == 0.0 and off.requires_grad, "inactive regulariser must be 0, attached")
    p = photo({}, {"rgb": rgb}, target)
    pref = 0.7 * (rgb - target).abs().mean() + 0.3 * (1 - ssim(rgb, target))
    _check(abs(float(p) - float(pref)) < 1e-6, "photometric is not 0.7 L1 + 0.3 D-SSIM")
    return {"regulariser": float(val), "photometric": float(p)}


# -- C4: kernel schedule -----------------------------------------------------------------------
def kernel_schedule(method, device):
    _require_cuda(device)
    kern = CRiMKernel(_cameras(device, 2), {}, device)
    lr0, lrf, total = 1e-4, 1e-6, 40000
    _check(abs(kern.lr(1001) - lr0) < 1e-15, "first kernel step lr != 1e-4")
    exp = lr0 - 1000 * (lr0 - lrf) / total
    _check(abs(kern.lr(2001) - exp) < 1e-15, "kernel lr not linear")
    _check(abs(kern.lr(40000) - (lr0 - 38999 * (lr0 - lrf) / total)) < 1e-15, "final lr")
    before = kern.embedding.weight.detach().clone()
    kern.embedding.weight.grad = torch.ones_like(before)
    kern.after_step(1000)
    _check(torch.equal(before, kern.embedding.weight.detach()), "kernel updated at step 1000")
    kern.embedding.weight.grad = torch.ones_like(before)
    kern.after_step(1001)
    _check(not torch.equal(before, kern.embedding.weight.detach()), "kernel not updated at 1001")
    _check(abs(kern.optimizer.param_groups[0]["lr"] - lr0) < 1e-15, "lr not applied")
    return {"lr_1001": kern.lr(1001), "lr_2001": kern.lr(2001), "lr_40000": kern.lr(40000),
            "kernel_steps": kern.counts["kernel_steps"]}


# -- Base: density schedule -----------------------------------------------------------------
def density_schedule(method, device):
    _require_cuda(device)
    from splatstudio.components import build

    def run(cfg_update, steps):
        comps = tuple(replace(c, config={**c.config, **cfg_update}) if c.name == "crimgs-density"
                      else c for c in method.components)
        m = replace(method, components=comps)
        renderer, density, _ = build(m)
        params, opts = _scene(m, device)
        cams = _cameras(device, 2)
        density.initialize(params, opts, 1.0)
        density.set_training_cameras(cams)
        counts = []
        try:
            for s in steps:
                density.kernel.step = 0  # keep the sharp path: isolates the density schedule
                out = renderer.render(params, cams[0], 0)
                density.before_backward(params, opts, s, out)
                (out["rgb"] - 0.5).square().mean().backward()
                for o in opts.values():
                    o.step()
                    o.zero_grad(set_to_none=True)
                before = params["opacities"].detach().clone()
                n0 = len(before)
                density.after_backward(params, opts, s, out)
                counts.append((s, n0, len(params["means"]), before,
                               params["opacities"].detach().clone()))
        finally:
            kernel_mod.ACTIVE = None
        return counts, density

    # No reset: no growth / pruning, so a reset would show as opacities clamped to <= 0.01.
    res, _ = run({"grow_grad2d": 1e9, "prune_opa": 0.0}, [3000, 6000])
    for s, n0, n1, before, after in res:
        _check(n0 == n1, "unexpected population change")
        _check(torch.equal(before, after), f"opacities changed at step {s} (reset?)")
    # Growth boundary: every Gaussian qualifies (threshold 0), densify at 600 but not >= 20000.
    res, dens = run({"grow_grad2d": 0.0, "prune_opa": 0.0}, [600, 20000, 20100])
    grow = {s: n1 - n0 for s, n0, n1, _, _ in res}
    _check(grow[600] > 0, "no densification at step 600")
    _check(grow[20000] == 0 and grow[20100] == 0, "densification at/after step 20000")
    official = [s for s in range(1, 40001) if s % 100 == 0 and (500 < s < 20000 or
                                                                20000 < s < 39900)]
    ours = [s for s in range(1, 40001) if is_filter_step(s, 100, 500, 20000, 40000)]
    _check(ours == official, "3D filter steps differ from the official schedule")
    return {"growth": grow, "filter_steps": len(ours), "no_reset_steps": [3000, 6000]}
