"""Mechanism probes for the DeblurGS port. Each raises if its mechanism is removed or wrong."""

import math

import numpy as np
import torch
from gsplat import rasterization
from splatstudio.registry import resolve

from . import bank as bank_mod
from .bank import alignment_lr, bezier, curve_lr
from .pytorch3d_se3 import hat, se3_exp_map


def _component(method, name):
    spec = next(c for c in method.components if c.name == name)
    return spec, resolve(spec.factory)(spec)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _camera(device, width=64, height=48, z=0.0, x=0.0):
    K = torch.tensor([[60.0, 0, width / 2], [0, 60.0, height / 2], [0, 0, 1]], device=device)
    vm = torch.eye(4, device=device)
    vm[0, 3], vm[2, 3] = x, z
    return {"K": K, "viewmat": vm, "width": width, "height": height}


def _scene(method, device, n=300, seed=3):
    """Random Gaussians in front of the camera, initialised through the method's density
    (raw opacity convention)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([2.0, 1.5, 1.0]) - torch.tensor(
        [1.0, 0.75, -2.5])
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means.to(device)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.05), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.zeros(n, device=device)),
        "sh0": torch.nn.Parameter((torch.rand(n, 1, 3, generator=g).to(device) - 0.5) / 0.2821),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    return params, opts


def _live_bank(method, device, cameras):
    spec, density = _component(method, "deblurgs-density")
    params, opts = _scene(method, device)
    density.initialize(params, opts, 1.0)
    density.set_training_cameras(cameras)
    return density, params, opts


def _plain_render(params, viewmats, K, w, h, deg, bg):
    """Independent sharp render: clamp opacity, SH up to degree 2, gsplat, background bg."""
    colors = torch.cat((params["sh0"], params["shN"][:, :8]), 1)
    out = []
    for vm in viewmats:
        r, a, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                params["opacities"].clamp(0, 1).clamp(max=0.999), colors,
                                vm[None], K[None], w, h, sh_degree=deg, near_plane=0.2)
        out.append(r[0] + (1 - a[0]) * bg)
    return torch.stack(out)


# -------------------------------------------------------------------------------------------
def bezier_se3(method, device):
    g = torch.Generator().manual_seed(0)
    cp = torch.randn(10, 3, generator=g, dtype=torch.float64)
    t = torch.rand(7, generator=g, dtype=torch.float64)

    def casteljau(points, s):  # standard Bezier; DeblurGS pairs c_k with t^(C-k): reversed
        p = list(points.flip(0))
        while len(p) > 1:
            p = [(1 - s) * a + s * b for a, b in zip(p[:-1], p[1:])]
        return p[0]

    ref = torch.stack([casteljau(cp, float(s)) for s in t])
    err_bez = float((bezier(cp, t) - ref).abs().max())
    _check(err_bez < 1e-10, f"Bezier mismatch {err_bez}")
    v = torch.randn(5, 6, generator=g, dtype=torch.float64) * 0.6
    m = se3_exp_map(v)
    err_exp = 0.0
    for i in range(5):
        a = torch.zeros(4, 4, dtype=torch.float64)
        a[:3, :3] = hat(v[i:i + 1, 3:])[0]
        a[:3, 3] = v[i, :3]
        err_exp = max(err_exp, float((m[i] - torch.linalg.matrix_exp(a).T).abs().max()))
    _check(err_exp < 1e-9, f"se3 exp mismatch {err_exp}")
    # initial trajectory (noise 0) reproduces the COLMAP pose for every t
    cams = [_camera("cpu", z=0.3, x=0.1)]
    ang = 0.4
    cams[0]["viewmat"][:3, :3] = torch.tensor([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0],
                                               [-math.sin(ang), 0, math.cos(ang)]])
    bank = bank_mod.TrajectoryBank(cams, {"curve_init_noise": 0.0}, "cpu")
    poses = bank.poses(0, torch.linspace(0, 1, 5))
    err_pose = float((poses - cams[0]["viewmat"]).abs().max())
    _check(err_pose < 1e-4, f"initial trajectory != COLMAP pose ({err_pose})")
    c = bank.cfg
    lr_a, lr_b = curve_lr(999, 1e-2, c), curve_lr(1000 + 14999, 1e-2, c)
    _check(lr_a == 1e-2 and abs(lr_b - 5e-3) < 1e-12, "curve lr schedule")
    return {"bezier_err": err_bez, "se3_exp_err": err_exp, "init_pose_err": err_pose,
            "curve_lr_16k": lr_b}


def blur_synthesis(method, device):
    saved = bank_mod.ACTIVE
    try:
        cam = _camera(device)
        density, params, _ = _live_bank(method, device, [cam])
        _, renderer = _component(method, "deblurgs-renderer")
        bank = density.bank
        with torch.no_grad():  # spread the trajectory: translation control points along x
            bank.trans[0] += torch.linspace(-0.06, 0.06, bank.trans.shape[1], device=device)[
                :, None] * torch.tensor([1.0, 0.3, 0.0], device=device)
        bank.step = 4999
        out = renderer.render(params, cam, sh_degree=1)
        n_sub = out["subframes"].shape[0]
        _check(n_sub == bank.cfg["num_subframes"], f"{n_sub} sub-frames rendered")
        with torch.no_grad():
            vms = bank.subframe_viewmats(0, 5000)
            ref = _plain_render(params, vms, cam["K"], 64, 48, 1, out["background"]).mean(0)
        err = float((out["blur_linear"] - ref).abs().max())
        _check(err < 1e-5, f"blur != mean of sub-frame renders ({err})")
        sharp = renderer.render_sharp(params, cam, 1)["rgb_linear"]
        sharp_bg = sharp - (1 - renderer.render_sharp(params, cam, 1)["alpha"]) * (
            1 - out["background"])
        diff = float((out["blur_linear"] - sharp_bg).abs().mean())
        _check(diff > 1e-3, f"blurred render equals the sharp render ({diff})")
        out["blur_linear"].sum().backward()
        _check(bank.rot.grad is not None and bank.trans.grad is not None
               and float(bank.trans.grad.abs().sum()) > 0, "curve receives no gradient")
        bank.optimizer.zero_grad(set_to_none=True)
        bank.step = 10
        early = renderer.render(params, cam, sh_degree=0)
        _check(early["subframes"].shape[0] == 1, "more than one sub-frame before step 1000")
        t0 = bank.poses(0, torch.zeros(1, device=device))
        _check(float((early["subframe_viewmats"] - t0).abs().max()) < 1e-6, "early pose != t=0")
        early["blur_linear"].sum().backward()
        _check(bank.trans.grad is None, "curve trained before curve_start_iter")
        bank.step = 120000
        late = renderer.render(params, cam, sh_degree=0)
        late["blur_linear"].sum().backward()
        _check(late["subframes"].shape[0] == n_sub and bank.trans.grad is None,
               "curve trained after curve_end_iter")
        with torch.no_grad():
            sharp_eval = renderer.render(params, cam, sh_degree=0)
        _check("subframes" not in sharp_eval, "evaluation render used the trajectory")
        return {"n_subframes": n_sub, "blur_vs_reference": err, "blur_vs_sharp": diff}
    finally:
        bank_mod.ACTIVE = saved


def alignment(method, device):
    saved = bank_mod.ACTIVE
    try:
        cam = _camera(device)
        density, params, _ = _live_bank(method, device, [cam])
        _, renderer = _component(method, "deblurgs-renderer")
        bank = density.bank
        t = bank.times(0).detach()
        f = bank.cfg["num_subframes"]
        _check(float((t - torch.linspace(0, 1, f, device=device)).abs().max()) < 1e-5,
               "initial times != linspace")
        _check(float(t[0]) == 0.0 and float(t[-1]) == 1.0, "end points not fixed")
        lr0 = bank.cfg["curve_alignment_lr"]
        _check(alignment_lr(30000, bank.cfg) == 0.0, "nu lr nonzero before 30000")
        if lr0 > 1e-8:
            _check(abs(alignment_lr(30001, bank.cfg) - lr0) / lr0 < 1e-3, "nu lr start")
            _check(abs(alignment_lr(150000, bank.cfg) - 1e-7) < 1e-12, "nu lr end")
        with torch.no_grad():
            bank.trans[0] += torch.linspace(-0.06, 0.06, bank.trans.shape[1], device=device)[
                :, None] * torch.tensor([1.0, 0.0, 0.0], device=device)
        before = bank.nu.detach().clone()
        bank.step = 39999
        out = renderer.render(params, cam, sh_degree=0)
        target = torch.rand_like(out["blur_linear"])
        (out["blur_linear"] - target).abs().mean().backward()
        gnorm = float(bank.nu.grad.abs().sum())
        _check(gnorm > 0, "nu receives no gradient")
        bank.after_step(40000)
        moved = float((bank.nu.detach() - before).abs().max())
        if lr0 > 1e-8:
            _check(moved > 1e-5, "nu did not move after 30000")
        else:
            _check(moved == 0.0, "nu moved with alignment lr 0")
        return {"nu_grad_l1": gnorm, "nu_moved": moved, "alignment_lr": lr0}
    finally:
        bank_mod.ACTIVE = saved


def gamma(method, device):
    from .losses import create_rgb

    spec = next(c for c in method.components if c.name == "deblurgs-rgb")
    loss = create_rgb(spec)
    g = torch.Generator().manual_seed(1)
    blur = torch.rand(8, 9, 3, generator=g)
    gt = torch.rand(8, 9, 3, generator=g)
    val = float(loss(None, {"blur_linear": blur}, gt))
    ref = float(np.abs(blur.numpy() - np.maximum(gt.numpy(), 1e-8) ** 2.2).mean())
    plain = float(np.abs(blur.numpy() - gt.numpy()).mean())
    _check(abs(val - ref) < 1e-6 and abs(val - plain) > 1e-3, "loss target is not GT^2.2")
    saved = bank_mod.ACTIVE
    bank_mod.ACTIVE = None
    try:
        cam = _camera(device)
        params, _ = _scene(method, device)
        with torch.no_grad():
            params["opacities"].fill_(0.6)
            _, renderer = _component(method, "deblurgs-renderer")
            rgb = renderer.render(params, cam, sh_degree=0)["rgb"]
            lin = _plain_render(params, cam["viewmat"][None], cam["K"], 64, 48, 0,
                                torch.ones(3, device=device))[0]
            ref_img = lin.clamp_min(1e-8).cpu().numpy() ** (1 / 2.2)
        err = float(np.abs(rgb.cpu().numpy() - ref_img).max())
        _check(err < 1e-5, f"evaluation image != render^(1/2.2) ({err})")
        return {"loss_err": abs(val - ref), "eval_err": err}
    finally:
        bank_mod.ACTIVE = saved


def smoothness(method, device):
    from .losses import create_hinge, create_smoothness, lambda_smooth

    spec = next(c for c in method.components if c.name == "deblurgs-smooth")
    loss = create_smoothness(spec)
    g = torch.Generator().manual_seed(2)
    sub = torch.rand(21, 6, 5, 3, generator=g, requires_grad=True)
    step = 50000
    lam = 1e-3 * (1e-5 / 1e-3) ** (step / 150000)
    s = sub.detach().numpy()
    ref = lam * np.abs(s[1:] - s[:-1]).mean()
    val = float(loss(None, {"subframes": sub, "step": step}, None))
    _check(abs(val - ref) < 1e-9, f"smoothness {val} != {ref}")
    l1, lend = lambda_smooth(1, spec.config), lambda_smooth(150000, spec.config)
    _check(abs(l1 - 1e-3 * 0.01 ** (1 / 150000)) < 1e-12 and abs(lend - 1e-5) < 1e-12,
           f"lambda_t schedule {l1} {lend}")
    hspec = next(c for c in method.components if c.name == "deblurgs-hinge")
    x = torch.tensor([-0.5, 0.2, 1.3, 1.0, 0.0])
    hv = float(create_hinge(hspec)({"opacities": x}, {}, None))
    href = 0.1 * np.mean([0.25, 0.0, 0.09, 0.0, 0.0])
    _check(abs(hv - href) < 1e-7, f"hinge {hv} != {href}")
    return {"smooth": val, "hinge": hv}


def densify_annealing(method, device):
    from .density import densify_and_prune, densify_threshold, official_extent, reset_opacity

    spec, density = _component(method, "deblurgs-density")
    cfg = density.cfg
    th = [densify_threshold(s, cfg) for s in (1, 12500, 25000, 60000)]
    _check(abs(th[0] - 4e-4 * (2e-4 / 4e-4) ** (1 / 25000)) < 1e-12, "theta(1)")
    _check(abs(th[1] - math.sqrt(4e-4 * 2e-4)) < 1e-10, "theta(12500)")
    _check(abs(th[2] - 2e-4) < 1e-12 and th[3] == 2e-4, f"theta after annealing {th}")

    def fixture():
        params, opts = _scene(method, device, n=20)
        with torch.no_grad():
            params["opacities"].fill_(0.5)
            params["opacities"][0] = 0.001  # prune
            params["scales"][1] = math.log(50.0)  # huge but opaque: kept (no size prune)
        for p, o in zip(params.values(), opts.values()):
            p.grad = torch.ones_like(p)
            o.step()
        return params, opts

    grads = torch.full((20,), 3e-4, device=device)
    params, opts = fixture()
    early = densify_and_prune(params, opts, grads, densify_threshold(600, cfg), 1.0, cfg)
    params, opts = fixture()
    late = densify_and_prune(params, opts, grads, densify_threshold(25000, cfg), 1.0, cfg)
    _check(early[0] == 0 and early[1] == 0, f"densified at step 600: {early}")
    _check(late[0] + late[1] == 20, f"no densification at step 25000: {late}")
    _check(early[2] == 1 and late[2] >= 1, "opacity prune")
    big = params["scales"].exp().max(dim=1).values.max()
    _check(float(big) > 10, "size-based prune removed an opaque Gaussian")
    with torch.no_grad():
        params["opacities"][:3] = torch.tensor([0.7, -0.2, 0.05], device=device)
    reset_opacity(params, opts, 0.1)
    o = params["opacities"][:3].tolist()
    _check(np.allclose(o, [0.1, 0.0, 0.05], atol=1e-7), f"reset {o}")
    _check(float(opts["opacities"].state[params["opacities"]]["exp_avg"].abs().sum()) == 0,
           "reset kept opacity moments")
    g = torch.Generator().manual_seed(5)
    cams = torch.randn(12, 3, generator=g, dtype=torch.float64)
    pts = torch.randn(500, 3, generator=g, dtype=torch.float64) * 3 + 1
    ext = official_extent(cams, pts)
    c, p = cams.numpy(), pts.numpy()
    r1 = np.percentile(np.linalg.norm(c - p.mean(0), axis=1), 10.0)
    r2 = 1.1 * np.linalg.norm(c - c.mean(0), axis=1).max()
    _check(abs(ext - min(r1, r2)) < 1e-9, "official extent")
    return {"theta": th, "early": early, "late": late, "extent": ext}


def test_pose_opt(method, device):
    from .evaluate import optimize_test_poses

    saved = bank_mod.ACTIVE
    bank_mod.ACTIVE = None
    try:
        _, renderer = _component(method, "deblurgs-renderer")
        params, _ = _scene(method, device, n=600)
        with torch.no_grad():
            params["opacities"].fill_(0.8)
        cam = _camera(device)
        with torch.no_grad():
            gt = renderer.render_sharp(params, cam, 0)["rgb"].clamp(0, 1)
        moved = dict(cam, image=gt)
        vm = cam["viewmat"].clone()
        vm[0, 3] += 0.03
        vm[1, 3] -= 0.02
        a = 0.01
        rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                           [0, 0, 1.0]], device=device)
        vm[:3, :3] = rz @ vm[:3, :3]
        moved["viewmat"] = vm
        poses, hist = optimize_test_poses(renderer, params, [moved], 0,
                                          {"epochs": 400, "step_size": 20})

        def psnr(v):
            with torch.no_grad():
                img = renderer.render_sharp(params, moved, 0, viewmat=v)["rgb"].clamp(0, 1)
            return float(-10 * torch.log10((img - gt).square().mean().clamp_min(1e-12)))

        before, after = psnr(vm), psnr(poses[0])
        e0 = float((vm[:3, 3] - cam["viewmat"][:3, 3]).norm())
        e1 = float((poses[0][:3, 3] - cam["viewmat"][:3, 3]).norm())
        _check(after > before + 3.0 and e1 < 0.5 * e0,
               f"pose optimisation failed: psnr {before:.2f}->{after:.2f}, t err {e0}->{e1}")
        return {"psnr_before": before, "psnr_after": after, "t_err_before": e0,
                "t_err_after": e1, "l1": [hist[0], hist[-1]]}
    finally:
        bank_mod.ACTIVE = saved
