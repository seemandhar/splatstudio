"""Mechanism probes for the SplatFields port. Each raises if its mechanism is removed or wrong."""

import math

import torch
import torch.nn.functional as F
from splatstudio.components import build
from splatstudio.state import sample_scene

from . import density as density_mod
from . import losses, network


def _density_spec(method):
    return next(c for c in method.components if c.kind == "density")


def _net(method, device="cpu"):
    cfg = _density_spec(method).config
    return network.SplatFieldsNet(cfg["net"], cfg["net_seed"]).to(device)


def _close(a, b, tol, what):
    err = float((a - b).abs().max())
    if not math.isfinite(err) or err > tol:
        raise AssertionError(f"{what}: max abs err {err} > {tol}")
    return err


def _bilinear_zero(plane, u, v):
    """Independent grid_sample(bilinear, zeros, align_corners=False) of one [C,H,W] plane."""
    C, H, W = plane.shape
    x = ((u + 1) * W - 1) / 2
    y = ((v + 1) * H - 1) / 2
    x0, y0 = torch.floor(x), torch.floor(y)
    out = torch.zeros(C, len(u), dtype=plane.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            vals = plane[:, yi.clamp(0, H - 1).long(), xi.clamp(0, W - 1).long()]
            out += vals * (w * ok)[None]
    return out


def triplane_generator(method, device):
    net = _net(method)
    tri = net.encoder
    if tri is None:
        raise AssertionError("tri-plane CNN generator is disabled")
    with torch.no_grad():
        planes = tri.planes()
        again = tri.planes()
    if tuple(planes.shape) != (3, 16, 160, 160) or tuple(tri.noise.shape) != (3, 1, 8, 20, 20):
        raise AssertionError(f"planes {tuple(planes.shape)}, noise {tuple(tri.noise.shape)}")
    if not torch.equal(planes, again) or float(planes.std()) == 0:
        raise AssertionError("planes must be a deterministic, non-constant function of noise")
    zero = [m for m in tri.modules() if isinstance(m, network.ResnetBlock2D)]
    if len(zero) != 3 * (2 + 4 * 2) or any(float(m.conv2.weight.abs().max()) != 0 for m in zero):
        raise AssertionError("expected 30 zero-initialised residual blocks (official init)")
    # attention block vs an independent softmax formula (random output projection)
    att = network.AttentionBlock(32, 32)
    torch.nn.init.normal_(att.to_out[0].weight, std=0.1)
    x = torch.randn(2, 32, 5, 6)
    with torch.no_grad():
        t = F.group_norm(x.reshape(2, 32, 30), 32, att.group_norm.weight, att.group_norm.bias,
                         1e-6).transpose(1, 2)
        q, k, v = att.to_q(t), att.to_k(t), att.to_v(t)
        a = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(32), dim=-1) @ v
        ref = att.to_out[0](a).transpose(1, 2).reshape(2, 32, 5, 6) + x
        err_att = _close(att(x), ref, 1e-5, "attention")
    # feature sampling vs an independent bilinear formula, incl. points outside [-1, 1]
    rand_planes = torch.randn(3, 16, 160, 160)
    tri.planes = lambda: rand_planes
    xyz = torch.rand(500, 3) * 2.4 - 1.2
    with torch.no_grad():
        got = tri(xyz)
    ref = torch.cat([_bilinear_zero(rand_planes[i], xyz[:, a], xyz[:, b]).T
                     for i, (a, b) in enumerate(network.TriPlane.AXES)], dim=1)
    err_s = _close(got, ref, 1e-4, "tri-plane sampling")
    n_params = sum(p.numel() for p in net.parameters())
    return {"planes": list(planes.shape), "attention_err": err_att, "sampling_err": err_s,
            "network_params": n_params}


def _ref_mlp(mlp, xyz, feat):
    h_in = torch.cat([xyz] + [fn(xyz * (2.0 ** i)) for i in range(mlp.multires)
                              for fn in (torch.sin, torch.cos)] + [feat], dim=-1)
    h = h_in
    for i, lin in enumerate(mlp.net):
        h = F.leaky_relu(h @ lin.weight.T + lin.bias, 0.01)
        if i in mlp.skips and i != len(mlp.net) - 1:
            h = torch.cat([h_in, h], -1)
    return h


def splat_fields(method, device):
    net = _net(method)
    xyz = torch.rand(300, 3) * 2 - 1
    base = torch.rand(300, 3) * 0.02 + 0.001
    pred = net(xyz, base)
    with torch.no_grad():
        f = net.features(xyz)
        if f is None or f.shape[1] != 48:
            raise AssertionError("tri-plane features must be 48-dimensional")
        d = _ref_mlp(net.mlp_deform, xyz, f)
        e1 = _close(pred["means"], xyz + d, 1e-5, "means = x + D(x, f)")
        e2 = _close(pred["scales"], _ref_mlp(net.mlp_scale, pred["means"], f) + base, 1e-5,
                    "scales = S(p, f) + base")
        e3 = _close(pred["opacities"], torch.sigmoid(_ref_mlp(net.mlp_opacity, pred["means"],
                                                              f))[:, 0], 1e-5, "opacity")
        e4 = _close(pred["rgb"], torch.sigmoid(_ref_mlp(net.mlp_rgb, pred["means"], f)), 1e-5,
                    "rgb")
        e5 = _close(pred["quats"], F.normalize(_ref_mlp(net.mlp_rotation, pred["means"], f)),
                    1e-5, "quats")
    if (pred["means"] - xyz).abs().max() == 0:
        raise AssertionError("deform field has no effect")
    sum(v.sum() for v in pred.values()).backward()
    for name in ("mlp_deform", "mlp_rgb", "mlp_scale", "mlp_opacity", "mlp_rotation", "refine"):
        g = [p.grad for p in getattr(net, name).parameters()]
        if any(x is None for x in g) or sum(float(x.abs().sum()) for x in g) == 0:
            raise AssertionError(f"{name}: no gradient")
    enc = [p.grad for p in net.encoder.parameters() if p.grad is not None]
    if not enc or sum(float(x.abs().sum()) for x in enc) == 0:
        raise AssertionError("tri-plane CNN receives no gradient")
    return {"err_means": e1, "err_scales": e2, "err_opacity": e3, "err_rgb": e4,
            "err_quats": e5, "mean_offset": float((pred["means"] - xyz).abs().mean())}


def _camera(device):
    # camera at z = -3 looking at the origin (OpenCV), 64 x 64
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 3.0
    K = torch.tensor([[80.0, 0, 32], [0, 80.0, 32], [0, 0, 1]], device=device)
    return {"viewmat": viewmat, "K": K, "width": 64, "height": 64}


def _scene(method, device, n=400):
    params, opts = sample_scene(method, device, n=n)
    with torch.no_grad():
        params["means"].copy_(torch.rand(n, 3, device=device) * 1.2 - 0.6)
        params["scales"].fill_(math.log(0.03))
    return params, opts


def renderer_uses_fields(method, device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, _, _ = build(method)
    renderer.net = _net(method, device)
    params, _ = _scene(method, device)
    cam = _camera(device)
    a = renderer.render(params, cam)
    if "sf_means" not in a:
        raise AssertionError("renderer did not use the neural fields")
    with torch.no_grad():
        params["sh0"].normal_()
        params["opacities"].normal_()
        params["quats"].normal_()
    b = renderer.render(params, cam)
    e_base = float((a["rgb"] - b["rgb"]).abs().max())
    if e_base > 1e-6:
        raise AssertionError(f"render depends on base attributes ({e_base})")
    b["rgb"].mean().backward()
    if params["means"].grad is not None or params["scales"].grad is not None:
        raise AssertionError("anchors must be detached (official get_xyz.detach())")
    grads = [p.grad for p in renderer.net.mlp_rgb.parameters()]
    if any(g is None for g in grads) or sum(float(g.abs().sum()) for g in grads) == 0:
        raise AssertionError("no gradient reaches the colour field")
    with torch.no_grad():
        renderer.net.mlp_rgb.net[-1].bias.add_(1.0)
        c = renderer.render(params, cam)
    e_net = float((a["rgb"] - c["rgb"]).abs().max())
    if e_net < 1e-3:
        raise AssertionError("render does not depend on the network")
    return {"base_attr_effect": e_base, "network_effect": e_net,
            "coverage": float(a["alpha"].mean())}


def bake(method, device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    params, opts = _scene(method, device)
    cam = _camera(device)
    try:
        density.initialize(params, opts, 1.0)
        with torch.no_grad():
            train = renderer.render(params, cam)
            if "sf_means" not in train:
                raise AssertionError("training render must use the network")
            density.bake(params)
            if not density.net.baked:
                raise AssertionError("network not marked baked")
            baked = renderer.render(params, cam, sh_degree=3)
        if "sf_means" in baked:
            raise AssertionError("baked render still uses the network")
        network.ACTIVE["net"] = None
        with torch.no_grad():
            fresh = build(method)[0].render(params, cam, sh_degree=3)
        err = _close(fresh["rgb"], train["rgb"], 2e-4, "baked vs network render")
        err_a = _close(fresh["alpha"], train["alpha"], 2e-4, "baked vs network alpha")
    finally:
        network.ACTIVE["net"] = None
    return {"rgb_err": err, "alpha_err": err_a, "coverage": float(train["alpha"].mean())}


def norm_loss(method, device):
    _, _, ls = build(method)
    loss = dict(ls)["splat-norm"]
    net = _net(method)
    xyz = torch.rand(200, 3) * 2 - 1
    pred = net(xyz, torch.full((200, 3), 0.01))
    value = loss({}, {"sf_means": pred["means"], "rgb": None}, None)
    ref = 0.01 * torch.sqrt((pred["means"] ** 2).sum(1)).mean()
    err = _close(value, ref, 1e-7, "splat norm")
    value.backward()
    g = sum(float(p.grad.abs().sum()) for p in net.mlp_deform.parameters())
    if g == 0 or any(p.grad is not None for p in net.mlp_rgb.parameters()):
        raise AssertionError("norm loss must act through the deform field only")
    return {"value": float(value), "err": err, "deform_grad_l1": g}


def mask_loss(method, device):
    _, _, ls = build(method)
    loss = dict(ls)["mask"]
    g = torch.Generator().manual_seed(0)
    vms = torch.stack([torch.eye(4), torch.eye(4)])
    vms[1, 0, 3] = 0.5
    masks = (torch.rand(2, 8, 9, generator=g) * 255).to(torch.uint8)
    loss.state["table"] = losses.MaskTable(vms.numpy(), masks.numpy())
    alpha = (torch.rand(8, 9, 1, generator=g) * 1.4 - 0.2).requires_grad_(True)
    value = loss({}, {"alpha": alpha, "viewmat": vms[1]}, None)
    ref = 0.1 * (alpha.detach()[..., 0].clamp(0, 1) - masks[1].float() / 255).abs().mean()
    err = _close(value, ref, 1e-7, "mask L1")
    wrong = 0.1 * (alpha.detach()[..., 0].clamp(0, 1) - masks[0].float() / 255).abs().mean()
    if abs(float(wrong - ref)) < 1e-4:
        raise AssertionError("fixture masks are not distinguishable")
    value.backward()
    if alpha.grad is None or float(alpha.grad.abs().sum()) == 0:
        raise AssertionError("mask loss has no gradient on the rendered alpha")
    other = torch.eye(4)
    other[1, 3] = 1.0
    try:
        loss({}, {"alpha": alpha, "viewmat": other}, None)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown camera must raise")
    return {"value": float(value), "err": err}


def _fake_info(n, device="cpu"):
    means2d = torch.zeros(1, n, 2, device=device, requires_grad=True)
    means2d.grad = torch.zeros_like(means2d)  # no densification growth in the fixture
    return {"means2d": means2d, "width": 64, "height": 64, "n_cameras": 1,
            "radii": torch.ones(1, n, 2, dtype=torch.int32, device=device),
            "gaussian_ids": None}


def _no_reset(density, method, steps=(3000, 6000)):
    """Run the density's after_backward at reset-schedule steps; opacity must not change."""
    params, opts = sample_scene(method, "cpu")
    with torch.no_grad():
        params["opacities"].fill_(0.0)
        params["scales"].fill_(math.log(0.01))  # below every grow/prune size threshold
    density.initialize(params, opts, 1.0)
    before = params["opacities"].detach().clone()
    try:
        for step in steps:
            density.before_backward(params, opts, step, {"info": _fake_info(len(before))})
            density.after_backward(params, opts, step, {"info": _fake_info(len(before))})
    finally:
        network.ACTIVE["net"] = None
    if len(params["opacities"]) != len(before):
        raise AssertionError("fixture unexpectedly densified")
    return float((params["opacities"].detach() - before).abs().max())


def _final_step_refine(density, method):
    """Huge screen gradients at step total_steps - 100 must refine, at total_steps must not."""
    total = int(density.cfg["total_steps"])
    counts = {}
    for step in (total - 100, total):
        params, opts = sample_scene(method, "cpu")
        with torch.no_grad():
            params["opacities"].fill_(0.0)
            params["scales"].fill_(math.log(0.01))
        density.initialize(params, opts, 1.0)
        n = len(params["means"])
        info = _fake_info(n)
        info["means2d"].grad = torch.ones_like(info["means2d"])
        try:
            density.before_backward(params, opts, step, {"info": info})
            density.after_backward(params, opts, step, {"info": info})
        finally:
            network.ACTIVE["net"] = None
        counts[step] = len(params["means"]) - n
    if counts[total - 100] <= 0:
        raise AssertionError("fixture does not trigger a refine before the final step")
    if counts[total] != 0:
        raise AssertionError(f"refine fired at total_steps {total} (+{counts[total]})")
    return {str(k): v for k, v in counts.items()}


def schedule(method, device):
    d = density_mod.SplatFieldsDensity(_density_spec(method))
    lrs = [d.lr(s) for s in (0, 20000, 40000, 50000)]
    ref = [8e-4, math.sqrt(8e-4 * 1.6e-6), 1.6e-6, 1.6e-6]
    if any(abs(a - b) > 1e-12 + 1e-6 * b for a, b in zip(lrs, ref)):
        raise AssertionError(f"network LR schedule {lrs} != {ref}")
    s = d.strategy
    got = (s.refine_start_iter, s.refine_every, s.refine_stop_iter, s.grow_grad2d,
           s.grow_scale3d, s.prune_scale3d, s.prune_opa, s.reset_every)
    if got != (500, 100, 40000, 0.0002, 0.01, 0.1, 0.005, 3000):
        raise AssertionError(f"densification schedule {got}")
    if d.cfg["total_steps"] != 40000:
        raise AssertionError("bake must happen at the paper's 40k iterations")
    reset = _no_reset(density_mod.SplatFieldsDensity(_density_spec(method)), method)
    if reset != 0:
        raise AssertionError("opacity must not be reset (official train.py never resets)")
    final = _final_step_refine(density_mod.SplatFieldsDensity(_density_spec(method)), method)
    return {"lr": lrs, "refine": list(got), "opacity_change_at_3000": reset,
            "final_step_refine": final}


def stage1_schedule(method, device):
    spec = _density_spec(method)
    d = density_mod.Stage1Density(spec)
    lrs = [d.means_lr(s) for s in (0, 15000, 30000, 40000)]
    ref = [8e-4, math.sqrt(8e-4 * 8e-6), 8e-6, 8e-6]
    if any(abs(a - b) > 1e-12 + 1e-6 * b for a, b in zip(lrs, ref)):
        raise AssertionError(f"xyz LR schedule {lrs} != {ref}")
    params, opts = sample_scene(method, "cpu")
    d.initialize(params, opts, 3.0)
    if abs(opts["means"].param_groups[0]["lr"] - 8e-4) > 1e-12:
        raise AssertionError("initial xyz LR must be 1.6e-4 * 5 (not scene-scaled)")
    d.after_backward(params, opts, 100, {"info": _fake_info(len(params["means"]))})
    if abs(opts["means"].param_groups[0]["lr"] - d.means_lr(100)) > 1e-15:
        raise AssertionError("xyz LR not overridden after the step")
    s = d.strategy
    if (s.refine_start_iter, s.refine_every, s.refine_stop_iter) != (500, 100, 40000):
        raise AssertionError("densification schedule differs from the official 3DGS run")
    reset = _no_reset(density_mod.Stage1Density(spec), method)
    if reset != 0:
        raise AssertionError("stage-1 3DGS must not reset opacity")
    final = _final_step_refine(density_mod.Stage1Density(spec), method)
    return {"lr": lrs, "opacity_change_at_3000": reset, "final_step_refine": final}
