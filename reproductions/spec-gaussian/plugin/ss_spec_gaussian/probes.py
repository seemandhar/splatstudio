"""Mechanism probes for the Spec-Gaussian port. Each raises AssertionError on failure."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization, spherical_harmonics
from splatstudio.components import build
from splatstudio.state import sample_scene

from . import bank as bank_mod
from .asg import asg_encoding, build_network, init_predefined_omega, pack, unpack
from .renderer import BLOB, FEAT, c2f_scale, scaled_camera


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _cfg(method):
    return dict(_spec(method, "renderer").config)


def _camera(device, w=96, h=72):
    K = torch.tensor([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": w, "height": h}


def _scene(method, device, n=1024, seed=3):
    params, optimizers = sample_scene(method, device=device, n=n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        params["means"].copy_((torch.rand(n, 3, generator=g) - 0.5).to(device) * 1.2
                              + torch.tensor([0.0, 0.0, 3.0], device=device))
        params["scales"].fill_(math.log(0.08))
        params["opacities"].fill_(1.0)
        params["shN"].copy_(0.2 * torch.randn(n, 15, 3, generator=g).to(device))
        params[FEAT].copy_(torch.randn(n, params[FEAT].shape[1], generator=g).to(device))
    return params, optimizers


def _setup(method, device):
    renderer, density, losses = build(method)
    params, optimizers = _scene(method, device)
    density.initialize(params, optimizers, scene_scale=2.0)
    bank_mod.ACTIVE = None  # probes use explicit banks only
    renderer.bank = density.bank
    return renderer, density, losses, params, optimizers


def _render(params, camera, colors, sh_degree=None):
    out, alpha, _ = rasterization(
        params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        colors, camera["viewmat"][None], camera["K"][None], camera["width"], camera["height"],
        sh_degree=sh_degree, packed=False)
    return out[0] + (1 - alpha[0])


# ------------------------------------------------------------------------ CPU probes
def _np_axes(n_theta, n_phi):
    zs, xs, ys = [], [], []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2 * np.pi, n_phi):
            z = np.array([np.cos(p) * np.sin(t), np.sin(p) * np.sin(t), np.cos(t)])
            x = np.array([np.cos(p) * np.sin(t + np.pi / 2), np.sin(p) * np.sin(t + np.pi / 2),
                          np.cos(t + np.pi / 2)])
            # Rodrigues rotation of x by pi/2 about z (quaternion rotate in the official code)
            y = x * np.cos(np.pi / 2) + np.cross(z, x) * np.sin(np.pi / 2) \
                + z * np.dot(z, x) * (1 - np.cos(np.pi / 2))
            zs.append(z)
            xs.append(x)
            ys.append(y)
    return np.array(zs), np.array(xs), np.array(ys)


def _np_softplus(v):
    return np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0)


def _np_asg(d, a, la, mu, z, x, y):
    d = d[:, None, :]
    s = np.maximum((d * z[None]).sum(-1), 0)
    e = np.exp(-_np_softplus(la - 1) * (d * x[None]).sum(-1) ** 2
               - _np_softplus(mu - 1) * (d * y[None]).sum(-1) ** 2)
    return a * (s * e)[..., None]  # [N, n, 2]


def asg_basis(method, device="cpu"):
    c = _cfg(method)
    T, P = c["num_theta"], c["num_phi"]
    z, x, y = (t.double().numpy() for t in init_predefined_omega(T, P))
    zr, xr, yr = _np_axes(T, P)
    err_axes = max(np.abs(z - zr).max(), np.abs(x - xr).max(), np.abs(y - yr).max())
    assert err_axes < 1e-5, f"ASG axes differ from the official construction ({err_axes})"
    gram = np.stack([z, x, y], 1)  # [n, 3, 3]
    ortho = np.abs(np.einsum("nij,nkj->nik", gram, gram) - np.eye(3)).max()
    assert ortho < 1e-5, f"ASG frames are not orthonormal ({ortho})"
    rng = np.random.default_rng(0)
    n = 32
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a = rng.normal(size=(n, T, P, 2))
    la = rng.normal(size=(n, T, P, 1)) * 3
    mu = rng.normal(size=(n, T, P, 1)) * 3
    zt, xt, yt = (torch.tensor(v, dtype=torch.float64).view(T, P, 3) for v in (z, x, y))
    got = asg_encoding(torch.tensor(d), torch.tensor(a), torch.tensor(la), torch.tensor(mu),
                       zt, xt, yt).numpy()
    ref = _np_asg(d, a.reshape(n, T * P, 2), la.reshape(n, -1), mu.reshape(n, -1),
                  z, x, y).reshape(n, -1)
    err_enc = np.abs(got - ref).max()
    assert err_enc < 1e-10, f"ASG encoding differs from the formula ({err_enc})"
    # anisotropy: lobe 0 (z = +Z); tilt by the same angle along x and along y
    ang = 0.4
    dx = np.cos(ang) * z[0] + np.sin(ang) * x[0]
    dy = np.cos(ang) * z[0] + np.sin(ang) * y[0]
    one = lambda v: np.ones((1, T, P, 1)) * v  # noqa: E731
    ex, ey = (asg_encoding(torch.tensor(dd[None]), torch.tensor(np.ones((1, T, P, 2))),
                           torch.tensor(one(8.0)), torch.tensor(one(-2.0)), zt, xt, yt)[0, 0].item()
              for dd in (dx, dy))
    assert ex < 0.5 * ey, f"lambda != mu must give an anisotropic lobe ({ex} vs {ey})"
    ix, iy = (asg_encoding(torch.tensor(dd[None]), torch.tensor(np.ones((1, T, P, 2))),
                           torch.tensor(one(3.0)), torch.tensor(one(3.0)), zt, xt, yt)[0, 0].item()
              for dd in (dx, dy))
    assert abs(ix - iy) < 1e-6 * max(abs(ix), 1e-3), "lambda == mu must be rotationally symmetric about z"
    return {"axes_max_err": float(err_axes), "orthonormal_err": float(ortho),
            "encoding_max_err": float(err_enc), "aniso_x": ex, "aniso_y": ey, "n_asg": T * P}


def _np_forward(net, feat, d, view_pe):
    sd = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    T, P = net.num_theta, net.num_phi
    p = (feat @ sd["gaussian_feature.weight"].T + sd["gaussian_feature.bias"]).reshape(-1, T, P, 4)
    z, x, y = (t.double().numpy() for t in init_predefined_omega(T, P))
    kappa = _np_asg(d, p[..., :2].reshape(len(d), -1, 2), p[..., 2].reshape(len(d), -1),
                    p[..., 3].reshape(len(d), -1), z, x, y).reshape(len(d), -1)
    bands = 2.0 ** np.arange(view_pe)
    pts = (d[..., None] * bands).reshape(len(d), -1)
    h = np.concatenate([kappa, d, np.sin(pts), np.cos(pts)], -1)
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    for i, li in enumerate(lin):
        h = h @ sd[f"mlp.{li}.weight"].T + sd[f"mlp.{li}.bias"]
        if i < len(lin) - 1:
            h = np.maximum(h, 0)
    return h


def decoder_architecture(method, device="cpu"):
    c = _cfg(method)
    torch.manual_seed(0)
    net = build_network(c).double()
    lins = [m for m in net.mlp if isinstance(m, torch.nn.Linear)]
    shapes = [tuple(m.weight.shape) for m in lins]
    in_ch = 2 * c["view_pe"] * 3 + 3 + c["num_theta"] * c["num_phi"] * 2
    want = [(c["hidden"], in_ch)] + ([(c["hidden"], c["hidden"])] if c["indoor"] else []) \
        + [(3, c["hidden"])]
    assert in_ch == 31 and shapes == want, f"decoder shapes {shapes} != {want}"
    assert tuple(net.gaussian_feature.weight.shape) == (32, c["feat_dim"]) == (32, 12)
    assert float(lins[-1].bias.abs().max()) == 0.0, "last decoder bias must start at zero"
    rng = np.random.default_rng(1)
    feat = rng.normal(size=(16, c["feat_dim"]))
    d = rng.normal(size=(16, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    with torch.no_grad():
        for m in lins:  # non-zero biases so the reference check covers them
            m.bias.normal_()
        got = net(torch.tensor(feat), torch.tensor(d)).numpy()
    ref = _np_forward(net, feat, d, c["view_pe"])
    err = np.abs(got - ref).max()
    assert err < 1e-10, f"decoder forward differs from the reference ({err})"
    return {"mlp_shapes": shapes, "indoor": bool(c["indoor"]), "forward_err": float(err)}


# ------------------------------------------------------------------------ CUDA probes
def colour_composition(method, device="cuda"):
    renderer, density, _, params, _ = _setup(method, device)
    cam = _camera(device)
    bank = density.bank
    start = _cfg(method)["spec_start"]
    # step <= spec_start: pure SH colour == gsplat's own SH path
    bank.step = start - 1
    off = renderer.render(params, cam, sh_degree=3)
    c = _cfg(method)

    def ref_cam(cur):  # camera the renderer must use at the current step (c2f variants)
        if not c["c2f"]:
            return cam
        return scaled_camera(cam, c2f_scale(cur, c["c2f_start_scale"], c["c2f_until"]))

    ref_off = _render(params, ref_cam(start), torch.cat((params["sh0"], params["shN"]), 1),
                      sh_degree=3)
    err_off = (off["rgb"] - ref_off).abs().max().item()
    assert not off["info"]["spec"]["spec_on"] and err_off < 1e-4, f"SH-only colour ({err_off})"
    off["rgb"].sum().backward()
    g_feat = params[FEAT].grad
    assert g_feat is None or float(g_feat.abs().max()) == 0.0, "spec_feat got gradient <= 3000"
    assert all(p.grad is None for p in bank.network.parameters()), "decoder got gradient early"
    for p in params.values():
        p.grad = None
    # step > spec_start: c = clamp_min(SH + 0.5, 0) + c_s
    bank.step = start
    with torch.no_grad():  # random decoder output layer so c_s is clearly non-zero
        bank.network.mlp[-1].bias.normal_(0, 0.3)
    on = renderer.render(params, cam, sh_degree=3)
    dirs = params["means"] - torch.zeros(3, device=device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    coeffs = torch.cat((params["sh0"], params["shN"]), 1)
    cd = torch.clamp_min(spherical_harmonics(3, dirs, coeffs) + 0.5, 0)
    cs = bank.network(params[FEAT], dirs)
    ref_on = _render(params, ref_cam(start + 1), cd + cs)
    err_on = (on["rgb"] - ref_on).abs().max().item()
    with torch.no_grad():
        gap = (on["rgb"] - _render(params, ref_cam(start + 1), cd)).abs().max().item()
    assert on["info"]["spec"]["spec_on"] and err_on < 1e-4, f"SH + c_s colour ({err_on})"
    assert gap > 1e-2, "specular term has no effect on the render"
    on["rgb"].sum().backward()
    gf = float(params[FEAT].grad.abs().sum())
    gn = sum(float(p.grad.abs().sum()) for p in bank.network.parameters() if p.grad is not None)
    assert gf > 0 and gn > 0, "spec_feat / decoder receive no gradient after spec_start"
    return {"err_sh_only": err_off, "err_sh_plus_spec": err_on, "spec_effect": gap,
            "grad_feat": gf, "grad_decoder": gn}


def decoder_blob(method, device="cuda"):
    renderer, density, _, params, _ = _setup(method, device)
    cam = _camera(device)
    bank = density.bank
    bank.step = 10000
    with torch.no_grad():
        for p in bank.network.parameters():
            p.add_(0.05 * torch.randn_like(p))
    density.bake(params)
    net2 = unpack(params[BLOB].detach(), _cfg(method))
    same = all(torch.equal(a, b) for a, b in zip(bank.network.state_dict().values(),
                                                 net2.state_dict().values()))
    assert same, "unpacked decoder differs from the live decoder"
    with torch.no_grad():
        live = renderer.render(params, cam, 3)["rgb"]
        renderer.bank = None
        fresh = renderer.render(params, cam, 3)["rgb"]  # ss-eval path (decoder from blob)
    err = (live - fresh).abs().max().item()
    assert err < 1e-6, f"eval render from spec_blob differs from training render ({err})"
    # a model baked before spec_start (e.g. a short run) must evaluate without c_s, like training
    bank.step = 1000
    density.bake(params)
    c2f = renderer.cfg["c2f"]
    renderer.cfg["c2f"] = False  # evaluation is always full resolution; compare like for like
    with torch.no_grad():
        renderer.bank = bank
        live_early = renderer.render(params, cam, 3)
        renderer.bank = None
        fresh_early = renderer.render(params, cam, 3)
    renderer.cfg["c2f"] = c2f
    assert not fresh_early["info"]["spec"]["spec_on"], "eval enabled an untrained specular branch"
    err_early = (live_early["rgb"] - fresh_early["rgb"]).abs().max().item()
    assert err_early < 1e-6, f"early-baked eval differs from training render ({err_early})"
    bank.step = 10000
    density.bake(params)
    bad = params[BLOB].detach().clone()
    bad[[2, 3]] = bad[[3, 2]] + 1.0
    try:
        unpack(bad, _cfg(method))
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected, "corrupted blob was accepted"
    return {"floats": int(pack(bank.network).numel()), "eval_vs_live_err": err,
            "early_eval_vs_live_err": err_early}


def absgrad_densify(method, device="cuda"):
    renderer, density, _, params, _ = _setup(method, device)
    cfg = _spec(method, "density").config
    strat = density.base.strategy
    assert density.requires == {"gradient.screen.absolute"} and strat.absgrad
    assert abs(strat.grow_grad2d - 0.0005) < 1e-12 and cfg["grow_grad2d"] == 0.0005
    cam = _camera(device)
    out = renderer.render(params, cam, 3)
    out["info"]["means2d"].retain_grad()
    h, w = out["rgb"].shape[:2]
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    sign = (((xx // 3) + (yy // 3)) % 2 * 2 - 1).float()[..., None]
    (out["rgb"] * sign).sum().backward()
    signed = out["info"]["means2d"].grad[0].abs()
    absg = out["info"]["means2d"].absgrad[0].abs()
    assert bool((absg + 1e-6 >= signed).all()), "absgrad must bound the signed gradient"
    ratio = float(absg.sum() / signed.sum().clamp_min(1e-12))
    assert ratio > 1.05, f"absgrad equals the signed gradient (ratio {ratio})"
    return {"absgrad_over_signed": ratio, "grow_grad2d": strat.grow_grad2d}


def official_schedules(method, device="cuda"):
    renderer, density, _, params, optimizers = build(method) + _scene(method, device)
    lr0 = optimizers["means"].param_groups[0]["lr"]
    optimizers["means"].param_groups[0]["lr"] = lr0 * 2.0  # runtime: 1.6e-4 * scene_scale
    density.initialize(params, optimizers, scene_scale=2.0)
    bank_mod.ACTIVE = None
    means_lr = optimizers["means"].param_groups[0]["lr"]
    assert abs(means_lr - 1.6e-4 * 5) < 1e-12, f"means lr {means_lr} != 1.6e-4 * 5"
    f = next(x for x in method.fields if x.name == FEAT)
    assert f.learning_rate == 0.0025 and f.initial_value == 0.0 and f.shape == (12,)
    bank = density.bank
    sched = {s: bank.schedule(s) for s in (0, 15000, 30000, 40000)}
    want = {0: 0.0025, 15000: (0.0025 + 0.000125) / 2, 30000: 0.000125, 40000: 0.000125}
    for s, v in want.items():
        assert abs(sched[s] - v) < 1e-12, f"decoder lr at {s}: {sched[s]} != {v}"
    before = [p.detach().clone() for p in bank.network.parameters()]
    lr = bank.after_step(3000)  # no gradient: no update
    assert all(torch.equal(a, b) for a, b in zip(before, bank.network.parameters()))
    assert abs(lr - (0.0025 * 0.9 + 0.000125 * 0.1)) < 1e-12
    bank.step = 5000
    renderer.bank = bank
    out = renderer.render(params, _camera(device), 3)
    out["rgb"].sum().backward()
    bank.after_step(5001)
    moved = any(not torch.equal(a, b) for a, b in zip(before, bank.network.parameters()))
    assert moved and bank.updates == 1, "decoder must be stepped once per training step"
    assert _cfg(method)["spec_start"] == 3000
    return {"means_lr": means_lr, "decoder_lr": sched, "updates": bank.updates}


def c2f_resolution(method, device="cuda"):
    renderer, density, losses, params, _ = _setup(method, device)
    c = _cfg(method)
    cam = _camera(device, 192, 144)
    sizes = {}
    for cur in (1, 2501, 5001, 20000):
        density.bank.step = cur - 1
        with torch.no_grad():
            sizes[cur] = tuple(renderer.render(params, cam, 3)["rgb"].shape[:2])
    if c["c2f"]:
        for cur, hw in sizes.items():
            s = min(0.25 + 0.75 * cur / 5000, 1.0)
            want = (max(int(round(144 * s)), 16), max(int(round(192 * s)), 16))
            assert hw == want, f"step {cur}: size {hw} != {want}"
            assert abs(c2f_scale(cur, 0.25, 5000) - s) < 1e-12
        name, loss = losses[0]
        assert name == "spec-photometric"
        target = torch.rand(144, 192, 3, device=device)
        rgb = torch.rand(36, 48, 3, device=device, requires_grad=True)
        tgt = F.interpolate(target.permute(2, 0, 1)[None], size=(36, 48), mode="bilinear",
                            align_corners=False)[0].permute(1, 2, 0)
        from splatstudio.losses import ssim
        ref = 0.8 * (rgb - tgt).abs().mean() + 0.2 * (1 - ssim(rgb, tgt))
        got = loss(params, {"rgb": rgb}, target)
        assert abs(float(got - ref)) < 1e-6, "c2f loss must use the bilinearly resized target"
        assert sizes[1][0] < 144, "coarse-to-fine must start at low resolution"
    else:
        assert all(hw == (144, 192) for hw in sizes.values()), \
            f"resolution must stay constant without c2f ({sizes})"
    return {"c2f": bool(c["c2f"]), "sizes": {str(k): list(v) for k, v in sizes.items()}}
