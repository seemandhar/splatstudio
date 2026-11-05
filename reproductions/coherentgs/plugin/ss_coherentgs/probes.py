"""Mechanism probes for CoherentGS; each raises if its mechanism is removed or wrong."""

import functools
import math
import operator

import numpy as np
import torch
import torch.nn.functional as F

from . import bank as bank_mod
from .bank import Bank, masked_channel_sum
from .decoder import ImplicitDecoder, layer_specs, posenc, upsampling_factor_padding


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _density_cfg(method, **over):
    from .density import DEFAULTS

    return {**DEFAULTS, **_spec(method, "coherent-density").config, **over}


def geometry_fixture(n=2, h=128, w=160, c=3, f=160.0, mono_scale=(1.0, 1.6), mono_offset=(0.0, -0.8)):
    """Two cameras (baseline 0.3 along x) viewing the plane Z = 3 + 0.2 X.

    Returns a sidecar dict (as prepare_cgs_llff.py writes) whose monocular depth is the true
    depth distorted per view by z_mono = (z_true - o_i) / s_i, exact flows from the geometry,
    all-ones consistency weights, depth-layer masks, and the true depth.
    """
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    centers = [np.array([0.3 * i, 0.0, 0.0]) for i in range(n)]
    viewmats = []
    for cc in centers:
        vm = np.eye(4)
        vm[:3, 3] = -cc
        viewmats.append(vm)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rx, ry = (u - K[0, 2]) / f, (v - K[1, 2]) / f
    z_true, world = [], []
    for cc in centers:  # ray X = c + z (rx, ry, 1); plane Z = 3 + 0.2 X
        z = (3 + 0.2 * cc[0]) / (1 - 0.2 * rx)
        z_true.append(z)
        world.append(np.stack([cc[0] + z * rx, cc[1] + z * ry, z], -1))
    flows = np.zeros((n, n, h, w, 2))
    inside = np.zeros((n, n, h, w))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = world[i] - centers[j]
                flows[i, j, ..., 0] = f * p[..., 0] / p[..., 2] + K[0, 2] - u
                flows[i, j, ..., 1] = f * p[..., 1] / p[..., 2] + K[1, 2] - v
                qx, qy = u + flows[i, j, ..., 0], v + flows[i, j, ..., 1]
                inside[i, j] = (qx >= 1) & (qx <= w - 2) & (qy >= 1) & (qy <= h - 2)
    z_true = np.stack(z_true)
    z_mono = np.stack([(z_true[i] - mono_offset[i]) / mono_scale[i] for i in range(n)])
    edges = np.quantile(z_true, np.linspace(0, 1, c + 1))
    edges[-1] += 1
    dep_mask = np.stack([np.stack([(z_true[i] >= edges[k]) & (z_true[i] < edges[k + 1])
                                   for k in range(c)]) for i in range(n)]).astype(np.uint8)
    return {"viewmats": np.stack(viewmats).astype(np.float32), "K": K.astype(np.float32),
            "z_mono": z_mono.astype(np.float32), "dep_mask": dep_mask,
            "flows": flows.astype(np.float32), "flow_w": inside.astype(np.float32),
            "z_true": z_true.astype(np.float32)}


def _bank(method, device, data=None, **over):
    data = data or geometry_fixture()
    cfg = _density_cfg(method, n_views=data["z_mono"].shape[0],
                       mask_channels=data["dep_mask"].shape[1], **over)
    return Bank(data["viewmats"], data["K"], data["z_mono"], data["dep_mask"], data["flows"],
                data["flow_w"], cfg, device), data


def ray_structure(method, device):
    """Eq. 3: every Gaussian lies on its own pixel ray at depth D_init + 50 chsum(tanh S)."""
    bank, data = _bank(method, device, ngf_depth=2, ngf_opac=2)
    bank.build_decoders(0)
    with torch.no_grad():
        for p in bank.decoder.parameters():  # non-trivial residuals
            p.add_(torch.randn_like(p) * 0.3)
        means, _ = bank.gaussians()
        res = bank.residual()
    n, h, w = data["z_mono"].shape
    K = data["K"]
    z = torch.as_tensor(data["z_mono"], device=device).reshape(-1, 1) + res
    if float(res.abs().max()) < 1e-3:
        raise AssertionError("decoder residual is trivially zero")
    # independent: project means back to each camera -> integer pixel grid and depth z
    pts = means.reshape(n, h * w, 3)
    vm = torch.as_tensor(data["viewmats"], device=device)
    cam = pts @ vm[:, :3, :3].transpose(1, 2) + vm[:, None, :3, 3]
    u = K[0, 0] * cam[..., 0] / cam[..., 2] + K[0, 2]
    v = K[1, 1] * cam[..., 1] / cam[..., 2] + K[1, 2]
    gv, gu = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    err_uv = max(float((u - gu.reshape(1, -1)).abs().max()), float((v - gv.reshape(1, -1)).abs().max()))
    err_z = float((cam[..., 2].reshape(-1, 1) - z).abs().max())
    if err_uv > 1e-3 or err_z > 1e-4:
        raise AssertionError(f"Gaussians left their pixel rays (uv err {err_uv}, z err {err_z})")
    return {"gaussians": int(means.shape[0]), "max_pixel_error": err_uv, "max_depth_error": err_z,
            "residual_absmax": float(res.abs().max())}


def decoder_architecture(method, device):
    """Official decoder: output H x W for any size, parameter count of the official layer spec."""
    cfg = _density_cfg(method)
    out = {}
    for h, w, ngf in ((128, 160, 2), (131, 257, 2), (378, 504, int(cfg["ngf_depth"]))):
        dec = ImplicitDecoder(h, w, ngf=ngf, in_ch=20, out_ch=int(cfg["mask_channels"]))
        px, _, _, _ = upsampling_factor_padding(h, w)
        specs = layer_specs(ngf, len(px))
        expected = 20 * specs[0] + specs[0]  # 1x1 first conv
        size = specs[0] + 2
        for s in specs[1:]:
            expected += size * s * 9 + s
            size = s
        expected += size * int(cfg["mask_channels"]) * 9 + int(cfg["mask_channels"])
        count = sum(p.numel() for p in dec.parameters())
        if count != expected:
            raise AssertionError(f"decoder {h}x{w} has {count} params, official spec {expected}")
        if h * w <= 50000:
            y = dec(torch.randn(2, 20, 1, 1))
            if tuple(y.shape) != (2, int(cfg["mask_channels"]), h, w):
                raise AssertionError(f"decoder output {tuple(y.shape)} for {h}x{w}")
        out[f"{h}x{w}"] = count
    pe = posenc(3)
    x = np.linspace(0, 1, 3)
    ref = np.stack(functools.reduce(operator.iadd, [[np.sin(x * 2.0 ** k), np.cos(x * 2.0 ** k)] for k in range(10)], []), -1)
    if pe.shape != (3, 20, 1, 1) or np.abs(pe[..., 0, 0].numpy() - ref).max() > 1e-5:
        raise AssertionError("posenc differs from sin/cos of 2^k x, k = 0..9")
    if (int(cfg["ngf_depth"]), int(cfg["ngf_opac"])) != (15, 10):
        raise AssertionError("3-view capacities must be 15 (depth) / 10 (opacity), supp. Sec. 1")
    return {"param_counts": out, "posenc_shape": list(pe.shape)}


def masked_residual(method, device):
    """chsum(S . tanh(f)) with S normalised: one-hot picks a channel, empty masks give 0."""
    torch.manual_seed(0)
    raw = torch.randn(2, 3, 4, 5, device=device) * 2
    mask = torch.zeros(2, 3, 4, 5, device=device)
    mask[:, 1, :2] = 1
    mask[:, 2, 2:3] = 1  # row 3 stays empty
    norm = mask / (mask.sum(1, keepdim=True) + 1e-10)
    cfg = _density_cfg(method)
    got = masked_channel_sum(raw, norm, cfg["depth_scale"]).reshape(2, 4, 5)
    ref = torch.zeros(2, 4, 5, device=device)
    ref[:, :2] = 50 * torch.tanh(raw[:, 1, :2])
    ref[:, 2] = 50 * torch.tanh(raw[:, 2, 2])
    if not torch.allclose(got, ref, atol=1e-5):
        raise AssertionError("masked residual differs from 50 tanh(f_c) of the pixel's layer")
    if float(cfg["depth_scale"]) != 50 or float(cfg["opac_scale"]) != 100:
        raise AssertionError("official residual scales are 50 (depth) / 100 (opacity)")
    bank, _ = _bank(method, device, ngf_depth=2, ngf_opac=2)
    bank.build_decoders(0)
    with torch.no_grad():
        _, logits = bank.gaussians()
        ref_logits = math.log(0.5 / 0.5) + masked_channel_sum(
            bank.decoder_opac(bank.inp), bank.mask_norm, 100.0)[:, 0]
    if not torch.allclose(logits, ref_logits, atol=1e-5) or float(cfg["init_opacity"]) != 0.5:
        raise AssertionError("opacity = sigmoid(logit(0.5) + 100 chsum(...)) violated")
    return {"max_error": float((got - ref).abs().max()), "init_opacity": cfg["init_opacity"]}


def alignment(method, device):
    """Eq. 8 on a two-view plane with distorted monocular depth: loss drops, depth recovered."""
    bank, data = _bank(method, device, ngf_depth=2, ngf_opac=2)
    before = float(bank.warping_loss(bank.xyz(bank.z_mono), order=[0, 1]))
    history = bank.align()
    after = float(bank.warping_loss(bank.xyz(bank.z), order=[0, 1]))
    z_true = torch.as_tensor(data["z_true"], device=device).reshape(-1, 1)
    rel_before = float(((bank.z_mono - z_true).abs() / z_true).mean())
    rel_after = float(((bank.z - z_true).abs() / z_true).mean())
    floor = float(bank.warping_loss(bank.xyz(z_true), order=[0, 1]))  # backwarp half-pixel shift
    if not (after < 0.2 * before and after < 1.5 * floor + 1e-4):
        raise AssertionError(f"alignment did not minimise the flow loss ({before} -> {after}, "
                             f"true-depth floor {floor})")
    if not rel_after < 0.25 * rel_before:
        raise AssertionError(f"alignment did not recover depth ({rel_before} -> {rel_after})")
    if len(history) != int(bank.cfg["align_steps"]) or int(bank.cfg["align_steps"]) != 1000:
        raise AssertionError("official alignment runs 1000 iterations")
    return {"loss_before": before, "loss_after": after, "loss_true_depth": floor,
            "depth_rel_err_before": rel_before,
            "depth_rel_err_after": rel_after, "scale": bank.align_scale.tolist(),
            "offset": bank.align_offset.tolist()}


def radius(method, device):
    """Eq. 9 / set_scaling: log(0.6 D / f) and the reset schedule (every 100 its before 15000)."""
    from splatstudio.state import initialize

    from .density import Density

    data = geometry_fixture()
    spec = _spec(method, "coherent-density")
    from dataclasses import replace

    dens = Density(replace(spec, config={**spec.config, "n_views": 2, "mask_channels": 3,
                                         "ngf_depth": 2, "ngf_opac": 2, "align_steps": 50}))
    b0 = Bank(data["viewmats"], data["K"], data["z_mono"], data["dep_mask"], data["flows"],
              data["flow_w"], dens.cfg, device)
    pts = b0.xyz(b0.z_mono)
    params, opts = initialize(method, pts, torch.full_like(pts, 0.5))
    try:
        dens.initialize(params, opts, sidecar=data)
        bank = dens.bank
        f = float(data["K"][1, 1])
        ref = torch.log(0.6 * bank.z.abs() / f).repeat(1, 3)  # independent: 1.2 * D / (2 f)
        if not torch.allclose(params["scales"], ref, atol=1e-5):
            raise AssertionError("initial radius != log(1.2 * D_init / (2 f))")
        with torch.no_grad():
            for p in bank.decoder.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        resets = []
        for step in (98, 99, 100, 7998, 7999, 8000):
            bank.step = step
            before = params["scales"].detach().clone()
            dens.after_backward(params, opts, step, {})
            resets.append(not torch.equal(before, params["scales"]))
        expect = [False, True, False, False, False, False]  # it 7100 resets; 15000 does not
        if resets != expect:
            raise AssertionError(f"reset schedule {resets} != {expect}")
        ref2 = torch.log(0.6 * (bank.z + bank.residual()).abs() / f).repeat(1, 3)
        bank.step = 199
        dens.after_backward(params, opts, 199, {})
        if not torch.allclose(params["scales"], ref2.detach(), atol=1e-4, rtol=1e-4):
            raise AssertionError("radius reset must use D_init + decoder residual")
        for name, lr in (("shN", 5e-5), ("scales", 1e-4), ("quats", 1e-5), ("means", 0.0),
                         ("opacities", 0.0), ("sh0", 0.0)):
            if abs(opts[name].param_groups[0]["lr"] - lr) > 1e-12:
                raise AssertionError(f"{name} lr {opts[name].param_groups[0]['lr']} != {lr}")
    finally:
        bank_mod.ACTIVE["bank"] = None
    return {"resets_at": [s for s, r in zip((98, 99, 100, 7998, 7999, 8000), resets) if r]}


def tv_loss(method, device):
    """Eq. 4-5 with the code's gamma = it / 20000 (it = 7000 + step), weight 5."""
    from .losses import smooth_value

    torch.manual_seed(1)
    disp = torch.rand(1, 6, 7, device=device)
    mask = (torch.rand(3, 6, 7, device=device) > 0.5).float()
    gamma = 0.4
    def ref_tv(x):  # independent finite-difference L1
        return (np.abs(np.diff(x, axis=2)).mean() + np.abs(np.diff(x, axis=1)).mean())
    d, m = disp.cpu().numpy(), mask.cpu().numpy()
    ref = gamma * ref_tv(m * d) + (1 - gamma) * ref_tv(d)
    got = float(smooth_value(disp, mask, gamma))
    if abs(got - ref) > 1e-6:
        raise AssertionError(f"L_multi {got} != {ref}")
    cfg = _spec(method, "coherent-smooth").config
    g1 = (cfg["iteration_offset"] + 1) / cfg["gamma_denominator"]
    g_end = (cfg["iteration_offset"] + 13000) / cfg["gamma_denominator"]
    if abs(g1 - 0.35005) > 1e-9 or abs(g_end - 1.0) > 1e-9 or cfg["weight"] != 5.0:
        raise AssertionError("gamma must ramp 0.35005 -> 1 and beta_m = 5")
    return {"value": got, "gamma_first": g1, "gamma_last": g_end}


def _ref_warp(img, flow):
    """Independent bilinear sampling at (x + u - 0.5, y + v - 0.5), zeros outside."""
    _c, h, w = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            sx, sy = x + flow[0, y, x] - 0.5, y + flow[1, y, x] - 0.5
            x0, y0 = math.floor(sx), math.floor(sy)
            for yy, wy in ((y0, 1 - (sy - y0)), (y0 + 1, sy - y0)):
                for xx, wx in ((x0, 1 - (sx - x0)), (x0 + 1, sx - x0)):
                    if 0 <= xx < w and 0 <= yy < h:
                        out[:, y, x] += wy * wx * img[:, yy, xx]
    return out


def flow_loss(method, device):
    """Eq. 6 (official warping_loss): mean L1 between X_i and X_j warped by F_{i->j}, masked."""
    data = geometry_fixture(h=8, w=10, f=10.0)
    rng = np.random.default_rng(0)
    data["flow_w"] = rng.random(data["flow_w"].shape).astype(np.float32)
    data["flows"] = data["flows"] + rng.normal(0, 0.7, data["flows"].shape).astype(np.float32)
    bank, _ = _bank(method, device, data=data)
    xyz = torch.as_tensor(rng.normal(0, 1, (2 * 80, 3)), dtype=torch.float32, device=device)
    got = float(bank.warping_loss(xyz, order=[1, 0]))
    img = xyz.reshape(2, 8, 10, 3).permute(0, 3, 1, 2).cpu().numpy()
    m = (data["flow_w"][1, 0] > 0.5).astype(np.float32)[None]
    warped = _ref_warp(img[0], np.moveaxis(data["flows"][1, 0], -1, 0))
    ref = float(np.abs(warped * m - img[1] * m).mean())
    if abs(got - ref) > 1e-5:
        raise AssertionError(f"warping loss {got} != independent {ref}")
    w = _spec(method, "coherent-flow").config["weight"]
    if abs(w - 0.01) > 1e-12:
        raise AssertionError("official effective flow weight is 0.1 * 0.1")
    return {"value": got, "reference": ref, "weight": w}


def photometric_terms(method, device):
    """0.8 ML1 + 0.2 (1 - SSIM_same) + 0.05/32 L1(relu1_1) against independent formulas."""
    from splatstudio.components import build

    from .losses import PHOTO_DEFAULTS, VGGRelu11, create_photometric, ssim_same

    torch.manual_seed(2)
    rgb = torch.rand(32, 40, 3, device=device)
    gt = torch.rand(32, 40, 3, device=device)
    spec = _spec(method, "photometric")
    value = float(create_photometric(spec)({}, {"rgb": rgb}, gt))
    x, y = rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
    ml1 = 0.0
    for k in range(5):
        ml1 += [1, 1 / 4, 1 / 8, 1 / 16, 1 / 32][k] * float((x - y).abs().mean())
        x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    xs, ys = rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
    # independent SSIM via unfold on zero-padded input
    g = torch.exp(-((torch.arange(11.0, device=device) - 5) ** 2) / 4.5)
    g = (g / g.sum())
    win = (g[:, None] * g[None]).reshape(-1)
    def local(t):
        u = F.unfold(F.pad(t, (5, 5, 5, 5)), 11)  # [1, C*121, L]
        return (u.reshape(1, 3, 121, -1) * win[None, None, :, None]).sum(2)
    mx, my = local(xs), local(ys)
    vx, vy, cxy = local(xs * xs) - mx ** 2, local(ys * ys) - my ** 2, local(xs * ys) - mx * my
    ssim_ref = float((((2 * mx * my + 1e-4) * (2 * cxy + 9e-4))
                      / ((mx ** 2 + my ** 2 + 1e-4) * (vx + vy + 9e-4))).mean())
    if abs(ssim_ref - float(ssim_same(xs, ys))) > 1e-5:
        raise AssertionError("SSIM (same padding) mismatch")
    cfg = {**PHOTO_DEFAULTS, **spec.config}
    net = VGGRelu11(cfg["vgg_weights"], cfg["vgg_sha256"]).to(device)
    vgg = float((net(xs) - net(ys)).abs().mean()) / 32
    ref = 0.8 * ml1 + 0.2 * (1 - ssim_ref) + 0.05 * vgg
    if abs(value - ref) > 1e-5 or vgg <= 0:
        raise AssertionError(f"photometric {value} != {ref}")
    build(method)  # all components construct
    return {"value": value, "ml1": ml1, "ssim": ssim_ref, "vgg": vgg}


def _cuda_setup(method, device):
    from dataclasses import replace

    from splatstudio.state import initialize

    from .density import Density

    data = geometry_fixture()
    spec = _spec(method, "coherent-density")
    dens = Density(replace(spec, config={**spec.config, "n_views": 2, "mask_channels": 3,
                                         "ngf_depth": 2, "ngf_opac": 2, "align_steps": 20}))
    b0 = Bank(data["viewmats"], data["K"], data["z_mono"], data["dep_mask"], data["flows"],
              data["flow_w"], dens.cfg, device)
    pts = b0.xyz(b0.z_mono)
    params, opts = initialize(method, pts, torch.rand_like(pts))
    dens.initialize(params, opts, sidecar=data)
    camera = {"viewmat": torch.as_tensor(data["viewmats"][0], device=device),
              "K": torch.as_tensor(data["K"], device=device), "width": 160, "height": 128}
    return dens, params, opts, camera


def multisample_render(method, device):
    """Sec. 4.2: s x supersampled render pooled to H x W for it > 12000; s = 1 before; SH schedule."""
    from splatstudio.verification import ProbeUnavailable

    from .renderer import Renderer, pool, raster, scaled_camera

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    dens, params, _opts, camera = _cuda_setup(method, device)
    try:
        bank = dens.bank
        r = Renderer(_spec(method, "coherent-renderer"))
        bank.step = 5000
        if r.render(params, camera)["coherent"]["samples"] != 1:
            raise AssertionError("no multisampling before official iteration 12000")
        seen = set()
        for _ in range(40):
            bank.step = 5001
            seen.add(r.render(params, camera)["coherent"]["samples"])
        if seen != {1, 2, 3, 4}:
            raise AssertionError(f"samples {sorted(seen)} != randint(1, 4)")
        bank.rng.seed(123)
        bank.step = 6000
        out = r.render(params, camera)
        s = out["coherent"]["samples"]
        bank.rng.seed(123)
        assert bank.samples(6000) == s
        with torch.no_grad():
            means, logits = bank.gaussians()
            colors = torch.cat((params["sh0"], params["shN"]), 1)
            rgb, _d, _a, _ = raster(means, params["quats"], params["scales"], logits.sigmoid(),
                                  colors, scaled_camera(camera, s), 0, 0.999, 0.01)
            ref = pool(rgb, s)
        if out["rgb"].shape != (128, 160, 3) or not torch.allclose(out["rgb"], ref, atol=1e-5):
            raise AssertionError("multisampled render != AvgPool(s) of the s x render")
        if s > 1:
            with torch.no_grad():
                one, _, _, _ = raster(means, params["quats"], params["scales"], logits.sigmoid(),
                                      colors, camera, 0, 0.999, 0.01)
            if torch.allclose(one, out["rgb"], atol=1e-4):
                raise AssertionError("supersampled render identical to s = 1")
        degs = [bank.sh_degree(t) for t in (7999, 8000, 9000, 10000, 13000)]
        if degs != [0, 1, 2, 3, 3]:
            raise AssertionError(f"SH schedule {degs}")
    finally:
        bank_mod.ACTIVE["bank"] = None
    return {"samples_seen": sorted(seen), "checked_s": s, "sh_degrees": degs}


def persistence(method, device):
    """Training render writes the derived Gaussians to params; a no-grad render of params alone
    (fresh-process evaluation) reproduces the training render."""
    from splatstudio.verification import ProbeUnavailable

    from .renderer import Renderer

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    dens, params, _opts, camera = _cuda_setup(method, device)
    try:
        bank = dens.bank
        with torch.no_grad():
            for p in bank.decoder.parameters():
                p.add_(torch.randn_like(p) * 0.2)
            for p in bank.decoder_opac.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        r = Renderer(_spec(method, "coherent-renderer"))
        bank.step = 100
        out = r.render(params, camera)
        means, logits = bank.gaussians()
        if not torch.allclose(params["means"], means) or not torch.allclose(params["opacities"], logits):
            raise AssertionError("derived Gaussians were not written back to params")
        out["rgb"].mean().backward()
        if params["means"].grad is not None or params["opacities"].grad is not None:
            raise AssertionError("means/opacities must only change through the decoders")
        if params["sh0"].grad is not None:
            raise AssertionError("DC colour is fixed in the official code")
        g = [p.grad for p in bank.decoder.parameters() if p.grad is not None]
        if not g or float(sum(x.abs().sum() for x in g)) == 0:
            raise AssertionError("photometric gradient does not reach the depth decoder")
        bank_mod.ACTIVE["bank"] = None
        with torch.no_grad():
            ev = r.render(params, camera, sh_degree=0)
        err = float((ev["rgb"] - out["rgb"].detach()).abs().max())
        if err > 1e-5:
            raise AssertionError(f"params-only render differs from the training render ({err})")
    finally:
        bank_mod.ACTIVE["bank"] = None
    return {"render_max_diff": err}
