"""Mechanism probes for the EAGLES port. Each raises if its mechanism is removed or wrong."""

import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from splatstudio.backends.gsplat import Renderer
from splatstudio.specs import ComponentSpec
from splatstudio.state import initialize, sample_scene
from splatstudio.verification import ProbeUnavailable

from . import codec, decoders, density, kernels, losses
from .renderer import EaglesRenderer, scaled_camera


def _cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA device required")
    return torch.device(device)


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _density(method, **overrides):
    spec = _component(method, "density")
    cfg = {**spec.config, **overrides}
    return density.EaglesDensity(ComponentSpec(spec.name, "density", spec.factory,
                                               spec.requires, (), cfg))


def _scene(method, device, n=200, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    points = (torch.rand(n, 3, generator=g) - 0.5) * 1.2
    points[:, 2] += 3.0
    colors = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, points.to(device), colors.to(device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.05))
    return params, opts


def _camera(device, W=48, H=40, f=50.0):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": W, "height": H}


def _reference_dft(rows, cols):
    """DCT-II basis written from the closed form sqrt(c_j/k) cos(pi (2i+1) j / (2k))."""
    i = torch.arange(rows, dtype=torch.float64)[:, None]
    j = torch.arange(cols, dtype=torch.float64)[None, :]
    c = torch.where(j > 0, 2.0, 1.0)
    return (torch.sqrt(c / cols) * torch.cos(math.pi * (2 * i + 1) * j / (2 * cols))).float()


def decoder_ste(method, device):
    """Decoder forward vs independent formulas, quantized forward, straight-through backward."""
    bank = decoders.DecoderBank(_component(method, "density").config["attributes"], seed=3)
    dft_err = float((bank.decoders["shN"].dft - _reference_dft(16, 45)).abs().max())
    if dft_err > 1e-6:
        raise AssertionError(f"DFT matrix differs from DCT-II: {dft_err}")
    g = torch.Generator(device="cpu").manual_seed(1)
    n = 64
    params = {f: torch.nn.Parameter(torch.randn(n, d, generator=g) * 3)
              for f, d in (("q_shN", 16), ("q_rot", 8), ("q_opa", 1))}
    out = {}
    for name, (field, lat) in decoders.TARGETS.items():
        dec = bank.decoders[name]
        q = params[lat]
        S, b = dec.scale.detach(), dec.shift.detach()
        r = torch.round(q.detach())
        ref = (r @ _reference_dft(16, 45)) * S + b if name == "shN" else r @ S + b
        got = bank.decode(params, name).reshape(n, -1)
        fwd_err = float((got - ref).abs().max())
        # moving q inside its rounding bin must not change the decoded value
        jitter = torch.nn.Parameter(r + (torch.rand(r.shape, generator=g) - 0.5) * 0.9)
        same = float((bank.decode({lat: jitter}, name).reshape(n, -1) - ref).abs().max())
        w = torch.randn(ref.shape, generator=g)
        q.grad = None
        (bank.decode(params, name).reshape(n, -1) * w).sum().backward()
        expect = (w * S) @ _reference_dft(16, 45).T if name == "shN" else w @ S.T
        grad_err = float((q.grad - expect).abs().max())
        if fwd_err > 1e-5 or same > 1e-5 or grad_err > 1e-5:
            raise AssertionError(f"{name}: forward {fwd_err}, bin-invariance {same}, "
                                 f"straight-through grad {grad_err}")
        out[name] = {"forward_err": fwd_err, "grad_err": grad_err}
    shapes = {k: list(v.shape) for k, v in bank.decode_all(params).items()}
    if shapes != {"shN": [n, 15, 3], "quats": [n, 4], "opacities": [n]}:
        raise AssertionError(f"decoded shapes {shapes}")
    return {"dft_err": dft_err, **out}


def latent_init_lr(method, device):
    """q = D^{-1}(a) at initialization; latent LR = lr_3dgs / ||S||_F (div) or lr_3dgs; decoder
    Adam LR 1e-4 (effective official values)."""
    decoders.ACTIVE["bank"] = None
    params, opts = sample_scene(method, "cpu", n=32)
    d = _density(method)
    d.initialize(params, opts, 1.0)
    try:
        bank = d.bank
        rot, opa = bank.decoders["rot"], bank.decoders["opa"]
        q_rot = params["q_rot"].detach()
        unrounded_rot = q_rot @ rot.scale.detach() + rot.shift.detach()
        quat = torch.tensor([1.0, 0, 0, 0]).expand_as(unrounded_rot)
        rot_err = float((unrounded_rot - quat).abs().max())
        S, b = float(opa.scale), float(opa.shift)
        expect_q = (math.log(0.1 / 0.9) - b) / (S + 1e-6)
        opa_err = float((params["q_opa"].detach() - expect_q).abs().max())
        sh_zero = float(params["q_shN"].detach().abs().max())
        lr = {f: opts[f].param_groups[0]["lr"] for f in ("q_shN", "q_rot", "q_opa")}
        ref = {"q_shN": 0.0025 / 20 / float(bank.decoders["shN"].scale.detach().norm()),
               "q_rot": 0.001 / float(torch.sqrt((rot.scale.detach() ** 2).sum())),
               "q_opa": 0.05}
        lr_err = max(abs(lr[k] - ref[k]) / ref[k] for k in ref)
        dec_lr = sorted(g["lr"] for g in bank.optimizer.param_groups)
        if dec_lr != [1e-4] * 3:
            raise AssertionError(f"decoder LRs {dec_lr}")
        if rot_err > 1e-4 or opa_err > 1e-4 or sh_zero != 0 or lr_err > 1e-6:
            raise AssertionError(f"init/LR wrong: rot {rot_err}, opa {opa_err}, shN {sh_zero}, "
                                 f"lr {lr_err}")
        return {"rot_inversion_err": rot_err, "opacity_inversion_err": opa_err,
                "latent_lr": lr, "lr_rel_err": lr_err}
    finally:
        decoders.ACTIVE["bank"] = None


def entropy_size(method, device):
    """size() vs analytic entropies; decoder bit count; range-coder roundtrip and efficiency."""
    n = 4096
    a = torch.stack((torch.arange(n) % 4, torch.full((n,), 7)), 1).float()  # 2 bits, 0 bits
    b = torch.cat((torch.zeros(n // 2), torch.ones(n // 4), torch.full((n // 4,), -3.0)))[:, None]
    bank = decoders.DecoderBank(_component(method, "density").config["attributes"], seed=0)
    size = codec.paper_size_bits({"q_rot": a, "q_opa": b}, bank, n)
    expect = {"q_rot": 2.0 * n, "q_opa": 1.5 * n}
    lat_err = max(abs(size["latent_bits"][k] - v) for k, v in expect.items())
    # shN: scale 45 + shift 45 + DFT 16x45 + div 16; rot: 8x4 + 4 + 8; opa: 1 + 1 + 1
    dec_expect = 32 * ((45 + 45 + 720 + 16) + (32 + 4 + 8) + (1 + 1 + 1))
    if lat_err > 1e-2 or size["decoder_bits"] != dec_expect or \
            size["unquantized_bits"] != 32 * 9 * n:
        raise AssertionError(f"size(): latent err {lat_err}, decoder {size['decoder_bits']} vs "
                             f"{dec_expect}, unquantized {size['unquantized_bits']}")
    g = np.random.default_rng(0)
    m = 20000
    q = torch.from_numpy(np.stack((g.geometric(0.4, m) - 3, g.integers(-2, 3, m),
                                   np.zeros(m, dtype=np.int64)), 1).astype(np.float32))
    stream = codec.encode({"q_shN": q})
    back = codec.decode(stream)["q_shN"]
    if not torch.equal(back, q):
        raise AssertionError("range coder roundtrip is not lossless")
    shannon = sum(codec.column_entropy_bits(q[:, d]) for d in range(3))
    coded = 32 * len(stream["q_shN"]["words"])
    if coded > 1.01 * shannon + 64:
        raise AssertionError(f"coded {coded} bits vs entropy {shannon}")
    return {"latent_bits_err": lat_err, "decoder_bits": size["decoder_bits"],
            "coded_bits": coded, "entropy_bits": shannon, "overhead": coded / shannon - 1}


def progressive_schedule(method, device):
    """s(t) = 1 + 0.5 (0.3 - 1)(1 + cos(pi t / 21000)) for t <= 21000, then 1."""
    cfg = _component(method, "density").config
    T = 21000  # int(0.7 * (30000 + 1))
    checks = {1: 1 - 0.35 * (1 + math.cos(math.pi / T)), 10500: 0.65, 21000: 1.0,
              21001: 1.0, 30000: 1.0}
    err = max(abs(density.resize_scale(t, cfg) - v) for t, v in checks.items())
    if err > 1e-9 or density.resize_scale(1, {**cfg, "progressive": False}) != 1.0:
        raise AssertionError(f"progressive scale schedule wrong ({err})")
    cam = _camera("cpu", W=101, H=77)
    s = 0.37
    small = scaled_camera(cam, s)
    target = torch.rand(77, 101, 3, generator=torch.Generator().manual_seed(0))
    down = F.interpolate(target.permute(2, 0, 1)[None], scale_factor=s, mode="bilinear",
                         align_corners=False)[0].permute(1, 2, 0)
    if (small["height"], small["width"]) != tuple(down.shape[:2]) or \
            not torch.allclose(small["K"][:2], cam["K"][:2] * s):
        raise AssertionError(f"render size {small['height']}x{small['width']} vs target "
                             f"{tuple(down.shape[:2])}")
    loss = losses.create_photometric(_component(method, "loss"))
    rgb = torch.rand(*down.shape, generator=torch.Generator().manual_seed(1), requires_grad=True)
    value = loss({}, {"rgb": rgb, "eagles_scale": s}, target)
    from splatstudio.losses import ssim

    ref = 0.8 * (rgb - down).abs().mean() + 0.2 * (1 - ssim(rgb, down))
    if abs(float(value - ref)) > 1e-6:
        raise AssertionError("photometric loss does not use the downsampled target")
    return {"scale_err": err, "render_hw": [small["height"], small["width"]],
            "s_at_1": density.resize_scale(1, cfg)}


def influence_kernel(method, device):
    """Kernel sum_p alpha*T == per-pixel front-to-back compositing in torch (gsplat numerics)."""
    dev = _cuda(device)
    g = torch.Generator(device="cpu").manual_seed(5)
    n = 40
    params = {
        "means": torch.randn(n, 3, generator=g) * 0.4 + torch.tensor([0, 0, 3.0]),
        "scales": (torch.randn(n, 3, generator=g) * 0.2 - 2.0),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.rand(n, generator=g) * 0.6 + 0.05),
        "sh0": torch.rand(n, 1, 3, generator=g),
        "shN": torch.zeros(n, 15, 3),
    }
    params = {k: v.to(dev) for k, v in params.items()}
    cam = _camera(dev, W=40, H=30, f=40.0)
    with torch.no_grad():
        out = Renderer(ComponentSpec("r", "renderer", "x", config={})).render(params, cam, 3)
    info = out["info"]
    w = kernels.influence(info)
    m2 = info["means2d"][0].double().cpu()
    con = info["conics"][0].double().cpu()
    opa = info["opacities"][0].double().cpu()
    depth = info["depths"][0].double().cpu()
    vis = (info["radii"][0] > 0).all(-1).cpu()
    order = [i for i in torch.argsort(depth).tolist() if vis[i]]
    ref = torch.zeros(n, dtype=torch.float64)
    final_t = torch.ones(cam["height"], cam["width"], dtype=torch.float64)
    for py in range(cam["height"]):
        for px in range(cam["width"]):
            T = 1.0
            for i in order:
                dx, dy = m2[i, 0] - (px + 0.5), m2[i, 1] - (py + 0.5)
                sig = 0.5 * (con[i, 0] * dx * dx + con[i, 2] * dy * dy) + con[i, 1] * dx * dy
                a = min(0.999, float(opa[i] * torch.exp(-sig)))
                if sig < 0 or a < 1 / 255:
                    continue
                nt = T * (1 - a)
                if nt <= 1e-4:
                    break
                ref[i] += a * T
                T = nt
            final_t[py, px] = T
    alpha_err = float((out["alpha"][..., 0].double().cpu() - (1 - final_t)).abs().max())
    if alpha_err > 1e-3:
        raise ProbeUnavailable(f"torch reference does not reproduce gsplat alpha ({alpha_err})")
    w_err = float((w.double().cpu() - ref).abs().max())
    # sum_i W_i = sum_p (1 - T_final(p)) = total rendered alpha
    alpha_sum_err = abs(float(w.sum()) - float(out["alpha"].sum()))
    if w_err > 1e-2 or ref.sum() == 0 or alpha_sum_err > 1e-2 * float(out["alpha"].sum()):
        raise AssertionError(f"influence mismatch: {w_err}, alpha-sum {alpha_sum_err}")
    return {"alpha_err": alpha_err, "influence_err": w_err, "alpha_sum_err": alpha_sum_err,
            "total_influence": float(ref.sum())}


def influence_prune(method, device):
    """Quantile pruning removes exactly W <= q15 and resets W; new Gaussians start at W = 0."""
    dev = _cuda(device)
    decoders.ACTIVE["bank"] = None
    try:
        params, opts = _scene(method, dev, n=100)
        d = _density(method)
        d.initialize(params, opts, 1.0)
        w = torch.arange(100, device=dev, dtype=torch.float32).flip(0) + 1.0
        d.strategy_state[density.INFLUENCE] = w.clone()
        tag = params["means"].detach()[:, 0].clone()
        info = d._influence_prune(params, opts)
        thr = float(torch.quantile(w, 0.15))
        keep = tag[w > thr]
        if not torch.equal(params["means"].detach()[:, 0], keep) or \
                float(d.strategy_state[density.INFLUENCE].abs().max()) != 0.0:
            raise AssertionError("influence pruning kept the wrong set or did not reset W")
        n = len(params["means"])
        st = d.strategy_state
        st[density.INFLUENCE] = torch.full((n,), 5.0, device=dev)
        st["grad2d"] = torch.zeros(n, device=dev)
        st["count"] = torch.ones(n, device=dev)
        st["grad2d"][:3] = 1.0  # three high-gradient Gaussians
        with torch.no_grad():
            params["scales"].fill_(math.log(0.005))  # <= 0.01 * scene_scale: clone
            params["scales"][1] = math.log(10.0)  # large -> split
        n_d, n_s = d.strategy._grow_gs(params, opts, st, 1000)
        tail = st[density.INFLUENCE][-(n_d + 2 * n_s):]
        head = st[density.INFLUENCE][: len(params["means"]) - (n_d + 2 * n_s)]
        if (n_d, n_s) != (2, 1) or float(tail.abs().max()) != 0 or bool((head != 5.0).any()):
            raise AssertionError(f"new-Gaussian influence not zeroed (dupli {n_d}, split {n_s})")
        return {"pruned": info["pruned"], "threshold": info["threshold"], "dupli": n_d,
                "split": n_s}
    finally:
        decoders.ACTIVE["bank"] = None


def opacity_reset(method, device):
    """q_opa <- (logit(min(sigmoid(D(q)), 0.01)) - b)/(S + 1e-6); its Adam moments zeroed."""
    decoders.ACTIVE["bank"] = None
    params, opts = sample_scene(method, "cpu", n=32)
    d = _density(method)
    d.initialize(params, opts, 1.0)
    try:
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            params["q_opa"].copy_(torch.randn(32, 1, generator=g) * 6)
        params["q_opa"].grad = torch.randn(32, 1, generator=g)
        opts["q_opa"].step()
        before_q = params["q_opa"].detach().clone()
        before_mirror = params["opacities"].detach().clone()
        d._reset_opacity(params, opts)
        dec = d.bank.decoders["opa"]
        S, b = float(dec.scale), float(dec.shift)
        op = torch.sigmoid(torch.round(before_q) * S + b)
        x = torch.minimum(op, torch.full_like(op, 0.01))
        expect = (torch.log(x / (1 - x)) - b) / (S + 1e-6)
        err = float((params["q_opa"].detach() - expect).abs().max())
        moments = opts["q_opa"].state[params["q_opa"]]
        zeroed = float(moments["exp_avg"].abs().max() + moments["exp_avg_sq"].abs().max())
        if err > 1e-4 or zeroed != 0 or torch.equal(params["q_opa"].detach(), before_q):
            raise AssertionError(f"opacity reset wrong: latent err {err}, moments {zeroed}")
        if not torch.equal(params["opacities"].detach(), before_mirror):
            raise AssertionError("reset should act on the latent, not the mirrored field")
        return {"latent_err": err, "moments_after": zeroed}
    finally:
        decoders.ACTIVE["bank"] = None


def codec_bake(method, device):
    """Training-time decoded render == bank-free render of the baked checkpoint params."""
    dev = _cuda(device)
    decoders.ACTIVE["bank"] = None
    try:
        params, opts = _scene(method, dev, n=300, seed=4)
        d = _density(method)
        d.initialize(params, opts, 1.0)
        g = torch.Generator(device="cpu").manual_seed(9)
        with torch.no_grad():
            params["q_rot"].add_((torch.randn(300, 8, generator=g) * 20).to(dev))
            params["q_shN"].copy_((torch.randn(300, 16, generator=g) * 4).to(dev))
            params["q_opa"].add_((torch.randn(300, 1, generator=g) * 2).to(dev))
            d.bank.decoders["shN"].scale.mul_(200.0)
        renderer = EaglesRenderer(_component(method, "renderer"))
        cam = _camera(dev)
        d.bank.scale = 0.5
        small = renderer.render(params, cam, 3)
        if tuple(small["rgb"].shape[:2]) != (20, 24) or small["eagles_scale"] != 0.5:
            raise AssertionError(f"training render not at the progressive scale: "
                                 f"{tuple(small['rgb'].shape)}")
        d.bank.scale = 1.0
        train_rgb = renderer.render(params, cam, 3)["rgb"].detach()
        with torch.no_grad():  # garbage mirrors: the bake must overwrite them
            params["quats"].copy_(torch.randn_like(params["quats"]))
            params["opacities"].fill_(3.0)
            params["shN"].zero_()
        report = d._encode(params)
        state = d.state
        buf = io.BytesIO()
        torch.save({"density_state": state, "params": dict(params)}, buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location=dev, weights_only=True)
        decoders.ACTIVE["bank"] = None  # a fresh process (ss-eval) has no bank
        eval_rgb = EaglesRenderer(_component(method, "renderer")).render(
            loaded["params"], cam, 3)["rgb"]
        err = float((eval_rgb - train_rgb).abs().max())
        decoders.ACTIVE["bank"] = d.bank  # baked bank: renders base fields too
        err_baked = float((renderer.render(params, cam, 3)["rgb"].detach() - train_rgb)
                          .abs().max())
        ints = all(torch.equal(params[f], torch.round(params[f])) for f in d.bank.fields)
        if err > 1e-5 or err_baked > 1e-5 or not d.bank.baked or not ints:
            raise AssertionError(f"decoded render differs: {err} / {err_baked}, ints {ints}")
        if loaded["density_state"]["eagles"]["encoded"]["gaussians"] != 300:
            raise AssertionError("encoded report not checkpointed")
        return {"max_abs_diff": err, "paper_size_MB": report["paper_size_MB"],
                "actual_MB": report["actual_MB"]}
    finally:
        decoders.ACTIVE["bank"] = None


def schedule(method, device):
    """Event steps of the 30k official loop (configs/efficient-3dgs.yaml)."""
    cfg = _component(method, "density").config
    d = _density(method)
    s = d.strategy
    counts = {"opacity_reset": [], "influence_prune": [], "encode": []}
    refine = []
    for step in range(1, cfg["total_steps"] + 1):
        for e in density.schedule_events(step, cfg):
            counts[e].append(step)
        if (step < s.refine_stop_iter and step > s.refine_start_iter
                and step % s.refine_every == 0 and step % s.reset_every >= 0):
            refine.append(step)
    official_refine = [t for t in range(1, 30001) if t > 500 and t % 125 == 0 and t < 18000]
    expect = {"opacity_reset": list(range(2500, 18000, 2500)),
              "influence_prune": [5100, 10200, 15300, 20400, 25500], "encode": [30000]}
    gsplat_reset = any(step % s.reset_every == 0 & step > 0 for step in range(1, 30001))
    if counts != expect or refine != official_refine or gsplat_reset or \
            (s.grow_grad2d, s.prune_opa, s.reset_every) != (0.0002, 0.005, 2500):
        raise AssertionError(f"schedule differs: {counts}, refine {len(refine)}")
    return {"refine_steps": len(refine), **{k: v for k, v in counts.items()}}
