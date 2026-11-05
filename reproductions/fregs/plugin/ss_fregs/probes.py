"""FreGS mechanism probes. The oracles here are written independently of frequency.py."""

import math

import torch
from splatstudio.registry import resolve


def _component(method, name):
    return next(c for c in method.components if c.name == name)


def _frequency_fn(method, **overrides):
    spec = _component(method, "fregs-frequency")
    if overrides:
        from dataclasses import replace

        spec = replace(spec, config={**spec.config, **overrides})
    return resolve(spec.factory)(spec), spec.config


# ----------------------------------------------------------------------------- oracle
def _oracle_dft(image):
    """Eq. 2 by explicit DFT matrices in float64, output rows/cols ordered u = -H//2 .. ."""
    h, w, _ = image.shape
    x = image.permute(2, 0, 1).to(torch.float64)  # [C,H,W]

    def tables(n):
        k = torch.arange(n, dtype=torch.int64) - n // 2
        m = (k[:, None] * torch.arange(n, dtype=torch.int64)[None, :]) % n
        angle = 2 * math.pi * m.to(torch.float64) / n
        cos, sin = torch.cos(angle), -torch.sin(angle)  # e^{-i angle}
        sin = torch.where((2 * m) % n == 0, torch.zeros_like(sin), sin)  # exact 0 at 0 and pi
        return cos, sin

    ch, sh = tables(h)
    cw, sw = tables(w)
    re = torch.einsum("ux,cxy,vy->cuv", ch, x, cw) - torch.einsum("ux,cxy,vy->cuv", sh, x, sw)
    im = torch.einsum("ux,cxy,vy->cuv", sh, x, cw) + torch.einsum("ux,cxy,vy->cuv", ch, x, sw)
    return re, im + 0.0


def _oracle_masks(h, w, cfg, t):
    """Masks built with python loops, independent of frequency.py."""
    radii = {}
    for i in range(h):
        for j in range(w):
            u, v = i - h // 2, j - w // 2
            if cfg["radius_units"] == "normalized":
                u, v = u / (h / 2), v / (w / 2)
            radii[i, j] = math.hypot(u, v)
    big_d = max(radii.values())
    d0 = cfg["d0_frac"] * big_d
    if not cfg["anneal"]:
        dt = big_d
    elif t > cfg["t0"]:
        dt = d0 + (t - cfg["t0"]) * (big_d - d0) / (cfg["t_end"] - cfg["t0"])
        dt = min(max(dt, d0), big_d)
    else:
        dt = d0
    low = torch.zeros(h, w, dtype=torch.bool)
    high = torch.zeros(h, w, dtype=torch.bool)
    for (i, j), r in radii.items():
        low[i, j] = r <= d0
        high[i, j] = d0 < r <= dt
    return low, high


def _oracle_loss(rgb, target, t, cfg):
    h, w, _ = rgb.shape
    if t > cfg["stop_iter"]:
        return rgb.sum() * 0
    a_re, a_im = _oracle_dft(rgb)
    with torch.no_grad():
        b_re, b_im = _oracle_dft(target)
    n = math.sqrt(h * w)

    def term(mask):
        total = 0.0
        for c in range(rgb.shape[2]):
            for i, j in mask.nonzero().tolist():
                ar, ai, br, bi = a_re[c, i, j], a_im[c, i, j], b_re[c, i, j], b_im[c, i, j]
                amp_a = torch.sqrt(ar**2 + ai**2 + 1e-20)
                amp_b = torch.sqrt(br**2 + bi**2 + 1e-20)
                total = total + (amp_a - amp_b).abs()
                if amp_a.item() < cfg["phase_eps"] * n:
                    continue
                d = torch.atan2(ai, ar) - torch.atan2(bi, br)
                if cfg["phase"] == "wrapped":
                    if amp_b.item() < cfg["phase_eps"] * n:
                        continue
                    # Independent wrap of the raw difference into (-pi, pi].
                    while d.item() > math.pi:
                        d = d - 2 * math.pi
                    while d.item() <= -math.pi:
                        d = d + 2 * math.pi
                total = total + d.abs()
        return total / n

    low, high = _oracle_masks(h, w, cfg, t)
    value = cfg["w_l"] * term(low)
    if t > cfg["t0"] or not cfg["anneal"]:
        value = value + cfg["w_h"] * term(high)
    return value


def freq_oracle(method, device):
    fn, cfg = _frequency_fn(method)
    if cfg["phase"] not in ("raw", "wrapped"):
        raise AssertionError("The oracle implements phase='raw' and phase='wrapped' only")
    g = torch.Generator().manual_seed(7)
    evidence = {}
    cases = [(24, 40, cfg["t0"]), (24, 40, (cfg["t0"] + cfg["t_end"]) // 2), (23, 37, cfg["t_end"])]
    for h, w, t in cases:
        rgb0 = torch.rand(h, w, 3, generator=g, dtype=torch.float64)
        target = torch.rand(h, w, 3, generator=g, dtype=torch.float64)
        oracle_in = rgb0.clone().requires_grad_(True)
        expected = _oracle_loss(oracle_in, target, t, cfg)
        expected_grad = torch.autograd.grad(expected, oracle_in)[0]
        for dtype, rtol in ((torch.float64, 1e-7), (torch.float32, 2e-4)):
            x = rgb0.to(device, dtype).requires_grad_(True)
            value = fn({}, {"rgb": x, "step": t}, target.to(device, dtype))
            grad = torch.autograd.grad(value, x)[0].cpu().to(torch.float64)
            if not math.isclose(value.item(), expected.item(), rel_tol=rtol):
                raise AssertionError(f"L_f {value.item()} != oracle {expected.item()} ({h}x{w}, t={t})")
            err = (grad - expected_grad).abs().max().item()
            scale = expected_grad.abs().max().item()
            if not err <= 50 * rtol * scale:
                raise AssertionError(f"dL_f/drgb mismatch {err:.3e} (scale {scale:.3e}, {dtype})")
            evidence[f"{h}x{w}_t{t}_{str(dtype).split('.')[-1]}_rel_grad_err"] = err / scale
        evidence[f"{h}x{w}_t{t}_loss"] = expected.item()
        if not expected.item() > 0 or not torch.count_nonzero(expected_grad):
            raise AssertionError("L_f is inactive on a differing pair")
    same = torch.rand(24, 40, 3, generator=g).to(device).requires_grad_(True)
    t = cfg["t0"] + 1
    value = fn({}, {"rgb": same, "step": t}, same.detach().clone())
    grad = torch.autograd.grad(value, same)[0]
    if value.item() != 0 or torch.count_nonzero(grad):
        raise AssertionError("Identical images must give L_f = 0 and zero gradient")
    evidence["identical_loss"] = value.item()
    return evidence


# ----------------------------------------------------------------------------- schedule
def _sinusoid(h, w, v, amplitude=0.1):
    y = torch.arange(w, dtype=torch.float64)
    wave = amplitude * torch.cos(2 * math.pi * v * y / w)
    return wave[None, :, None].expand(h, w, 3)


def freq_schedule(method, device):
    """Known frequency content along the v axis: bins (0, +-v) at radius r(v)."""
    from .frequency import high_mask, radius_grid

    fn, cfg = _frequency_fn(method, w_l=1.0, w_h=1.0)
    h = w = 64
    units = cfg["radius_units"]

    def radius(v):
        return v / (w / 2) if units == "normalized" else float(v)

    big_d = math.hypot(h // 2 / ((h / 2) if units == "normalized" else 1), radius(w // 2))
    d0 = cfg["d0_frac"] * big_d
    g = torch.Generator().manual_seed(3)
    base = torch.rand(h, w, 3, generator=g, dtype=torch.float64) * 0.5 + 0.25

    def lf(target, t):
        x = base.to(device).requires_grad_(True)
        return fn({}, {"rgb": x, "step": t}, target.to(device)).item()

    v_hi = next(v for v in range(w // 2 - 1, 0, -1) if radius(v) <= (d0 + big_d) / 2)
    v_lo = next((v for v in range(1, w // 2) if radius(v) < d0), None)
    if radius(v_hi) <= d0 or v_lo is None:
        raise AssertionError("Probe geometry needs a bin below D0 and one in (D0, D)")
    hi_target = base + _sinusoid(h, w, v_hi)
    lo_target = base + _sinusoid(h, w, v_lo)
    visible, invisible = 1e-3, 1e-8
    evidence = {"D": big_d, "D0": d0, "r_high": radius(v_hi), "r_low": radius(v_lo)}
    if lf(lo_target, 1) <= visible:
        raise AssertionError("Low-frequency difference is not seen at step 1")
    if cfg["anneal"]:
        # Independent inversion of D_t = D0 + (t - T0)(D - D0)/(T - T0) at D_t = r*.
        t_cross = cfg["t0"] + (radius(v_hi) - d0) * (cfg["t_end"] - cfg["t0"]) / (big_d - d0)
        t_after = math.ceil(t_cross)
        t_before = t_after - 1
        for t in (1, cfg["t0"], t_before):
            value = lf(hi_target, t)
            if value >= invisible:
                raise AssertionError(f"High band visible too early at t={t} (L_f={value})")
        value = lf(hi_target, t_after)
        if value <= visible:
            raise AssertionError(f"High band not visible at t={t_after} (L_f={value})")
        evidence.update(t_cross=t_cross, loss_before=lf(hi_target, t_before), loss_after=value)
        r, grid_d = radius_grid(h, w, "cpu", units)
        if abs(grid_d - big_d) > 1e-9:
            raise AssertionError("Max spectrum radius D differs from the independent value")
        for t in (cfg["t0"] + 1, (cfg["t0"] + cfg["t_end"]) // 2, cfg["t_end"]):
            dt = min(d0 + (t - cfg["t0"]) * (big_d - d0) / (cfg["t_end"] - cfg["t0"]), big_d)
            expected = sum(
                d0 < math.hypot((i - h // 2) / ((h / 2) if units == "normalized" else 1),
                                radius(j - w // 2)) <= dt
                for i in range(h)
                for j in range(w)
            )
            count = int(high_mask(r, grid_d, cfg, t).sum())
            if count != expected:
                raise AssertionError(f"H_h({t}) passes {count} bins, expected {expected}")
            evidence[f"high_bins_t{t}"] = count
    else:
        value = lf(hi_target, 1)
        if value <= visible:
            raise AssertionError("Base+FR must see the full spectrum from step 1")
        evidence["loss_high_t1"] = value
    if lf(hi_target, cfg["stop_iter"]) <= visible:
        raise AssertionError("Full band must be active at stop_iter")
    x = base.to(device).requires_grad_(True)
    after = fn({}, {"rgb": x, "step": cfg["stop_iter"] + 1}, hi_target.to(device))
    if after.item() != 0 or not after.requires_grad:
        raise AssertionError("L_f must be 0 (graph-attached) after stop_iter")
    evidence["loss_after_stop"] = after.item()
    return evidence


# ----------------------------------------------------------------------------- threshold
def threshold(method, device):
    from gsplat import DefaultStrategy
    from splatstudio.state import sample_scene

    from .defaults import DEFAULTS

    spec = _component(method, "fregs-density")
    density = resolve(spec.factory)(spec)
    strategy = density.strategy
    value = float(spec.config["grow_grad2d"])
    if strategy.grow_grad2d != value or not value > 0.0002:
        raise AssertionError("Density does not apply the raised grow_grad2d")
    if method.name in ("fregs", "fregs-fr") and value != DEFAULTS["density"]["grow_grad2d"]:
        raise AssertionError("fregs/fregs-fr must use the calibrated DEFAULTS threshold")
    if strategy.refine_stop_iter != 15000 or strategy.absgrad:
        raise AssertionError("Densification must stop at 15000 with signed gradients")
    n, step = 64, 600  # a refine step: > refine_start_iter, multiple of refine_every

    def grown(strat, avg):
        params, optimizers = sample_scene(method, device, n=n)
        sum(p.square().sum() for p in params.values()).backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        state = strat.initialize_state(scene_scale=10.0)
        state["grad2d"] = torch.full((n,), avg * 10, device=device)
        state["count"] = torch.full((n,), 10.0, device=device)
        means2d = torch.zeros(1, n, 2, device=device, requires_grad=True)
        means2d.grad = torch.zeros_like(means2d)
        info = {
            "width": 16,
            "height": 16,
            "n_cameras": 1,
            "radii": torch.zeros(1, n, 2, device=device, dtype=torch.int32),
            "gaussian_ids": None,
            "means2d": means2d,
        }
        strat.step_post_backward(params, optimizers, state, step, info, packed=False)
        return len(params["means"]) - n

    between = (0.0002 + value) / 2
    fregs_growth = grown(strategy, between)
    base_growth = grown(DefaultStrategy(), between)
    above_growth = grown(strategy, value * 1.5)
    if fregs_growth != 0:
        raise AssertionError("Gaussians with gradient below the raised threshold grew")
    if base_growth <= 0 or above_growth <= 0:
        raise AssertionError("Fixture self-check failed: growth expected above the threshold")
    return {
        "grow_grad2d": value,
        "fixture_avg_grad": between,
        "fregs_growth": fregs_growth,
        "base_3dgs_growth": base_growth,
        "fregs_growth_above_threshold": above_growth,
    }


# ----------------------------------------------------------------------------- gradient gain
def _synthetic(method, device):
    """Rendered synthetic scene at benchmark-like resolution (L_f scales with HW)."""
    from splatstudio.components import build
    from splatstudio.state import initialize
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is required for the gsplat rasterizer")
    renderer, _, losses = build(method)
    photometric = dict(losses)["photometric"]
    g = torch.Generator(device=device).manual_seed(0)
    n, width, height = 20000, 960, 640  # benchmark-like resolution: L_f scales with HW
    points = torch.rand(n, 3, generator=g, device=device) * 2 - 1
    points[:, :2] *= 1.5
    points[:, 2] += 4
    colors = torch.rand(n, 3, generator=g, device=device)
    params, _ = initialize(method, points, colors)
    with torch.no_grad():
        params["scales"].fill_(math.log(0.03))
        params["opacities"].fill_(2.0)
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[800.0, 0, width / 2], [0, 800.0, height / 2], [0, 0, 1]], device=device),
        "width": width,
        "height": height,
    }
    with torch.no_grad():
        moved = {k: v.detach().clone() for k, v in params.items()}
        moved["means"] += 0.02 * torch.randn(n, 3, generator=g, device=device)
        moved["sh0"] += 0.3 * torch.randn(n, 1, 3, generator=g, device=device)
        target = renderer.render(moved, camera, sh_degree=0)["rgb"].clamp(0, 1)
    from .frequency import parse_config

    cfg = parse_config(_component(method, "fregs-frequency").config)
    return renderer, photometric, params, camera, target, cfg


def _weights(cfg, w_h=None):
    w_h = cfg["w_h"] if w_h is None else w_h
    return {"photo": 1.0, "low_a": cfg["w_l"], "low_p": cfg["w_l"], "high_a": w_h, "high_p": w_h}


def _noise_floor(grads, weights, value):
    """Float noise floor of a pooled ratio: |float32 - float64 recomputation|, at least float32
    machine epsilon times the value (one rounding of the result)."""
    from .gradstats import pooled_ratio

    f64 = {k: (a.double(), b.double()) for k, (a, b) in grads.items()}
    return max(abs(pooled_ratio([f64], weights)[1] - value), 1.2e-7 * abs(value))


# Re-registered after calibration (evidence log in README): mechanism probes contrast each
# mechanism with its own ablation in the same run; magnitudes vs the paper are checked on real
# data by fregs-calibration-ratio.
GAIN_MARGIN = 0.05  # t=1 ratio must exceed 1 + 5%; the no-op ablation (w_l = 0) gives exactly 1
NOISE_FACTOR = 10  # high-band gain must exceed 10x the float noise floor of the w_h = 0 ratio


def gradient_gain(method, device):
    from splatstudio.verification import ProbeUnavailable

    from .defaults import CALIBRATED
    from .gradstats import magnitudes, pooled_ratio, term_gradients

    renderer, photometric, params, camera, target, cfg = _synthetic(method, device)
    if not CALIBRATED["w_l"]:
        raise ProbeUnavailable("w_l is an uncalibrated placeholder")
    weights = _weights(cfg)
    noop = {**weights, "low_a": 0.0, "low_p": 0.0, "high_a": 0.0, "high_p": 0.0}
    evidence = {}
    for label, t in (("t1", 1), ("tT", cfg["t_end"])):
        out = renderer.render(params, camera, sh_degree=0)
        grads = term_gradients(out, target, t, cfg, photometric)
        rgb_ratio, m2d_ratio = pooled_ratio([grads], weights)
        evidence[f"{label}_rgb_ratio"] = rgb_ratio
        evidence[f"{label}_means2d_ratio"] = m2d_ratio
        evidence[f"{label}_noop_means2d_ratio"] = pooled_ratio([grads], noop)[1]
        # Unit-weight magnitude of each term (phase vs amplitude dominance, review-1).
        evidence[f"{label}_unit_weight_terms"] = magnitudes([grads])
    if abs(evidence["t1_noop_means2d_ratio"] - 1) > 1e-6:
        raise AssertionError(f"No-op ablation must give ratio 1: {evidence}")
    if not evidence["t1_means2d_ratio"] > 1 + GAIN_MARGIN:
        raise AssertionError(f"L_f raises t=1 screen gradients by <= {GAIN_MARGIN:.0%}: {evidence}")
    return evidence


def high_band_gain(method, device):
    """C2 weight is not a no-op: at t = T (t = 1 without annealing) w_h > 0 and the high band's
    gain over the w_h = 0 ablation exceeds NOISE_FACTOR x the float noise floor."""
    from .gradstats import band_magnitudes, pooled_ratio, term_gradients

    renderer, photometric, params, camera, target, cfg = _synthetic(method, device)
    if not cfg["w_h"] > 0:
        raise AssertionError("w_h must be positive: the high band would be switched off")
    t = cfg["t_end"] if cfg["anneal"] else 1
    out = renderer.render(params, camera, sh_degree=0)
    grads = term_gradients(out, target, t, cfg, photometric)
    if "high_a" not in grads:
        raise AssertionError(f"High band inactive at t={t}")
    full = pooled_ratio([grads], _weights(cfg))[1]
    ablation = _weights(cfg, 0.0)
    low_only = pooled_ratio([grads], ablation)[1]
    repeat = pooled_ratio([grads], ablation)[1]
    floor = max(_noise_floor(grads, ablation, low_only), abs(repeat - low_only))
    gain = full - low_only
    evidence = {"t": t, "w_h": cfg["w_h"], "means2d_ratio": full, "means2d_ratio_w_h_0": low_only,
                "high_band_gain": gain, "noise_floor": floor,
                "gain_over_noise": gain / floor,
                "high_to_low_share": gain / (low_only - 1) if low_only > 1 else None,
                "band_magnitudes_unit_weight": band_magnitudes([grads])}
    if not gain > NOISE_FACTOR * floor:
        raise AssertionError(f"High-band gain is not above {NOISE_FACTOR}x noise: {evidence}")
    return evidence


REFERENCE_DIR = "/home/sejain/splatify/runs/fregs-calib-reference"
REAL_SCENE = "garden"
REAL_DATA = "~/datasets/mipnerf360/native/garden"


def calibration_ratio(method, device, views=4):
    """Real-data calibration fidelity (garden, TRAINING views only): t=1 screen-gradient ratio on
    the step-1 state is 1.7 +- 0.15, and the t=T ratio on the cached 15k reference is in the Fig. 2
    12k-15k envelope [1.15, 2.3]. Measured with annealing semantics (the weights are shared by all
    variants). Unverified when the reference or dataset is absent."""
    from dataclasses import replace
    from pathlib import Path

    from splatstudio.components import build
    from splatstudio.data import initial_points, load_dataset, on_device
    from splatstudio.runtime import knn_log_scales
    from splatstudio.state import initialize
    from splatstudio.verification import ProbeUnavailable

    from .defaults import CALIBRATED, WEIGHT_TARGETS
    from .frequency import parse_config
    from .gradstats import pooled_ratio, term_gradients

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is required for the gsplat rasterizer")
    if not (CALIBRATED["w_l"] and CALIBRATED["w_h"]):
        raise ProbeUnavailable("w_l/w_h are not calibrated")
    ref_path = Path(REFERENCE_DIR) / f"{REAL_SCENE}.pt"
    data = Path(REAL_DATA).expanduser()
    if not ref_path.exists() or not (data / "transforms.json").exists():
        raise ProbeUnavailable(f"Needs {ref_path} and {data} (calibration machine only)")
    spec = _component(method, "fregs-frequency")
    cfg = parse_config({**spec.config, "anneal": True})
    renderer, _, losses = build(method)
    photometric = dict(losses)["photometric"]
    dataset = load_dataset(data)
    names = dataset["splits"]["train"]  # training views only
    names = [names[round(i * (len(names) - 1) / (views - 1))] for i in range(views)]
    weights = _weights(cfg)

    def ratio(params, degree, t):
        params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
        per_view = []
        for name in names:
            frame = on_device(dataset["frames"][name], device)
            out = renderer.render(params, frame, sh_degree=degree)
            per_view.append(term_gradients(out, frame["image"], t, cfg, photometric))
            del out
        value = pooled_ratio(per_view, weights)[1]
        del per_view
        torch.cuda.empty_cache()
        return value

    torch.manual_seed(0)
    points, colors = initial_points(dataset, 10_000_000, 0, device)
    params, _ = initialize(replace(method, fields=()), points, colors)
    with torch.no_grad():
        params["scales"].copy_(knn_log_scales(points))
    t1 = ratio(dict(params), 0, 1)
    del params
    reference = torch.load(ref_path, map_location=device, weights_only=True)
    if reference["protocol_id"] != dataset["protocol_id"]:
        raise AssertionError("Cached reference was trained on a different dataset/protocol")
    t_t = ratio(reference["params"], int(reference["sh_degree"]), cfg["t_end"])
    lo, hi = WEIGHT_TARGETS["fig2_envelope_tT"]
    evidence = {"scene": REAL_SCENE, "train_views": names, "t1_means2d_ratio": t1,
                "tT_means2d_ratio": t_t, "t1_target": [1.7 - 0.15, 1.7 + 0.15],
                "tT_envelope": [lo, hi], "reference_step": int(reference["step"])}
    if not abs(t1 - WEIGHT_TARGETS["w_l"]) <= 0.15:
        raise AssertionError(f"t=1 ratio outside 1.7 +- 0.15: {evidence}")
    if not lo <= t_t <= hi:
        raise AssertionError(f"t=T ratio outside the Fig. 2 envelope: {evidence}")
    return evidence
