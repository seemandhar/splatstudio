"""Mechanism probes for AD-GS; each raises if its mechanism is removed or wrong."""

import numpy as np
import torch
from splatstudio.state import sample_scene


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _density(method):
    from .density import create_density

    return create_density(_spec(method, "adgs-density"))


def alternating_densification(method, device):
    from .density import thresholds

    density = _density(method)
    cfg = density.config
    # Independent statement of the official schedule (train.py:168-169, 384-401).
    for step in range(1, 10001):
        expected = step > 1500 and step % 200 >= 100
        if density.schedule.regularize(step) != expected:
            raise AssertionError(f"regularisation gate wrong at step {step}")
    phases = {s: density.schedule.phase(s) for s in (600, 1500, 1600, 1700, 9800, 9900)}
    if phases != {600: "warmup", 1500: "warmup", 1600: "high", 1700: "low", 9800: "high",
                  9900: "low"}:
        raise AssertionError(f"phase map wrong: {phases}")
    if thresholds(cfg, "high") != (2e-4, 5e-3) or thresholds(cfg, "low") != (5e-4, 0.1):
        raise AssertionError("per-phase thresholds differ from supp. Table 7")

    def fixture():
        params, opts = sample_scene(method, device)
        with torch.no_grad():
            params["opacities"].fill_(0.0)  # sigmoid 0.5: survives both prune thresholds
            params["opacities"][1] = torch.logit(torch.tensor(0.05))  # between eps_a and eps_L
            params["means"][2] = torch.tensor([40.0, 0.0, 3.0])  # isolated
            params["scales"][2] = np.log(2.0)  # larger than the extent (1.0)
        d = _density(method)
        d.initialize(params, opts, scene_scale=1.0)
        n = len(params["means"])
        d.main.state["grad2d"] = torch.zeros(n, device=device)
        d.main.state["count"] = torch.ones(n, device=device)
        d.main.state["grad2d"][0] = 3e-4  # tau_pos <= g < tau_L
        return d

    out = {}
    for phase in ("high", "low"):
        d = fixture()
        tau, eps = thresholds(d.config, phase)
        out[phase] = d.main.densify_and_prune(tau, eps, 1.0, d.config)
    high, low = out["high"], out["low"]
    if high["cloned"] + high["split"] - high["fsgs_split"] != 1:
        raise AssertionError(f"high phase must densify the g=3e-4 Gaussian: {high}")
    if low["cloned"] + low["split"] - low["fsgs_split"] != 0:
        raise AssertionError(f"low phase must not densify the g=3e-4 Gaussian: {low}")
    if high["pruned"] != 0 or low["pruned"] != 1:
        raise AssertionError(f"opacity 0.05 must be pruned only in the low phase: {high} {low}")
    if high["fsgs_split"] != 1 or low["fsgs_split"] != 1:
        raise AssertionError("FSGS isolated-large-Gaussian split criterion inactive")
    return {"high": high, "low": low, "phases": {str(k): v for k, v in phases.items()}}


def phase_gated_losses(method, device):
    from .losses import create_depth_smoothness

    fn = create_depth_smoothness(_spec(method, "adgs-depth"))
    g = torch.Generator(device=device).manual_seed(0)
    depth = torch.rand(16, 16, 1, generator=g, device=device).requires_grad_(True)
    image = torch.rand(16, 16, 3, generator=g, device=device)
    rgb = torch.rand(16, 16, 3, generator=g, device=device).requires_grad_(True)
    values = {}
    for step in (1000, 1600, 1699, 1700, 1799, 1800):
        v = fn({}, {"depth": depth, "rgb": rgb, "step": step}, image)
        if not v.requires_grad:
            raise AssertionError("gated loss detached from the graph")
        values[step] = v.item()
    active = {s for s, v in values.items() if v != 0.0}
    if active != {1700, 1799}:
        raise AssertionError(f"depth regulariser active at {sorted(active)}; expected 1700, 1799")
    return {"values": {str(k): v for k, v in values.items()}}


def edge_aware_depth(method, device):
    from .losses import depth_regularizer

    spec = _spec(method, "adgs-depth")
    w, lr = spec.config["weight"], spec.config["range_weight"]
    if (w, lr) != (0.01, 0.001):
        raise AssertionError("omega_1 / lambda_r differ from the paper values")
    g = torch.Generator(device="cpu").manual_seed(1)
    depth = torch.rand(7, 9, 1, generator=g, dtype=torch.float64)
    image = torch.rand(7, 9, 3, generator=g, dtype=torch.float64)
    d = depth.clone().requires_grad_(True)
    value = depth_regularizer(d, image, w, lr)
    (grad,) = torch.autograd.grad(value, d)
    # Independent per-pixel loop of the official edge_aware_smoothness + range term.
    H, W = 7, 9
    sx = sum(abs(depth[y, x + 1, 0] - depth[y, x, 0])
             * np.exp(-float((image[y, x + 1] - image[y, x]).abs().mean()))
             for y in range(H) for x in range(W - 1)) / (H * (W - 1))
    sy = sum(abs(depth[y + 1, x, 0] - depth[y, x, 0])
             * np.exp(-float((image[y + 1, x] - image[y, x]).abs().mean()))
             for y in range(H - 1) for x in range(W)) / ((H - 1) * W)
    expected = w * (float(sx + sy) - lr * float(depth.max() - depth.min()))
    if abs(value.item() - expected) > 1e-10:
        raise AssertionError(f"Eq. 7 mismatch: {value.item()} vs {expected}")
    # Range term pushes the max depth up and the min depth down.
    i_max, i_min = depth.argmax(), depth.argmin()
    eps = 1e-6
    for idx in (i_max, i_min):
        bumped = depth.clone().view(-1)
        bumped[idx] += eps
        num = (depth_regularizer(bumped.view(7, 9, 1), image, w, lr).item() - value.item()) / eps
        if abs(num - grad.view(-1)[idx].item()) > 1e-4:
            raise AssertionError("gradient mismatch against finite differences")
    return {"value": value.item(), "expected": expected,
            "grad_at_max": grad.view(-1)[i_max].item()}


def pseudo_view_consistency(method, device):
    from splatstudio.components import build
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")
    from dataclasses import replace

    # Register the synthetic cameras below with official-style bounds that differ from the
    # init-cloud percentiles (z in [2.5, 3.5]), so using the wrong source is detected.
    centers = [[-x, 0.0, 0.0] for x in (-0.2, 0.0, 0.2)]
    comps = list(method.components)
    k = next(i for i, c in enumerate(comps) if c.name == "adgs-density")
    table = {**comps[k].config["scene_bounds"],
             "probe": {"near": 1.0, "far": 50.0, "train_centers": centers}}
    comps[k] = replace(comps[k], config={**comps[k].config, "scene_bounds": table})
    method = replace(method, components=tuple(comps))
    renderer, density, losses = build(method)
    params, opts = sample_scene(method, device, n=64)
    density.initialize(params, opts, scene_scale=1.0)
    with torch.no_grad():  # make the two fields differ (official: via random splits)
        density.partner.params["sh0"].add_(0.3)
    K = torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=device)
    cams = []
    for x in (-0.2, 0.0, 0.2):
        v = torch.eye(4, device=device)
        v[0, 3] = x
        cams.append({"viewmat": v, "K": K, "width": 32, "height": 32})
    density.set_training_cameras(cams)
    info = density.pseudo_info
    scale = 1 / (1.0 * 0.75)
    focus = 1 / (0.25 / (0.9 * 1.0 * scale) + 0.75 / (5 * 50.0 * scale)) / scale
    if info["scene"] != "probe" or info["bounds"] != [[1.0, 50.0]] or abs(
        info["focus_distance"] - focus
    ) > 1e-9:
        raise AssertionError(f"pseudo poses do not use the official near/far bounds: {info}")
    poses = density.pseudo_viewmats.double().cpu()
    R = poses[:, :3, :3]
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    if (R @ R.transpose(1, 2) - eye).abs().max() > 1e-4:
        raise AssertionError("pseudo poses are not rotations")
    centers = -(R.transpose(1, 2) @ poses[:, :3, 3:4])[..., 0]
    if centers[:, 0].abs().max() > 0.2 + 1e-3 or centers[:, 2].abs().max() > 1e-3:
        raise AssertionError("pseudo camera centres leave the training-camera box")
    losses = dict(losses)
    target = torch.rand(32, 32, 3, device=device)
    evidence = {"focus_distance": density.pseudo_info["focus_distance"]}
    for step, expect in ((1600, False), (1700, True)):
        for p in list(params.values()) + list(density.partner.params.values()):
            p.grad = None
        outputs = renderer.render(params, cams[1], sh_degree=0)
        outputs["step"] = step
        density.before_backward(params, opts, step, outputs)
        before = len(density._order)
        term = losses["adgs-cotrain"](params, outputs, target)
        g2 = density.partner.params["sh0"].grad
        if g2 is None or not torch.count_nonzero(g2):
            raise AssertionError("G2 received no gradient from the co-training loss")
        (g1,) = torch.autograd.grad(term, params["sh0"], allow_unused=True)
        used = before != len(density._order) or (before == 0 and len(density._order) > 0)
        active = g1 is not None and bool(torch.count_nonzero(g1))
        if active != expect or used != expect:
            raise AssertionError(f"pseudo loss gating wrong at step {step}: active={active}")
        evidence[f"g1_pseudo_grad_norm_{step}"] = 0.0 if g1 is None else g1.norm().item()
        evidence[f"g2_grad_norm_{step}"] = g2.norm().item()
    # Ablation: without detach, G1's term would also push G2 — check G2 is untouched by it.
    density.partner.params["sh0"].grad = None
    outputs = renderer.render(params, cams[1], sh_degree=0)
    outputs["step"] = 1700
    density.before_backward(params, opts, 1700, outputs)
    term = losses["adgs-cotrain"](params, outputs, target)
    g2_before = density.partner.params["sh0"].grad.clone()
    term.backward()
    if not torch.equal(g2_before, density.partner.params["sh0"].grad):
        raise AssertionError("G1's pseudo loss leaks gradient into G2 (target not detached)")
    density.after_backward(params, opts, 1700, outputs)
    return evidence


def sh_schedule(method, device):
    from .renderer import create_renderer

    spec = _spec(method, "adgs-renderer")
    r = create_renderer(spec)
    seen = {}
    with torch.enable_grad():
        for n in range(1, 3001):
            d = r.degree()
            if d != min(n // 500, 3):  # official train.py:191 oneupSHdegree every 500
                raise AssertionError(f"SH degree {d} at training render {n}")
            if n in (499, 500, 1000, 1500):
                seen[n] = d
    if seen[500] == min((500 - 1) // 1000, 3):
        raise AssertionError("renderer follows the runtime's 1000-step SH schedule")
    with torch.no_grad():
        if r.degree() != 3:
            raise AssertionError("evaluation must use the final SH degree 3")
    evidence = {"degrees": {str(k): v for k, v in seen.items()}}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        r = create_renderer(spec)
        r.training_renders = 1499
        params, _ = sample_scene(method, device)
        cam = {"viewmat": torch.eye(4, device=device), "width": 16, "height": 16,
               "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device)}
        out = r.render(params, cam, sh_degree=1)  # runtime value at step 1500
        if out["sh_degree"] != 3:
            raise AssertionError("render() does not apply the official SH degree")
        evidence["render_degree_at_1500"] = out["sh_degree"]
    return evidence
