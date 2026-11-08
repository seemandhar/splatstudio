"""Perceptual-GS mechanism probes. Each one fails if its mechanism is removed or wrong.

Reference computations are written independently of the plugin code paths they check
(explicit-loop Sobel / block pooling, alpha-map algebra for the sensitivity branch, sorting-based
quartile for gamma, brute-force blending for the CUDA statistics).
"""

import math
from dataclasses import replace

import numpy as np
import torch
from splatstudio.registry import resolve
from torch import nn

from . import core, kernels, loss, sidecar

FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN", "sensitivity")


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, kind=None, name=None, **overrides):
    spec = next(c for c in method.components if (c.kind == kind if kind else c.name == name))
    if overrides:
        spec = replace(spec, config={**spec.config, **overrides})
    return resolve(spec.factory)(spec)


def _scene(means, scales, opacities, sens, quats=None, device="cuda"):
    n = len(means)
    if quats is None:
        quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    values = {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
        "quats": torch.as_tensor(quats, dtype=torch.float32),
        "opacities": torch.logit(torch.as_tensor(opacities, dtype=torch.float32)),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 15, 3),
        "sensitivity": torch.logit(torch.as_tensor(sens, dtype=torch.float32)),
    }
    params = nn.ParameterDict({k: nn.Parameter(v.to(device)) for k, v in values.items()})
    rates = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 0.05, "sh0": 2.5e-3,
             "shN": 1.25e-4, "sensitivity": 0.05}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    return params, opts


def _camera(width, height, f, c2w=None, device="cuda", image=None):
    if c2w is None:
        c2w = torch.eye(4)
    cam = {
        "viewmat": torch.linalg.inv(c2w).to(device),
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device),
        "width": width,
        "height": height,
    }
    if image is not None:
        cam["image"] = image.to(device)
    return cam


def _cams_only(cams):
    return [{k: c[k] for k in ("K", "viewmat", "width", "height")} for c in cams]


def _trainable(params):
    bad = [k for k, p in params.items() if not (p.is_leaf and p.requires_grad)]
    if bad:
        raise AssertionError(f"Fields no longer trainable: {bad}")


# ---------------------------------------------------------------------- C1
def _reference_map(lum, tau_e, tau_s, pool):
    """Explicit-loop numpy Sobel (zero padding), threshold, non-overlapping block mean, repeat."""
    h, w = lum.shape
    pad = np.pad(lum.astype(np.float64), 1)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    mag = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            win = pad[i:i + 3, j:j + 3]
            mag[i, j] = math.hypot((win * kx).sum(), (win * ky).sum())
    binary = (mag > tau_e).astype(np.float64)
    hb, wb = h // pool, w // pool
    # nearest upsample of the floor-pooled map covers the remainder with the last blocks
    rows = np.minimum((np.arange(h) * hb) // h, hb - 1)
    cols = np.minimum((np.arange(w) * wb) // w, wb - 1)
    pooled = np.array([[binary[r * pool:(r + 1) * pool, c * pool:(c + 1) * pool].mean()
                        for c in range(wb)] for r in range(hb)])
    out = pooled[rows][:, cols]
    return (out > tau_s).astype(np.float64), mag


def extraction(method, device):
    g = np.random.default_rng(0)
    h, w = 42, 53
    lum = np.full((h, w), 0.5)
    lum[:, 32:] = 0.9  # strong edge between columns 31 and 32 (inside pooling block 30-34)
    lum[5:20, 5:20] += g.uniform(-0.03, 0.03, size=(15, 15))  # faint texture
    got = core.sensitivity_map(torch.tensor(lum, dtype=torch.float32)[None, None])[0, 0].numpy()
    ref, mag = _reference_map(lum, 0.05, 0.3, 5)
    mismatch = int((got != ref).sum())
    if mismatch:
        raise AssertionError(f"Sensitivity map differs from the reference at {mismatch} pixels")
    texture = got[5:20, 5:20].mean()
    flat = got[25:40, 10:25].mean()
    edge = got[:40, 30:35].mean()
    # Without enhancement (pooling raw magnitudes) the faint texture is lost (Sec. 3.3 claim).
    raw = torch.nn.functional.avg_pool2d(torch.tensor(mag)[None, None], 5)
    raw_texture = float(raw[0, 0, 1:4, 1:4].mean())
    if not (texture > 0.5 and flat == 0 and edge > 0.5 and raw_texture < 0.3):
        raise AssertionError(f"Unexpected map: texture {texture}, flat {flat}, edge {edge}, "
                             f"raw-texture pooled {raw_texture}")
    return {"pixels": h * w, "mismatch": mismatch, "texture_sensitive_fraction": float(texture),
            "flat_sensitive_fraction": float(flat), "raw_magnitude_pooled_texture": raw_texture}


# ---------------------------------------------------------------------- C2
def dual_branch(method, device):
    _require_cuda()
    from gsplat import rasterization

    w, h = 48, 40
    params, _ = _scene([[0.0, 0.0, 3.0], [0.4, 0.2, 4.0]], [[0.5, 0.4, 0.1], [0.6, 0.6, 0.1]],
                       [0.8, 0.6], [0.7, 0.2], device=device)
    cam = _camera(w, h, 40.0, device=device)
    # Independent algebra: R[1 - s] = sum_i w_i (1 - s_i) = A - R[s]
    ones = torch.ones(2, 1, device=device)
    alpha = rasterization(params["means"], params["quats"], params["scales"].exp(),
                          params["opacities"].sigmoid(), ones, cam["viewmat"][None],
                          cam["K"][None], w, h, sh_degree=None)[0][0, ..., 0]
    r_s = loss.render_sensitivity(params, cam, params["sensitivity"].sigmoid())
    pred = core.predicted_sensitivity(loss.render_sensitivity(params, cam))
    ref = (1 - alpha) + r_s
    err = (pred - ref).abs().max().item()
    if err > 1e-5:
        raise AssertionError(f"Sensitivity branch differs from (1-A) + R[sigma(eps)] by {err}")
    bce = _component(method, name="pgs-bce")
    photo = _component(method, name="pgs-photometric")
    gt = torch.zeros(h, w)
    gt[:, : w // 2] = 1.0
    bce.table = sidecar.Table(cam["viewmat"].cpu()[None], (gt * 255).byte()[None])
    target = torch.full((h, w, 3), 0.5, device=device)
    out = {"rgb": torch.full((h, w, 3), 0.4, device=device, requires_grad=True), "step": 100,
           "camera": cam}
    value = bce(params, out, target)
    value.backward()
    grad = params["sensitivity"].grad.clone()
    expected = 0.1 * core.bce(ref.detach().clamp(0, 1), gt.to(device)).item()
    if abs(value.item() - expected) > 1e-5 or not (grad.abs() > 0).all():
        raise AssertionError(f"BCE {value.item()} vs {expected}; grad {grad.tolist()}")
    # gradient vs a finite difference
    with torch.no_grad():
        eps = 1e-2
        params["sensitivity"][0] += eps
        up = bce(params, out, target).item()
        params["sensitivity"][0] -= eps
    fd = (up - value.item()) / eps
    if abs(fd - grad[0].item()) > 0.05 * abs(grad[0].item()) + 1e-5:
        raise AssertionError(f"BCE gradient {grad[0].item()} vs finite difference {fd}")
    photo_val = photo(params, out, target)
    full = _component(method, name="pgs-photometric", lambda_s=0.0)(params, out, target)
    late = {**out, "step": 15001}
    off = bce(params, late, target)
    photo_late = photo(params, late, target)
    if abs(photo_val.item() - 0.9 * full.item()) > 1e-6 or off.item() != 0:
        raise AssertionError("Eq. 10 weighting or the 15k schedule boundary is wrong")
    if abs(photo_late.item() - full.item()) > 1e-6 or not off.requires_grad:
        raise AssertionError("Photometric loss must be unweighted after 15k")
    return {"branch_max_error": err, "bce": value.item(), "grad_eps0": grad[0].item(),
            "finite_difference": fd, "photometric_weight": photo_val.item() / full.item()}


# ---------------------------------------------------------------------- C3
def _guided_run(method, device, sens, step, scene_scale=1.0, split_only=False, **overrides):
    renderer = _component(method, kind="renderer")
    density = _component(method, kind="density", **overrides)
    params, opts = _scene([[0.0, 0.0, 3.0]], [[0.5, 0.5, 0.5]], [0.9], [sens], device=device)
    density.initialize(params, opts, scene_scale=scene_scale)
    cam = _camera(64, 48, 40.0, device=device)
    density.set_training_cameras(_cams_only([cam]))
    density.split_only = split_only
    out = renderer.render(params, cam, sh_degree=0)
    out["step"] = step
    density.before_backward(params, opts, step, out)
    (out["rgb"] * 0).sum().backward()  # zero viewspace gradient: only guided masks can fire
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    density.after_backward(params, opts, step, out)
    _trainable(params)
    return params, density


def guided_densify(method, device):
    _require_cuda()
    cases = {
        "high_1000": (0.95, 1000, 2),
        "high_1100": (0.95, 1100, 1),
        "medium_1000": (0.5, 1000, 1),
        "medium_1500": (0.5, 1500, 2),
        "low_1500": (0.1, 1500, 1),
        "high_1000_weight_disabled": (0.95, 1000, 1),
    }
    got = {}
    for name, (s, step, want) in cases.items():
        kw = {"complex_weight": 1e9} if name.endswith("weight_disabled") else {}
        params, density = _guided_run(method, device, s, step, **kw)
        got[name] = len(params["means"])
        if got[name] != want:
            raise AssertionError(f"{name}: {got[name]} Gaussians, expected {want}")
    # Small high-sensitivity Gaussian: clone unless split_only (beta > 0.85) -> split.
    p_clone, _ = _guided_run(method, device, 0.95, 1000, scene_scale=1000.0)
    p_split, _ = _guided_run(method, device, 0.95, 1000, scene_scale=1000.0, split_only=True)
    s_clone = p_clone["scales"].exp().max().item()
    s_split = p_split["scales"].exp().max().item()
    if len(p_clone["means"]) != 2 or abs(s_clone - 0.5) > 1e-5:
        raise AssertionError("Small complex Gaussian should be cloned when not split_only")
    if len(p_split["means"]) != 2 or abs(s_split - 0.5 / 1.6) > 1e-5:
        raise AssertionError("split_only must split small complex Gaussians")
    # The sensitivity field must stay trainable through densification (ResGS pitfall).
    cam = _camera(64, 48, 40.0, device=device)
    loss.render_sensitivity(p_split, cam).sum().backward()
    g = p_split["sensitivity"].grad
    if g is None or not (g.abs() > 0).all():
        raise AssertionError("Sensitivity lost its gradient after densification")
    return {**got, "clone_scale": s_clone, "split_only_scale": s_split}


# ---------------------------------------------------------------------- C5
def opacity_decline(method, device):
    out = {}
    for k in (1.2, None):
        density = _component(method, kind="density", od_k=k)
        params, opts = _scene([[0.0, 0.0, 3.0]], [[0.01, 0.01, 0.01]], [0.6], [0.5], device="cpu")
        density.initialize(params, opts, scene_scale=100.0)
        density.grad2d[:] = 1.0
        density.count[:] = 1.0
        density._densify(params, opts, 600)
        if len(params["means"]) != 2:
            raise AssertionError("Gradient clone did not fire")
        out[str(k)] = params["opacities"].sigmoid().tolist()
    expected = 1 - math.sqrt(1 - 0.6**1.2)
    if any(abs(a - expected) > 1e-5 for a in out["1.2"]):
        raise AssertionError(f"OD opacities {out['1.2']} != {expected}")
    if any(abs(a - 0.6) > 1e-5 for a in out["None"]):
        raise AssertionError("Without OD the clone must keep its opacity")
    region = 1 - (1 - expected) ** 2
    if abs(region - 0.6**1.2) > 1e-5:
        raise AssertionError("Two clones must composite to OD(alpha) = alpha^k (Eq. 20)")
    return {"declined": out["1.2"][0], "expected": expected, "composited": region}


# ---------------------------------------------------------------------- C4
def scene_decision(method, device):
    g = torch.Generator().manual_seed(0)
    n = 400
    scales = torch.rand(n, 3, generator=g) * 0.1 + 0.01
    sens = torch.rand(n, generator=g)
    # Independent gamma: sort by longest axis, take the top quarter
    m = scales.max(dim=1).values
    top = torch.argsort(m, descending=True)[: n // 4]
    ref_gamma = float(((sens[top] > 0.3) & (sens[top] < 0.9)).float().mean())
    gamma = core.sparse_indicator(m, sens)
    if abs(gamma - ref_gamma) > 1e-6:
        raise AssertionError(f"gamma {gamma} vs reference {ref_gamma}")
    results = {}
    for label, s_top in (("sparse", 0.6), ("dense", 0.97)):
        density = _component(method, kind="density")
        s = sens.clone()
        s[top] = s_top
        params, opts = _scene(torch.rand(n, 3, generator=g), scales, [0.5] * n,
                              s.clamp(1e-4, 1 - 1e-4), device="cpu")
        density.initialize(params, opts, scene_scale=1.0)
        cams = [_camera(20, 10, 10.0, device="cpu"), _camera(20, 10, 10.0, device="cpu")]
        cams[1]["viewmat"] = cams[1]["viewmat"].clone()
        cams[1]["viewmat"][0, 3] = 0.5
        density.set_training_cameras(cams)
        maps = torch.zeros(2, 10, 20, dtype=torch.uint8)
        maps[0, :, :18] = 3  # 90% non-zero (bicubic halo counts, code: > 0)
        maps[1, :, :17] = 255  # 85%
        density.table = sidecar.Table(torch.stack([c["viewmat"] for c in cams]), maps)
        density._decide(params, 600)
        results[label] = {"sparse": density.sparse, "split_only": density.split_only,
                          "gamma": density.state["gamma"], "beta": density.state["beta"]}
    if not results["sparse"]["sparse"] or results["dense"]["sparse"]:
        raise AssertionError(f"gamma > 0.55 must select depth reinit: {results}")
    if abs(results["sparse"]["beta"] - 0.875) > 1e-6 or not results["sparse"]["split_only"]:
        raise AssertionError(f"beta must be mean(#sens>0/#pix) = 0.875 > 0.85: {results}")
    return {"gamma_reference": ref_gamma, **{f"{k}_{a}": v for k, d in results.items()
                                             for a, v in d.items()}}


def stats_kernel(method, device):
    _require_cuda()
    from gsplat import rasterization

    g = torch.Generator().manual_seed(3)
    n, w, h = 24, 40, 32
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1,
                       torch.rand(n, 1, generator=g) * 3 + 2], 1)
    means[:, :2] *= means[:, 2:3] * 0.45
    scales = torch.rand(n, 3, generator=g) * 0.25 + 0.05
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.8 + 0.15
    params, _ = _scene(means, scales, opac, [0.5] * n, quats, device=device)
    cam = _camera(w, h, 40.0, device=device)
    meta = kernels.project(params, cam)
    got = kernels.blend_stats(meta, weights=True)
    ref = core.blend_reference(meta["means2d"][0].cpu().double(), meta["conics"][0].cpu().double(),
                               meta["opacities"][0].cpu().double(), meta["depths"][0].cpu(), w, h)
    _, alpha, _ = rasterization(
        params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        params["sh0"], cam["viewmat"][None], cam["K"][None], w, h, sh_degree=0, packed=False)
    agree = (got["argmax"].cpu().long() == ref["argmax"]).float().mean().item()
    err_w = (got["accum_weights"].cpu().double() - ref["accum_weights"]).abs().max().item()
    err_t = (got["T"].cpu().double() - ref["T"]).abs().max().item()
    err_alpha = (1 - got["T"] - alpha[0, ..., 0]).abs().max().item()
    if int((ref["argmax"] >= 0).sum()) < w * h // 2:
        raise AssertionError("Synthetic scene too sparse")
    if agree < 0.995 or err_w > 1e-3 or err_t > 1e-4 or err_alpha > 1e-5:
        raise AssertionError(f"Kernel disagrees with reference: argmax {agree}, w {err_w}, "
                             f"T {err_t}, alpha {err_alpha}")
    return {"argmax_agreement": agree, "max_weight_error": err_w, "max_T_error": err_t,
            "gsplat_alpha_error": err_alpha}


def depth_reinit(method, device):
    _require_cuda()
    renderer = _component(method, kind="renderer")
    density = _component(method, kind="density", num_depth=600)
    params, opts = _scene([[0.0, 0.0, 3.0]], [[2.0, 2.0, 0.005]], [0.8], [0.9], device=device)
    density.initialize(params, opts, scene_scale=1.0)
    w, h = 40, 30
    colors = [torch.tensor([0.9, 0.2, 0.1]), torch.tensor([0.1, 0.3, 0.8])]
    c2w1 = torch.eye(4)
    c2w1[:3, 3] = torch.tensor([0.3, 0.0, 0.0])
    cams = [_camera(w, h, 30.0, c, device=device, image=col.expand(h, w, 3).clone())
            for c, col in zip((torch.eye(4), c2w1), colors)]
    density.set_training_cameras(_cams_only(cams))
    for cam in cams:
        density._cache_target(renderer.render(params, cam, sh_degree=0), 1)
    density._depth_reinit(params, opts, 5000)
    n = len(params["means"])
    z = params["means"][:, 2]
    rgb = params["sh0"][:, 0] * core.SH_C0 + 0.5
    color_err = torch.minimum(*[(rgb - c.to(device)).abs().max(dim=1).values
                                for c in colors]).max().item()
    if not (100 < n <= 600) or (z - 3.0).abs().max().item() > 0.02 or color_err > 2e-3:
        raise AssertionError(f"Depth reinit: n {n}, z dev {(z - 3).abs().max().item()}, "
                             f"colour err {color_err}")
    if (params["opacities"].sigmoid() - 0.1).abs().max() > 1e-6 or \
            (params["sensitivity"].sigmoid() - 0.1).abs().max() > 1e-6:
        raise AssertionError("Reinit must set opacity and sensitivity to 0.1")
    if any(len(o.state) for o in opts.values()):
        raise AssertionError("Reinit must reset the optimizer state")
    _trainable(params)
    return {"gaussians_after": n, "max_depth_deviation": (z - 3).abs().max().item(),
            "max_colour_error": color_err}
