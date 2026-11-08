"""Mechanism probes for the AAA-Gaussians port. Each raises on failure and returns measurements.

The CUDA probes compare the vendored official kernels against the independent float64 dense
renderer in reference.py and against ablated variants of it, so they fail if the mechanism
(3D evaluation, perpendicular amplitude scaling, adaptive filter, view-space bounding, frustum
culling) is removed or wrong.
"""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from . import reference as ref
from .filters import perpendicular_scaling, train_filter_3d, volume_scaling

C0 = ref.C0


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _cpu_params(means, scales, quats, opac, colors):
    means = torch.as_tensor(means, dtype=torch.float32)
    n = len(means)
    return {"means": means,
            "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
            "quats": F.normalize(torch.as_tensor(quats, dtype=torch.float32), dim=-1),
            "opacities": torch.logit(torch.as_tensor(opac, dtype=torch.float32)),
            "sh0": ((torch.as_tensor(colors, dtype=torch.float32) - 0.5) / C0)[:, None, :],
            "shN": torch.zeros(n, 15, 3)}


def _gpu(p, device):
    return {k: torch.nn.Parameter(v.to(device)) for k, v in p.items()}


def _camera(W, H, f, device, view=None):
    v = torch.eye(4) if view is None else torch.as_tensor(view, dtype=torch.float32)
    return {"viewmat": v.to(device), "width": W, "height": H,
            "K": torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)}


def _settings(method, **kw):
    from .raster import make_settings

    spec = next(c for c in method.components if c.kind == "renderer")
    return make_settings(spec.config.get("splatting"), **kw)


def _render(p, camera, settings, bg, filter_3d=None):
    from .raster import render

    color, _, radii = render(p, camera, 0, bg, settings, filter_3d=filter_3d)
    return color.permute(1, 2, 0), radii


def _random_scene(seed=0, n=24):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([2.0, 1.6, 2.0]) + torch.tensor(
        [-1.0, -0.8, 3.0])
    scales = torch.rand(n, 3, generator=g) * 0.25 + 0.01
    scales[:4, 2] = 0.002  # flat, strongly anisotropic splats (perpendicular scaling matters)
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.7 + 0.2
    cols = torch.rand(n, 3, generator=g)
    return _cpu_params(means, scales, quats, opac, cols), g


def _max_err(a, b):
    return (a.detach().cpu().double() - b.detach().cpu().double()).abs().max().item()


# ---------------------------------------------------------------------------- probes
def forward_reference(method, device):
    """CUDA image == dense 3D reference (perpendicular scaling, adaptive filter, per-pixel
    depth order); the volume-scaled (Eq. 8), unscaled and 2D-EWA (eval_3D off) variants are
    far off."""
    _cuda(device)
    W, H, f = 64, 48, 60.0
    cam = _camera(W, H, f, device)
    bg = torch.tensor([0.1, 0.2, 0.3], device=device)
    res = {}
    for seed in (0, 1):
        p, _ = _random_scene(seed)
        img, _ = _render(_gpu(p, device), cam, _settings(method), bg)
        L = ref.leaves_from(p)
        err = _max_err(img, ref.composite(L, cam, bg.cpu()))
        err_vol = _max_err(img, ref.composite(L, cam, bg.cpu(), scaling="volume"))
        err_none = _max_err(img, ref.composite(L, cam, bg.cpu(), scaling="none"))
        img2d, _ = _render(_gpu(p, device), cam, _settings(method, eval_3D=False), bg)
        err_2d = _max_err(img, img2d)
        res[f"seed{seed}"] = {"max_err": err, "volume_scaling_err": err_vol,
                              "no_scaling_err": err_none, "ewa2d_vs_3d": err_2d}
        if err > 2e-4:
            raise AssertionError(f"CUDA vs dense 3D reference: max err {err}")
        if min(err_vol, err_none, err_2d) < 50 * max(err, 1e-5):
            raise AssertionError(f"ablations are not distinguishable: {res}")
    return res


def backward_reference(method, device):
    """Kernel gradients (means, log-scales, quats, opacity logits, sh0) == autograd of the
    dense reference; the volume-scaled reference's gradients differ."""
    _cuda(device)
    W, H, f = 64, 48, 60.0
    cam = _camera(W, H, f, device)
    bg = torch.tensor([0.1, 0.2, 0.3], device=device)
    p, g = _random_scene(2)
    pc = _gpu(p, device)
    w = torch.rand(H, W, 3, generator=g)
    img, _ = _render(pc, cam, _settings(method), bg)
    (img * w.to(device)).sum().backward()
    res = {}
    for scaling in ("perpendicular", "volume"):
        L = ref.leaves_from(p)
        (ref.composite(L, cam, bg.cpu(), scaling=scaling) * w.double()).sum().backward()
        for k in ("means", "scales", "quats", "opacities", "sh0"):
            a, b = pc[k].grad.cpu().double(), L[k].grad
            res[f"{scaling}/{k}"] = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)
    worst = max(v for k, v in res.items() if k.startswith("perpendicular/"))
    ablated = min(v for k, v in res.items() if k.startswith("volume/") and "sh0" not in k)
    if worst > 1e-3:
        raise AssertionError(f"kernel vs reference gradient rel err {worst}: {res}")
    if ablated < 20 * worst:
        raise AssertionError(f"volume-scaling ablation not distinguishable by gradients: {res}")
    res["worst_rel"] = worst
    return res


def perp_scaling(method, device):
    """Thin disk (s3 = 1e-3) facing the camera: peak alpha = o * Eq.(11) factor (perpendicular,
    ~0.995), not Eq.(8) (volume, ~0.07); edge-on the two factors are checked too."""
    _cuda(device)
    W, H, f = 33, 33, 50.0
    cam = _camera(W, H, f, device)
    bg = torch.zeros(3, device=device)
    s = torch.tensor([[0.3, 0.3, 1e-3]])
    z = 3.0
    v = torch.tensor([0.3 * (z / f) ** 2])
    out = {}
    for name, quat in (("face_on", [1.0, 0, 0, 0]),
                       ("edge_on", [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])):
        p = _cpu_params([[0.0, 0.0, z]], s, [quat], [0.8], [[1.0, 1.0, 1.0]])
        img, _ = _render(_gpu(p, device), cam, _settings(method), bg)
        # pixel (16, 16) centre ray x/z = 0.5/f: evaluate the reference's G there
        L = ref.leaves_from(p)
        r = ref.composite(L, cam, bg.cpu())
        R = ref.rotmat(p["quats"])
        perp = perpendicular_scaling(s.double(), R, torch.tensor([[0, 0, 1.0]], dtype=ref.DD),
                                     v.double()).item()
        vol = volume_scaling(s.double(), v.double()).item()
        peak = img[16, 16, 0].item()
        G = r[16, 16, 0].item() / (0.8 * perp)
        out[name] = {"peak": peak, "perp_factor": perp, "volume_factor": vol,
                     "peak_if_perp": 0.8 * perp * G, "peak_if_volume": 0.8 * vol * G}
        if abs(peak - 0.8 * perp * G) > 1e-4:
            raise AssertionError(f"{name}: peak {peak} != o*perp*G {0.8 * perp * G}")
    fo = out["face_on"]
    if not (fo["perp_factor"] > 0.95 and fo["volume_factor"] < 0.2
            and abs(fo["peak"] - fo["peak_if_volume"]) > 0.5):
        raise AssertionError(f"face-on disk does not separate Eq. 11 from Eq. 8: {fo}")
    return out


def _moments(img):
    a = img[..., 0].double().cpu()
    H, W = a.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=a.dtype), torch.arange(W, dtype=a.dtype),
                            indexing="ij")
    m = a.sum()
    cx, cy = (a * xs).sum() / m, (a * ys).sum() / m
    var = (a * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / (2 * m)
    return m.item(), var.item()


def adaptive_filter(method, device):
    """Small isotropic Gaussian (s = 0.006) at z = 2 and z = 4: the per-view filter
    v = k z^2/f^2 keeps the rendered mass ~ s^2 f^2/z^2 (ratio 4, energy-preserving
    anti-aliasing) and the image equals the reference; a training-view filter_3d from z = 4
    floors v at z = 2 (v' = min(v_train, v), Eq. 12): pixel variance grows ~3.5x and the image
    equals the floored reference, not the unfloored one."""
    _cuda(device)
    W, H, f = 41, 41, 80.0
    cam = _camera(W, H, f, device)
    bg = torch.zeros(3, device=device)
    out = {}
    imgs = {}

    def scene(z):
        return _cpu_params([[0.0, 0.0, z]], [[0.006] * 3], [[1.0, 0, 0, 0]], [0.9], [[1.0] * 3])

    for z in (2.0, 4.0):
        p = scene(z)
        img, _ = _render(_gpu(p, device), cam, _settings(method), bg)
        imgs[z] = img
        err = _max_err(img, ref.composite(ref.leaves_from(p), cam, bg.cpu()))
        mass, var = _moments(img)
        out[f"z{z:g}"] = {"mass": mass, "pixel_var": var, "max_err": err}
        if not (mass > 0 and err < 1e-5):
            raise AssertionError(f"z={z}: {out}")
    ratio = out["z2"]["mass"] / out["z4"]["mass"]
    out["mass_ratio_z2_z4"] = ratio
    if abs(ratio - 4.0) > 0.6:
        raise AssertionError(f"per-view filter does not preserve energy: ratio {ratio}")
    p = scene(2.0)
    floor = torch.tensor([math.sqrt(0.3) * 4.0 / f])
    img, _ = _render(_gpu(p, device), cam, _settings(method), bg, filter_3d=floor.to(device))
    mass, var = _moments(img)
    L = ref.leaves_from(p)
    err = _max_err(img, ref.composite(L, cam, bg.cpu(), filter_3d=floor))
    err_unfloored = _max_err(img, ref.composite(L, cam, bg.cpu()))
    out["floored"] = {"mass": mass, "pixel_var": var, "max_err": err,
                      "err_vs_unfloored_ref": err_unfloored}
    if err > 1e-5 or err_unfloored < 100 * max(err, 1e-6) or var < 2.5 * out["z2"]["pixel_var"]:
        raise AssertionError(f"training-view floor not applied: {out}")
    return out


def frustum_bounds(method, device):
    """A large Gaussian whose mean is behind the camera / far outside the view but which
    extends into the frustum is rendered (== reference); HTGS-style projective bounding
    (new_aabb off) drops it (pop-in); a Gaussian fully outside the frustum is culled (0 tiles)."""
    _cuda(device)
    W, H, f = 48, 40, 40.0
    cam = _camera(W, H, f, device)
    bg = torch.zeros(3, device=device)
    cases = {
        "mean_behind_camera": ([[1.0, 0.0, -0.1]], [[0.15, 0.3, 0.6]]),
        "mean_left_outside_fov": ([[-2.5, 0.0, 2.0]], [[0.8, 0.3, 0.3]]),
    }
    out = {}
    for name, (m, s) in cases.items():
        p = _cpu_params(m, s, [[1.0, 0, 0, 0]], [0.9], [[1.0, 0.5, 0.2]])
        img, radii = _render(_gpu(p, device), cam, _settings(method), bg)
        r = ref.composite(ref.leaves_from(p), cam, bg.cpu())
        img_old, _ = _render(_gpu(p, device), cam, _settings(method, new_aabb=False), bg)
        out[name] = {"ref_mass": r.sum().item(), "mass": img.sum().item(),
                     "max_err": _max_err(img, r), "old_bounding_mass": img_old.sum().item()}
        if r.sum().item() < 1.0 or out[name]["max_err"] > 2e-4:
            raise AssertionError(f"{name}: view-space bounding failed {out[name]}")
    if out["mean_behind_camera"]["old_bounding_mass"] > 0.5 * out["mean_behind_camera"]["mass"]:
        raise AssertionError(f"projective bounding should discard the behind-camera case: {out}")
    p = _cpu_params([[3.0, 0.0, 2.0]], [[0.05] * 3], [[1.0, 0, 0, 0]], [0.9], [[1.0] * 3])
    img, radii = _render(_gpu(p, device), cam, _settings(method), bg)
    out["outside"] = {"mass": img.sum().item(), "radii": int(radii[0])}
    if img.sum().item() != 0 or int(radii[0]) != 0:
        raise AssertionError(f"Gaussian outside the frustum not culled: {out['outside']}")
    # tile culling only removes work: same image with tile_based_culling off
    p, _ = _random_scene(4)
    a, _ = _render(_gpu(p, device), cam, _settings(method), bg)
    b, _ = _render(_gpu(p, device), cam, _settings(method, tile_based_culling=False), bg)
    from .raster import forward_raw
    pg = _gpu(p, device)
    n_on = forward_raw(pg, cam, 0, bg, _settings(method))[0]
    n_off = forward_raw(pg, cam, 0, bg, _settings(method, tile_based_culling=False))[0]
    out["tile_culling"] = {"image_diff": _max_err(a, b), "entries_on": n_on, "entries_off": n_off}
    if _max_err(a, b) > 1e-5 or n_on >= n_off:
        raise AssertionError(f"tile-based 3D culling: {out['tile_culling']}")
    return out


def filter_train(method, device):
    """filter_3d == an independent per-Gaussian loop of mip_filter.py (k = 0.3, z > 0.2, 15%
    margin, max focal_x, 0 for unseen); the density writes it only at the final step."""
    from .density import create_density

    g = torch.Generator().manual_seed(5)
    means = torch.randn(40, 3, generator=g) * 2
    cams = []
    for i in range(3):
        view = torch.eye(4)
        view[:3, 3] = torch.tensor([0.3 * i, 0.0, 4.0 + i])
        cams.append({"viewmat": view, "width": 64, "height": 48,
                     "K": torch.tensor([[50.0 + 10 * i, 0, 32], [0, 55.0, 24], [0, 0, 1]])})
    got = train_filter_3d(means, cams)
    fmax = max(float(c["K"][0, 0]) for c in cams)
    want = torch.zeros(40)
    for n in range(40):
        zs = []
        for c in cams:
            pc = c["viewmat"][:3, :3] @ means[n] + c["viewmat"][:3, 3]
            if pc[2] <= 0.2:
                continue
            x = pc[0] / pc[2] * c["K"][0, 0] + c["K"][0, 2]
            y = pc[1] / pc[2] * c["K"][1, 1] + c["K"][1, 2]
            if -0.15 * 64 <= x <= 1.15 * 64 and -0.15 * 48 <= y <= 1.15 * 48:
                zs.append(float(pc[2]))
        want[n] = min(zs) / fmax * math.sqrt(0.3) if zs else 0.0
    err = (got - want).abs().max().item()
    if err > 1e-6 or (want == 0).sum() == 0 or (want > 0).sum() == 0:
        raise AssertionError(f"train filter err {err}, unseen {(want == 0).sum()}")
    spec = next(c for c in method.components if c.kind == "density")
    d = create_density(replace(spec, config={**spec.config, "cap_max": 100, "total_steps": 7}))
    params = {k: torch.nn.Parameter(v) for k, v in
              _cpu_params(means, torch.full((40, 3), 0.05), [[1.0, 0, 0, 0]] * 40,
                          [0.5] * 40, torch.full((40, 3), 0.5)).items()}
    params["filter_3d"] = torch.nn.Parameter(torch.zeros(40))
    d.cameras = cams
    if params["filter_3d"].abs().max() != 0:
        raise AssertionError("filter_3d must be zero before the final step")
    d.update_filter(params)
    if (params["filter_3d"].detach() - want).abs().max() > 1e-6:
        raise AssertionError("density final-step filter differs from mip_filter.py")
    return {"max_err": err, "unseen": int((want == 0).sum()), "n": 40}


def mcmc_schedule(method, device):
    """Official 3DGS-MCMC settings: init opacity 0.5 and scale x0.1; relocation/addition only at
    500 < s < 25000, s % 100 == 0, +5% up to the per-scene cap; noise every step; filter_3d
    stays 0 until the final step and is then the training-view filter; reg loss values."""
    _cuda(device)
    from splatstudio import ComponentSpec

    from .density import CAPS, create_density, resolve_cap
    from .losses import create_reg

    spec = next(c for c in method.components if c.kind == "density")
    n = 200
    g = torch.Generator().manual_seed(7)
    means = torch.randn(n, 3, generator=g) + torch.tensor([0, 0, 5.0])
    base = _cpu_params(means, torch.full((n, 3), 0.05), [[1.0, 0, 0, 0]] * n, [0.1] * n,
                       torch.full((n, 3), 0.5))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in
                                     base.items()})
    params["filter_3d"] = torch.nn.Parameter(torch.zeros(n, device=device))
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    opts["means"].param_groups[0]["lr"] = 1.6e-4
    for p in params.values():
        p.grad = torch.zeros_like(p)
    for o in opts.values():
        o.step()
    d = create_density(replace(spec, config={**spec.config, "cap_max": 230, "total_steps": 600}))
    d.initialize(params, opts, 1.0)
    view = torch.eye(4, device=device)
    d.set_training_cameras([{"viewmat": view, "width": 64, "height": 48,
                             "K": torch.tensor([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]],
                                               device=device)}])
    out = {"init_opacity": params["opacities"].sigmoid().mean().item(),
           "init_scale": params["scales"].exp().mean().item()}
    if abs(out["init_opacity"] - 0.5) > 1e-6 or abs(out["init_scale"] - 0.005) > 1e-7:
        raise AssertionError(f"MCMC init: {out}")
    with torch.no_grad():
        params["opacities"][:10] = torch.logit(torch.tensor(0.001))  # dead
    counts = {}
    for step in (400, 500, 550, 600):
        m0 = params["means"].detach().clone()
        d.after_backward(params, opts, step, {"info": {}})
        counts[step] = len(params["means"])
        moved = (params["means"][: len(m0)].detach() - m0).abs().max().item()
        if step in (400, 550) and moved == 0:
            raise AssertionError(f"no MCMC noise at step {step}")
        if step < 600 and params["filter_3d"].abs().max() != 0:
            raise AssertionError(f"filter_3d written before the final step ({step})")
    out["counts"] = counts
    if counts[400] != n or counts[500] != n or counts[550] != n or counts[600] != 210:
        raise AssertionError(f"MCMC add cadence/growth wrong: {counts}")
    if (params["opacities"].sigmoid() <= 0.005).any():
        raise AssertionError("dead Gaussians not relocated at step 600")
    f = params["filter_3d"].detach()
    out["filter_after_final"] = f.abs().max().item()
    if out["filter_after_final"] == 0:
        raise AssertionError("filter_3d not written at total_steps")
    d2 = create_density(replace(spec, config={**spec.config, "cap_max": 215}))
    d2.strategy.cap_max = 215
    d2.state = d2.strategy.initialize_state()
    d2.total_steps = 30000
    d2.after_backward(params, opts, 700, {"info": {}})
    if len(params["means"]) != 215:
        raise AssertionError(f"cap not respected: {len(params['means'])}")
    d2.after_backward(params, opts, 25000, {"info": {}})
    if len(params["means"]) != 215:
        raise AssertionError("refined at 25000 (official stops before densify_until_iter)")
    if resolve_cap(123) != 123 or CAPS["garden"] != 5834784:
        raise AssertionError("cap table")
    reg = create_reg(ComponentSpec("r", "loss", "x", (), (), {}))
    val = reg(params, {}, None).item()
    want = (0.01 * params["opacities"].sigmoid().mean()
            + 0.01 * params["scales"].exp().mean()).item()
    out["reg"] = val
    if abs(val - want) > 1e-7:
        raise AssertionError(f"reg loss {val} != {want}")
    return out
