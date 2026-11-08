"""Mechanism probes for LPM. Each raises when its mechanism is removed or altered."""

import math
from types import SimpleNamespace

import numpy as np
import torch

from . import core
from .density import DEFAULTS, LPMDensity


def _density(**cfg):
    return LPMDensity(SimpleNamespace(config={**cfg}))


def _camera(center, width=128, height=96, f=100.0, device="cpu"):
    """OpenCV camera at `center` looking along +z (identity rotation)."""
    vm = torch.eye(4, device=device)
    vm[:3, 3] = -torch.as_tensor(center, dtype=torch.float32, device=device)
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], device=device)
    return {"viewmat": vm, "K": K, "width": width, "height": height}


def _project(cam, X):
    x = (X - core.camera_center(cam)) @ cam["K"].T
    return x[:, :2] / x[:, 2:]


def _params(means, log_scales, logits, device="cpu"):
    n = len(means)
    p = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means.clone().to(device)),
        "scales": torch.nn.Parameter(log_scales.clone().to(device)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(logits.clone().to(device)),
        "sh0": torch.nn.Parameter(torch.full((n, 1, 3), 0.5, device=device)),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })
    opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
    return p, opts


# ------------------------------------------------------------------ C1
def error_map_probe(method, device):
    g = torch.Generator().manual_seed(0)
    h, w = 70, 90  # not multiples of 16: exercises the padding
    gt = 0.4 + 0.05 * torch.rand(h, w, 3, generator=g)
    img = gt + 0.01 * torch.rand(h, w, 3, generator=g)
    img[16:48, 32:64] += 0.4  # two by two patches of strong error
    got = core.error_map(img, gt).numpy()
    # independent numpy reference of the official get_errormap
    a, b = img.numpy(), gt.numpy()
    e = np.abs(a / (a.mean() + 0.01) - b / (b.mean() + 0.01)).sum(-1)
    m = e > np.quantile(e, 0.4)
    ref = np.zeros((h, w), np.float32)
    for py in range(0, h, 16):
        for px in range(0, w, 16):
            blk = np.zeros((16, 16), bool)
            sub = m[py:py + 16, px:px + 16]
            blk[: sub.shape[0], : sub.shape[1]] = sub
            if blk.sum() > 0.85 * 256:
                ref[py:py + 16, px:px + 16] = 1
    if not np.array_equal(got, ref):
        raise AssertionError("error map differs from the independent reference")
    if got[16:48, 32:64].min() != 1 or got.sum() != 32 * 32:
        raise AssertionError(f"error map should mark exactly the 32x32 error block, got {got.sum()} px")
    regions = core.error_regions(torch.from_numpy(got))
    if regions != [(32, 16, 64, 48)]:
        raise AssertionError(f"connected-component box wrong: {regions}")
    return {"marked_pixels": float(got.sum()), "regions": regions}


# ------------------------------------------------------------------ C3
def zone_projection_probe(method, device):
    # closest points vs independent least squares on random line pairs
    g = torch.Generator().manual_seed(1)
    ra, rb = torch.randn(20, 6, generator=g), torch.randn(20, 6, generator=g)
    pts = core.closest_points(ra, rb)
    worst = 0.0
    for k in range(20):
        oa, da, ob, db = ra[k, :3].numpy(), ra[k, 3:].numpy(), rb[k, :3].numpy(), rb[k, 3:].numpy()
        t = np.linalg.lstsq(np.stack([da, -db], 1), ob - oa, rcond=None)[0]
        pa, pb = oa + t[0] * da, ob + t[1] * db
        worst = max(worst, float(np.abs(pts[k].numpy() - pa).max()), float(np.abs(pts[20 + k].numpy() - pb).max()))
    if worst > 1e-3:
        raise AssertionError(f"closest points differ from least squares by {worst}")
    # a 3D rectangle at z=5 seen from two cameras: the zone must enclose it tightly
    ca, cb = _camera([-0.5, 0, 0]), _camera([0.5, 0, 0])
    rect = torch.tensor([[x, y, 5.0] for x in (-0.6, 0.6) for y in (-0.4, 0.4)])
    def box(cam):
        uv = _project(cam, rect)
        return (int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max()))
    zones = core.zones3d(ca, cb, [box(ca)], [box(cb)])
    if len(zones) != 1:
        raise AssertionError("expected one zone")
    z = zones[0]
    # rays through matching border pixels meet on the plane: zone = rectangle up to pixel rounding
    lo, hi = torch.tensor([-0.6, -0.4, 5.0]), torch.tensor([0.6, 0.4, 5.0])
    if (z[:3] - lo).abs().max() > 0.06 or (z[3:] - hi).abs().max() > 0.06:
        raise AssertionError(f"zone {z.tolist()} is not the rectangle {lo.tolist()}..{hi.tolist()}")
    return {"closest_point_err": worst, "zone": [round(v, 3) for v in z.tolist()]}


# ------------------------------------------------------------------ C4
def _densify_case(ratio):
    tau = DEFAULTS["grad_threshold"]
    means = torch.tensor([
        [0.0, 0, 0],    # A: zone, small, 0.6 tau  -> LPM clone (extra)
        [5.0, 0, 0],    # B: outside, small, 0.6 tau -> nothing
        [0.1, 0, 0],    # C: zone, small, 1.2 tau  -> 3DGS clone
        [-0.1, 0, 0],   # D: zone, large, 0.6 tau  -> LPM split (extra)
        [6.0, 0, 0],    # E: low opacity
        [7.0, 0, 0],    # F: low opacity
        [8.0, 0, 0],    # G
    ])
    log_s = torch.full((7, 3), math.log(0.005))
    log_s[3] = math.log(0.05)
    logits = torch.tensor([0.0, 0, 0, 0, -4.0, -3.5, 0])
    params, opts = _params(means, log_s, logits)
    d = _density(zone_grad_ratio=ratio)
    d.initialize(params, opts, scene_scale=1.0)
    d.state["grad2d"] = torch.tensor([0.6, 0.6, 1.2, 0.6, 0, 0, 0]) * tau
    d.state["count"] = torch.ones(7)
    zone = torch.tensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
    d.densify_and_prune(params, opts, step=1000, zones=[zone])
    return d, params


def localized_densify_probe(method, device):
    d, p = _densify_case(0.5)
    n_lpm, extra = len(p["means"]), d.state["lpm_extra"]
    # +1 (A clone) +1 (C clone) +1 (D split: -1 +2) -2 (E,F: the n_extra lowest logits)
    if extra != 2 or n_lpm != 8:
        raise AssertionError(f"LPM densify: n_extra={extra}, N={n_lpm} (expected 2, 8)")
    if (p["opacities"] < -3).any():
        raise AssertionError("the n_extra lowest-opacity Gaussians were not pruned")
    if not ((p["means"] - torch.tensor([5.0, 0, 0])).norm(dim=1) < 1e-6).sum() == 1:
        raise AssertionError("Gaussian outside the zone must not be densified")
    for k, v in p.items():
        if not v.requires_grad or not v.is_leaf:
            raise AssertionError(f"field {k} frozen after densify")
    d0, p0 = _densify_case(1.0)  # no localized lowering == 3DGS densify
    if d0.state["lpm_extra"] != 0 or len(p0["means"]) != 8:
        raise AssertionError(f"ratio 1.0 should be plain 3DGS (C clone only): N={len(p0['means'])}")
    return {"n_lpm": n_lpm, "n_extra": extra, "n_plain": len(p0["means"])}


# ------------------------------------------------------------------ C5
def localized_reset_probe(method, device):
    means = torch.tensor([
        [0.0, 0, 2],     # P1 in front, high opacity -> reset
        [0.0, 0, 8],     # P2 behind
        [3.0, 0, 2],     # P3 outside the cone
        [0.05, 0, 3],    # P4 in front, opacity below threshold
        [0.0, 0, 5], [0.1, 0, 5], [0, 0.1, 5], [0, 0, 5.1],  # inside the zone
    ])
    logits = torch.tensor([3.0, 3.0, 3.0, 0.5, 2.0, 1.0, 0.0, -1.0])
    params, opts = _params(means, torch.full((8, 3), -3.0), logits)
    d = _density()
    d.initialize(params, opts, scene_scale=1.0)
    d.set_training_cameras([_camera([0, 0, 0])])
    zone = torch.tensor([-0.5, -0.5, 4.5, 0.5, 0.5, 5.5])
    d.localized_reset(params, 0, [zone])
    out = params["opacities"].detach()
    want = logits.clone()
    want[0] = math.log(0.01 / 0.99)
    if not torch.allclose(out, want, atol=1e-5):
        raise AssertionError(f"reset result {out.tolist()} != expected {want.tolist()}")
    if not params["opacities"].requires_grad:
        raise AssertionError("opacities frozen by localized reset")
    return {"reset": d.state["lpm_reset"], "opacities": [round(v, 3) for v in out.tolist()]}


# ------------------------------------------------------------------ C2 (CUDA, real network)
def matching_probe(method, device):
    from splatstudio.verification import ProbeUnavailable

    from .matching import Matcher, weights_available

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    if not weights_available():
        raise ProbeUnavailable("LightGlue source or SuperPoint/LightGlue weights missing")
    g = torch.Generator().manual_seed(0)
    big = torch.nn.functional.interpolate(torch.rand(1, 3, 40, 52, generator=g), size=(400, 520),
                                          mode="bilinear", align_corners=False)[0]
    big = (big + 0.5 * (torch.rand(1, 3, 400, 520, generator=g)[0] > 0.97).float()).clamp(0, 1)
    sx, sy = 12, 7
    a = big[:, 20:340, 20:460].permute(1, 2, 0)
    b = big[:, 20 + sy:340 + sy, 20 + sx:460 + sx].permute(1, 2, 0)  # content moves by (-sx, -sy)
    m = Matcher(device)
    p0, p1 = m.match(m.features(a), m.features(b))
    if len(p0) < 50:
        raise AssertionError(f"only {len(p0)} matches")
    disp = (p0 - p1).median(dim=0).values
    if (disp - torch.tensor([sx, sy], dtype=torch.float32)).abs().max() > 1.0:
        raise AssertionError(f"median displacement {disp.tolist()} != ({sx},{sy})")
    emap = torch.zeros(320, 440)
    emap[96:192, 160:288] = 1
    rc, rr = core.paired_regions(emap, p0, p1)
    if len(rc) != 1:
        raise AssertionError("expected one paired region")
    shift = np.subtract(rc[0], rr[0])
    if np.abs(shift - [sx, sy, sx, sy]).max() > 3:
        raise AssertionError(f"paired regions {rc[0]} -> {rr[0]} not shifted by ({sx},{sy})")
    return {"matches": len(p0), "median_disp": disp.tolist(), "regions": [rc[0], rr[0]]}


# ------------------------------------------------------------------ integration (CUDA)
def density_cycle_probe(method, device):
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    dev = "cuda"
    xs = torch.linspace(-1.2, 1.2, 40)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), -1).reshape(-1, 2)
    means = torch.cat([grid, torch.full((len(grid), 1), 5.0)], 1)
    g = torch.Generator().manual_seed(0)
    # distinct logits (official prunes every logit <= the n_extra-th smallest, ties included)
    logits = 2.0 + 0.1 * torch.rand(len(means), generator=g)
    params, opts = _params(means, torch.full((len(means), 3), math.log(0.03)), logits, dev)
    with torch.no_grad():
        params["sh0"].copy_(torch.rand(len(means), 1, 3, generator=g).to(dev))
    cams = [_camera([-0.3, 0, 0], device=dev), _camera([0.3, 0, 0], device=dev)]
    renderer = Renderer(SimpleNamespace(config={}))
    result = {}
    for ratio in (0.5, 1.0):
        p = torch.nn.ParameterDict({k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()})
        o = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
        d = _density(zone_grad_ratio=ratio)
        d.initialize(p, o, scene_scale=1.0)
        d.set_training_cameras(cams)
        with torch.no_grad():
            target = renderer.render(p, cams[0])["rgb"].clone()
            target[32:64, 48:80] = 1 - target[32:64, 48:80]
            d.images = [(target.clamp(0, 1) * 255).round().to(torch.uint8).cpu()] * 2
        # matches: projections of the plane's grid into both views (true correspondences)
        X = means.to(dev)
        u0, u1 = _project(cams[0], X), _project(cams[1], X)
        ok = ((u0 >= 0) & (u0 < torch.tensor([128, 96], device=dev))).all(1) & \
             ((u1 >= 0) & (u1 < torch.tensor([128, 96], device=dev))).all(1)
        d.pairs[0] = [(u0[ok], u1[ok], 1)]
        d.neighbours = [[1], [0]]
        d.pairs[1] = [(u1[ok], u0[ok], 0)]
        out = renderer.render(p, cams[0])
        out["target_image"], out["viewmat"], out["step"] = target, cams[0]["viewmat"], 600
        d.before_backward(p, o, 600, out)
        (out["rgb"] - target).abs().mean().backward()
        n0 = len(p["means"])
        d.after_backward(p, o, 600, out)
        for k, v in p.items():
            if not (v.is_leaf and v.requires_grad) or len(v) != len(p["means"]):
                raise AssertionError(f"field {k} broken after LPM step")
            if o[k].param_groups[0]["params"][0] is not v:
                raise AssertionError(f"optimizer for {k} references a stale parameter")
        result[ratio] = {"zones": d.state["lpm_zones"], "extra": d.state["lpm_extra"],
                         "reset": d.state["lpm_reset"], "n_before": n0, "n_after": len(p["means"])}
    if result[0.5]["zones"] < 1:
        raise AssertionError(f"no error zone found for a clear error block: {result}")
    if result[0.5]["extra"] <= result[1.0]["extra"]:
        raise AssertionError(f"localized addition added nothing beyond 3DGS: {result}")
    return {str(k): v for k, v in result.items()}
