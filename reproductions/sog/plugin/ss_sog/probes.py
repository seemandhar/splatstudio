"""Mechanism probes for the SOG plugin. Each raises if its mechanism is removed or wrong."""

import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from splatstudio.backends.gsplat import Renderer
from splatstudio.components import build
from splatstudio.state import check_state, initialize

from . import codec, sorting
from .density import DEFAULTS, LogSpaceStrategy, SOGDensity, schedule_events, split_world
from .losses import DEFAULTS as NB_DEFAULTS
from .losses import neighbor_terms
from .renderer import inverse_log_transform, log_transform


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _scene(method, device, n, seed=0):
    """n Gaussians on a smooth 3D surface in front of an identity camera (world space)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    uv = torch.rand(n, 2, generator=g)
    pts = torch.stack([uv[:, 0] - 0.5, uv[:, 1] - 0.5,
                       3.0 + 0.3 * torch.sin(4 * uv[:, 0]) * torch.cos(3 * uv[:, 1])], 1)
    cols = torch.stack([uv[:, 0], uv[:, 1], 0.5 + 0.0 * uv[:, 0]], 1) * 0.8 + 0.1
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.02))
        params["opacities"].copy_(torch.randn(n, generator=g).to(device))
        params["quats"].add_(0.2 * torch.randn(n, 4, generator=g).to(device))
    return params, opts


def _camera(device, size=32):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[40.0, 0, size / 2], [0, 40.0, size / 2], [0, 0, 1]],
                              device=device),
            "width": size, "height": size}


def _adam_moments(params, opts):
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)


def _quat_to_rotmat(q):
    """Independent unit quaternion (wxyz) -> rotation matrix."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


# --------------------------------------------------------------------------------- C3 log xyz
def log_xyz(method, device):
    _cuda(device)
    x = torch.linspace(-60, 60, 1001, device=device, dtype=torch.float64)
    torch.testing.assert_close(inverse_log_transform(log_transform(x)), x, rtol=1e-9, atol=1e-9)
    y = torch.tensor([-2.0, 0.0, 1.0], device=device)
    torch.testing.assert_close(inverse_log_transform(y), torch.tensor(
        [-(math.e ** 2 - 1), 0.0, math.e - 1], device=device))

    renderer, _, _ = build(method)
    params, _ = _scene(method, device, 300)
    world = {k: v.detach() for k, v in params.items()}
    stored = dict(world, means=log_transform(world["means"]))
    cam = _camera(device)
    ref = Renderer(replace(method.components[0], config={})).render(world, cam, 0)["rgb"]
    out = renderer.render(stored, cam, 0)["rgb"]
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    naive = Renderer(replace(method.components[0], config={})).render(stored, cam, 0)["rgb"]
    if (naive - ref).abs().max() < 1e-2:
        raise AssertionError("rendering log means without activation should differ")

    # split: children = world parent + R S n, then mapped back to log space
    params, opts = _scene(method, device, 50)
    with torch.no_grad():
        params["means"].copy_(log_transform(params["means"]))
    _adam_moments(params, opts)
    parent_world = inverse_log_transform(params["means"][:5].detach().clone())
    R = _quat_to_rotmat(params["quats"][:5].detach())
    s = params["scales"][:5].detach().exp()
    mask = torch.zeros(50, dtype=torch.bool, device=device)
    mask[:5] = True
    torch.manual_seed(123)
    noise = torch.randn(2, 5, 3, device=device)
    torch.manual_seed(123)
    split_world(params, opts, {}, mask)
    check_state(method, params, opts)
    expected = (parent_world[None] + torch.einsum("nij,nj,bnj->bni", R, s, noise)).reshape(-1, 3)
    got = inverse_log_transform(params["means"][-10:].detach())
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(params["scales"][-10:].exp(), (s / 1.6).repeat(2, 1))

    # initialize converts the SfM (world) init to log space
    density = SOGDensity(next(c for c in method.components if c.kind == "density"))
    params, opts = _scene(method, device, 17 * 17 + 3)
    world_init = params["means"].detach().clone()
    density.initialize(params, opts, 1.0)
    d = torch.cdist(inverse_log_transform(params["means"].detach()), world_init,
                    compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values
    if d.max() > 1e-4 or len(params["means"]) != 17 * 17:
        raise AssertionError("initialize did not store the SfM init as log-space means")
    return {"render_max_diff": float((out - ref).abs().max()),
            "naive_render_diff": float((naive - ref).abs().max()),
            "split_children_max_err": float((got - expected).abs().max()),
            "init_world_max_err": float(d.max())}


# ------------------------------------------------------------------------- C2 neighbor loss
def _independent_blur(img, k, sigma):
    """Separable Gaussian blur: x pass with circular padding, y pass with reflect padding."""
    r = k // 2
    xs = torch.arange(k, dtype=img.dtype, device=img.device) - r
    w = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    S = img.shape[-1]
    cols = torch.cat([img[..., -r:], img, img[..., :r]], -1)  # circular in x
    bx = sum(w[i] * cols[..., i:i + S] for i in range(k))
    H = img.shape[-2]
    rows = list(range(r, 0, -1)) + list(range(H)) + list(range(H - 2, H - 2 - r, -1))
    padded = bx[..., rows, :]  # reflect (edge not repeated) in y
    return sum(w[i] * padded[..., i:i + H, :] for i in range(k))


def _independent_huber(a, b):
    d = (a - b).abs()
    return torch.where(d < 1, 0.5 * d ** 2, d - 0.5).mean()


def _independent_neighbor(params, cfg):
    total = 0
    wsum = sum(cfg["weights"].values())
    for name, w in cfg["weights"].items():
        if w <= 0:
            continue
        t = params[name].reshape(len(params[name]), -1)
        side = math.isqrt(len(t))
        img = t.reshape(side, side, -1).permute(2, 0, 1)[None]
        target = _independent_blur(img.detach(), cfg["kernel_size"], cfg["sigma"])
        total = total + cfg["lambda_neighbor"] * w / wsum * _independent_huber(target, img)
    return total


def neighbor_loss(method, device):
    cfg = next(c for c in method.components if c.name == "sog-neighbor").config
    if cfg["weights"]["opacities"] / sum(cfg["weights"].values()) != 1 / 11:
        raise AssertionError("opacity weight is not 1/11 (supp. 0.09)")
    g = torch.Generator().manual_seed(0)
    n = 20 * 20
    params = {"opacities": torch.nn.Parameter(3 * torch.randn(n, generator=g)),
              "quats": torch.nn.Parameter(torch.randn(n, 4, generator=g))}
    _, _, losses = build(method)
    fn = dict(losses)["sog-neighbor"]
    value = fn(params, {}, None)
    grads = torch.autograd.grad(value, list(params.values()))
    ref = _independent_neighbor(params, cfg)
    ref_grads = torch.autograd.grad(ref, list(params.values()))
    torch.testing.assert_close(value, ref, rtol=1e-5, atol=1e-7)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-8)
    # the target must be detached: gradient through the blur branch would differ
    no_detach = 0
    for name, w in cfg["weights"].items():
        if w > 0:
            t = params[name].reshape(n, -1).reshape(20, 20, -1).permute(2, 0, 1)[None]
            no_detach = no_detach + w / 11 * _independent_huber(
                _independent_blur(t, cfg["kernel_size"], cfg["sigma"]), t)
    nd = torch.autograd.grad(no_detach, params["quats"])[0]
    if torch.allclose(nd, grads[1], atol=1e-6):
        raise AssertionError("detach of the blurred target has no effect")
    const = {"opacities": torch.full((n,), 2.0, requires_grad=True),
             "quats": torch.ones(n, 4, requires_grad=True)}
    if abs(float(sum(neighbor_terms(const, cfg).values()))) > 1e-7:
        raise AssertionError("constant grid must have zero smoothness loss")
    if NB_DEFAULTS["kernel_size"] != 5 or NB_DEFAULTS["sigma"] != 3.0:
        raise AssertionError("blur parameters differ from supp. Table")
    return {"loss": float(value), "independent": float(ref),
            "grad_quats_norm": float(grads[1].norm())}


# --------------------------------------------------------------------------------- C1 sorting
def grid_sort(method, device):
    _cuda(device)
    n = 40 * 40 + 37
    params, opts = _scene(method, device, n)
    with torch.no_grad():
        params["means"].copy_(log_transform(params["means"]))
    _adam_moments(params, opts)
    state = {"grad2d": torch.arange(n, device=device, dtype=torch.float32),
             "count": torch.ones(n, device=device), "scene_scale": 1.0}
    before = {k: v.detach().clone() for k, v in params.items()}
    alpha = params["opacities"].detach().sigmoid()
    side, removed = sorting.prune_to_square(params, opts, state)
    keep = torch.sort(torch.topk(alpha, side * side).indices).values
    if side != 40 or removed != 37:
        raise AssertionError("prune to square kept the wrong count")
    for k in params:
        torch.testing.assert_close(params[k].detach(), before[k][keep])
    torch.testing.assert_close(state["grad2d"], keep.float())
    check_state(method, params, opts)

    pre = {k: v.detach().clone() for k, v in params.items()}
    pre_m = {k: opts[k].state[p]["exp_avg"].clone() for k, p in params.items()}
    pre_g = state["grad2d"].clone()
    feats0 = sorting.sort_features(params, DEFAULTS["sort_weights"])
    torch.manual_seed(0)
    shuffled = feats0[torch.randperm(len(feats0), device=device)]
    d_random = sorting.neighbour_distance(shuffled, side)
    idx = sorting.sort_into_grid(params, opts, state, DEFAULTS)
    if not torch.equal(torch.sort(idx).values, torch.arange(side * side, device=device)):
        raise AssertionError("sort result is not a permutation")
    for k, p in params.items():
        torch.testing.assert_close(p.detach(), pre[k][idx])
        torch.testing.assert_close(opts[k].state[p]["exp_avg"], pre_m[k][idx])
    torch.testing.assert_close(state["grad2d"], pre_g[idx])
    check_state(method, params, opts)
    feats = sorting.sort_features(params, DEFAULTS["sort_weights"])
    d_sorted = sorting.neighbour_distance(feats, side)
    if not d_sorted < 0.6 * d_random:
        raise AssertionError(f"sorting did not organise the grid ({d_sorted} vs {d_random})")
    # key channels: standardized world xyz, DC and exp(scale) -> 9 channels
    if feats.shape[1] != 9:
        raise AssertionError("sort keys must be xyz + DC + scale (9 channels)")
    return {"grid_side": side, "pruned": removed, "neighbour_dist_shuffled": d_random,
            "neighbour_dist_sorted": d_sorted}


# ----------------------------------------------------------------------------------- C4 codec
def _smooth_params(side, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = side * side
    u = torch.linspace(0, 1, side)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    b = torch.stack([xx, yy, xx * yy], -1).reshape(n, 3)
    return {"means": log_transform(b * 6 - 3), "sh0": (b * 2 - 0.5)[:, None, :],
            "shN": 0.05 * torch.randn(n, 15, 3, generator=g),
            "scales": -4 + b, "quats": torch.cat([1 + 0 * b[:, :1], 0.3 * b], 1),
            "opacities": 6 * b[:, 0] - 2}


def _expected_lossless(a, bits, lo=None, hi=None):
    """Independent float64 reference of quantize -> (clip-)normalize -> denormalize."""
    a = a.double()
    mn, mx = a.min(), a.max()
    q = torch.round((a - mn) / (mx - mn) * 2 ** bits) / 2 ** bits * (mx - mn) + mn
    if lo is None:
        return q
    return (q.clamp(lo, hi) - lo) / (hi - lo) * (hi - lo) + lo


def codec_roundtrip(method, device):
    side = 32
    params = _smooth_params(side)
    dec, rep, files = codec.encode_decode(params)
    if rep["codec"]["imagecodecs"] != "2023.9.18":
        raise AssertionError("wrong imagecodecs version")
    expected = {
        "means": _expected_lossless(params["means"], 14),
        "shN": _expected_lossless(params["shN"], 5, -1, 1),
        "scales": _expected_lossless(params["scales"], 6),
        "quats": _expected_lossless(params["quats"], 6, -1, 2),
        "opacities": _expected_lossless(params["opacities"], 6, -6, 12),
    }
    errs = {}
    for k, e in expected.items():
        errs[k] = float((dec[k].double() - e).abs().max())
        if errs[k] > 1e-5 * max(1.0, float(e.abs().max())):
            raise AssertionError(f"{k}: decoded grid differs from the quantization formula")
    dc_err = (dec["sh0"] - params["sh0"].clamp(-2, 4)).abs()
    if dc_err.max() > 0.1 or dc_err.mean() > 0.02 or dc_err.max() == 0:
        raise AssertionError(f"DC lossy error out of range ({float(dc_err.max())})")
    names = {"_xyz.jxl", "_features_dc.jxl", "_features_rest.jxl", "_scaling.jxl",
             "_rotation.jxl", "_opacity.jxl", "compression_info.csv", "compression_config.yml"}
    if set(files) != names or rep["size_bytes"] != sum(len(b) for b in files.values()):
        raise AssertionError("size must be the sum of all compressed-directory files")
    perm = torch.randperm(side * side, generator=torch.Generator().manual_seed(1))
    _, rep_shuffled, _ = codec.encode_decode({k: v[perm] for k, v in params.items()})
    smooth_bytes = rep["size_bytes"] - rep["file_bytes"]["_features_rest.jxl"]
    shuffled_bytes = rep_shuffled["size_bytes"] - rep_shuffled["file_bytes"]["_features_rest.jxl"]
    if not smooth_bytes < 0.7 * shuffled_bytes:
        raise AssertionError("smooth grids should compress much better than shuffled ones")
    return {"size_bytes": rep["size_bytes"], "max_err": errs, "dc_max_err": float(dc_err.max()),
            "smooth_vs_shuffled_bytes": [smooth_bytes, shuffled_bytes], "codec": rep["codec"]}


# ------------------------------------------------------------------------------ C5 schedule
def schedule(method, device):
    spec = next(c for c in method.components if c.kind == "density")
    cfg = spec.config
    total = cfg["total_steps"]
    density = SOGDensity(spec)
    strat = density.base.strategy
    if not isinstance(strat, LogSpaceStrategy):
        raise AssertionError("density must use the log-space split strategy")
    checks = {"grow_grad2d": 7e-5, "prune_opa": 0.1, "grow_scale3d": 0.1, "refine_every": 1000,
              "refine_start_iter": 500, "refine_stop_iter": 15000}
    for k, v in checks.items():
        if getattr(strat, k) != v:
            raise AssertionError(f"{k} = {getattr(strat, k)} != {v}")
    if density.base.opacity_reset or strat.reset_every <= total:
        raise AssertionError("opacity reset / size pruning must be disabled")
    steps = range(1, total + 1)
    densify = [s for s in steps if s > strat.refine_start_iter and s % strat.refine_every == 0
               and s < strat.refine_stop_iter]
    sorts = [s for s in steps if "sort" in schedule_events(s, cfg)]
    encodes = [s for s in steps if "encode" in schedule_events(s, cfg)]
    # official train.py: densify 500 < it < 15000, it % 1000 == 0; sort it > 500, it % 1000 == 0
    off_densify = [it for it in steps if 500 < it < 15000 and it % 1000 == 0]
    off_sort = [it for it in steps if it > 500 and it % 1000 == 0 and it != total]
    if densify != off_densify or sorts != off_sort or encodes != [total]:
        raise AssertionError("event schedule differs from the official train.py")
    if not set(densify) - {total} <= set(sorts):  # the final step prunes to square in encode
        raise AssertionError("every densification step must re-grid")
    return {"densify_steps": [densify[0], densify[-1], len(densify)],
            "sort_steps": [sorts[0], sorts[-1], len(sorts)], "encode_step": encodes}


# ----------------------------------------------------------------------- C4 decoded model eval
def _read_info_csv(text):
    rows = {}
    for line in text.strip().splitlines()[1:]:
        name, lo, hi, f = line.split(",")
        rows[name] = (float(lo) if lo else None, float(hi) if hi else None, f)
    return rows


def decoded_eval(method, device):
    _cuda(device)
    spec = next(c for c in method.components if c.kind == "density")
    density = SOGDensity(replace(spec, config={**spec.config, "total_steps": 7}))
    renderer, _, _ = build(method)
    params, opts = _scene(method, device, 24 * 24 + 5)
    density.initialize(params, opts, 1.0)
    cam = _camera(device)
    out = renderer.render(params, cam, 3)
    density.before_backward(params, opts, 7, out)
    out["rgb"].mean().backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    pre = {k: v.detach().clone() for k, v in params.items()}
    pre_rgb = renderer.render(params, cam, 3)["rgb"].detach()
    density.after_backward(params, opts, 7, out)
    check_state(method, params, opts)
    st = density.state["sog"]
    files = {k: bytes(v.numpy()) for k, v in st["files"].items()}
    if st["encoded"]["size_bytes"] != sum(len(b) for b in files.values()):
        raise AssertionError("stored files do not sum to the reported size")
    # independently decode the stored files and compare with the replaced parameters
    with tempfile.TemporaryDirectory() as tmp:
        for k, b in files.items():
            (Path(tmp) / k).write_bytes(b)
        info = _read_info_csv(files["compression_info.csv"].decode())
        jobs = [{"file": str(Path(tmp) / f), "decoded": str(Path(tmp) / f"{a}.npy")}
                for a, (_, _, f) in info.items()]
        codec.run_worker(jobs, tmp, codec.resolve_codec_env(spec.config["codec_env"]))
        for attr, (lo, hi, _) in info.items():
            img = np.load(Path(tmp) / f"{attr}.npy").astype(np.float64)
            if lo is not None:
                img = img * (hi - lo) + lo
            field = codec.ATTR_TO_FIELD[attr]
            ref = torch.tensor(img.reshape(params[field].shape), dtype=torch.float32,
                               device=device)
            torch.testing.assert_close(params[field].detach(), ref, atol=1e-6, rtol=1e-6)
    if len(torch.unique(params["opacities"].detach())) > 65:
        raise AssertionError("opacities were not replaced by their 6-bit decoded values")
    if torch.equal(params["opacities"].detach(), pre["opacities"]):
        raise AssertionError("parameters unchanged by the encode step")
    fresh, _, _ = build(method)
    post_rgb = fresh.render({k: v.detach() for k, v in params.items()}, cam, 3)["rgb"]
    mse = float((post_rgb - pre_rgb).square().mean())
    psnr = -10 * math.log10(max(mse, 1e-12))
    if psnr < 25:
        raise AssertionError(f"decoded model renders far from the trained one ({psnr:.1f} dB)")
    return {"size_bytes": st["encoded"]["size_bytes"], "decoded_vs_trained_psnr": psnr,
            "files": sorted(files)}


def sog_renderer_gradient(method, device):
    _cuda(device)
    renderer, density, _ = build(method)
    params, opts = _scene(method, device, 20 * 20 + 7)
    density.initialize(params, opts, 1.0)
    cam = _camera(device, 16)
    outputs = renderer.render(params, cam, sh_degree=0)
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(16, device=device)[None, :, None] % 2).expand(16, 16, 3).float()
    (outputs["rgb"] - target).square().mean().backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    if params["means"].grad is None or not torch.count_nonzero(params["means"].grad):
        raise AssertionError("no gradient reaches the log-space means")
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    return {"screen_gradient_norm": float(xy.grad.norm()),
            "means_gradient_norm": float(params["means"].grad.norm())}
