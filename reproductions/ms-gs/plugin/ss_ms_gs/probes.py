"""Mechanism probes. Each returns measurements and raises AssertionError when the mechanism is
missing or wrong. CPU probes use independent re-derivations; `selective_render` runs the real
density/renderer path on CUDA."""

import math
import random
from dataclasses import replace

import torch

from .aggregate import contract, voxel_aggregate
from .coverage import pixel_coverage, select, update_ranges
from .losses import create_photometric, resize_target, ssim_same
from .schedule import MultiScaleSampler, position_lr


def _scene(n, seed=0, device="cpu", scale=(-4.0, -2.0)):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * 2 - 1
    means[:, 2] += 4
    params = {
        "means": means,
        "quats": torch.randn(n, 4, generator=g),
        "scales": torch.rand(n, 3, generator=g) * (scale[1] - scale[0]) + scale[0],
        "opacities": torch.randn(n, generator=g) * 2,
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g),
        "ms_level": torch.zeros(n),
        "ms_max_px": -torch.ones(n),
        "ms_min_px": -torch.ones(n),
    }
    return {k: v.to(device) for k, v in params.items()}


def _camera(device="cpu", f=300.0, w=200, h=150):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device),
            "width": w, "height": h}


def _reference_coverage(params, cam):
    """Independent float64 re-derivation (explicit quaternion matrix, scalar loop)."""
    out = []
    K, V = cam["K"].double(), cam["viewmat"].double()
    W, H = cam["width"], cam["height"]
    fx, fy = K[0, 0].item(), K[1, 1].item()
    for i in range(len(params["means"])):
        w, x, y, z = (params["quats"][i].double() / params["quats"][i].double().norm()).tolist()
        Rq = torch.tensor([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=torch.float64)
        s = params["scales"][i].double().exp()
        cov3 = Rq @ torch.diag(s * s) @ Rq.T
        t = V[:3, :3] @ params["means"][i].double() + V[:3, 3]
        tz = t[2].item()
        if tz <= 0.2:
            out.append(0.0)
            continue
        lx, ly = 1.3 * W / (2 * fx), 1.3 * H / (2 * fy)
        tx = min(max(t[0].item() / tz, -lx), lx) * tz
        ty = min(max(t[1].item() / tz, -ly), ly) * tz
        J = torch.tensor([[fx / tz, 0, -fx * tx / tz**2], [0, fy / tz, -fy * ty / tz**2]],
                         dtype=torch.float64)
        c2 = J @ V[:3, :3] @ cov3 @ V[:3, :3].T @ J.T
        Q = torch.linalg.inv(c2)
        o = torch.sigmoid(params["opacities"][i].double()).item()
        ls = max(0.0, 2 * math.log(255 * o))
        out.append(min(math.sqrt(ls / Q[0, 0].item()), math.sqrt(ls / Q[1, 1].item())))
    return torch.tensor(out)


def pixel_coverage_probe(method, device):
    params = _scene(300, seed=1)
    cam = _camera()
    ours = pixel_coverage(params, cam).double()
    ref = _reference_coverage(params, cam)
    err = (ours - ref).abs() / ref.clamp_min(1e-3)
    if err.max() > 1e-3:
        raise AssertionError(f"pixel coverage differs from float64 reference: {err.max():.2e}")
    # Analytic: isotropic sigma_s, fronto-parallel on the axis -> S = f sigma_s / z sqrt(2 ln 255o)
    f, z, sig, o = 300.0, 5.0, 0.004, 0.8
    one = {"means": torch.tensor([[0.0, 0.0, z]]), "quats": torch.tensor([[1.0, 0, 0, 0]]),
           "scales": torch.full((1, 3), math.log(sig)), "opacities": torch.logit(torch.tensor([o]))}
    s = pixel_coverage(one, _camera(f=f)).item()
    expect = f * sig / z * math.sqrt(2 * math.log(255 * o))
    dilated = math.sqrt(2 * math.log(255 * o)) * math.sqrt((f * sig / z) ** 2 + 0.3)
    if abs(s - expect) > 1e-4 * expect:
        raise AssertionError(f"analytic coverage {s} != {expect}")
    if abs(dilated - expect) < 0.1 * expect:
        raise AssertionError("analytic case does not discriminate the 0.3 px^2 dilation")
    # behind the 3DGS near plane (z <= 0.2) and fully transparent -> 0
    one["means"][0, 2] = 0.1
    if pixel_coverage(one, _camera()).item() != 0:
        raise AssertionError("Gaussian in front of z=0.2 must have coverage 0")
    return {"max_rel_err_vs_float64": err.max().item(), "analytic": s, "analytic_expected": expect,
            "with_dilation_would_be": dilated}


def selection_rule_probe(method, device):
    #            px   max  min  -> keep  (rule: cull if min-rel<0.5 & px<2, or max-rel>2)
    cases = [(1.0, -1, -1, True),  # uninitialised never culled
             (0.9, -1, 2.0, False),  # 0.45 < 0.5 and < 2 px
             (1.1, -1, 2.0, True),  # 0.55
             (3.0, -1, 8.0, True),  # rel 0.375 but >= S_T = 2 px kept
             (1.9, -1, 8.0, False),
             (4.1, 2.0, -1, False),  # rel_max 2.05 > 2
             (3.9, 2.0, -1, True),  # 1.95 (would be culled with the paper's 1.5)
             (0.0, 1.0, 1.0, False)]
    px, mx, mn, want = (torch.tensor([c[i] for c in cases], dtype=torch.float32) for i in range(4))
    keep = select(px, mx, mn)
    if not torch.equal(keep, want.bool()):
        raise AssertionError(f"selection rule wrong: {keep.tolist()} vs {want.bool().tolist()}")
    only_small = select(px, mx, mn, filter_large=True, filter_small=False)
    if not only_small.all():
        raise AssertionError("filter_large must be nested inside filter_small (forward.cu)")
    return {"cases": len(cases), "keep": keep.tolist()}


def range_update_probe(method, device):
    level = torch.tensor([0.0, 0, 2, 2, 2, 6])
    mx = torch.tensor([-1.0, -1, -1, 3.0, 3.0, 5.0])
    mn = torch.tensor([-1.0, 4.0, -1, 2.0, 2.0, -1])
    vis = torch.tensor([True, True, True, True, False, True])
    px = torch.tensor([1.5, 5.0, 1.0, 1.0, 9.0, 0.5])
    a_mx, a_mn = mx.clone(), mn.clone()
    update_ranges(a_mx, a_mn, level, vis, px, 0, 7)  # 1x: only min of level 0
    want_mn0 = torch.tensor([1.5, min(4 * 1.05, 5.0), -1, 2.0, 2.0, -1])
    if not (torch.allclose(a_mn, want_mn0) and torch.equal(a_mx, mx)):
        raise AssertionError(f"1x update wrong: {a_mx.tolist()} {a_mn.tolist()}")
    b_mx, b_mn = mx.clone(), mn.clone()
    update_ranges(b_mx, b_mn, level, vis, px, 2, 7)
    want_mx = torch.tensor([-1.0, -1, 1.0, max(3 * 0.95, 1.0), 3.0, 5.0])
    want_mn = torch.tensor([-1.0, 4.0, 1.0, min(2 * 1.05, 1.0), 2.0, -1])
    if not (torch.allclose(b_mx, want_mx) and torch.allclose(b_mn, want_mn)):
        raise AssertionError(f"level-2 update wrong: {b_mx.tolist()} {b_mn.tolist()}")
    c_mx, c_mn = mx.clone(), mn.clone()
    update_ranges(c_mx, c_mn, level, vis, px, 6, 7)  # coarsest index: no min update
    if not (c_mn[5] == -1 and abs(c_mx[5] - 4.75) < 1e-6):
        raise AssertionError("coarsest-level exemption / max decay wrong")
    return {"max_after_level2": b_mx.tolist(), "min_after_level2": b_mn.tolist()}


def aggregation_probe(method, device):
    g = torch.Generator().manual_seed(3)
    n = 400
    extent = 2.0
    raw = {"means": (torch.rand(n, 3, generator=g) * 6 - 3),
           "sh0": torch.randn(n, 1, 3, generator=g), "shN": torch.randn(n, 15, 3, generator=g),
           "opacities": torch.randn(n, generator=g), "quats": torch.randn(n, 4, generator=g),
           "scales": torch.randn(n, 3, generator=g) - 4}
    raw["means"][:50] = torch.tensor([0.101, 0.2, -0.3]) * extent  # many in one voxel
    min_px = torch.rand(n, generator=g) * 1.2
    level = 4
    new = voxel_aggregate(raw, min_px, extent, level)
    size = 0.02 * level / 4
    groups = {}
    for i in range(n):
        p = []
        for v in (raw["means"][i] / extent).tolist():
            p.append(2 - 1 / v if v > 1 else v)
        key = tuple(math.floor(v / size) for v in p)
        groups.setdefault(key, []).append(i)
    if len(groups) != len(new["means"]):
        raise AssertionError(f"voxel count {len(new['means'])} != reference {len(groups)}")
    by_mean = {tuple(round(v, 5) for v in new["means"][j].tolist()): j
               for j in range(len(new["means"]))}
    worst = 0.0
    for idx in groups.values():
        ii = torch.tensor(idx)
        key = tuple(round(v, 5) for v in raw["means"][ii].mean(0).tolist())
        j = by_mean[key]
        fac = 2.0 / min(max(min_px[ii].mean().item(), 0.25), 2.0)
        exp_scale = raw["scales"][ii].mean(0) + math.log(fac)
        for name, ref in (("scales", exp_scale), ("opacities", raw["opacities"][ii].mean(0)),
                          ("quats", raw["quats"][ii].mean(0)), ("sh0", raw["sh0"][ii].mean(0))):
            worst = max(worst, (new[name][j] - ref).abs().max().item())
    if worst > 1e-5:
        raise AssertionError(f"pooled attributes differ from reference by {worst}")
    if not (torch.all(new["ms_level"] == level) and torch.all(new["ms_max_px"] == -1)):
        raise AssertionError("new Gaussians must carry the new level and an uninitialised range")
    c = contract(torch.tensor([[4.0, -4.0, 0.5]]), 2.0)
    if not torch.allclose(c, torch.tensor([[1.5, -2.0, 0.25]])):
        raise AssertionError("contraction must follow the official one-sided 2 - 1/x")
    return {"chosen": n, "voxels": len(new["means"]), "max_abs_err": worst,
            "largest_group": max(len(v) for v in groups.values())}


def ms_schedule_probe(method, device):
    s = MultiScaleSampler(max_scale=7, seed=5)
    s.n_train = 10
    levels = [s.next_level(t) for t in range(1, 40001)]
    if any(v != 0 for v in levels[:1010]):
        raise AssertionError("only 1x before the first insertion")
    if 1 in levels:
        raise AssertionError("2x (index 1) is never trained with --insert_large")
    first = {lv: levels.index(lv) + 1 for lv in range(2, 8) if lv in levels}
    if not (min(first[2], first[3]) > 1010 and min(first[4], first[5]) > 1020
            and min(first[6], first[7]) > 1030):
        raise AssertionError(f"levels enabled too early: {first}")
    blocks = [levels[i:i + 10] for i in range(0, 40000, 10)]
    if any(len(set(b)) != 1 for b in blocks):
        raise AssertionError("resolution must stay fixed for an epoch of n_train views")
    p1 = levels.count(0) / len(levels)
    if abs(p1 - 0.75) > 0.03:
        raise AssertionError(f"P(1x)={p1}")
    counts = [levels[1100:].count(lv) for lv in range(2, 8)]
    if max(counts) - min(counts) > 10:
        raise AssertionError(f"least-trained choice must balance levels: {counts}")
    loss = create_photometric(replace(method.components[2], config={}))
    rgb = torch.full((16, 16, 3), 0.3)
    gt = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(1))
    full = loss({}, {"rgb": rgb, "ms_level": 0}, gt).item()
    low = loss({}, {"rgb": rgb[:8, :8], "ms_level": 1}, gt).item()
    gt8 = resize_target(gt, 8, 8)
    ref = 0.8 * (rgb[:8, :8] - gt8).abs().mean() + 0.2 * (1 - ssim_same(rgb[:8, :8], gt8))
    ref_full = 0.8 * (rgb - gt).abs().mean() + 0.2 * (1 - ssim_same(rgb, gt))
    if abs(low - 0.1 * ref.item()) > 1e-6 or abs(full - ref_full.item()) > 1e-6:
        raise AssertionError("non-1x loss must be weighted by 0.1 against a resized target")
    lr = [position_lr(1.0, t) for t in (0, 15000, 30000, 40000)]
    if not (abs(lr[1] - 0.1) < 1e-9 and lr[2] == lr[3] == 0.01):
        raise AssertionError(f"position LR schedule wrong: {lr}")
    return {"p_full_res": p1, "first_step_per_level": first, "level_counts_after_1100": counts,
            "lr": lr}


def densify_fields_probe(method, device):
    from splatstudio.components import build
    from splatstudio.state import sample_scene

    random.seed(0)
    torch.manual_seed(0)
    params, opts = sample_scene(method, "cpu", n=12)
    _, density, _ = build(method)
    density.initialize(params, opts, scene_scale=1.0)
    with torch.no_grad():
        params["ms_level"][6:] = 2
        params["ms_max_px"][:] = 3.2
        params["ms_min_px"][:] = 1.6
        params["scales"][:3] = math.log(0.001)  # clone
        params["scales"][3:] = math.log(0.05)  # split
        params["opacities"][[2, 8]] = torch.logit(torch.tensor(0.001))  # prune candidates
    density.state["stats"]["grad2d"][:] = 1.0
    density.state["stats"]["count"][:] = 1.0
    out = density.densify_and_prune(params, opts, step=600)
    lv = params["ms_level"]
    # level-0: 3 cloned (+3), 3 split (+3 net), Gaussian 2 (and its clone) pruned; level 2 kept
    if out["clone"] != 3 or out["split"] != 3:
        raise AssertionError(f"only level-0 Gaussians may densify: {out}")
    if int((lv == 2).sum()) != 6:
        raise AssertionError("level-2 Gaussians must be neither densified nor pruned")
    children = (params["ms_max_px"] - 2.0).abs() < 1e-6
    if int(children.sum()) != 6 or not torch.allclose(params["ms_min_px"][children],
                                                      torch.tensor(1.0)):
        raise AssertionError("split children must carry S range / 1.6")
    with torch.no_grad():
        params["opacities"][:] = 3.0
    density.reset_opacity(params, opts)
    op = params["opacities"].sigmoid()
    if not (torch.all(op[lv == 0] <= 0.0100001) and torch.all(op[lv == 2] > 0.9)):
        raise AssertionError("opacity reset must touch only level-0 Gaussians")
    return {**out, "gaussians": len(lv)}


def selective_render(method, device):
    from splatstudio.components import build

    from .renderer import scaled_camera

    if not torch.cuda.is_available():
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    g = torch.Generator().manual_seed(0)
    n = 4000
    params = _scene(n, seed=2, device=device, scale=(-6.0, -5.0))
    params["means"] = torch.cat([torch.rand(n, 2, generator=g) * 1.6 - 0.8,
                                 torch.full((n, 1), 4.0)], 1).to(device)
    params["opacities"] = torch.full((n,), 2.0, device=device)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in params.items()})
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    cams = [_camera(device, f=400.0, w=256, h=256) for _ in range(3)]
    density.initialize(params, opts, scene_scale=1.0)
    density.set_training_cameras(cams)
    with torch.no_grad():
        for cam in cams:  # S_min of level 0 from 1x renders
            px, vis = density.coverage_and_visibility(params, cam)
            update_ranges(params["ms_max_px"].data, params["ms_min_px"].data,
                          params["ms_level"].data, vis, px, 0, density.n_levels)
    info = density.insert_large(params, opts, 2)
    if info["inserted"] == 0:
        raise AssertionError("aggregation must insert coarse Gaussians for tiny ones")
    level = params["ms_level"]
    with torch.no_grad():
        _, keep1 = renderer.selection(params, cams[0])
        _, keep4 = renderer.selection(params, scaled_camera(cams[0], 2))
        out1 = renderer.render(params, cams[0])
        out4 = renderer.render(params, scaled_camera(cams[0], 2))
    coarse = level == 2
    stats = {
        "inserted": info["inserted"], "selected_small": info["selected"],
        "coarse_kept_at_1x": keep1[coarse].float().mean().item(),
        "coarse_kept_at_4x": keep4[coarse].float().mean().item(),
        "fine_kept_at_1x": keep1[~coarse].float().mean().item(),
        "fine_kept_at_4x": keep4[~coarse].float().mean().item(),
        "rgb_4x_shape": list(out4["rgb"].shape),
        "rendered_1x": int((out1["info"]["radii"][0] > 0).all(-1).sum()),
        "rendered_4x": int((out4["info"]["radii"][0] > 0).all(-1).sum()),
    }
    if not (stats["coarse_kept_at_1x"] < 0.05 and stats["coarse_kept_at_4x"] > 0.95):
        raise AssertionError(f"coarse Gaussians must be culled at 1x and drawn at 4x: {stats}")
    if not (stats["fine_kept_at_1x"] > 0.95 and stats["fine_kept_at_4x"] < 0.5):
        raise AssertionError(f"tiny fine Gaussians must be culled at 4x: {stats}")
    return stats


def cuda_kernel_probe(method, device):
    """Fused CUDA kernel == torch reference (coverage + mask) on a random scene with ranges."""
    if not torch.cuda.is_available():
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("CUDA is unavailable")
    from .renderer import DEFAULTS, coverage_and_mask

    params = _scene(20000, seed=4, scale=(-6.0, -1.0))
    params["means"][:100, 2] = 0.1  # behind the near plane
    g = torch.Generator().manual_seed(5)
    ref_px = pixel_coverage(params, _camera())
    params["ms_min_px"] = torch.where(torch.rand(20000, generator=g) < 0.7,
                                      ref_px * (torch.rand(20000, generator=g) * 4), -torch.ones(20000))
    params["ms_max_px"] = torch.where(torch.rand(20000, generator=g) < 0.7,
                                      ref_px * (torch.rand(20000, generator=g) * 1.5), -torch.ones(20000))
    cpu_px, cpu_keep = coverage_and_mask(params, _camera(), DEFAULTS)
    cuda_params = {k: v.to(device) for k, v in params.items()}
    gpu_px, gpu_keep = coverage_and_mask(cuda_params, _camera(device), DEFAULTS)
    rel = ((gpu_px.cpu() - cpu_px).abs() / cpu_px.clamp_min(1e-3)).max().item()
    flips = (gpu_keep.cpu() != cpu_keep).sum().item()
    if rel > 1e-3 or flips > 5:
        raise AssertionError(f"CUDA kernel differs from torch reference: rel {rel}, flips {flips}")
    return {"max_rel_err": rel, "mask_flips": flips, "culled_fraction": 1 - cpu_keep.float().mean().item()}
