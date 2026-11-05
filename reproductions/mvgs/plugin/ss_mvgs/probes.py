"""Mechanism probes for the MVGS port. Each raises AssertionError when its mechanism is removed or
wrong. Reference computations are written independently of geometry.py/density.py where possible.
"""

import math
from dataclasses import replace

import numpy as np
import torch
from splatstudio.state import check_state, sample_scene

from . import geometry as G
from .density import INDOOR_STAGES, MVGSDensity
from .losses import create_photometric, ssim_same
from .renderer import Renderer, render_view


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return device if str(device).startswith("cuda") else "cuda"


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _parts(method, **density_config):
    dspec = _spec(method, "density")
    density = MVGSDensity(replace(dspec, config={**dspec.config, **density_config}))
    renderer = Renderer(_spec(method, "renderer"))
    loss = create_photometric(_spec(method, "loss"))
    return renderer, density, loss


def look_at(center, target, device, width, height, focal, cx=None, cy=None):
    """OpenCV world-to-camera (x right, y down = world +y, z forward)."""
    c = torch.tensor(center, dtype=torch.float32)
    z = torch.tensor(target, dtype=torch.float32) - c
    z = z / z.norm()
    x = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = -R @ c
    K = torch.tensor([[focal, 0, width / 2 if cx is None else cx],
                      [0, focal, height / 2 if cy is None else cy], [0, 0, 1]])
    return {"viewmat": viewmat.to(device), "K": K.to(device), "width": width, "height": height}


def ring(n, device, width=32, height=24, focal=30.0, spread=40.0, radius=3.0):
    """n cameras on an arc around (0, 0, 3), looking at it (sample_scene's Gaussians)."""
    cams = []
    for a in np.linspace(-spread, spread, n):
        r = math.radians(a)
        cams.append(look_at((radius * math.sin(r), 0.0, 3.0 - radius * math.cos(r)), (0, 0, 3),
                            device, width, height, focal))
    return cams


def _targets(cams, device, seed=0):
    """Random 8-bit images (the density stores training images as uint8)."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.rand(c["height"], c["width"], 3, generator=g) * 255).round().div(255).to(device)
            for c in cams]


# ---------------------------------------------------------------------------------------------
def multiview_sum(method, device):
    device = _cuda(device)
    renderer, density, loss = _parts(method, stages=[[4, -1, 10]], full_width_ratio=1.0)
    params, opts = sample_scene(method, device)
    cams = ring(6, device)
    density.initialize(params, opts)
    density.set_training_frames(cams, _targets(cams, device))
    out = renderer.render(params, cams[0], sh_degree=0)
    idx = [v["index"] for v in out["mv_views"]]
    if len(idx) != 4 or len(set(idx)) != 4:
        raise AssertionError(f"batch is not 4 distinct views: {idx}")
    value = loss(params, out, torch.zeros(24, 32, 3, device=device))
    keys = ("means", "sh0", "opacities")
    grads = torch.autograd.grad(value, [params[k] for k in keys])

    targets = _targets(cams, device)
    singles, single_grads = [], []
    for i in idx:
        o = render_view(params, cams[i], 0)["rgb"]
        t = targets[i]
        li = 0.8 * (o - t).abs().mean() + 0.2 * (1 - ssim_same(o, t))
        singles.append(li)
        single_grads.append(torch.autograd.grad(li, [params[k] for k in keys]))
    total = sum(singles)
    if not torch.allclose(value, total, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"multi-view loss {value.item()} != sum of views {total.item()}")
    mean = total / len(singles)
    if torch.allclose(value, mean, rtol=1e-3):
        raise AssertionError("multi-view loss equals the mean, not the sum")
    for k, g, *gs in zip(keys, grads, *single_grads):
        if not torch.allclose(g, sum(gs), rtol=1e-4, atol=1e-7):
            raise AssertionError(f"{k}: gradient is not the sum of per-view gradients")
    return {"views": idx, "loss": value.item(), "per_view": [s.item() for s in singles],
            "gradient_norm": float(grads[0].norm())}


# ---------------------------------------------------------------------------------------------
def cig_schedule(method, device):
    st = G.check_stages(INDOOR_STAGES)
    expect = {1: (0, 1), 3000: (0, 3000), 3001: (1, 1), 6000: (1, 3000), 9000: (2, 3000),
              9001: (3, 1), 39000: (3, 30000)}
    for step, ref in expect.items():
        if G.stage_at(st, step) != ref:
            raise AssertionError(f"stage_at({step}) = {G.stage_at(st, step)}, expected {ref}")
    try:
        G.stage_at(st, 39001)
        raise AssertionError("step beyond the schedule must raise")
    except RuntimeError:
        pass
    views = [s[0] for s in st]
    if views != [48, 24, 12, 8]:
        raise AssertionError(f"views per stage {views}")
    # Official loader: round(W_full / r) for the counter scene (3115 x 2077 -> w1600 = 1600 x 1066).
    sizes = [G.stage_size(1600, 1066, r, 3115 / 1600) for _, r, _ in st]
    official_w = [round(3115 / 8), round(3115 / 4), round(3115 / 2), 1600]
    if [s[0] for s in sizes] != official_w:
        raise AssertionError(f"stage widths {sizes} != official {official_w}")
    if any(abs(s[1] - h) > 1 for s, h in zip(sizes, [round(2077 / 8), round(2077 / 4),
                                                    round(2077 / 2), 1066])):
        raise AssertionError(f"stage heights {sizes} differ from official by > 1 px")
    d = MVGSDensity(_spec(method, "density"))
    checks = {"densify_500": d.is_densify(500), "densify_600": d.is_densify(600),
              "densify_14900": d.is_densify(14900), "densify_15000": d.is_densify(15000),
              "reset_500": d.is_reset(500), "reset_3000": d.is_reset(3000),
              "reset_15000": d.is_reset(15000), "reset_2900": d.is_reset(2900)}
    want = {"densify_500": False, "densify_600": True, "densify_14900": True,
            "densify_15000": False, "reset_500": True, "reset_3000": True, "reset_15000": False,
            "reset_2900": False}
    if checks != want:
        raise AssertionError(f"stage-local schedule {checks} != {want}")

    device = _cuda(device)
    renderer, density, loss = _parts(method, stages=[[2, 2, 3], [2, -1, 3]], full_width_ratio=1.0)
    params, opts = sample_scene(method, device)
    cams = ring(4, device, width=64, height=48, focal=60.0)
    density.initialize(params, opts)
    density.set_training_frames(cams, _targets(cams, device))
    out = renderer.render(params, cams[0], sh_degree=0)
    for v in out["mv_views"]:
        if tuple(v["rgb"].shape[:2]) != (24, 32) or tuple(v["target"].shape[:2]) != (24, 32):
            raise AssertionError("training render/target not at the stage resolution")
        cam = cams[v["index"]]
        K = cam["K"].clone()
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = K[0, 0] / 2, K[1, 1] / 2, K[0, 2] / 2, K[1, 2] / 2
        ref = render_view(params, {**cam, "K": K, "width": 32, "height": 24}, 0)["rgb"]
        if not torch.allclose(ref, v["rgb"], atol=1e-6):
            raise AssertionError("stage render differs from gsplat with hand-scaled intrinsics")
    with torch.no_grad():
        full = renderer.render(params, cams[0], sh_degree=0)["rgb"]
    if tuple(full.shape[:2]) != (48, 64):
        raise AssertionError("evaluation render is not full resolution")
    return {"stage_sizes_counter": sizes, "schedule": checks}


# ---------------------------------------------------------------------------------------------
def mvad(method, device):
    beta = 2e-4

    def centers(angles, radius=5.0):
        return torch.tensor([[radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)),
                              0.3 * radius * math.cos(math.radians(a))] for a in angles])

    def reference(c):
        t = c / c.norm(dim=-1, keepdim=True)
        ang = torch.arccos((t @ t.T).clamp(-1, 1))
        off = ~torch.eye(len(c), dtype=torch.bool)
        return beta / 2 if bool((ang[off] > math.pi / 3 + 1e-9).any()) else beta

    rows = {}
    for angles in ([0, 50], [0, 70], [0, 30, 55], [0, 30, 65], [10, -45, 5, 12], [0]):
        c = centers(angles)
        got, ref = G.mvad_threshold(c, beta, 1.0), reference(c)
        if got != ref:
            raise AssertionError(f"MVAD threshold for {angles}: {got} != {ref}")
        rows[str(angles)] = got
    if rows["[0, 70]"] != beta / 2 or rows["[0, 50]"] != beta:
        raise AssertionError("MVAD does not halve the threshold for distinct views")
    return {"thresholds": rows}


# ---------------------------------------------------------------------------------------------
def _official_window(l1map_chw, h_box=20, w_box=40):
    """Literal port of train.py:152-171."""
    c, h, w = l1map_chw.shape
    loss_max, max_id = 0, (0, 0)
    for i in range(0, h - h_box, h_box):
        for j in range(0, w - w_box, w_box):
            r = torch.mean(l1map_chw[:, i:i + h_box, j:j + w_box])
            if r > loss_max:
                loss_max, max_id = r, [i, j]
    i, j = max_id
    return float(loss_max), [i, j, i + h_box, j + w_box]


def _closest_midpoint_np(o1, d1, o2, d2):
    """Least-squares closest points of two lines (normal equations), numpy float64."""
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
    b = np.array([(o2 - o1) @ d1, (o2 - o1) @ d2])
    t1, t2 = np.linalg.solve(A, b)
    return 0.5 * ((o1 + t1 * d1) + (o2 + t2 * d2))


def _unproject_np(cam, row, col):
    K = cam["K"].cpu().double().numpy()
    c2w = np.linalg.inv(cam["viewmat"].cpu().double().numpy())
    d = c2w[:3, :3] @ (np.linalg.inv(K) @ np.array([col + 0.5, row + 0.5, 1.0]))
    return c2w[:3, 3], d / np.linalg.norm(d)


def crd(method, device):
    g = torch.Generator().manual_seed(3)
    for h, w in ((100, 200), (107, 173), (60, 128)):
        m = torch.rand(3, h, w, generator=g)
        ref = _official_window(m)
        got = G.max_loss_window(m.permute(1, 2, 0))
        if got[1] != ref[1] or abs(got[0] - ref[0]) > 1e-5:
            raise AssertionError(f"window {got} != official {ref} for {h}x{w}")

    cam = look_at((1.0, -0.5, -2.0), (0.2, 0.1, 3.0), "cpu", 128, 60, 50.0, 61.3, 29.7)
    o, d = G.pixel_rays(cam, [3, 40], [7, 100])
    for k, (r, c) in enumerate(((3, 7), (40, 100))):
        o_ref, d_ref = _unproject_np(cam, r, c)
        if not np.allclose(d[k].double().numpy(), d_ref, atol=1e-5) or not np.allclose(
                o[k].double().numpy(), o_ref, atol=1e-5):
            raise AssertionError("pixel ray differs from independent unprojection")

    q = torch.tensor([0.3, -0.2, 4.0])
    o1, o2 = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-2.0, 1.0, 0.5])
    p, valid = G.intersect_lines(o1, q - o1, o2, q - o2)
    if not bool(valid) or not torch.allclose(p, q, atol=1e-5):
        raise AssertionError("intersection of rays through a common point is not that point")
    _, valid = G.intersect_lines(o1, q - o1, o1 + 1, q - o1)
    if bool(valid):
        raise AssertionError("parallel rays must be rejected")

    device = _cuda(device)
    W, H = 128, 60
    # Asymmetric pair (a symmetric one gives a flat cuboid with no strict interior).
    cams = [look_at((r * math.sin(math.radians(a)), 0.1 * r, 3 - r * math.cos(math.radians(a))),
                    (0, 0, 3), device, W, H, 50.0, 55.5, 27.5) for a, r in ((-25.0, 3.0),
                                                                           (35.0, 4.0))]
    box = [20, 40, 40, 80]
    views = []
    for cam in cams:
        target = torch.zeros(H, W, 3, device=device)
        target[20:40, 40:80] = 1.0
        views.append({"rgb": torch.zeros(H, W, 3, device=device), "target": target,
                      "camera": cam})
    # Independent cuboid (paper/official indexing coincide here: both views share the box).
    corners = [(box[0], box[1]), (box[2], box[3]), (box[2], box[1]), (box[0], box[3])]
    mids = []
    for r, c in corners:
        oa, da = _unproject_np(cams[0], r, c)
        ob, db = _unproject_np(cams[1], r, c)
        mids.append(_closest_midpoint_np(oa, da, ob, db))
    mids = np.array(mids)
    lo, hi = mids.min(0), mids.max(0)
    if not (hi - lo > 1e-3).all():
        raise AssertionError("degenerate probe cuboid")
    center = torch.tensor((lo + hi) / 2, dtype=torch.float32)

    def run(**cfg):
        _, density, _ = _parts(method, **cfg)
        params, opts = sample_scene(method, device, n=3)
        with torch.no_grad():
            params["means"][0] = center.to(device)
            params["means"][1] = torch.tensor([100.0, 100.0, 100.0], device=device)
            params["means"][2] = torch.tensor([-50.0, 3.0, 40.0], device=device)
            params["scales"][:] = 0.0  # scale 1 = extent -> "large", never small-cloned
        for p in params.values():
            p.grad = torch.ones_like(p)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        density.initialize(params, opts, scene_scale=1.0)
        density.densify(params, opts, 600, {"mv_views": views})
        check_state(method, params, opts)
        return density, params, opts

    density, params, opts = run()
    hist = density.history[-1]
    if len(params["means"]) != 4 or hist["inbox"] != 1 or hist["cloned"] != 1:
        raise AssertionError(f"cross-ray clone wrong: {hist}")
    if not torch.equal(params["means"][3], params["means"][0]) or float(
            (params["means"][0] - center.to(device)).norm()) > 1e-2:
        raise AssertionError("the clone is not the in-cuboid Gaussian")
    m = opts["means"].state[params["means"]]["exp_avg"]
    if m[3].abs().sum() != 0 or m[0].abs().sum() == 0:
        raise AssertionError("clone must have zero Adam moments and the original keeps its own")
    if hist["threshold"] != 1e-4:
        raise AssertionError("MVAD should halve the threshold for these distinct views")
    off, params_off, _ = run(crd=False)
    if len(params_off["means"]) != 3:
        raise AssertionError("Gaussians cloned with cross-ray densification disabled")
    nomvad, _, _ = run(mvad=False)
    if nomvad.history[-1]["threshold"] != 2e-4:
        raise AssertionError("threshold changed with MVAD disabled")
    return {"cuboid": [lo.tolist(), hi.tolist()], "history": hist,
            "crd_off_gaussians": len(params_off["means"])}


# ---------------------------------------------------------------------------------------------
def cycle(method, device):
    device = _cuda(device)
    cfg = dict(stages=[[3, 2, 4], [2, -1, 6]], full_width_ratio=1.0, densify_from=1,
               densify_every=2, densify_until=100, reset_every=1000, white_reset_at=-1,
               grow_grad=1e-9)
    renderer, density, loss = _parts(method, **cfg)
    params, opts = sample_scene(method, device)
    cams = ring(6, device)
    density.initialize(params, opts, scene_scale=1.0)
    lr0 = density.lr0
    density.set_training_frames(cams, _targets(cams, device))
    sizes, lrs = [], {}
    for step in range(1, 11):
        out = renderer.render(params, cams[0], sh_degree=0)
        out["step"] = step
        density.before_backward(params, opts, step, out)
        value = loss(params, out, torch.zeros(24, 32, 3, device=device))
        value.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opts["means"].param_groups[0]["lr"] = 123.0  # the runtime's own schedule; must be replaced
        sizes.append(tuple(out["rgb"].shape[:2]))
        density.after_backward(params, opts, step, out)
        lrs[step] = opts["means"].param_groups[0]["lr"]
        if step == 1:
            radii = out["mv_views"][-1]["info"]["radii"][0]
            vis = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
            if not torch.equal(density.count > 0, vis) or float(density.count.max()) != 1.0:
                raise AssertionError("statistics must count only the last view's visible Gaussians")
        check_state(method, params, opts)
    if sizes[:4] != [(12, 16)] * 4 or sizes[4:] != [(24, 32)] * 6:
        raise AssertionError(f"stage sizes {sizes}")
    if not math.isclose(lrs[4], lr0 * 0.01 ** (1 / 30000), rel_tol=1e-9):
        raise AssertionError("means LR does not restart at the stage boundary")
    if not math.isclose(lrs[3], lr0 * 0.01 ** (4 / 30000), rel_tol=1e-9):
        raise AssertionError("means LR does not follow the stage-local 3DGS schedule")
    if not density.history:
        raise AssertionError("no densification ran")
    return {"sizes": sizes, "history": density.history[-3:], "gaussians": len(params["means"]),
            "lr": {k: lrs[k] for k in (3, 4, 10)}}

