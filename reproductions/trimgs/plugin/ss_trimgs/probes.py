"""Trim3DGS mechanism probes. Each one raises if its mechanism is removed or wrong.

Reference computations are written independently of the CUDA kernel and of the component
code paths they check (explicit per-pixel blending, analytic plane normals, literal replays).
"""

import math
from dataclasses import replace

import torch
from splatstudio.registry import resolve
from torch import nn

from . import core, kernels

EPS = 1e-8


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, kind, name=None, **overrides):
    spec = next(c for c in method.components if c.kind == kind and (name is None or c.name == name))
    if overrides:
        spec = replace(spec, config={**spec.config, **overrides})
    return resolve(spec.factory)(spec)


def _params(means, scales, opacities, quats=None, device="cuda"):
    n = len(means)
    if quats is None:
        quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    s = torch.as_tensor(scales, dtype=torch.float32).clone()
    values = {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "scales": s.log(),
        "quats": torch.as_tensor(quats, dtype=torch.float32),
        "opacities": torch.logit(torch.as_tensor(opacities, dtype=torch.float32)),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 15, 3),
    }
    params = nn.ParameterDict({k: nn.Parameter(v.to(device)) for k, v in values.items()})
    rates = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 0.05, "sh0": 2.5e-3,
             "shN": 1.25e-4}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    return params, opts


def _camera(width, height, f, device="cuda", c2w=None):
    c2w = torch.eye(4) if c2w is None else c2w
    return {
        "viewmat": torch.linalg.inv(c2w).to(device),
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device),
        "width": width,
        "height": height,
    }


def _facing_z_quat(n, tilt_deg=0.0):
    """wxyz quats whose first rotation column is +z (tilted about y by tilt_deg)."""
    a = math.radians(-90.0 + tilt_deg) / 2
    return torch.tensor([[math.cos(a), 0.0, math.sin(a), 0.0]]).repeat(n, 1)


# ---------------------------------------------------------------------------- C1 kernel
def _reference_contribution(meta, gamma):
    """Independent per-pixel loop of the official record_transmittance branch."""
    m2, con, op = meta["means2d"].cpu().double(), meta["conics"].cpu().double(), meta["opacities"]
    op = op.cpu().double()
    r = meta["radii"].cpu().double()
    W, H = meta["width"], meta["height"]
    n = len(m2)
    depth_order = meta["depth_order"]
    s = torch.zeros(n, dtype=torch.float64)
    cnt = torch.zeros(n, dtype=torch.int64)
    tiles = {}
    for g in range(n):
        if r[g] <= 0:
            continue
        x0 = max(int(math.floor((m2[g, 0] - r[g]) / 16)), 0)
        x1 = min(int(math.ceil((m2[g, 0] + r[g]) / 16)), (W + 15) // 16)
        y0 = max(int(math.floor((m2[g, 1] - r[g]) / 16)), 0)
        y1 = min(int(math.ceil((m2[g, 1] + r[g]) / 16)), (H + 15) // 16)
        for ty in range(y0, y1):
            for tx in range(x0, x1):
                tiles.setdefault((tx, ty), []).append(g)
    for (tx, ty), gs in tiles.items():
        gs = sorted(gs, key=lambda g: depth_order[g])
        for py in range(ty * 16, min(ty * 16 + 16, H)):
            for px in range(tx * 16, min(tx * 16 + 16, W)):
                T = 1.0
                for g in gs:
                    dx, dy = m2[g, 0] - (px + 0.5), m2[g, 1] - (py + 0.5)
                    a, b, c = con[g]
                    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
                    if power > 0:
                        continue
                    alpha = min(0.99, op[g] * math.exp(power))
                    s[g] += alpha ** (2 * gamma) * T ** (2 - 2 * gamma)
                    cnt[g] += 1
                    if alpha < 1 / 255:
                        continue
                    test_T = T * (1 - alpha)
                    if test_T < 1e-4:
                        break
                    T = test_T
    return s, cnt


def contribution_kernel(method, device="cuda"):
    _require_cuda()
    g = torch.Generator().manual_seed(3)
    n = 24
    means = torch.cat((torch.rand(n, 2, generator=g) * 2 - 1, 2 + 2 * torch.rand(n, 1, generator=g)), 1)
    scales = torch.rand(n, 3, generator=g) * 0.2 + 0.03
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.9 + 0.05
    params, _ = _params(means, scales, opac, quats, device)
    cam = _camera(48, 40, 30.0, device)
    out = {}
    for gamma in (0.5, 0.25):
        r = kernels.contribution(params, cam, EPS, gamma)
        meta = r["meta"]
        meta["depth_order"] = {i: float(means[i, 2]) for i in range(n)}
        ref_s, ref_c = _reference_contribution(meta, gamma)
        err = (r["sum"].cpu().double() - ref_s).abs().max().item()
        cnt_err = (r["covered"].cpu() - ref_c).abs().max().item()
        if err > 1e-3 * max(1.0, ref_s.max().item()) or cnt_err != 0:
            raise AssertionError(f"gamma {gamma}: sum err {err}, count err {cnt_err}")
        out[f"gamma{gamma}"] = {"max_sum_err": err, "total_covered": int(ref_c.sum())}
    # gamma 0.5 must equal the plain blend weight alpha*T, gamma 0.25 must differ
    r5 = kernels.contribution(params, cam, EPS, 0.5)["sum"]
    r25 = kernels.contribution(params, cam, EPS, 0.25)["sum"]
    if torch.allclose(r5, r25):
        raise AssertionError("gamma has no effect on the contribution")
    return out


# ---------------------------------------------------------------------------- C1 trimming
def trimming(method, device="cuda"):
    _require_cuda()
    # literal replay of the official insertion list
    seq = [1.0, 3.0, 2.0, 5.0, 4.0, 0.0, 6.0]
    top = [None] * 5
    for v in seq:
        top = core.official_topk_insert(top, torch.tensor([v]))
    got = [float(t) for t in top]
    if got != [6.0, 5.0, 3.0, 1.0, 1.0]:
        raise AssertionError(f"record insertion list {got} != official [6,5,3,1,1]")
    # scene: 9 visible Gaussians on a grid + 1 high-opacity Gaussian hidden behind the centre one
    xs = torch.linspace(-0.6, 0.6, 3)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    means = torch.cat((torch.cat((grid, torch.full((9, 1), 3.0)), 1), torch.tensor([[0.0, 0, 3.5]])))
    scales = torch.tensor([[EPS, 0.15, 0.15]] * 9 + [[EPS, 0.1, 0.1]])
    scales[4] = torch.tensor([EPS, 0.3, 0.3])
    opac = torch.tensor([0.6] * 9 + [0.99])
    opac[4] = 0.99  # opaque occluder in front of the hidden one
    params, opts = _params(means, scales, opac, _facing_z_quat(10), device)
    density = _component(method, "density")
    density.c["flat_eps"] = EPS
    cams = [_camera(64, 64, 60.0, device) for _ in range(3)]
    density.set_training_cameras(cams)
    contribution = density.contribution(params)
    if int(contribution.argmin()) != 9:
        raise AssertionError(f"occluded Gaussian not the lowest contribution: {contribution}")
    mask = core.trim_mask(contribution, 0.1)
    if mask.tolist() != [False] * 9 + [True]:
        raise AssertionError(f"trim mask {mask.tolist()}")
    # opacity-based pruning (3DGS default) would keep the hidden Gaussian (it is the most opaque)
    if bool(core.trim_mask(params["opacities"].sigmoid(), 0.1)[9]):
        raise AssertionError("fixture does not discriminate contribution from opacity")
    n = core.remove(params, opts, mask)
    return {"contribution": contribution.tolist(), "trimmed": n, "left": len(params["means"])}


# ---------------------------------------------------------------------------- C2 split
def scale_split(method, device="cuda"):
    _require_cuda()
    means = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [50.0, 0, 0]])
    scales = torch.tensor([[EPS, 0.5, 0.2], [EPS, 0.01, 0.01], [EPS, 0.5, 0.5]])
    params, opts = _params(means, scales, torch.tensor([0.5, 0.5, 0.5]), None, device)
    params["means"].grad = torch.ones_like(params["means"])
    for k, p in params.items():
        p.grad = torch.ones_like(p)
        opts[k].step()
    density = _component(method, "density", trim=False)
    density.c["flat_eps"] = EPS
    c2w = [torch.eye(4) for _ in range(2)]
    c2w[1][:3, 3] = torch.tensor([2.0, 2.0, 2.0])
    density.set_training_cameras([_camera(32, 32, 30.0, device, m) for m in c2w])
    density.extent = 10.0  # tau_s = 0.01 * 10 = 0.1 -> only Gaussians 0 and 2 are large
    density.max_radii = torch.zeros(3, device=device)
    kept_moment = opts["sh0"].state[params["sh0"]]["exp_avg"][1].clone()
    parent_scale = core.flat_scales(params["scales"].detach(), EPS)[0, 1:].clone()
    parent_mu = params["means"][0].detach().clone()
    parent_axis = core.gaussian_normals(params["quats"][:1].detach())[0]
    density._densify(params, opts, 1000)
    n = len(params["means"])
    if n != 5:
        raise AssertionError(f"expected 2 untouched + 3 children, got {n}")
    child_scales = params["scales"][2:].exp()
    want = parent_scale / 2.4
    if not torch.allclose(child_scales[:, 1:], want.expand(3, 2), rtol=1e-4):
        raise AssertionError(f"child scales {child_scales}")
    if not torch.allclose(params["scales"][:, 0], torch.full((5,), math.log(EPS), device=device)):
        raise AssertionError("flat axis not pinned after split")
    offsets = params["means"][2:] - parent_mu
    if float((offsets @ parent_axis).abs().max()) > 1e-5 or float(offsets.norm(dim=1).max()) < 1e-3:
        raise AssertionError("children not sampled in the flat Gaussian's plane")
    if float((params["means"][1] - torch.tensor([50.0, 0, 0], device=device)).abs().max()) > 1e-2:
        raise AssertionError("out-of-box Gaussian was split")
    st = opts["sh0"].state[params["sh0"]]
    if st["exp_avg"].shape[0] != 5 or st["exp_avg"][2:].abs().sum() != 0:
        raise AssertionError("children Adam moments not zero / misaligned")
    if not torch.equal(st["exp_avg"][0], kept_moment):
        raise AssertionError("kept Gaussian lost its Adam moments")
    return {"gaussians": n, "child_scales": child_scales.tolist(), "event": density.events[-1]}


# ---------------------------------------------------------------------------- C3 normals
def _plane_scene(device, slope=0.0, tilt=0.0, side=1.6, n=40):
    xs = torch.linspace(-side, side, n)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    z = 3.0 + slope * grid[:, 0:1]
    means = torch.cat((grid, z), 1)
    step = 2 * side / (n - 1)
    scales = torch.tensor([[EPS, 1.2 * step, 1.2 * step]]).repeat(len(means), 1)
    a = math.atan(slope)  # analytic plane tilt about y
    quats = _facing_z_quat(len(means), math.degrees(-a) + tilt)
    return _params(means, scales, torch.full((len(means),), 0.95), quats, device)


def normal_loss(method, device="cuda"):
    _require_cuda()
    loss = _component(method, "loss", name="trim-normal")
    cam = _camera(64, 64, 50.0, device)
    outputs = {"camera": cam, "flat_eps": EPS, "step": loss.c["start_step"] + 1}
    res = {}
    for slope in (0.0, 0.4):
        params, _ = _plane_scene(device, slope=slope)
        ln, ls, dn = loss.terms(params, outputs)
        want = torch.nn.functional.normalize(torch.tensor([slope, 0.0, -1.0], device=device), dim=0)
        centre = dn[24:40, 24:40].reshape(-1, 3).mean(0)
        if (centre - want).abs().max() > 0.02:
            raise AssertionError(f"slope {slope}: depth normal {centre} != analytic {want}")
        res[f"slope{slope}"] = {"normal": float(ln), "smooth": float(ls)}
    params, _ = _plane_scene(device, slope=0.0, tilt=35.0)
    ln_t, ls_t, _ = loss.terms(params, outputs)
    if not float(ln_t) > 5 * res["slope0.0"]["normal"] + 1e-3:
        raise AssertionError(f"tilted normals not penalized: {float(ln_t)} vs {res}")
    total = loss(params, outputs, None)
    total.backward()
    g = params["quats"].grad
    if g is None or not torch.isfinite(g).all() or g.abs().sum() == 0:
        raise AssertionError("no finite gradient to the Gaussian rotations")
    off = loss(params, {**outputs, "step": loss.c["start_step"]}, None)
    if float(off) != 0.0:
        raise AssertionError("normal loss active before the fine-tune")
    res["tilted35"] = {"normal": float(ln_t), "smooth": float(ls_t)}
    return res


# ---------------------------------------------------------------------------- C0 schedule
def flat_schedule(method, device="cpu"):
    density = _component(method, "density")
    renderer_spec = next(c for c in method.components if c.kind == "renderer")
    eps = renderer_spec.config["flat_eps"]
    params, opts = _params(torch.rand(6, 3), torch.full((6, 3), 0.05), torch.full((6,), 0.5),
                           None, "cpu")
    opts["means"].param_groups[0]["lr"] = 1.6e-4 * 4.0
    density.initialize(params, opts, 4.0)
    if not torch.allclose(params["scales"][:, 0], torch.full((6,), math.log(eps))):
        raise AssertionError("flat axis not pinned at initialize")
    a = core.flat_scales(params["scales"], eps)
    b = core.flat_scales(params["scales"] + torch.tensor([3.0, 0, 0]), eps)
    if not torch.equal(a, b) or not torch.all(a[:, 0] == eps):
        raise AssertionError("rendered scale depends on the pinned column")
    P = density.c["pretrain_steps"]
    lr0 = density.lr0
    checks = {1: lr0 * 0.01 ** (1 / P), P // 2: lr0 * 0.1, P: lr0 * 0.01, P + 3000: lr0 * 0.01}
    for s, want in checks.items():
        got = core.pretrain_means_lr(lr0, s, P)
        if abs(got - want) > 1e-6 * want:
            raise AssertionError(f"means lr at {s}: {got} != {want}")
    for k, p in params.items():
        p.grad = torch.ones_like(p)
        opts[k].step()
    density._start_finetune(params, opts)
    if any(len(o.state) for o in opts.values()):
        raise AssertionError("Adam state not reset at the fine-tune start")
    info = {"radii": torch.ones(1, 6, 2, dtype=torch.int32),
            "conics": torch.tensor([[1.0, 0.0, 1.0]]).repeat(1, 6, 1)}
    density.after_backward(params, opts, P + 5, {"info": info})
    if abs(opts["means"].param_groups[0]["lr"] - lr0 * 0.01) > 1e-12:
        raise AssertionError("fine-tune means LR not the constant final LR")
    # 3DGS: lambda = mid + sqrt(max(0.1, mid^2 - det)) = 1 + sqrt(0.1) -> ceil(3 * 1.147) = 4
    if not torch.all(density.max_radii == 4.0):
        raise AssertionError(f"official radius tracking wrong: {density.max_radii}")
    return {"lr0": lr0, "finetune_lr": opts["means"].param_groups[0]["lr"]}
