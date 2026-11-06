"""Mechanism probes for the Laplacian-GS port. Each raises when its mechanism is removed."""

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from . import pyramid as P
from .renderer import LEVEL_FIELD, render_field

C0 = 0.28209479177387814


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


# ---------------------------------------------------------------- independent references
def np_reflect_conv(img, stride):
    k1 = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    k = np.outer(k1, k1) / 256.0
    pad = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    h, w = img.shape[:2]
    out_h, out_w = (h - 1) // stride + 1, (w - 1) // stride + 1
    out = np.zeros((out_h, out_w, img.shape[2]))
    for y in range(out_h):
        for x in range(out_w):
            win = pad[y * stride:y * stride + 5, x * stride:x * stride + 5]
            out[y, x] = (win * k[:, :, None]).sum(axis=(0, 1))
    return out


def np_bilinear(img, size):
    h, w = img.shape[:2]
    oh, ow = size
    out = np.zeros((oh, ow, img.shape[2]))
    for y in range(oh):
        sy = max((y + 0.5) * h / oh - 0.5, 0.0)
        y0 = min(int(math.floor(sy)), h - 1)
        y1, fy = min(y0 + 1, h - 1), sy - y0
        for x in range(ow):
            sx = max((x + 0.5) * w / ow - 0.5, 0.0)
            x0 = min(int(math.floor(sx)), w - 1)
            x1, fx = min(x0 + 1, w - 1), sx - x0
            out[y, x] = ((1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x1])
                         + fy * ((1 - fx) * img[y1, x0] + fx * img[y1, x1]))
    return out


def lap_pyramid(method, device):
    g = np.random.default_rng(0)
    img = g.random((13, 18, 3))
    t = torch.tensor(img, dtype=torch.float64)
    d = P.down(t).numpy()
    ref_d = np_reflect_conv(img, 2)
    if d.shape != (7, 9, 3) or not np.allclose(d, ref_d, atol=1e-10):
        raise AssertionError("Down differs from the reflect-pad 5x5 binomial stride-2 reference")
    u = P.up(torch.tensor(ref_d), (13, 18), blur_up=True).numpy()
    ref_u = np_reflect_conv(np_bilinear(ref_d, (13, 18)), 1)
    if not np.allclose(u, ref_u, atol=1e-10):
        raise AssertionError("Up differs from bilinear + binomial blur reference")
    renderer_cfg = _component(method, "renderer").config
    if not renderer_cfg.get("blur_up", True):
        raise AssertionError("blur_up disabled; official scripts pass --blur_up")
    sizes = P.level_sizes(1066, 1600, 3)
    if sizes != [(1066, 1600), (533, 800), (267, 400)]:
        raise AssertionError(f"pyramid sizes {sizes}")
    recon = P.up(torch.tensor(ref_d), (13, 18)) + (t - P.up(torch.tensor(ref_d), (13, 18)))
    err = float((recon - t).abs().max())
    return {"down_max_err": float(np.abs(d - ref_d).max()),
            "up_max_err": float(np.abs(u - ref_u).max()), "laplacian_recon_err": err,
            "sizes_1600": sizes}


def _camera(device, w=32, h=24, f=30.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": w, "height": h}


def lap_signed_residual(method, device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    cam = _camera(device)
    color = -0.3  # c(d) for degree 0 = C0 * sh0
    g = {"means": torch.tensor([[0.0, 0.0, 3.0]], device=device),
         "quats": torch.tensor([[1.0, 0, 0, 0]], device=device),
         "scales": torch.full((1, 3), math.log(0.3), device=device),
         "opacities": torch.tensor([2.0], device=device),
         "sh0": torch.full((1, 1, 3), color / C0, device=device),
         "shN": torch.zeros(1, 15, 3, device=device)}
    res = render_field(g, cam, 0, signed=True, background=0.0)
    base = render_field(g, cam, 0, signed=False, background=0.0)
    alpha = base["alpha"]
    # base colour = c + 0.5 (= 0.2 > 0): base = 0.2*A, residual = -0.3*A  =>  R = B - 0.5*A
    err = float((res["rgb"] - (base["rgb"] - 0.5 * alpha)).abs().max())
    if err > 1e-5 or float(res["rgb"].min()) > -0.1:
        raise AssertionError(f"residual is not the signed (unshifted) colour: err {err}")
    g["sh0"] = torch.full((1, 1, 3), -3.0 / C0, device=device)
    clamped = render_field(g, cam, 0, signed=True, background=0.0)["rgb"]
    if float(clamped.min()) < -1.0 - 1e-6 or float(clamped.min()) > -0.99:
        raise AssertionError("residual render must be clamped to [-1, 1] and stay signed")
    return {"signed_err": err, "residual_min": float(res["rgb"].min()),
            "peak_alpha": float(alpha.max()), "clamped_min": float(clamped.min())}


def _two_level_scene(method, device):
    from splatstudio.state import sample_scene

    params, opts = sample_scene(method, device, n=24)
    with torch.no_grad():
        params[LEVEL_FIELD][12:] = 0.0
        params["sh0"][12:] = torch.randn_like(params["sh0"][12:]) * 0.3
        params["shN"].copy_(torch.randn_like(params["shN"]) * 0.05)
        params["scales"].fill_(math.log(0.15))
        params["opacities"].fill_(1.0)
    return params, opts


def lap_composition(method, device):
    from gsplat import rasterization, spherical_harmonics
    from splatstudio.components import build
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    if renderer.levels != 2:
        raise ProbeUnavailable("composition probe is written for the paper's 2 levels")
    params, opts = _two_level_scene(method, device)
    cam = _camera(device)
    with torch.no_grad():
        out = renderer.render(params, cam, 3)["rgb"]
    # independent reconstruction from raw gsplat calls
    lv = params[LEVEL_FIELD].detach()
    b, r = lv == 1, lv == 0
    K1 = cam["K"].clone()
    K1[0] *= 16 / 32
    K1[1] *= 12 / 24
    d1, d0 = renderer.degrees[1], renderer.degrees[0]

    def args(mask):
        return dict(means=params["means"][mask], quats=params["quats"][mask],
                    scales=params["scales"][mask].exp(),
                    opacities=params["opacities"][mask].sigmoid(),
                    viewmats=cam["viewmat"][None])

    with torch.no_grad():
        coeffs = torch.cat((params["sh0"], params["shN"]), 1)
        rb, ab, _ = rasterization(colors=coeffs[b], sh_degree=d1, Ks=K1[None], width=16,
                                  height=12, **args(b))
        base = (rb[0] + (1 - ab[0])).clamp(0, 1)
        cols = spherical_harmonics(d0, params["means"][r], coeffs[r][:, :(d0 + 1) ** 2])
        rr, _, _ = rasterization(colors=cols, Ks=cam["K"][None], width=32, height=24, **args(r))
        up = F.interpolate(base.permute(2, 0, 1)[None], size=(24, 32), mode="bilinear",
                           align_corners=False)[0].permute(1, 2, 0)
        ref = P.blur(up) + rr[0].clamp(-1, 1)
    err = float((out - ref).abs().max())
    if err > 1e-5:
        raise AssertionError(f"evaluation composition differs from Eq. 7 reference: {err}")
    # training render at the residual stage: archived base frozen, residual trained
    density.stage, density.level, density.stage_start, density.local_step = 1, 0, 100, 0
    o = renderer.render(params, cam, 3)
    o["rgb"].square().sum().backward()
    gm = params["means"].grad
    frozen = float(gm[b].abs().max())
    active = float(gm[r].abs().max())
    if frozen != 0.0 or active == 0.0:
        raise AssertionError(f"archived grad {frozen}, active grad {active}")
    with torch.no_grad():
        cache = P.up(renderer.compose(params, cam, 1), (24, 32))
        cache_err = float((cache - P.blur(up)).abs().max())
    if cache_err > 1e-5:
        raise AssertionError(f"training cache is not Up(archived base): {cache_err}")
    if o["lap_level"] != 0 or o["sh_degree"] != 0:
        raise AssertionError("training render must use the active level and stage-local SH")
    return {"eval_err": err, "cache_err": cache_err, "archived_grad": frozen, "active_grad": active,
            "train_level": o["lap_level"], "train_sh_degree": o["sh_degree"]}


def _small_density(method, plan, until):
    from .density import create_density

    spec = _component(method, "density")
    spec = replace(spec, config={**spec.config, "iter_plan": plan, "densify_until": until})
    return create_density(spec)


def lap_staging(method, device):
    from splatstudio.state import check_state, sample_scene

    levels = _component(method, "density").config.get("levels", 2)
    plan = [3] * levels
    density = _small_density(method, plan, [0] * levels)
    params, opts = sample_scene(method, device, n=12)
    density.initialize(params, opts, 1.0)
    for name, lr in density.c["lr"].items():
        if abs(opts[name].param_groups[0]["lr"] - lr) > 1e-12:
            raise AssertionError(f"{name} LR not set to the official value")
    lr0 = density.lr_init
    top = levels - 1

    def train_step(step):
        active = params[LEVEL_FIELD].detach() == float(density.level)
        loss = sum((params[k][active] - 0.37).square().sum()
                   for k in ("means", "scales", "quats", "opacities", "sh0", "shN"))
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        density.after_backward(params, opts, step, {"info": {}})

    for step in (1, 2):
        train_step(step)
    expect = lr0 * 0.01 ** (3 / 3)
    if abs(opts["means"].param_groups[0]["lr"] - expect) > 1e-12 * max(1, lr0):
        raise AssertionError("position LR is not log-linear over the stage")
    before = {k: v.detach().clone() for k, v in params.items()}
    train_step(3)  # stage boundary
    trained = {k: v.detach().clone() for k, v in params.items()}
    check_state(method, params, opts)
    n_new = len(params["means"]) - 12
    if n_new != 12 or density.level != top - 1 or density.stage != 1:
        raise AssertionError(f"inheritance created {n_new} Gaussians at level {density.level}")
    new = slice(12, 24)
    if not torch.equal(params["means"][new], trained["means"][:12]):
        raise AssertionError("new field must inherit the archived xyz")
    d2 = torch.cdist(trained["means"][:12], trained["means"][:12]).square()
    d2.fill_diagonal_(float("inf"))
    ref = d2.topk(3, largest=False).values.mean(1).clamp_min(1e-7).sqrt().log()
    if not torch.allclose(params["scales"][new], ref[:, None].expand(-1, 3), atol=1e-5):
        raise AssertionError("new scales are not the 3-NN initialisation")
    if (params["sh0"][new].abs().max() != 0 or params["shN"][new].abs().max() != 0
            or not torch.allclose(params["opacities"][new].sigmoid(), torch.tensor(0.1, device=device))
            or not torch.equal(params["quats"][new][:, 0], torch.ones(12, device=device))
            or not bool((params[LEVEL_FIELD][new] == top - 1).all())):
        raise AssertionError("new field must re-initialise colour / opacity / rotation")
    for name, p in params.items():
        st = opts[name].state.get(p, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st and bool(st[key].abs().max() > 0):
                raise AssertionError(f"{name}.{key} not reset at the stage boundary")
        if "step" in st and float(st["step"]) != 0:
            raise AssertionError("Adam step not reset (fresh optimizer per stage)")
    if abs(opts["means"].param_groups[0]["lr"] - lr0 * 0.01 ** (1 / 3)) > 1e-12 * max(1, lr0):
        raise AssertionError("position LR must restart at the new stage")
    train_step(4)
    moved = max(float((params[k][:12] - trained[k][:12]).abs().max()) for k in params)
    learned = float((params["sh0"][new] - trained["sh0"][new]).abs().max())
    if moved != 0.0 or learned == 0.0:
        raise AssertionError(f"archived rows moved by {moved}; active learned {learned}")
    del before
    # ADC must never touch the archived field.
    density.scene_scale = 1.0
    n_before = len(params["means"])
    with torch.no_grad():
        params["opacities"][0] = -10.0  # archived, would be pruned
        params["opacities"][12] = -10.0  # active, must be pruned
    density._reset_stats(params)
    density.stats["grad2d"].fill_(1.0)
    density.stats["count"].fill_(1.0)
    archived_before = {k: v.detach()[:12].clone() for k, v in params.items()}
    density.prune_and_densify(params, opts)
    check_state(method, params, opts)
    lv = params[LEVEL_FIELD].detach()
    if int((lv == top).sum()) != 12:
        raise AssertionError("densification changed the archived field size")
    for k, v in archived_before.items():
        if not torch.equal(params[k].detach()[:12], v):
            raise AssertionError("densification modified archived Gaussians")
    n_active = int((lv == top - 1).sum())
    if n_active != 2 * 11:
        raise AssertionError(f"active field after prune+clone/split has {n_active}, expected 22")
    return {"inherited": n_new, "archived_drift": moved, "active_update": learned,
            "gaussians_after_adc": len(params["means"]), "n_before_adc": n_before,
            "active_after_adc": n_active}


def lap_loss(method, device):
    from .losses import create_loss, ssim_same

    loss = create_loss(_component(method, "loss"))
    g = np.random.default_rng(1)
    target_np = g.random((22, 30, 3))
    target = torch.tensor(target_np, dtype=torch.float32)
    gt1 = torch.tensor(np_reflect_conv(target_np, 2), dtype=torch.float32)  # (11, 15)
    rgb = (gt1 + 0.1 * torch.tensor(g.standard_normal(gt1.shape), dtype=torch.float32))
    rgb.requires_grad_(True)
    value = loss({}, {"rgb": rgb, "lap_level": 1}, target)
    ref = 0.8 * (rgb - gt1).abs().mean() + 0.2 * (1 - ssim_same(rgb, gt1))
    if abs(float(value) - float(ref)) > 1e-5:
        raise AssertionError("loss is not L1 + 0.2 D-SSIM against Down(GT)")
    exact = loss({}, {"rgb": gt1.clone().requires_grad_(True), "lap_level": 1}, target)
    if float(exact) > 1e-5:
        raise AssertionError("composed image equal to the pyramid level must give zero loss")
    full = loss({}, {"rgb": target.clone().requires_grad_(True), "lap_level": 0}, target)
    if float(full) > 1e-5:
        raise AssertionError("level 0 must supervise against the full-resolution GT")
    grad = torch.autograd.grad(value, rgb)[0]
    if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
        raise AssertionError("loss gradient is zero or nonfinite")
    return {"loss": float(value), "reference": float(ref), "exact_level_loss": float(exact)}


def lap_renderer_gradient(method, device):
    """Built-in renderer-gradient check with the target at the active pyramid level."""
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    params, opts = sample_scene(method, device)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
              "width": 16, "height": 16}
    density.initialize(params, opts, 1.0)
    outputs = renderer.render(params, camera, sh_degree=0)
    density.before_backward(params, opts, 1, outputs)
    target = (torch.arange(16, device=device)[None, :, None] % 2).expand(16, 16, 3).float()
    target = P.target_at_level(target, outputs["lap_level"])
    (outputs["rgb"] - target).square().mean().backward()
    xy = outputs["info"]["means2d"]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    if float(density.stats["count"].sum()) == 0:
        raise AssertionError("density statistics were not accumulated for the active field")
    return {"rgb_shape": list(outputs["rgb"].shape), "screen_gradient_norm": xy.grad.norm().item(),
            "visible_counted": float(density.stats["count"].sum())}
