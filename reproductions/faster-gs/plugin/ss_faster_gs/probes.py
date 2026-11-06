"""Mechanism probes for the Faster-GS port. Each raises on failure and returns measurements."""

import math
from dataclasses import replace

import torch

C0 = 0.28209479177387814
TILE_W, TILE_H = 16, 12  # rasterization_config.h


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, kind, **config):
    spec = next(c for c in method.components if c.kind == kind)
    return replace(spec, config={**spec.config, **config})


def _renderer(method):
    from .renderer import create_renderer

    return create_renderer(_component(method, "renderer"))


def _density(method, **config):
    from .density import create_density

    return create_density(_component(method, "density", **config))


def _camera(device, W=96, H=72, f=70.0, cx=None, cy=None):
    cx = W / 2 + 3.3 if cx is None else cx  # off-centre principal point (supported by Faster-GS)
    cy = H / 2 - 2.1 if cy is None else cy
    return {"viewmat": torch.eye(4, device=device), "width": W, "height": H,
            "K": torch.tensor([[f, 0, cx], [0, f * 1.05, cy], [0, 0, 1]], device=device)}


def _fixture(device, n=40, seed=3, sh=True):
    """Random anisotropic Gaussians in front of the camera; opacities 0.02..0.99."""
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([2.4, 1.8, 2.0]) + torch.tensor(
        [-1.2, -0.9, 3.0])
    scales = (torch.rand(n, 3, generator=g) * torch.tensor([0.20, 0.10, 0.05]) + 0.015).log()
    quats = torch.randn(n, 4, generator=g) * 1.7  # unnormalised on purpose (fused activations)
    opac = torch.cat([torch.rand(n // 2, generator=g) * 0.09 + 0.9,
                      torch.rand(n - n // 2, generator=g) * 0.6 + 0.02])
    sh0 = (torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8
    shN = torch.randn(n, 15, 3, generator=g) * (0.15 if sh else 0.0)
    vals = {"means": means, "scales": scales, "quats": quats, "opacities": torch.logit(opac),
            "sh0": sh0, "shN": shN}
    return {k: v.float().to(device) for k, v in vals.items()}


def _leaf(params):
    return {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------- probes
def render_reference(method, device):
    """Image and all six raw-parameter gradients equal the dense float64 reference."""
    _cuda(device)
    from . import reference

    renderer = _renderer(method)
    bg = torch.tensor(renderer.background, device=device)
    out = {}
    for degree in (0, 3):
        params = _fixture(device, seed=3 + degree)
        cam = _camera(device)
        p_fast, p_ref = _leaf(params), _leaf(params)
        img = renderer.render(p_fast, cam, sh_degree=degree)["rgb"]
        ref = reference.render(p_ref, cam, degree, bg)
        err = float((img.double() - ref).abs().max())
        if err > 2e-3:
            raise AssertionError(f"degree {degree}: image differs from dense reference by {err}")
        w = torch.rand(img.shape, generator=torch.Generator().manual_seed(1)).to(device)
        (img * w).sum().backward()
        (ref * w.double()).sum().backward()
        grads = {}
        for k in params:
            if degree == 0 and k == "shN":
                if p_fast[k].grad is not None and torch.count_nonzero(p_fast[k].grad):
                    raise AssertionError("inactive SH bases received gradients")
                continue
            e = _rel(p_fast[k].grad.double(), p_ref[k].grad)
            grads[k] = e
            if not e < 2e-2:
                raise AssertionError(f"degree {degree}: gradient of {k} rel. error {e:.3g}")
        out[f"degree{degree}"] = {"image_max_abs_err": err, "grad_rel_err": grads,
                                  "image_mean": float(img.mean())}
    return out


def _single(device, mean, scale, quat, opacity):
    vals = {"means": torch.tensor([mean]), "scales": torch.tensor([scale]).log(),
            "quats": torch.tensor([quat]), "opacities": torch.logit(torch.tensor([opacity])),
            "sh0": torch.full((1, 1, 3), 0.3), "shN": torch.zeros(1, 15, 3)}
    return {k: v.float().to(device) for k, v in vals.items()}


def _tile_counts(params, cam):
    """Independent counts for one Gaussian: L = tiles (in-image) holding a pixel centre with
    alpha >= 1/255; R = tiles of the opacity-aware rectangular AABB; S = tiles of 3DGS's
    3-sigma square (radius ceil(3 sqrt(lambda_max)))."""
    from . import reference

    p = reference.project(params, cam)
    mu, cov, conic, o = p["mean2d"][0], p["cov2d"][0], p["conic"][0], p["opacity"][0]
    W, H = int(cam["width"]), int(cam["height"])
    gw, gh = math.ceil(W / TILE_W), math.ceil(H / TILE_H)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64) + 0.5,
                            torch.arange(W, dtype=torch.float64) + 0.5, indexing="ij")
    dx, dy = mu[0].cpu() - xs, mu[1].cpu() - ys
    c = conic.cpu()
    alpha = o.cpu() * torch.exp(-0.5 * (c[0, 0] * dx**2 + 2 * c[0, 1] * dx * dy + c[1, 1] * dy**2))
    hit = alpha >= 1.0 / 255.0
    tiles = {(int(x) // TILE_W, int(y) // TILE_H) for y, x in zip(*torch.nonzero(hit).T.tolist())}
    L = len(tiles)

    def rect_tiles(x0, x1, y0, y1):
        tx0, tx1 = max(0, math.floor(x0 / TILE_W)), min(gw, math.ceil(x1 / TILE_W))
        ty0, ty1 = max(0, math.floor(y0 / TILE_H)), min(gh, math.ceil(y1 / TILE_H))
        return max(0, tx1 - tx0) * max(0, ty1 - ty0)

    k = 2 * math.log(float(o) * 255.0)
    ex = max(math.sqrt(float(cov[0, 0]) * k) - 0.5, 0.0)
    ey = max(math.sqrt(float(cov[1, 1]) * k) - 0.5, 0.0)
    R = rect_tiles(math.floor(mu[0] - ex), math.ceil(mu[0] + ex),
                   math.floor(mu[1] - ey), math.ceil(mu[1] + ey))
    lam = float(torch.linalg.eigvalsh(cov).max())
    r = math.ceil(3 * math.sqrt(lam))
    S = rect_tiles(float(mu[0]) - r, float(mu[0]) + r + 1, float(mu[1]) - r, float(mu[1]) + r + 1)
    return L, R, S


def tile_culling(method, device):
    """Exact opacity-aware tile assignment: L <= n <= R, and n < R, n < S for an elongated
    rotated splat (fails if exact culling or opacity-aware bounds are removed, or if the kernel
    drops tiles that contain visible fragments)."""
    _cuda(device)
    renderer = _renderer(method)
    cam = _camera(device, W=160, H=120, cx=80.0, cy=60.0, f=120.0)
    a = math.radians(35.0)
    cases = {
        "elongated_rotated_opaque": _single(device, [0.1, 0.0, 3.0], [0.9, 0.03, 0.03],
                                            [math.cos(a / 2), 0, 0, math.sin(a / 2)], 0.95),
        "elongated_rotated_faint": _single(device, [0.0, 0.1, 3.0], [0.8, 0.04, 0.04],
                                           [math.cos(-a / 2), 0, 0, math.sin(-a / 2)], 0.05),
        "isotropic_opaque": _single(device, [-0.3, 0.2, 4.0], [0.2, 0.2, 0.2], [1, 0, 0, 0], 0.99),
    }
    out = {}
    for name, params in cases.items():
        with torch.no_grad():
            n = renderer.render(params, cam, sh_degree=0)["info"]["n_instances"]
        L, R, S = _tile_counts(params, cam)
        out[name] = {"kernel": n, "lower_bound_visible": L, "rect_aabb": R, "square_3sigma": S}
        if not L <= n <= R:
            raise AssertionError(f"{name}: kernel tiles {n} outside [{L}, {R}]")
        if name.startswith("elongated") and not (n < R and n < S):
            raise AssertionError(f"{name}: no culling gain over AABB/3-sigma ({n}, {R}, {S})")
    if not out["elongated_rotated_faint"]["kernel"] < out["elongated_rotated_opaque"]["kernel"]:
        raise AssertionError("opacity-aware bounds: faint splat should touch fewer tiles")
    return out


def fused_adam(method, device):
    """FusedAdam == torch.optim.Adam(eps=1e-15) over steps with a changing lr; not SGD."""
    _cuda(device)
    from .optim import FusedAdam

    g = torch.Generator(device=device).manual_seed(0)
    x0 = torch.randn(1000, 3, device=device, generator=g)
    a = torch.nn.Parameter(x0.clone())
    b = torch.nn.Parameter(x0.clone())
    oa = FusedAdam([a], lr=1e-2, eps=1e-15)
    ob = torch.optim.Adam([b], lr=1e-2, eps=1e-15)
    for i in range(6):
        grad = torch.randn(1000, 3, device=device, generator=g) * (10.0 ** (i - 3))
        a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
        for o in (oa, ob):
            o.param_groups[0]["lr"] *= 0.7
    err = _rel(a.detach() - x0, b.detach() - x0)
    m_err = _rel(oa.state[a]["exp_avg_sq"], ob.state[b]["exp_avg_sq"])
    if err > 1e-3 or m_err > 1e-3:  # fmaf form of the official kernel; float32 rounding
        raise AssertionError(f"FusedAdam differs from torch Adam: update {err}, v {m_err}")
    sgd = x0 - 1e-2 * grad
    if _rel(a.detach(), sgd) < 1e-3:
        raise AssertionError("FusedAdam behaves like SGD")
    return {"update_rel_err": err, "exp_avg_sq_rel_err": m_err, "steps": int(oa.state[a]["step"])}


def _morton_py(p, lo, size):
    """Independent per-point Morton code (float32 quantisation as in the CUDA kernel, bitwise
    interleave in Python)."""
    rcp = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(size, dtype=torch.float32)
    q = ((p.float().cpu() - lo.float().cpu()) * rcp).clamp(0.0, 1.0) * 2097151.0
    q = [int(v) for v in q.tolist()]
    code = 0
    for bit in range(21):
        for axis in range(3):
            code |= ((q[axis] >> bit) & 1) << (3 * bit + axis)
    return code


def morton(method, device):
    """Density's Morton step sorts means by an independent bit-interleave Morton code and moves
    the Adam moments with them; schedule = {0, 5000, 10000, 15000}."""
    _cuda(device)
    from .density import schedule
    from .optim import FusedAdam

    density = _density(method)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in
                                     _fixture(device, n=300, seed=9).items()})
    opts = {k: FusedAdam([p], lr=1e-3) for k, p in params.items()}
    for p in params.values():
        p.grad = torch.randn_like(p)
    for o in opts.values():
        o.step()
    tag = params["means"].detach()[:, 0].clone()
    moment_tag = opts["means"].state[params["means"]]["exp_avg"][:, 0].clone()
    density.state = {"events": []}
    density.morton(params, opts, 5000)
    means = params["means"].detach().cpu()
    lo, size = means.amin(0), float((means.amax(0) - means.amin(0)).amax())
    codes = [_morton_py(m, lo, size) for m in means]
    if any(codes[i] > codes[i + 1] for i in range(len(codes) - 1)):
        raise AssertionError("means are not in Morton order after the Morton step")
    # moments follow the parameters
    perm = torch.stack([torch.nonzero(tag == v)[0, 0] for v in params["means"].detach()[:, 0]])
    m = opts["means"].state[params["means"]]["exp_avg"][:, 0]
    if not torch.equal(m, moment_tag[perm]):
        raise AssertionError("Adam moments were not permuted with the parameters")
    from splatstudio.state import check_state

    check_state(method, params, opts)
    _, msteps, _ = schedule(density.cfg)
    if msteps != [0, 5000, 10000, 15000]:
        raise AssertionError(f"Morton schedule {msteps}")
    return {"sorted": True, "n": len(codes), "schedule": msteps}


def adc(method, device):
    """3DGS ADC on a fixture equals an independent re-implementation of the official rule;
    moments of survivors are kept, new ones zero; schedule = 144 ADC steps 600..14900, resets
    3000..12000; densification info is consumed, re-zeroed and finally disabled."""
    _cuda(device)
    from splatstudio.state import check_state

    from .density import schedule
    from .optim import FusedAdam
    from .renderer import FasterGSInfo

    density = _density(method)
    n = 64
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in
                                     _fixture(device, n=n, seed=11).items()})
    opts = {k: FusedAdam([p], lr=1e-3) for k, p in params.items()}
    for p in params.values():
        p.grad = torch.ones_like(p)
    for o in opts.values():
        o.step()
    with torch.no_grad():  # some degenerate / low-opacity / huge Gaussians
        params["opacities"][0] = math.log(0.004 / 0.996)
        params["quats"][1] = 1e-5
        params["scales"][2] = math.log(2.0)
    extent = 8.0
    density.extent = extent
    density.state = {"events": []}
    g = torch.Generator().manual_seed(5)
    count = torch.randint(0, 6, (n,), generator=g).float()
    grad = torch.rand(n, generator=g) * 6e-4 * count
    holder = FasterGSInfo()
    holder.get(n, device).copy_(torch.stack([count, grad]).to(device))
    before = {k: v.detach().clone() for k, v in params.items()}
    m_before = opts["means"].state[params["means"]]["exp_avg"].clone()
    # independent expectation (official adaptive_density_control, prune_large since 3100 > 3000)
    avg = grad / count.clamp_min(1)
    sel = avg >= 2e-4
    small = before["scales"].cpu().max(1).values <= math.log(0.01 * extent)
    dup, spl = sel & small, sel & ~small
    opac_ok = before["opacities"].cpu() >= math.log(0.005 / 0.995)
    quat_ok = before["quats"].cpu().square().sum(1) >= 1e-8
    max_scale = before["scales"].cpu().max(1).values
    big = math.log(0.1 * extent)
    keep_old = ~spl & opac_ok & quat_ok & (max_scale <= big)
    keep_dup = dup & opac_ok & quat_ok & (max_scale <= big)
    keep_child = spl & opac_ok & quat_ok & (max_scale + math.log(0.625) <= big)
    density.after_backward(params, opts, 3100, {"info": {"fastergs": holder}})
    check_state(method, params, opts)
    n_old, n_dup, n_child = int(keep_old.sum()), int(keep_dup.sum()), 2 * int(keep_child.sum())
    if len(params["means"]) != n_old + n_dup + n_child:
        raise AssertionError(f"population {len(params['means'])} != {n_old}+{n_dup}+{n_child}")
    if n_dup == 0 or n_child == 0:
        raise AssertionError("fixture must exercise both duplicate and split")
    dev = lambda m: m.to(device)  # noqa: E731
    torch.testing.assert_close(params["sh0"][:n_old], before["sh0"][dev(keep_old)])
    torch.testing.assert_close(opts["means"].state[params["means"]]["exp_avg"][:n_old],
                               m_before[dev(keep_old)])
    if torch.count_nonzero(opts["means"].state[params["means"]]["exp_avg"][n_old:]):
        raise AssertionError("new Gaussians must start with zero Adam moments")
    torch.testing.assert_close(params["means"][n_old:n_old + n_dup], before["means"][dev(keep_dup)])
    exp_child = (before["scales"][dev(keep_child)].exp() * 0.625).log().repeat(2, 1)
    torch.testing.assert_close(params["scales"][n_old + n_dup:], exp_child)
    torch.testing.assert_close(params["opacities"][n_old + n_dup:],
                               before["opacities"][dev(keep_child)].repeat(2))
    if float(holder.buffer.abs().sum()) != 0.0 or holder.buffer.shape[1] != len(params["means"]):
        raise AssertionError("densification info must be re-zeroed and resized after ADC")
    density.after_backward(params, opts, 14900, {"info": {"fastergs": holder}})
    if holder.enabled:
        raise AssertionError("densification info must be disabled after the last ADC")
    adc_steps, msteps, resets = schedule(density.cfg)
    if not (len(adc_steps) == 144 and adc_steps[0] == 600 and adc_steps[-1] == 14900
            and resets == [3000, 6000, 9000, 12000]):
        raise AssertionError("ADC / reset schedule differs from the official callbacks")
    return {"duplicated": n_dup, "split_children": n_child, "survivors_old": n_old,
            "final": len(params["means"]), "adc_steps": len(adc_steps), "resets": resets}


def densification_info(method, device):
    """The backward accumulates count += 1 for visible Gaussians and grad += ||0.5 (W, H) *
    dL/d mu_2D|| (3DGS NDC scaling), equal to the dense reference's mean2d gradients."""
    _cuda(device)
    from . import reference

    renderer = _renderer(method)
    bg = torch.tensor(renderer.background, device=device)
    params = _fixture(device, seed=21)
    cam = _camera(device)
    p_fast, p_ref = _leaf(params), _leaf(params)
    renderer.densification.buffer = None
    renderer.densification.enabled = True
    w = torch.rand(int(cam["height"]), int(cam["width"]), 3,
                   generator=torch.Generator().manual_seed(2)).to(device)
    out = renderer.render(p_fast, cam, sh_degree=0)
    (out["rgb"] * w).sum().backward()
    info = out["info"]["fastergs"].buffer.clone()
    holder = {}

    def hook(m):
        m.retain_grad()
        holder["m"] = m
        return m

    (reference.render(p_ref, cam, 0, bg, mean2d_hook=hook) * w.double()).sum().backward()
    g = holder["m"].grad
    W, H = float(cam["width"]), float(cam["height"])
    ref_norm = (0.5 * torch.stack([g[:, 0] * W, g[:, 1] * H], -1)).norm(dim=-1)
    visible = info[0] > 0
    proj = reference.project(p_ref, cam)
    if not torch.equal(visible, proj["valid"]):
        raise AssertionError("visibility count differs from the reference validity")
    e = _rel(info[1][visible].double(), ref_norm[visible])
    if e > 2e-2:
        raise AssertionError(f"accumulated NDC gradient norm rel. error {e}")
    out2 = renderer.render(_leaf(params), cam, sh_degree=0)
    (out2["rgb"] * w).sum().backward()
    if not torch.allclose(out2["info"]["fastergs"].buffer, 2 * info, rtol=1e-4, atol=1e-9):
        raise AssertionError("densification info does not accumulate across iterations")
    return {"visible": int(visible.sum()), "grad_norm_rel_err": e}


def profile_smoke(method, device):
    """Forward/backward/update timings run (no claim; evidence that the profiler works)."""
    _cuda(device)
    from .profile import profile_params

    params = _fixture(device, n=2000, seed=1)
    cam = _camera(device, W=256, H=192, f=200.0)
    target = torch.rand(192, 256, 3, device=device)
    res = profile_params(params, [cam], [target], sh_degree=3, iters=3, warmup=1)
    for path in ("faster-gs", "gsplat"):
        for k in ("forward_ms", "backward_ms", "update_ms"):
            if not (res[path][k] > 0 and math.isfinite(res[path][k])):
                raise AssertionError(f"{path} {k} not measured")
    res.pop("images")
    return res
