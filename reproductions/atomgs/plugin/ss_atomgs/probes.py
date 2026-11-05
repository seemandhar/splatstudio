"""Mechanism probes for the AtomGS port. Each raises if its mechanism is removed or wrong."""

import math

import torch
import torch.nn.functional as F

from .density import Density, quantile_linear
from .losses import create_normal_loss, create_photometric, edge_aware_normal_loss


def _component(method, name):
    for c in method.components:
        if c.name == name:
            return c
    raise KeyError(name)


def _params(n, device, seed=0):
    from torch import nn

    g = torch.Generator(device="cpu").manual_seed(seed)
    p = nn.ParameterDict({
        "means": nn.Parameter(torch.randn(n, 3, generator=g).to(device)),
        "scales": nn.Parameter((torch.rand(n, 3, generator=g) * 2 - 4).to(device)),
        "quats": nn.Parameter(F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device)),
        "opacities": nn.Parameter((torch.rand(n, generator=g) * 2 - 1).to(device)),
        "sh0": nn.Parameter(torch.rand(n, 1, 3, generator=g).to(device)),
        "shN": nn.Parameter(torch.zeros(n, 15, 3).to(device)),
    })
    opt = {k: torch.optim.Adam([v], lr=1e-3, eps=1e-15) for k, v in p.items()}
    # allocate non-zero Adam moments
    loss = sum((v.float() ** 2).sum() for v in p.values())
    loss.backward()
    for o in opt.values():
        o.step()
        o.zero_grad(set_to_none=True)
    return p, opt


def _density(method, **override):
    spec = _component(method, "atomgs-density")
    from dataclasses import replace

    return Density(replace(spec, config={**spec.config, **override}))


# ------------------------------------------------------------------------------------------
# C3 feature rasterizer
# ------------------------------------------------------------------------------------------
def _quat_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def reference_render(means, scales, quats, opac, sh0, camera, bg):
    """Independent float64 Inria EWA splatting + front-to-back compositing (one pixel loop)."""
    K, V = camera["K"].double(), camera["viewmat"].double()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = K[0, 0], K[1, 1]
    pv = means @ V[:3, :3].T + V[:3, 3]
    x, y, z = pv.unbind(-1)
    tx, ty = W / (2 * fx), H / (2 * fy)
    xc = (x / z).clamp(-1.3 * tx, 1.3 * tx) * z
    yc = (y / z).clamp(-1.3 * ty, 1.3 * ty) * z
    zero = torch.zeros_like(z)
    J = torch.stack([fx / z, zero, -fx * xc / z ** 2, zero, fy / z, -fy * yc / z ** 2],
                    -1).view(-1, 2, 3)
    R = _quat_rot(quats)
    S = torch.diag_embed(scales)
    cov3 = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
    Wr = V[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2) + 0.3 * torch.eye(2, dtype=means.dtype)
    conic = torch.linalg.inv(cov2)
    px = fx * x / z + W / 2 - 0.5     # ndc2Pix((x_ndc + 1) * W - 1) / 2 with centred pp
    py = fy * y / z + H / 2 - 0.5
    color = (0.28209479177387814 * sh0[:, 0] + 0.5).clamp_min(0)
    order = torch.argsort(z.detach())
    img = torch.zeros(H, W, 3, dtype=means.dtype)
    dep = torch.zeros(H, W, dtype=means.dtype)
    med = torch.zeros(H, W, dtype=means.dtype)
    acc = torch.zeros(H, W, dtype=means.dtype)
    for i in range(H):
        for j in range(W):
            T = torch.ones((), dtype=means.dtype)
            c = torch.zeros(3, dtype=means.dtype)
            d = torch.zeros((), dtype=means.dtype)
            m = torch.zeros((), dtype=means.dtype)
            for k in order.tolist():
                dx, dy = px[k] - j, py[k] - i
                power = -0.5 * (conic[k, 0, 0] * dx * dx + conic[k, 1, 1] * dy * dy) - \
                    conic[k, 0, 1] * dx * dy
                if power > 0:
                    continue
                a = torch.clamp(opac[k] * torch.exp(power), max=0.99)
                if a < 1.0 / 255.0:
                    continue
                test_T = T * (1 - a)
                if test_T < 0.0001:
                    break
                c = c + color[k] * a * T
                d = d + z[k] * a * T
                if T > 0.5 and test_T < 0.5:
                    m = z[k].detach()
                T = test_T
            img[i, j] = c + T * bg
            dep[i, j] = d
            med[i, j] = m
            acc[i, j] = 1 - T
    return img, dep, med, acc


def feature_raster(method, device):
    from .raster import render

    if not str(device).startswith("cuda"):
        from splatstudio.probes import ProbeUnavailable

        raise ProbeUnavailable("feature-raster needs CUDA")
    g = torch.Generator().manual_seed(3)
    n, W, H = 5, 16, 16  # one 16x16 tile: every visible Gaussian covers all pixels
    means = torch.cat([torch.rand(n, 2, generator=g) - 0.5, 3 + torch.rand(n, 1, generator=g)], 1)
    scales = 0.7 + 0.4 * torch.rand(n, 3, generator=g)
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = 0.2 + 0.5 * torch.rand(n, generator=g)
    sh0 = torch.randn(n, 1, 3, generator=g)
    K = torch.tensor([[20.0, 0, W / 2], [0, 18.0, H / 2], [0, 0, 1]])
    camera = {"K": K, "viewmat": torch.eye(4), "width": W, "height": H}
    bg = torch.tensor([0.2, 0.4, 0.6])
    wts = torch.rand(H, W, generator=g)
    wrgb = torch.rand(H, W, 3, generator=g)

    params = {
        "means": means.to(device).requires_grad_(True),
        "scales": scales.log().to(device).requires_grad_(True),
        "quats": quats.to(device).requires_grad_(True),
        "opacities": torch.logit(opac).to(device).requires_grad_(True),
        "sh0": sh0.to(device).requires_grad_(True),
        "shN": torch.zeros(n, 15, 3, device=device),
    }
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in camera.items()}
    out = render(params, cam, 0, bg.to(device))
    radii = out["info"]["radii"]
    if int(radii.min()) <= 0:
        raise AssertionError(f"probe fixture Gaussians must all be visible, radii={radii.tolist()}")
    (out["mean_depth"][..., 0] * wts.to(device)).sum().backward(retain_graph=True)
    g_depth = params["means"].grad.detach().cpu().double()
    params["means"].grad = None
    (out["rgb"] * wrgb.to(device)).sum().backward()
    g_rgb = params["means"].grad.detach().cpu().double()

    m64 = means.double().requires_grad_(True)
    img, dep, med, acc = reference_render(m64, scales.double(), quats.double(), opac.double(),
                                          sh0.double(), camera, bg.double())
    (dep * wts.double()).sum().backward(retain_graph=True)
    r_depth = m64.grad.clone()
    m64.grad = None
    (img * wrgb.double()).sum().backward()
    r_rgb = m64.grad.clone()

    def err(a, b):
        return float((a.detach().cpu().double() - b.detach()).abs().max() /
                     b.detach().abs().max().clamp_min(1e-12))

    res = {
        "rgb_rel_err": err(out["rgb"], img),
        "mean_depth_rel_err": err(out["mean_depth"][..., 0], dep),
        "alpha_rel_err": err(out["alpha"][..., 0], acc),
        "median_rel_err": err(out["median_depth"][..., 0], med),
        "grad_means_from_depth_rel_err": err(g_depth, r_depth),
        "grad_means_from_rgb_rel_err": err(g_rgb, r_rgb),
        "depth_is_unnormalised": float((out["mean_depth"] - out["depth"]).abs().max()),
    }
    for k, v in res.items():
        if k.endswith("rel_err") and not v < 2e-3:
            raise AssertionError(f"{k} = {v}: kernel differs from the reference compositor")
    if not res["depth_is_unnormalised"] > 1e-3:
        raise AssertionError("mean depth looks alpha-normalised; official D = sum z w")
    return res


# ------------------------------------------------------------------------------------------
# C2 normals + edge-aware normal loss
# ------------------------------------------------------------------------------------------
def _independent_normals(depth, camera):
    """Same official grid normalisation, but via explicit camera-space rays (no 4x4 inverse)."""
    from .raster import projection

    H, W = depth.shape
    P = projection(camera).double()
    jj = torch.arange(W, dtype=torch.float64)
    ii = torch.arange(H, dtype=torch.float64)
    xn = (2 * jj / (W - 1) - 1)[None, :].expand(H, W)
    yn = (2 * ii / (H - 1) - 1)[:, None].expand(H, W)
    d = depth.double()
    cam = torch.stack([xn * d / P[0, 0], yn * d / P[1, 1], d], -1)
    c2w = torch.linalg.inv(camera["viewmat"].double())
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    pts = torch.zeros(H + 1, W + 1, 3, dtype=torch.float64)
    pts[:H, :W] = world
    n = torch.cross(pts[1:, :-1] - pts[:-1, :-1], pts[:-1, 1:] - pts[:-1, :-1], dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-8)


def normal_from_depth(method, device):
    from .raster import depth_to_normal, raster_settings

    W, H = 20, 16
    K = torch.tensor([[25.0, 0, W / 2], [0, 25.0, H / 2], [0, 0, 1]])
    ang = 0.3
    V = torch.eye(4)
    V[:3, :3] = torch.tensor([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0],
                              [-math.sin(ang), 0, math.cos(ang)]])
    V[:3, 3] = torch.tensor([0.1, -0.2, 0.5])
    camera = {"K": K, "viewmat": V, "width": W, "height": H}
    s = raster_settings(camera, torch.zeros(3), 0)
    g = torch.Generator().manual_seed(1)
    depth = 2 + torch.rand(H, W, generator=g)
    ours = depth_to_normal(depth, s)
    ref = _independent_normals(depth, camera)
    e_ind = float((ours.double()[:-1, :-1] - ref[:-1, :-1]).abs().max())
    # analytic: constant depth = fronto-parallel plane -> normal = +/- camera z axis (world)
    flat = depth_to_normal(torch.full((H, W), 2.5), s)[:-1, :-1]
    zaxis = torch.linalg.inv(V)[:3, 2]
    e_plane = float((1 - (flat @ zaxis).abs()).max())
    res = {"independent_max_err": e_ind, "plane_normal_err": e_plane}
    if not (e_ind < 2e-3 and e_plane < 1e-4):
        raise AssertionError(f"depth_to_normal mismatch {res}")
    return res


def _loss_reference(img, N, power):
    def sob(t):
        p = F.pad(t, (1, 1, 1, 1))
        a = lambda di, dj: p[:, 1 + di: p.shape[1] - 1 + di, 1 + dj: p.shape[2] - 1 + dj]  # noqa: E731
        gx = (a(-1, 1) + 2 * a(0, 1) + a(1, 1) - a(-1, -1) - 2 * a(0, -1) - a(1, -1)) / 4
        gy = (a(1, -1) + 2 * a(1, 0) + a(1, 1) - a(-1, -1) - 2 * a(-1, 0) - a(-1, 1)) / 4
        return gx, gy

    nx, ny = sob(N)
    ix, iy = sob(img)
    wx = (1 - ix.abs().mean(0)) ** power
    wy = (1 - iy.abs().mean(0)) ** power
    return torch.sqrt(((nx.abs() * wx + ny.abs() * wy) ** 2).sum(0)).mean()


def edge_aware_normal(method, device):
    spec = _component(method, "edge-aware-normal")
    power, until = int(spec.config["edge_power"]), int(spec.config["until"])
    lam = float(spec.config["lambda_normal"])
    g = torch.Generator().manual_seed(2)
    img = torch.rand(3, 12, 14, generator=g) * 0.3
    N = F.normalize(torch.randn(3, 12, 14, generator=g), dim=0)
    e_ref = abs(float(edge_aware_normal_loss(img, N, power)) -
                float(_loss_reference(img, N, power)))
    # edge alignment: a normal discontinuity is ignored where the image has a strong edge
    Nstep = torch.zeros(3, 12, 14)
    Nstep[2] = 1.0
    Nstep[:, :, 7:] = torch.tensor([1.0, 0, 0])[:, None, None]
    flat = torch.full((3, 12, 14), 0.5)
    edge = flat.clone()
    edge[:, :, 7:] = 1.0
    edge[:, :, :7] = 0.0
    Nconst = torch.zeros(3, 12, 14)
    Nconst[2] = 1.0

    def excess(img):  # loss added by the normal discontinuity (border terms cancel)
        return float(edge_aware_normal_loss(img, Nstep, power) -
                     edge_aware_normal_loss(img, Nconst, power))

    l_flat, l_edge = excess(flat), excess(edge)
    fn = create_normal_loss(spec)
    target = img.permute(1, 2, 0)
    normal = N.permute(1, 2, 0).clone().requires_grad_(True)
    before = float(fn(None, {"surface_normal": normal, "step": until - 1}, target))
    after = float(fn(None, {"surface_normal": normal, "step": until}, target))
    res = {"ref_abs_err": e_ref, "loss_flat_image": l_flat, "loss_edge_image": l_edge,
           "weighted_before_until": before, "after_until": after}
    if not e_ref < 1e-5:
        raise AssertionError(f"normal loss differs from the reference {res}")
    if not (l_flat > 1e-3 and l_edge < 0.05 * l_flat):
        raise AssertionError(f"edge-aware weighting inactive {res}")
    if not (abs(before - lam * float(_loss_reference(img, N, power))) < 1e-5 and after == 0.0):
        raise AssertionError(f"normal loss weight / schedule wrong {res}")
    return res


# ------------------------------------------------------------------------------------------
# C4 MS-SSIM photometric
# ------------------------------------------------------------------------------------------
def _ssim_separable(x, y):
    g = torch.exp(-((torch.arange(11, dtype=x.dtype) - 5) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()

    def blur(t):  # zero 'same' padding, separable
        c, h, w = t.shape
        t = F.conv1d(F.pad(t.reshape(c * h, 1, w), (5, 5)), g.view(1, 1, 11)).view(c, h, w)
        t = t.transpose(1, 2).reshape(c * w, 1, h)
        t = F.conv1d(F.pad(t, (5, 5)), g.view(1, 1, 11)).view(c, w, h).transpose(1, 2)
        return t

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + C1) * (2 * cxy + C2)) /
            ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))).mean()


def ms_ssim_probe(method, device):
    spec = _component(method, "atomgs-photometric")
    lam, scales = float(spec.config["lambda_ms_ssim"]), int(spec.config["ms_ssim_scales"])
    g = torch.Generator().manual_seed(4)
    x = torch.rand(3, 64, 48, generator=g, dtype=torch.float64)
    y = (x + 0.1 * torch.randn(3, 64, 48, generator=g, dtype=torch.float64)).clamp(0, 1)
    ms, a, b = 0.0, x, y
    for _ in range(scales):
        ms += _ssim_separable(a, b) / scales
        c, h, w = a.shape
        a = a.view(c, h // 2, 2, w // 2, 2).mean((2, 4))
        b = b.view(c, h // 2, 2, w // 2, 2).mean((2, 4))
    ref = (1 - lam) * (x - y).abs().mean() + lam * (1 - ms)
    ours = create_photometric(spec)(None, {"rgb": x.permute(1, 2, 0)}, y.permute(1, 2, 0))
    res = {"ours": float(ours), "reference": float(ref), "lambda": lam, "scales": scales}
    if not abs(res["ours"] - res["reference"]) < 1e-9:
        raise AssertionError(f"photometric differs from (1-l)L1 + l(1-MS-SSIM) {res}")
    if not (lam == 0.1 and scales == 3):
        raise AssertionError(f"paper values are lambda 0.1 / 3 scales {res}")
    return res


# ------------------------------------------------------------------------------------------
# C1 atomized proliferation
# ------------------------------------------------------------------------------------------
def _info(n, device):
    m2d = torch.zeros(n, 3, device=device, requires_grad=True)
    m2d.sum().backward()
    m2d.grad.zero_()
    return {"means2d": m2d, "radii": torch.ones(n, dtype=torch.int32, device=device)}


def atomize(method, device):
    dev = "cpu"
    p, opt = _params(6, dev, seed=5)
    d = _density(method)
    with torch.no_grad():
        p["scales"].copy_(torch.log(torch.tensor([[0.02] * 3, [0.05] * 3, [0.1] * 3,
                                                  [0.2] * 3, [0.4] * 3, [0.8] * 3])))
    d.initialize(p, opt, scene_scale=100.0)
    S = 0.1
    d.state["atom_scale"] = S
    with torch.no_grad():
        p["scales"].copy_(torch.log(torch.tensor([
            [0.05, 0.3, 0.6],   # min < S -> (S,S,S) even though elongated
            [0.1, 0.1, 0.5],    # min == S -> (S,S,S)
            [0.2, 0.3, 0.4],    # unchanged
            [0.11, 0.9, 0.2],   # unchanged
            [0.01, 0.01, 0.01],  # -> (S,S,S)
            [0.5, 0.5, 0.5]])))
    before = p["scales"].exp().clone()
    d.atomize(p, opt)
    after = p["scales"].exp()
    expect = before.clone()
    expect[[0, 1, 4]] = S
    e = float((after - expect).abs().max())
    st = opt["scales"].state[p["scales"]]
    moments = float(st["exp_avg"].abs().max() + st["exp_avg_sq"].abs().max())
    decay = d.state["atom_scale"] / S
    # schedule: atomize at 6900, not at 7000 (official `iteration < atom_proliferation_until`)
    c0 = d.state["counts"]["atomize"]
    d.after_backward(p, opt, 6900, {"info": _info(len(p["means"]), dev)})
    c1 = d.state["counts"]["atomize"]
    d.after_backward(p, opt, 7000, {"info": _info(len(p["means"]), dev)})
    c2 = d.state["counts"]["atomize"]
    res = {"scale_err": e, "moments_after": moments, "decay": decay,
           "atomize_calls_6900": c1 - c0, "atomize_calls_7000": c2 - c1,
           "geometric_ratio_at_7k": 0.99 ** 69}
    if not (e < 1e-6 and moments == 0.0 and abs(decay - 0.99) < 1e-12):
        raise AssertionError(f"atomize wrong {res}")
    if not (c1 - c0 == 1 and c2 - c1 == 0):
        raise AssertionError(f"atomize schedule wrong {res}")
    return res


def _official_densify(t, grads, atom, clone_thr, split_thr, prune_thr, extent, seed):
    """Literal plain-tensor transcription of official densify_and_prune (gaussian_model.py)."""
    t = {k: v.clone() for k, v in t.items()}
    N0 = len(t["means"])
    torch.manual_seed(seed)
    scaling = t["scales"].exp()
    split_mask = (grads >= split_thr) & (scaling.max(dim=1).values > atom)
    stds = scaling[split_mask].repeat(2, 1)
    samples = torch.normal(mean=torch.zeros((stds.size(0), 3)), std=stds)
    rots = _quat_rot(t["quats"][split_mask]).repeat(2, 1, 1)
    new = {k: v[split_mask].repeat(2, *([1] * (v.dim() - 1))) for k, v in t.items()}
    new["means"] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
        t["means"][split_mask].repeat(2, 1)
    new["scales"] = torch.log(scaling[split_mask].repeat(2, 1) / 1.6)
    t = {k: torch.cat([v, new[k]]) for k, v in t.items()}
    keep = ~torch.cat([split_mask, torch.zeros(2 * int(split_mask.sum()), dtype=torch.bool)])
    t = {k: v[keep] for k, v in t.items()}
    padded = torch.zeros(len(t["means"]))
    padded[:N0] = grads
    clone_mask = padded >= clone_thr
    t = {k: torch.cat([v, v[clone_mask]]) for k, v in t.items()}
    prune = (t["opacities"].sigmoid() < prune_thr) | (t["scales"].exp().max(dim=1).values >
                                                      0.1 * extent)
    return {k: v[~prune] for k, v in t.items()}, int(split_mask.sum()), int(clone_mask.sum())


def atom_proliferation(method, device):
    dev = "cpu"
    n = 10
    p, _ = _params(n, dev, seed=7)
    extent = 5.0
    with torch.no_grad():
        p["scales"].copy_(torch.log(torch.tensor([
            [0.3, 0.2, 0.1], [0.05, 0.05, 0.05], [0.4, 0.4, 0.4], [0.02, 0.03, 0.04],
            [0.6, 0.1, 0.1], [0.2, 0.2, 0.2], [0.7, 0.7, 0.7], [0.08, 0.09, 0.07],
            [0.3, 0.3, 0.3], [0.1, 0.5, 0.2]])))
        p["opacities"][3] = math.log(0.004 / 0.996)  # opacity prune
    S = 0.1
    grads = torch.tensor([5e-4, 5e-4, 1e-5, 3e-4, 5e-4, 0.0, 2.5e-4, 1e-3, 1.5e-5, 3e-4])
    # p[6] max 0.7 > 0.1*extent=0.5 -> world prune
    ref_in = {k: v.detach().clone() for k, v in p.items()}

    def run(step, shift=True):
        q = {k: torch.nn.Parameter(v.clone()) for k, v in ref_in.items()}
        qo = {k: torch.optim.Adam([v], lr=1e-3) for k, v in q.items()}
        dd = _density(method, official_clone_index_shift=shift)
        dd.extent, dd.state = extent, {"atom_scale": S, "counts": {"densify": 0}}
        dd.stats = {"grad_accum": grads.clone(), "denom": torch.ones(n)}
        thr = min(2e-4 * step / 7000, 2e-4)
        torch.manual_seed(11)
        dd.densify_and_prune(q, qo, 2e-4, thr)
        return q, dd.state["last"], thr

    out = {}
    for step in (200, 7000):
        q, last, thr = run(step)
        ref, n_split, n_clone = _official_densify(ref_in, grads, S, 2e-4, thr, 0.005, extent, 11)
        err = max(float((q[k].detach() - ref[k]).abs().max()) for k in ref)
        if len(q["means"]) != len(ref["means"]) or err > 1e-6:
            raise AssertionError(f"step {step}: densify differs from official transcription")
        out[f"step_{step}"] = {**last, "ref_split": n_split, "ref_clone": n_clone, "max_err": err}
    # warm-up: grads 1e-5 / 1.5e-5 (index 2, 8; max s > S) split at step 200 but not at 7000;
    # atom constraint: index 1 / 3 / 7 (max s <= S) never split despite grad >= threshold
    if not (out["step_200"]["split"] == 6 and out["step_7000"]["split"] == 4):
        raise AssertionError(f"warm-up / atom-constrained split wrong {out}")
    q_aligned, last_aligned, _ = run(7000, shift=False)
    out["aligned_clone"] = last_aligned["clone"]
    if last_aligned["clone"] == out["step_7000"]["clone"] and \
            len(q_aligned["means"]) == out["step_7000"]["after"]:
        raise AssertionError("fixture cannot distinguish the official clone index shift")
    return out


def init_schedules(method, device):
    dev = "cpu"
    p, opt = _params(200, dev, seed=9)
    g = torch.Generator().manual_seed(1)
    dist = torch.rand(200, generator=g) * 0.5 + 0.01
    with torch.no_grad():
        p["scales"].copy_(dist.log()[:, None].expand(-1, 3))
        p["opacities"].fill_(math.log(0.5 / 0.5))
    d = _density(method)
    d.initialize(p, opt, scene_scale=4.0)
    q = torch.quantile(dist, 0.01).item()
    e_q = abs(d.state["atom_scale"] - q)
    e_clamp = float((p["scales"].exp()[:, 0] - dist.clamp_min(q)).abs().max())
    lr7k, lr20k = d.xyz_lr(7000), d.xyz_lr(20000)
    final = 1.6e-6 * 4.0
    # opacity reset at 300 -> min(o, 0.1); not at 15000 (densify_until)
    d.after_backward(p, opt, 300, {"info": _info(len(p["means"]), dev)})
    o300 = float(p["opacities"].sigmoid().max())
    with torch.no_grad():
        p["opacities"].fill_(0.0)
    d.after_backward(p, opt, 15000, {"info": _info(len(p["means"]), dev)})
    o15k = float(p["opacities"].sigmoid().max())
    guard = False
    p2, opt2 = _params(5, dev)
    with torch.no_grad():
        p2["scales"].fill_(-3.0)
    try:
        _density(method).initialize(p2, opt2, 1.0)
    except ValueError:
        guard = True
    res = {"quantile_err": e_q, "clamp_err": e_clamp, "lr_7000": lr7k, "lr_20000": lr20k,
           "opacity_after_reset_300": o300, "opacity_at_15000": o15k, "recipe_guard": guard}
    if not (e_q < 1e-6 and e_clamp < 1e-6 and abs(lr7k - final) < 1e-12 and lr20k == lr7k):
        raise AssertionError(f"init / xyz schedule wrong {res}")
    if not (abs(o300 - 0.1) < 1e-6 and abs(o15k - 0.5) < 1e-6 and guard):
        raise AssertionError(f"opacity reset schedule wrong {res}")
    if abs(quantile_linear(dist, 0.37) - torch.quantile(dist, 0.37).item()) > 1e-7:
        raise AssertionError("quantile_linear differs from torch.quantile")
    return res
