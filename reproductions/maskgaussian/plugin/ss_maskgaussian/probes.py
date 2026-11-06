"""Mechanism probes for the MaskGaussian port. Each raises on failure and returns measurements."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _component(method, kind, name=None):
    return next(c for c in method.components if c.kind == kind and (name is None
                                                                    or c.name == name))


def _renderer(method, **config):
    from .renderer import create_renderer

    spec = _component(method, "renderer")
    return create_renderer(replace(spec, config={**spec.config, **config}))


def _density(method, **config):
    from .density import create_density

    spec = _component(method, "density")
    return create_density(replace(spec, config={**spec.config, **config}))


def _fixture(device, n=40, seed=5, W=64, H=48):
    """Random anisotropic Gaussians in front of a W x H camera, opacities 0.2..0.9 (no 0.99
    clamp since G <= 1), SH degree 0 colours."""
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([2.0, 1.6, 1.5]) + torch.tensor(
        [-1.0, -0.8, 3.0])
    scales = torch.rand(n, 3, generator=g) * torch.tensor([0.25, 0.15, 0.05]) + 0.03
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.7 + 0.2
    sh0 = (torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8
    vals = {"means": means, "scales": scales.log(), "quats": quats,
            "opacities": torch.logit(opac), "sh0": sh0, "shN": torch.zeros(n, 15, 3),
            "mask_scores": torch.zeros(n, 2)}
    params = {k: v.float().to(device) for k, v in vals.items()}
    camera = {"viewmat": torch.eye(4, device=device), "width": W, "height": H,
              "K": torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], device=device)}
    return params, camera


def _rot(q):
    """Rotation matrices by rotating basis vectors (v' = v + 2w(u x v) + 2u x (u x v))."""
    q = F.normalize(q.double(), dim=-1)
    w, u = q[:, :1], q[:, 1:]
    cols = []
    for e in torch.eye(3, dtype=torch.float64):
        v = e.expand_as(u)
        t = 2 * torch.cross(u, v, dim=-1)
        cols.append(v + w * t + torch.cross(u, t, dim=-1))
    return torch.stack(cols, dim=-1)


def _project(params, camera):
    """Independent float64 EWA projection with the Inria conventions (forward.cu)."""
    dd = torch.float64
    p = {k: v.detach().cpu().to(dd) for k, v in params.items()}
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].cpu().to(dd)
    view = camera["viewmat"].cpu().to(dd)
    t = p["means"] @ view[:3, :3].T + view[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tz = t[:, 2]
    tx = (t[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
    J = torch.zeros(len(t), 2, 3, dtype=dd)
    J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
    R = _rot(p["quats"])
    cov3 = R @ torch.diag_embed(p["scales"].exp() ** 2) @ R.transpose(1, 2)
    Wr = view[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2) + 0.3 * torch.eye(2, dtype=dd)
    conic = torch.linalg.inv(cov2)
    ndc = torch.stack([fx * t[:, 0] / tz * 2 / W + 2 * cx / W - 1,
                       fy * t[:, 1] / tz * 2 / H + 2 * cy / H - 1], -1)
    xy = ((ndc + 1) * torch.tensor([W, H], dtype=dd) - 1) * 0.5  # ndc2Pix
    opac = p["opacities"].sigmoid()
    col = (C0 * p["sh0"][:, 0] + 0.5).clamp_min(0)
    return {"xy": xy, "conic": conic, "opac": opac, "col": col, "depth": tz, "W": W, "H": H}


def _gauss(proj, i, pix):
    d = proj["xy"][i] - pix
    a, b, c = proj["conic"][i, 0, 0], proj["conic"][i, 0, 1], proj["conic"][i, 1, 1]
    power = -0.5 * (a * d[:, 0] ** 2 + c * d[:, 1] ** 2) - b * d[:, 0] * d[:, 1]
    return power, power.exp()


def _pixels(proj):
    dd = torch.float64
    ys, xs = torch.meshgrid(torch.arange(proj["H"], dtype=dd), torch.arange(proj["W"], dtype=dd),
                            indexing="ij")
    return torch.stack([xs, ys], -1).reshape(-1, 2)


def _dense_autograd(proj, masks, bg, weights):
    """Eq. 3-4 in float64, every Gaussian at every pixel in depth order with forward.cu's rules
    (power > 0 skipped, alpha < 1/255 skipped, stop when T(1 - M alpha) < 1e-4). Autograd of
    L = sum_p w_p . C_p gives the exact dL/dM and dL/dopacity (incl. the background term)."""
    dd = torch.float64
    pix = _pixels(proj)
    M = masks.detach().cpu().to(dd).clone().requires_grad_(True)
    o = proj["opac"].clone().requires_grad_(True)
    T = torch.ones(len(pix), dtype=dd)
    C = torch.zeros(len(pix), 3, dtype=dd)
    alive = torch.ones(len(pix), dtype=torch.bool)
    for i in torch.argsort(proj["depth"]).tolist():
        if proj["depth"][i] <= 0.2:
            continue
        power, G = _gauss(proj, i, pix)
        alpha = o[i] * G
        valid = alive & (power <= 0) & (alpha.detach() >= 1 / 255)
        test_T = T * (1 - M[i] * alpha)
        stop = valid & (test_T.detach() < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        C = C + torch.where(use[:, None], proj["col"][i] * (M[i] * alpha * T)[:, None], 0)
        T = torch.where(use, test_T, T)
    C = C + T[:, None] * bg.cpu().to(dd)
    (weights.reshape(-1, 3).to(dd) * C).sum().backward()
    return C.detach().reshape(proj["H"], proj["W"], 3), M.grad, o.grad


def _replica_backward(proj, masks, bg, weights, accum):
    """Independent float64 replica of backward.cu renderCUDA (back to front per pixel).
    accum='official': accum_rec uses the current Gaussian's mask (released kernel);
    accum='exact': uses the mask of the Gaussian behind (supplementary recursion b_i)."""
    dd = torch.float64
    pix = _pixels(proj)
    M = masks.detach().cpu().to(dd).reshape(-1)
    w = weights.reshape(-1, 3).to(dd)
    bgd = (bg.cpu().to(dd)[None] * w).sum(-1)
    T = torch.ones(len(pix), dtype=dd)
    alive = torch.ones(len(pix), dtype=torch.bool)
    rec = []  # forward pass: (i, alpha, G, use, T_i)
    for i in torch.argsort(proj["depth"]).tolist():
        if proj["depth"][i] <= 0.2:
            continue
        power, G = _gauss(proj, i, pix)
        alpha = proj["opac"][i] * G
        valid = alive & (power <= 0) & (alpha >= 1 / 255)
        test_T = T * (1 - M[i] * alpha)
        stop = valid & (test_T < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        rec.append((i, alpha, G, use, T.clone()))
        T = torch.where(use, test_T, T)
    T_final = T
    n = len(M)
    gM = torch.zeros(n, dtype=dd)
    go = torch.zeros(n, dtype=dd)
    acc = torch.zeros(len(pix), 3, dtype=dd)
    last_a = torch.zeros(len(pix), dtype=dd)
    last_c = torch.zeros(len(pix), 3, dtype=dd)
    last_m = torch.zeros(len(pix), dtype=dd)
    for i, alpha, G, use, Ti in reversed(rec):
        m = M[i]
        mm = (last_m if accum == "exact" else m.expand_as(last_m))[:, None]
        new_acc = mm * last_a[:, None] * last_c + (1 - mm * last_a[:, None]) * acc
        acc = torch.where(use[:, None], new_acc, acc)
        c = proj["col"][i][None]
        dot = ((c - acc) * w).sum(-1)
        dmask = alpha * Ti * dot - alpha * T_final / (1 - m * alpha) * bgd
        dalpha = m * Ti * dot - m * T_final / (1 - m * alpha) * bgd
        gM[i] = torch.where(use, dmask, 0).sum()
        go[i] = torch.where(use, G * dalpha, 0).sum()
        last_a = torch.where(use, alpha, last_a)
        last_c = torch.where(use[:, None], c.expand_as(last_c), last_c)
        last_m = torch.where(use, m.expand_as(last_m), last_m)
    return gM, go


def _kernel(params, camera, masks, bg, weights, accum):
    from .raster import rasterize

    p = {k: v.detach().clone() for k, v in params.items()}
    p["opacities"].requires_grad_(True)
    m = masks.detach().clone().float().requires_grad_(True)
    color, _, radii, _ = rasterize(p, camera, 0, bg, m, accum)
    (color.permute(1, 2, 0) * weights).sum().backward()
    o = p["opacities"].detach().sigmoid()
    g_opac = p["opacities"].grad / (o * (1 - o))  # back to d/d(sigmoid opacity)
    return color.detach().permute(1, 2, 0), m.grad[:, 0], g_opac, radii


def _rel(a, b):
    a, b = a.detach().cpu().double().reshape(-1), b.detach().cpu().double().reshape(-1)
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


# ---------------------------------------------------------------------------- C1 (CUDA)
def raster_exact(method, device):
    _cuda(device)
    params, cam = _fixture(device)
    n = len(params["means"])
    g = torch.Generator().manual_seed(11)
    masks = (torch.rand(n, 1, generator=g) > 0.4).float().to(device)
    bg = torch.tensor([0.2, 0.5, 0.3], device=device)
    weights = (torch.rand(cam["height"], cam["width"], 3, generator=g) - 0.5).to(device)
    proj = _project(params, cam)
    ref_img, ref_gM, ref_go = _dense_autograd(proj, masks, bg, weights.cpu())
    out = {"gaussians": n, "masked": int((masks == 0).sum())}
    tol = 2e-3
    for accum in ("exact", "official"):
        img, gM, go, radii = _kernel(params, cam, masks, bg, weights, accum)
        e_img = float((img.cpu().double() - ref_img).abs().max())
        if e_img > 2e-4:
            raise AssertionError(f"{accum}: masked image differs from Eq. 3-4 by {e_img:.2e}")
        rep_gM, rep_go = _replica_backward(proj, masks, bg, weights.cpu(), accum)
        e_m, e_o = _rel(gM, rep_gM), _rel(go, rep_go)
        if e_m > tol or e_o > tol:
            raise AssertionError(f"{accum}: kernel dL/dM or dL/do differ from the backward.cu "
                                 f"replica ({e_m:.2e}, {e_o:.2e})")
        out[f"{accum}_image_maxabs"] = e_img
        out[f"{accum}_mask_grad_rel"] = e_m
        out[f"{accum}_opacity_grad_rel"] = e_o
        out[f"{accum}_masked_nonzero_grads"] = int((gM[masks[:, 0] == 0] != 0).sum())
        if accum == "exact":
            # the exact replica is itself checked against autograd of Eq. 3-4
            a_m, a_o = _rel(rep_gM, ref_gM), _rel(rep_go, ref_go)
            k_m = _rel(gM, ref_gM)
            if a_m > 1e-9 or a_o > 1e-9 or k_m > tol:
                raise AssertionError(f"exact: replica/kernel differ from autograd ({a_m:.2e}, "
                                     f"{a_o:.2e}, kernel {k_m:.2e})")
            out["exact_kernel_vs_autograd_mask_rel"] = k_m
        else:
            out["official_vs_autograd_mask_rel"] = _rel(gM, ref_gM)
    # official == exact when no adjacent masks differ (all present)
    ones = torch.ones_like(masks)
    _, gM1, _, _ = _kernel(params, cam, ones, bg, weights, "official")
    _, gM2, _, _ = _kernel(params, cam, ones, bg, weights, "exact")
    _, ref1, _ = _dense_autograd(proj, ones, bg, weights.cpu())
    out["all_present_official_vs_autograd_rel"] = _rel(gM1, ref1)
    if _rel(gM1, ref1) > tol or _rel(gM2, ref1) > tol:
        raise AssertionError("with M = 1 both builds must equal Eq. 7")
    if out["official_vs_autograd_mask_rel"] < 1e-3:
        raise AssertionError("official build expected to deviate from Eq. 7 with mixed masks")
    return out


def _unsampled_check(renderer, params, cam, seed=3):
    """Render through the renderer's own Gumbel sampling (training mode) and check that
    unsampled, visible Gaussians (a) do not change the image and (b) get nonzero mask-score
    gradients. Returns measurements; raises if (b) fails."""
    from splatstudio.losses import ssim

    p = {k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()}
    with torch.no_grad():  # p(present) = 0.5 for every Gaussian
        p["mask_scores"].copy_(-torch.tensor(renderer.cfg["score_offset"], device=p["means"].device)
                               .expand_as(p["mask_scores"]))
    torch.manual_seed(seed)
    out = renderer.render(p, cam, sh_degree=0)
    hard = out["info"]["mask_hard"]
    vis = out["info"]["radii"] > 0
    target = torch.rand_like(out["rgb"])
    loss = 0.8 * (out["rgb"] - target).abs().mean() + 0.2 * (1 - ssim(out["rgb"], target))
    loss.backward()
    g = p["mask_scores"].grad
    sel = (~hard) & vis
    if int(sel.sum()) == 0:
        raise AssertionError("fixture has no visible unsampled Gaussian")
    nonzero = int((g[sel].abs().sum(-1) > 0).sum())
    with torch.no_grad():  # unsampled Gaussians act as pruned: image equals render without them
        kept = {k: v[hard].detach() for k, v in p.items()}
        img_pruned = renderer.render(kept, cam, sh_degree=0,
                                     masks=torch.ones((int(hard.sum()), 1),
                                                      device=p["means"].device))["rgb"]
    e_img = float((out["rgb"].detach() - img_pruned).abs().max())
    res = {"visible_unsampled": int(sel.sum()), "nonzero_mask_grads": nonzero,
           "unsampled_grad_mean_abs": float(g[sel].abs().mean()), "image_vs_pruned": e_img}
    if e_img > 1e-5:
        raise AssertionError(f"masked Gaussians changed the image ({e_img:.2e})")
    if nonzero < 0.9 * int(sel.sum()):
        raise AssertionError(f"only {nonzero}/{int(sel.sum())} visible unsampled Gaussians got "
                             "a mask gradient")
    return res


def unsampled_gradient(method, device):
    _cuda(device)
    params, cam = _fixture(device, n=60, seed=8)
    out = {"masked_raster": _unsampled_check(_renderer(method, mask_mode="raster"), params, cam)}
    try:
        naive = _unsampled_check(_renderer(method, mask_mode="opacity"), params, cam)
    except AssertionError as exc:
        out["naive_opacity_mask"] = f"fails as expected: {exc}"
    else:
        raise AssertionError(f"check did not discriminate: naive opacity mask passed {naive}")
    return out


# ---------------------------------------------------------------------------- C2
def gumbel(method, device):
    from .renderer import presence_probability, sample_mask

    cfg = _component(method, "renderer").config
    dens = _component(method, "density").config
    if list(dens["score_offset"]) != list(cfg["score_offset"]) or dens["tau"] != cfg["tau"]:
        raise AssertionError("density and renderer must sample the same mask distribution")
    offset, tau = cfg["score_offset"], float(cfg["tau"])
    p0 = float(presence_probability(torch.zeros(1, 2), offset)[0])
    if abs(p0 - 1 / (1 + math.exp(-9.0))) > 1e-6:
        raise AssertionError(f"init presence {p0} != sigmoid(9) (official [10, 1])")
    out = {"init_presence": p0, "rates": {}}
    torch.manual_seed(0)
    draws = 20000
    for d in (-3.0, -0.7, 0.0, 1.2, 4.0):  # logit difference l0 - l1
        s = torch.tensor([[d, 0.0]]) - torch.tensor(offset)
        s = s.expand(draws, 2).contiguous()
        p = float(presence_probability(s[:1], offset)[0])
        rate = float(sample_mask(s, offset, tau).mean())
        sigma = math.sqrt(p * (1 - p) / draws)
        if abs(rate - p) > 4 * sigma + 1e-4:
            raise AssertionError(f"P(M=1) {rate:.4f} != softmax {p:.4f} (logit gap {d})")
        out["rates"][str(d)] = [rate, p]
    # straight-through: forward is one-hot, gradient = soft-sample Jacobian (2 classes:
    # d y0/d s0 = y0 (1 - y0) / tau = -d y0/d s1, with y0 >= 0.5 iff M = 1)
    s = torch.nn.Parameter(torch.randn(256, 2))
    m = sample_mask(s, offset, tau)
    if not torch.all((m == 0) | (m == 1)):
        raise AssertionError("mask is not binary in the forward pass")
    m.sum().backward()
    g0, g1 = s.grad[:, 0] * tau, s.grad[:, 1] * tau
    if not torch.allclose(g0, -g1, atol=1e-6) or (g0 <= 0).any() or (g0 > 0.25 + 1e-6).any():
        raise AssertionError("ST gradient is not the soft Gumbel-Softmax Jacobian")
    y0 = 0.5 + torch.where(m[:, 0] > 0, 1.0, -1.0) * torch.sqrt((1 - 4 * g0).clamp_min(0)) / 2
    resample = (y0 >= 0.5 - 1e-4).float()
    if not torch.equal(resample, m[:, 0]):
        raise AssertionError("soft sample implied by the gradient disagrees with the hard mask")
    out["st_grad_range"] = [float(g0.min()), float(g0.max())]
    return out


# ---------------------------------------------------------------------------- C3
def mask_loss(method, device):
    from .losses import create_mask_loss

    spec = _component(method, "loss", "maskgs-mask-loss")
    cfg = spec.config
    fn = create_mask_loss(spec)
    m = torch.tensor([[1.0], [0.0], [1.0], [1.0], [0.0]], requires_grad=True)
    lam, lo, hi = float(cfg["lambda_mask"]), int(cfg["mask_from"]), int(cfg["mask_until"])
    out = {}
    for step, on in ((lo - 1, False), (lo, True), ((lo + hi) // 2, True), (hi, True),
                     (hi + 1, False)):
        if step < 1 and not on:
            continue
        m.grad = None
        v = fn({}, {"step": step, "info": {"mask": m}}, None)
        expect = lam * 0.6 ** 2 if on else 0.0
        if abs(float(v) - expect) > 1e-7:
            raise AssertionError(f"mask loss {float(v)} != {expect} at step {step}")
        v.backward()
        gexp = 2 * lam * 0.6 / 5 if on else 0.0
        if not torch.allclose(m.grad, torch.full_like(m, gexp), atol=1e-8):
            raise AssertionError(f"mask loss gradient wrong at step {step}")
        out[str(step)] = float(v)
    return out


# ---------------------------------------------------------------------------- C4
def _optimizable(params):
    params = {k: torch.nn.Parameter(v.clone()) for k, v in params.items()}
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for k, p in params.items():  # allocate Adam moments, then restore the values
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return torch.nn.ParameterDict(params), opts


def _outputs(n, device):
    m2 = torch.zeros(n, 3, device=device, requires_grad=True)
    m2.grad = torch.zeros_like(m2)
    return {"info": {"means2d": m2, "radii": torch.zeros(n, device=device)}}


def _scene(method, device, present):
    """20 Gaussians; `present` [20] bool: True -> p(present) ~ 1, False -> ~ e^-30."""
    params, _ = _fixture(device, n=20, seed=2)
    offset = torch.tensor(_component(method, "density").config["score_offset"], device=device)
    logits = torch.where(present[:, None].to(device), torch.tensor([10.0, 1.0], device=device),
                         torch.tensor([-30.0, 1.0], device=device))
    params["mask_scores"] = logits - offset
    params["opacities"] = torch.zeros(20, device=device)  # o = 0.5
    params["scales"] = torch.full((20, 3), math.log(0.05), device=device)
    return _optimizable(params)


def prune_schedule(method, device):
    from splatstudio.state import check_state

    from .density import mask_prune_steps

    cfg = _component(method, "density").config
    total = int(cfg["total_steps"])
    steps = mask_prune_steps(cfg, total)
    if steps[0] != cfg["densify_until"] or steps[-1] != total or any(
            b - a != cfg["mask_prune_every"] for a, b in zip(steps, steps[1:])):
        raise AssertionError(f"mask-prune steps {steps[:3]}..{steps[-2:]} wrong")
    present = torch.arange(20) % 2 == 0
    out = {"mask_prune_steps": [steps[0], steps[1], steps[-1]]}

    def run(step_list, dens=None, prm=None, opt=None):
        if dens is None:
            prm, opt = _scene(method, device, present)
            dens = _density(method)
            dens.initialize(prm, opt, 1.0)
        counts = []
        for s in step_list:
            dens.after_backward(prm, opt, s, _outputs(len(prm["means"]), device))
            check_state(method, prm, opt)
            counts.append(len(prm["means"]))
        return counts, prm, opt, dens

    # no pruning on non-densify steps during densification and off-cadence steps after
    c, *_ = run([1, 499, 550, 601, 15001, 15500, 15999])
    if c != [20] * 7:
        raise AssertionError(f"pruned off-schedule: {c}")
    # densify step: never-sampled removed (no densification since gradients are zero)
    c, prm, *_ = run([600])
    keep_p = F.softmax(prm["mask_scores"] + torch.tensor(cfg["score_offset"], device=device),
                       -1)[:, 0]
    if c != [10] or (keep_p < 0.99).any():
        raise AssertionError(f"densify_and_prune mask test wrong: {c}")
    # after densification: only at multiples of 1000
    c, *_ = run([15999, 16000])
    if c != [20, 10]:
        raise AssertionError(f"mask_prune at 16000 wrong: {c}")
    # control: with the mask test disabled the densify step keeps all
    prm, opt = _scene(method, device, present)
    d = _density(method, mask_prune_in_densify=False)
    d.initialize(prm, opt, 1.0)
    d.after_backward(prm, opt, 600, _outputs(20, device))
    if len(prm["means"]) != 20:
        raise AssertionError("control without mask test should keep all Gaussians")
    # final step keeps a single sample of p = 0.5 masks
    prm, opt = _scene(method, device, torch.ones(20, dtype=torch.bool))
    with torch.no_grad():
        prm["mask_scores"].copy_(-torch.tensor(cfg["score_offset"], device=device).expand(20, 2))
    d = _density(method)
    d.initialize(prm, opt, 1.0)
    torch.manual_seed(1)
    d.after_backward(prm, opt, total - 1, _outputs(20, device))
    n_before = len(prm["means"])
    d.after_backward(prm, opt, total, _outputs(20, device))
    n_final = len(prm["means"])
    if n_before != 20 or not 2 <= n_final <= 18:
        raise AssertionError(f"final single-sample keep wrong: {n_before} -> {n_final}")
    out["final_sample_kept"] = n_final
    # clone/split copy the mask scores
    prm, opt = _scene(method, device, torch.ones(20, dtype=torch.bool))
    with torch.no_grad():
        prm["scales"][:5] = math.log(0.005)  # <= percent_dense * extent -> clone
        prm["mask_scores"][:, 0] = torch.linspace(0, 1, 20, device=device)
    d = _density(method)
    d.initialize(prm, opt, 1.0)
    d.vgn[:] = 0
    d.vgn[[0, 1, 10, 11]] = 1.0  # 0,1 clone; 10,11 split
    d.denom[:] = 1
    src = prm["mask_scores"].detach().clone()
    d.after_backward(prm, opt, 600, _outputs(20, device))
    check_state(method, prm, opt)
    ms = prm["mask_scores"].detach()
    if len(ms) != 20 + 2 + 2:
        raise AssertionError(f"expected 24 Gaussians after clone 2 + split 2, got {len(ms)}")
    expect_new = torch.cat([src[[0, 1]], src[[10, 11]].repeat(2, 1)])
    if not torch.allclose(ms[-6:], expect_new):
        raise AssertionError("mask scores not copied by clone/split")
    out["clone_split_rows"] = len(ms)
    return out
