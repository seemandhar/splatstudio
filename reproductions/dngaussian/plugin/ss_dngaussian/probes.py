"""Mechanism probes for DNGaussian; each raises if its mechanism is removed or wrong."""

import math
import random

import torch
import torch.nn.functional as F
from splatstudio.state import sample_scene

from . import neural
from .depth import DepthTable, depth_regularization, global_loss, local_loss, smoothness


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _density(method):
    from .density import create_density

    return create_density(_spec(method, "dngaussian-density"))


def _renderer(method):
    from .renderer import create_renderer

    return create_renderer(_spec(method, "dngaussian-renderer"))


def _need_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


# ------------------------------------------------------------------ C3
def _ref_patch_loss(depth, mono, p, margin, use_global):
    """Independent loop: per-patch (x - mean) / (std + 0.01 * std_all), masked L2."""
    H, W = depth.shape[-2:]
    a_rows, b_rows = [], []
    for i in range(H // p):
        for j in range(W // p):
            a_rows.append(depth[0, 0, i * p:(i + 1) * p, j * p:(j + 1) * p].reshape(-1))
            b_rows.append(mono[0, 0, i * p:(i + 1) * p, j * p:(j + 1) * p].reshape(-1))
    out = []
    for rows, full in ((a_rows, depth), (b_rows, mono)):
        m = torch.stack(rows).double()
        eps = 0.01 * m.reshape(-1).std()
        std = full.double().std() if use_global else m.std(dim=1, keepdim=True)
        out.append((m - m.mean(dim=1, keepdim=True)) / (std + eps))
    diff = out[0] - out[1]
    sel = diff.abs() > margin
    return float(diff[sel].square().mean())


def glnorm_formula(method, device):
    g = torch.Generator().manual_seed(0)
    depth = torch.rand(1, 1, 37, 50, generator=g) * 20 + 5
    mono = torch.rand(1, 1, 37, 50, generator=g) * 3
    margin = float(_spec(method, "dngaussian-density").config["error_tolerance"])
    if margin != 0.00025:
        raise AssertionError(f"error tolerance {margin} != official 0.00025")
    res = {}
    for p in (5, 11, 17):
        loc = float(local_loss(depth, mono, p, margin))
        glo = float(global_loss(depth, mono, p, margin))
        ref_l = _ref_patch_loss(depth, mono, p, margin, False)
        ref_g = _ref_patch_loss(depth, mono, p, margin, True)
        if not (math.isclose(loc, ref_l, rel_tol=1e-4) and math.isclose(glo, ref_g, rel_tol=1e-4)):
            raise AssertionError(f"p={p}: local {loc}/{ref_l}, global {glo}/{ref_g}")
        if math.isclose(loc, glo, rel_tol=1e-3):
            raise AssertionError("local and global normalisation must differ")
        res[f"p{p}"] = {"local": loc, "global": glo}
    # scale / shift invariance of both terms (the DPT map is relative)
    for fn in (local_loss, global_loss):
        a = float(fn(depth, mono, 7, margin))
        b = float(fn(depth * 3.7 + 11.0, mono * 0.2 - 4.0, 7, margin))
        if not math.isclose(a, b, rel_tol=1e-3):
            raise AssertionError(f"{fn.__name__} not affine invariant: {a} vs {b}")
    # tolerance: identical maps up to an affine map -> loss 0; huge tolerance masks everything
    if float(local_loss(depth, depth * 2 + 1, 5, margin)) > 1e-8:
        raise AssertionError("affine-equal maps must give zero local loss")
    if float(global_loss(depth, mono, 5, 1e6)) != 0.0:
        raise AssertionError("error tolerance must mask small differences")
    # combination of one pass: 0.1 local + 1.0 global (+ 0.1 smoothness after 3000)
    cfg = dict(_spec(method, "dngaussian-density").config)
    r1, r2 = random.Random(3), random.Random(3)
    tot, terms = depth_regularization(depth, mono, 3000, r1, cfg)
    p_l, p_g = r2.randint(5, 17), r2.randint(5, 17)
    exp = 0.1 * float(local_loss(depth, mono, p_l, margin)) + float(global_loss(depth, mono, p_g, margin))
    if not math.isclose(float(tot), exp, rel_tol=1e-5) or "smooth" in terms:
        raise AssertionError("pass combination at 3000 must be 0.1 local + global")
    r1 = random.Random(3)
    tot2, terms2 = depth_regularization(depth, mono, 3001, r1, cfg)
    exp2 = exp + 0.1 * float(smoothness(depth, mono))
    if not math.isclose(float(tot2), exp2, rel_tol=1e-5):
        raise AssertionError("smoothness (0.1) must join after iteration 3000")
    res["pass_3001"] = terms2
    return res


# ------------------------------------------------------------------ C1 / C2
def _one_splat(device, z=4.0, opacity_logit=0.3):
    params, _ = sample_scene(_PROBE_METHOD[0], device, n=2)
    with torch.no_grad():
        params["means"].copy_(torch.tensor([[0.0, 0.0, z], [50.0, 50.0, z]], device=device))
        params["scales"].fill_(math.log(0.05))
        params["quats"].copy_(torch.tensor([[1.0, 0, 0, 0]] * 2, device=device))
        params["opacities"].fill_(opacity_logit)
    # pixel (8, 8) centre is (8.5, 8.5): put the principal point there so the mean projects on it
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[40.0, 0, 8.5], [0, 40.0, 8.5], [0, 0, 1]], device=device),
              "width": 17, "height": 17}
    return params, camera


_PROBE_METHOD = [None]


def _grad_fields(params):
    return sorted(k for k, p in params.items() if p.grad is not None and p.grad.abs().sum() > 0)


def hard_depth_probe(method, device):
    _need_cuda()
    from .renderer import hard_depth

    _PROBE_METHOD[0] = method
    params, cam = _one_splat(device)
    tau = float(_spec(method, "dngaussian-density").config["tau"])
    if tau != 0.95:
        raise AssertionError(f"tau {tau} != 0.95")
    depth, _, _ = hard_depth(params, cam, tau)
    centre = float(depth[0, 0, 8, 8])
    if not math.isclose(centre, tau * 4.0, rel_tol=1e-3):
        raise AssertionError(f"hard depth at the splat centre {centre} != tau * z = {tau * 4}")
    target = torch.linspace(0, 1, 17, device=device).expand(17, 17)
    (depth[0, 0] - target).square().mean().backward()
    got = _grad_fields(params)
    if got != ["means"]:
        raise AssertionError(f"hard depth gradients reached {got}, expected only means")
    return {"centre_depth": centre, "expected": tau * 4.0, "grad_fields": got}


def soft_depth_probe(method, device):
    _need_cuda()
    from .renderer import soft_depth

    _PROBE_METHOD[0] = method
    params, cam = _one_splat(device)
    depth, _, _ = soft_depth(params, cam)
    expect = torch.sigmoid(torch.tensor(0.3)).item() * 4.0
    centre = float(depth[0, 0, 8, 8])
    if not math.isclose(centre, expect, rel_tol=1e-3):
        raise AssertionError(f"soft depth {centre} != sigmoid(o) z = {expect}")
    depth[0, 0].sum().backward()
    got = _grad_fields(params)
    if got != ["opacities"]:
        raise AssertionError(f"soft depth gradients reached {got}, expected only opacities")
    return {"centre_depth": centre, "expected": expect, "grad_fields": got}


def _scene(method, device, n=400, seed=0):
    from splatstudio.state import initialize

    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([4.0, 3.0, 2.0])
    pts[:, 2] += 6.0
    params, opts = initialize(method, pts.to(device), torch.full((n, 3), 0.5, device=device))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.08))
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]], device=device),
           "width": 64, "height": 48}
    return params, opts, cam


def depth_pass_step(method, device):
    """The density's own depth passes: hard updates only means, soft only opacities."""
    _need_cuda()
    dens = _density(method)
    params, opts, cam = _scene(method, device)
    dens.initialize(params, opts, 5.0)
    mono = (torch.linspace(0, 255, 64, device=device)[None, :].expand(48, 64)
            + 5 * torch.rand(48, 64, device=device))
    dens.table = DepthTable(cam["viewmat"].cpu()[None].numpy(), mono.cpu()[None].numpy(),
                            torch.zeros(1, 3).numpy())
    before = {k: v.detach().clone() for k, v in params.items()}
    m = dens.table.lookup(cam["viewmat"], device)
    out_h = dens.depth_pass(params, opts, cam, m, 10, "hard")
    changed = sorted(k for k in params if not torch.equal(params[k], before[k]))
    if changed != ["means"]:
        raise AssertionError(f"hard pass changed {changed}")
    before = {k: v.detach().clone() for k, v in params.items()}
    out_s = dens.depth_pass(params, opts, cam, m, 1001, "soft")
    changed = sorted(k for k in params if not torch.equal(params[k], before[k]))
    if changed != ["opacities"]:
        raise AssertionError(f"soft pass changed {changed}")
    # schedule inside after_backward: soft only for iteration > 1000, smoothness > 3000
    cfg = dens.cfg
    if (cfg["soft_depth_start"], cfg["smooth_start"], cfg["hard_depth_start"]) != (1000, 3000, 0):
        raise AssertionError("depth schedule differs from the official 0 / 1000 / 3000")
    if "smooth" in out_h or "smooth" in out_s:
        raise AssertionError("smoothness must not be active before iteration 3001")
    neural.ACTIVE.update(net=None, baked=False)
    return {"hard": out_h, "soft": out_s}


# ------------------------------------------------------------------ C4
def _ref_hash_encode(enc, x01):
    """Independent torch hash-grid (torch-ngp 'hash', linear, align_corners False)."""
    primes = [1, 2654435761, 805459861]
    offs = enc.offsets.tolist()
    S = math.log2(enc.per_level_scale)
    outs = []
    for lvl in range(enc.num_levels):
        size = offs[lvl + 1] - offs[lvl]
        # float32 like the kernel: scale = exp2f(level * S) * H - 1, pos = x * scale + 0.5
        scale = torch.exp2(torch.tensor(lvl * S, dtype=torch.float32)) * enc.base_resolution - 1.0
        res = math.ceil(float(scale)) + 1
        pos = x01.float() * scale.to(x01.device) + 0.5
        g0 = torch.floor(pos)
        frac = pos - g0
        g0 = g0.long()
        acc = torch.zeros(len(x01), enc.level_dim, dtype=torch.float64, device=x01.device)
        for corner in range(8):
            w = torch.ones(len(x01), dtype=torch.float64, device=x01.device)
            idx_d = []
            for d in range(3):
                bit = (corner >> d) & 1
                w = w * (frac[:, d] if bit else 1 - frac[:, d]).double()
                idx_d.append(g0[:, d] + bit)
            dense = (res + 1) ** 3 <= size
            if dense:
                index = idx_d[0] + idx_d[1] * (res + 1) + idx_d[2] * (res + 1) ** 2
            else:
                h = torch.zeros_like(idx_d[0])
                for d in range(3):
                    h = h ^ ((idx_d[d] * primes[d]) & 0xFFFFFFFF)
                index = h
            index = index % size
            acc += w[:, None] * enc.embeddings.detach()[offs[lvl] + index].double()
        outs.append(acc)
    return torch.cat(outs, dim=1)


def _ref_sh4(d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack([
        torch.full_like(x, 0.28209479177387814), -0.48860251190291987 * y,
        0.48860251190291987 * z, -0.48860251190291987 * x,
        1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
        0.94617469575755997 * zz - 0.31539156525251999, -1.0925484305920792 * x * z,
        0.54627421529603959 * xx - 0.54627421529603959 * yy,
        0.59004358992664352 * y * (-3.0 * xx + yy), 2.8906114426405538 * x * y * z,
        0.45704579946446572 * y * (1.0 - 5.0 * zz), 0.3731763325901154 * z * (5.0 * zz - 3.0),
        0.45704579946446572 * x * (1.0 - 5.0 * zz), 1.4453057213202769 * z * (xx - yy),
        0.59004358992664352 * x * (-xx + 3.0 * yy)], dim=1)


def neural_color(method, device):
    _need_cuda()
    torch.manual_seed(0)
    xyz = (torch.rand(3000, 3, device=device) - 0.5) * 40 + 7
    net = neural.GridRenderer.from_points(xyz)
    if abs(float(net.bound) - float((xyz.max(0).values - xyz.min(0).values).max()) / 2 * 1.2) > 1e-4:
        raise AssertionError("hash-grid bound must be max extent / 2 * 1.2 of the initial cloud")
    with torch.no_grad():
        net.encoder_x.embeddings.uniform_(-1, 1)
    pts = (torch.rand(500, 3, device=device) - 0.5) * 30 + 7
    x01 = ((pts - net.coord_center) + net.bound) / (2 * net.bound)
    ours = net.encoder_x(pts - net.coord_center, bound=net.bound).double()
    ref = _ref_hash_encode(net.encoder_x, x01)
    # Fine levels reach resolution ~1e4, where float32 positions (the kernel contracts
    # x * scale + 0.5 into an FMA) carry ~5e-4 cell error; a wrong index or weight gives O(1).
    err = float((ours - ref).abs().max())
    mean_err = float((ours - ref).abs().mean())
    if err > 5e-3 or mean_err > 1e-4:
        raise AssertionError(f"CUDA hash grid differs from the reference: max {err}, mean {mean_err}")
    d = F.normalize(torch.randn(256, 3, device=device), dim=1)
    sh_err = float((neural.sh_encode(d) - _ref_sh4(d)).abs().max())
    if sh_err > 1e-5:
        raise AssertionError(f"SH4 encoder differs from the formula: {sh_err}")
    # colour gradient reaches the centres (official: the hash grid input requires grad)
    means = pts.clone().requires_grad_(True)
    cam = torch.eye(4, device=device)
    col = net(means, neural.view_dirs(means, cam))
    col.sum().backward()
    if means.grad is None or not torch.count_nonzero(means.grad):
        raise AssertionError("neural colours must be differentiable w.r.t. the centres")
    if not (col.min() >= -0.001 - 1e-6 and col.max() <= 1.001 + 1e-6):
        raise AssertionError("colour head must be sigmoid * 1.002 - 0.001")
    # bake / unpack reproduces live colours
    params, _ = sample_scene(method, device, n=700)
    with torch.no_grad():
        params["means"].copy_(pts[:1].expand(700, 3) + torch.randn(700, 3, device=device))
    live = net(params["means"], neural.view_dirs(params["means"], cam)).detach()
    neural.bake(params, net)
    mlp, deg = neural.unpack_color_net(params)
    d = neural.view_dirs(params["means"], cam)
    baked = neural.color_head(mlp, torch.cat([neural.sh_encode(d, deg), params["dng_geo"]], -1))
    berr = float((baked - live).abs().max())
    if berr > 1e-5:
        raise AssertionError(f"baked colours differ from live colours by {berr}")
    with torch.no_grad():
        params["means"][0, 0] += 1.0
    try:
        neural.unpack_color_net(params)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a stale bake must be rejected")
    return {"hash_max_err": err, "hash_mean_err": mean_err, "sh_max_err": sh_err, "bake_max_err": berr,
            "levels": net.encoder_x.num_levels, "per_level_scale": net.encoder_x.per_level_scale}


# ------------------------------------------------------------------ C5
def official_density(method, device):
    _need_cuda()
    from .density import expon_lr

    dens = _density(method)
    params, opts, cam = _scene(method, device, n=40)
    dens.initialize(params, opts, 10.0)
    ext = 10.0
    lr = [dens.means_lr(t) for t in (1, 500, 501, 3250, 6000)]
    exp = [0.016 * ext, 0.016 * ext, expon_lr(1, 0.016, 0.00016, 5500) * ext,
           math.exp(0.5 * math.log(0.016) + 0.5 * math.log(0.00016)) * ext, 0.00016 * ext]
    if not all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(lr, exp)):
        raise AssertionError(f"xyz LR schedule {lr} != {exp}")
    if opts["scales"].param_groups[0]["lr"] != 0.003:
        raise AssertionError("scaling lr must be 0.003")
    with torch.no_grad():
        logit = torch.logit(torch.tensor(0.05)).item()
        params["opacities"].fill_(torch.logit(torch.tensor(0.5)).item())
        params["opacities"][:10] = logit           # transparent: never cloned / split
        params["scales"][:20] = math.log(0.05)     # small (<= 0.01 * 10): clone
        params["scales"][20:] = math.log(0.5)      # large: split
        params["opacities"][30:32] = torch.logit(torch.tensor(0.005)).item()  # pruned
    dens.state["grad2d"] = torch.full((40,), 1.0, device=device)
    dens.state["count"] = torch.full((40,), 100.0, device=device)  # mean grad 0.01 > 0.0013
    out = dens.densify_and_prune(params, opts)
    # clone: 10 small opaque (10..19); split: 20..39 except 30, 31 (opacity 0.005 < 0.1) -> 18;
    # prune: 30, 31 (opacity < 0.01)
    if out["cloned"] != 10 or out["split"] != 18 or out["pruned"] != 2:
        raise AssertionError(f"opacity-gated densification wrong: {out}")
    n_after = len(params["means"])
    if n_after != 40 + 10 + 18 * 2 - 18 - 2:
        raise AssertionError(f"population {n_after} inconsistent with clone/split/prune")
    # near pruning with spiral centres
    dens.table = DepthTable(cam["viewmat"].cpu()[None].numpy(),
                            torch.zeros(1, 48, 64).numpy(), [[0.0, 0.0, 6.0]])
    d = (params["means"] - torch.tensor([0.0, 0.0, 6.0], device=device)).norm(dim=1)
    dens.cfg["near"] = float(d.median())
    expect = int((d < dens.cfg["near"]).sum())
    got = dens.near_prune(params, opts)
    if got != expect or got == 0:
        raise AssertionError(f"near pruning removed {got}, expected {expect}")
    # invisible pruning: move half behind the camera
    with torch.no_grad():
        params["means"][: len(params["means"]) // 2, 2] = -5.0
    dens.set_training_cameras([cam])
    removed = dens.clean_views(params, opts)
    if removed < len(params["means"]) // 2 - 1 or removed == 0:
        raise AssertionError(f"clean_views removed {removed}")
    neural.ACTIVE.update(net=None, baked=False)
    if dens.cfg["near"] == 10.0:
        raise AssertionError("probe did not exercise the near radius")
    return {"lr": lr, "densify": out, "near_pruned": got, "clean_removed": removed}


def photo_pass_losses(method, device):
    from .losses import create_photometric, create_regularization

    photo = create_photometric(_spec(method, "photometric"))
    reg = create_regularization(_spec(method, "dngaussian-regularization"))
    g = torch.Generator().manual_seed(1)
    a = torch.rand(20, 24, 3, generator=g, requires_grad=True)
    b = torch.rand(20, 24, 3, generator=g)
    # independent 3DGS SSIM (same padding) via avg of Gaussian-weighted stats
    import numpy as np

    k = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5**2))
    k = torch.tensor(np.outer(k, k) / np.outer(k, k).sum(), dtype=torch.float32)
    w = k.expand(3, 1, 11, 11)
    x, y = a.detach().permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mu1, mu2 = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    s1 = F.conv2d(x * x, w, padding=5, groups=3) - mu1**2
    s2 = F.conv2d(y * y, w, padding=5, groups=3) - mu2**2
    s12 = F.conv2d(x * y, w, padding=5, groups=3) - mu1 * mu2
    ssim = (((2 * mu1 * mu2 + 1e-4) * (2 * s12 + 9e-4))
            / ((mu1**2 + mu2**2 + 1e-4) * (s1 + s2 + 9e-4))).mean()
    exp = float((a.detach() - b).abs().mean() + 0.2 * (1 - ssim))
    got = float(photo({}, {"rgb": a}, b))
    if not math.isclose(got, exp, rel_tol=1e-5):
        raise AssertionError(f"photometric {got} != L1 + 0.2 D-SSIM {exp}")
    scales = torch.log(torch.tensor([[1.0, 2.0, 4.0], [0.5, 0.5, 0.5]]))
    op = torch.tensor([0.1, 0.5, 0.9, 0.15])
    shape = (4.0 + 1.0) / 2
    scale = (16 + 0.25) / 2
    opa = 1 - (0.25 + 0.81) / 2 + (0.81 + 0.7225) / 2
    exp_r = 0.002 * shape + 0.001 * scale + 0.001 * opa
    got_r = float(reg({"scales": scales}, {"opacity": op}, None))
    if not math.isclose(got_r, exp_r, rel_tol=1e-6):
        raise AssertionError(f"penalties {got_r} != {exp_r}")
    empty = float(reg({"scales": scales}, {"opacity": torch.full((4,), 0.1)}, None))
    if not math.isfinite(empty):
        raise AssertionError("empty opacity selection must not give NaN")
    return {"photometric": got, "penalties": got_r}


def renderer_gradient(method, device):
    _need_cuda()
    rend = _renderer(method)
    dens = _density(method)
    params, opts, cam = _scene(method, device)
    dens.initialize(params, opts, 5.0)
    out = rend.render(params, cam)
    out["info"]["means2d"].retain_grad()
    target = (torch.arange(64, device=device)[None, :, None] % 2).expand(48, 64, 3).float()
    (out["rgb"] - target).square().mean().backward()
    g = out["info"]["means2d"].grad
    if g is None or not torch.count_nonzero(g):
        raise AssertionError("no screen-space gradient")
    if params["sh0"].grad is not None:
        raise AssertionError("SH must not be used for colour")
    net_grad = sum(float(p.grad.abs().sum()) for p in dens.net.parameters() if p.grad is not None)
    if net_grad == 0:
        raise AssertionError("photometric loss must train the neural colour renderer")
    if float(out["rgb"][0, 0].sum()) > 1e-6 and float(out["alpha"][0, 0]) < 1e-6:
        raise AssertionError("background must be black")
    neural.ACTIVE.update(net=None, baked=False)
    return {"screen_grad_norm": float(g.norm()), "net_grad_l1": net_grad}


# ------------------------------------------------------------------ after_backward schedule
SCHEDULE_STEPS = (1, 2, 500, 600, 999, 1000, 1001, 2000, 2001, 2002, 2025, 2026, 2999, 3000, 3001,
                  5999, 6000)


def expected_schedule(step, total=6000):
    """Independent statement of the official per-iteration gates (train_llff.py), as seen from
    after_backward(step), which runs the depth passes of iteration step + 1."""
    nxt = step + 1
    clean = step in (1, total)
    return {
        "hard": nxt <= total,
        "soft": 1000 < nxt <= total,
        "smooth": 3000 < nxt <= total,
        "near_pruned": (step - 1) % 25 == 0 and step > 2000,
        "cleaned": clean,
        "densify": (not clean) and 500 < step < 6000 and step % 100 == 0,
        "baked": step % 1000 == 0 or step == total,
    }


def observed_schedule(row):
    return {
        "hard": "hard" in row,
        "soft": "soft" in row,
        "smooth": "smooth" in row.get("hard", {}),
        "near_pruned": "near_pruned" in row,
        "cleaned": "cleaned" in row,
        "densify": "cloned" in row,
        "baked": "baked" in row,
    }


def run_schedule(method, device, steps=SCHEDULE_STEPS, overrides=None):
    rend, dens = _renderer(method), _density(method)
    dens.cfg.update(overrides or {})
    params, opts, cam = _scene(method, device, n=800)
    dens.initialize(params, opts, 5.0)
    dens.set_training_cameras([cam])
    mono = (torch.linspace(0, 255, 64, device=device)[None, :].expand(48, 64)
            + 5 * torch.rand(48, 64, device=device))
    # spiral centre far away: near pruning runs but removes nothing
    dens.table = DepthTable(cam["viewmat"].cpu()[None].numpy(), mono.cpu()[None].numpy(),
                            [[0.0, 0.0, -1000.0]])
    target = torch.rand(48, 64, 3, device=device)
    rows = {}
    for step in steps:
        out = rend.render(params, cam)
        out["step"] = step
        dens.before_backward(params, opts, step, out)
        ((out["rgb"] - target).abs().mean()).backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        dens.after_backward(params, opts, step, out)
        rows[step] = observed_schedule(dens.last_row)
    neural.ACTIVE.update(net=None, baked=False)
    return rows


def after_backward_schedule(method, device):
    """Drive the real before/after_backward sequence over the boundary steps."""
    _need_cuda()
    depth_on = bool(_spec(method, "dngaussian-density").config["hard_depth"])
    rows = run_schedule(method, device)
    bad = {}
    for step, got in rows.items():
        exp = expected_schedule(step)
        if not depth_on:
            exp.update(hard=False, soft=False, smooth=False)
        if got != exp:
            bad[step] = {k: (got[k], exp[k]) for k in exp if got[k] != exp[k]}
    if bad:
        raise AssertionError(f"after_backward schedule differs (got, expected): {bad}")
    if depth_on:  # removing a pass must remove its key
        off = run_schedule(method, device, (1001,), {"soft_depth": False})[1001]
        if off["soft"] or not off["hard"]:
            raise AssertionError("soft_depth=False must remove only the soft pass")
    return {"steps": list(rows), "checked_gates": sorted(expected_schedule(1))}
