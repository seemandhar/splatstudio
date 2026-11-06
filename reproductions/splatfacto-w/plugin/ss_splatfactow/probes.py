"""Mechanism probes for Splatfacto-W. Each one compares the implementation with an independently
written formula or a fixture with a known answer, and raises if the mechanism is absent or wrong."""

import math

import numpy as np
import torch
from splatstudio.components import build
from splatstudio.state import check_state, sample_scene

from . import bank as bank_mod
from .bank import SfwBank
from .density import is_densify_step, is_reset_step
from .losses import (
    ALPHA_DEFAULTS,
    PHOTOMETRIC_DEFAULTS,
    RobustMask,
    alpha_loss_value,
    photometric_terms,
)
from .schedule import downscale_factor, lr_at, nerfstudio_normalisation, sh_degree


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _cameras(device, n=3):
    cams = []
    for i in range(n):
        vm = torch.eye(4, device=device)
        vm[0, 3] = 0.05 * i
        vm[1, 3] = -0.03 * i
        K = torch.tensor([[40.0, 0, 16], [0, 40.0, 12], [0, 0, 1]], device=device)
        cams.append({"viewmat": vm, "K": K, "width": 32, "height": 24})
    return cams


def _bank(method, device, cams):
    dspec = _spec(method, "sfw-density")
    cfg = {k[5:]: v for k, v in dspec.config.items() if k.startswith("bank_")}
    bank = SfwBank([c["viewmat"] for c in cams], [c["width"] for c in cams],
                   [c["K"] for c in cams], cfg, device)
    g = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():  # non-zero SH "rest" heads so that degrees >= 1 matter
        for p in bank.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.2).to(device))
    return bank


# -- independent real SH (3DGS / gsplat convention), degree <= 3 -----------------------------
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_by_hand(deg, dirs, sh):
    d = dirs / dirs.norm(dim=-1, keepdim=True)
    x, y, z = d[:, :1], d[:, 1:2], d[:, 2:3]
    out = C0 * sh[:, 0]
    if deg > 0:
        out = out - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if deg > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out = (out + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5]
               + C2[2] * (2 * zz - xx - yy) * sh[:, 6] + C2[3] * xz * sh[:, 7]
               + C2[4] * (xx - yy) * sh[:, 8])
    if deg > 2:
        out = (out + C3[0] * y * (3 * xx - yy) * sh[:, 9] + C3[1] * xy * z * sh[:, 10]
               + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
               + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
               + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + C3[5] * z * (xx - yy) * sh[:, 14]
               + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return out


def mlp_by_hand(head, x):
    lin = [m for m in head.encoder if isinstance(m, torch.nn.Linear)]
    for m in lin:
        x = torch.relu(x @ m.weight.T + m.bias)
    base = x @ head.sh_base_head.weight.T + head.sh_base_head.bias
    rest = x @ head.sh_rest_head.weight.T + head.sh_rest_head.bias
    return torch.cat([base, rest], -1).view(len(x), -1, 3)


def appearance_colour(method, device):
    _cuda()
    from gsplat import rasterization

    renderer, _, _ = build(method)
    cams = _cameras(device)
    params, _ = sample_scene(method, device)
    g = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        params["appearance_features"].copy_(torch.randn(params["appearance_features"].shape,
                                                        generator=g).to(device))
        params["opacities"].fill_(2.0)
    bank = _bank(method, device, cams)
    bank.eval()
    bank.step = 60000
    expect_in = bank.cfg["appearance_embed_dim"] + bank.cfg["appearance_features_dim"]
    first = next(m for m in bank.color_nn.encoder if isinstance(m, torch.nn.Linear))
    if first.in_features != expect_in:
        raise AssertionError("appearance MLP input is not exactly (l_j, f_i): view direction leaked")
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        with torch.no_grad():
            cam = cams[1]
            out = renderer.render(params, cam)
            emb = bank.appearance_embeds.weight[1]
            x = torch.cat([emb[None].expand(len(params["means"]), -1),
                           params["appearance_features"]], -1)
            sh = mlp_by_hand(bank.color_nn, x)
            campos = torch.linalg.inv(cam["viewmat"])[:3, 3]
            dirs = (params["means"] - campos) @ bank.ns_R.T
            colors = torch.clamp_min(sh_by_hand(3, dirs, sh) + 0.5, 0)
            q = params["quats"]
            img, alpha, _ = rasterization(
                params["means"], q / q.norm(dim=-1, keepdim=True), params["scales"].exp(),
                params["opacities"].sigmoid(), colors, cam["viewmat"][None], cam["K"][None],
                cam["width"], cam["height"], near_plane=0.01, far_plane=1e10,
                rasterize_mode="antialiased")
            expected = torch.clamp(img[0] + (1 - alpha[0]) * out["background"], 0, 1)
            err = float((expected - out["rgb"]).abs().max())
            if err > 1e-4:
                raise AssertionError(f"rendered colours differ from the independent formula: {err}")
            if float(alpha.max()) < 0.05:
                raise AssertionError("fixture renders nothing")
            bank.set_override(bank.appearance_embeds.weight[2])
            other = renderer.render(params, cam)["rgb"]
            bank.set_override(None)
            change = float((other - out["rgb"]).abs().max())
            if change < 1e-3:
                raise AssertionError("appearance embedding has no effect on the render")
    finally:
        bank_mod.ACTIVE = prev
    return {"max_abs_error": err, "embedding_change": change, "mlp_in_features": expect_in}


def background_model(method, device):
    _cuda()
    renderer, _, _ = build(method)
    cams = _cameras(device)
    params, _ = sample_scene(method, device)
    bank = _bank(method, device, cams)
    bank.eval()
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        with torch.no_grad():
            bank.step = 60000
            bank.set_override(bank.appearance_embeds.weight[0])
            a = renderer.render(params, cams[0])
            b = renderer.render(params, cams[2])  # same rotation, translated camera, same l_j
            bank.set_override(None)
            trans = float((a["background"] - b["background"]).abs().max())
            if trans > 1e-5:
                raise AssertionError(f"background depends on camera translation: {trans}")
            spread = float(a["background"].std())
            if spread < 1e-4:
                raise AssertionError("degree-4 background is constant over directions")
            comp = torch.clamp(a["foreground"] + (1 - a["alpha"]) * a["background"], 0, 1)
            cerr = float((comp - a["rgb"]).abs().max())
            if cerr > 1e-6:
                raise AssertionError("final colour is not C + (1 - alpha) C_bg")
            bank.step = 0  # degree 0 -> C0 * b0 everywhere
            z = renderer.render(params, cams[0])["background"]
            emb = bank.appearance_embeds.weight[0]
            b0 = C0 * bank.background_sh(emb)[0]
            derr = float((z - b0).abs().max())
            if derr > 1e-5:
                raise AssertionError("degree-0 background != C0 * b0")
            bank.step = 60000
            bank.set_override(bank.appearance_embeds.weight[1])
            o = renderer.render(params, cams[0])["background"]
            bank.set_override(None)
            echange = float((o - a["background"]).abs().max())
            if echange < 1e-4:
                raise AssertionError("background ignores the appearance embedding")
    finally:
        bank_mod.ACTIVE = prev
    return {"translation_diff": trans, "composite_err": cerr, "deg0_err": derr,
            "embedding_change": echange, "direction_std": spread}


def robust_mask_reference(errors, k, st, cfg):
    """Independent numpy loop version of the official ``robust_mask``."""
    e = errors.copy()
    H, W, _ = e.shape
    e[: int(H * cfg["never_mask_upper"])] = 0
    l1 = e.mean()
    if l1 > st["max"] or k % cfg["robust_mask_reset_interval"] == 0:
        st["max"] = l1
    st["min"] = min(st["min"], l1)
    lo, hi = cfg["robust_mask_percentage"]
    pct = (l1 - st["min"]) / ((st["max"] - st["min"]) + 1e-6) * (hi - lo) + lo
    per = e.mean(-1)
    thr = np.quantile(per, 1 - pct)  # linear interpolation, as torch.quantile
    inl = (per <= thr).astype(np.float64)
    f = cfg["box_size"]
    r = f // 2
    out = np.zeros_like(inl)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for di in range(-r, f - r):
                for dj in range(-r, f - r):
                    if 0 <= i + di < H and 0 <= j + dj < W:
                        acc += inl[i + di, j + dj]
            nb = acc / (f * f) > cfg["box_threshold"]
            out[i, j] = 1.0 if (nb or inl[i, j] > 0) else 0.0
    return out, pct


def robust_mask(method, device):
    cfg = {**PHOTOMETRIC_DEFAULTS, **_spec(method, "sfw-photometric").config}
    rng = np.random.default_rng(0)
    H, W = 30, 40
    rm = RobustMask(cfg)
    st = {"max": 0.0, "min": 1e10}
    results = []
    base = rng.uniform(0, 0.05, (H, W, 3)).astype(np.float32)
    trans = base.copy()
    trans[20:28, 10:22] += 0.8  # transient block in the lower part
    trans[2:8, 5:15] += 0.8      # high error in the upper 40%: must stay inlier
    for k, e in ((6000, base), (6001, trans), (6002, base * 1.5)):
        m, pct = rm(torch.from_numpy(e).to(device), k)
        ref, rpct = robust_mask_reference(e.astype(np.float64), k, st, cfg)
        mm = m[..., 0].cpu().numpy()
        diff = int((mm != ref).sum())
        if diff > 2 or abs(pct - rpct) > 1e-5:
            raise AssertionError(f"robust mask differs from the reference at k={k}: {diff} px")
        results.append({"k": k, "pct": pct, "masked": int((mm == 0).sum()), "diff": diff})
    m1 = rm.__class__(cfg)
    m1(torch.from_numpy(base).to(device), 6000)
    mk, pct = m1(torch.from_numpy(trans).to(device), 6001)
    mk = mk[..., 0].cpu().numpy()
    if pct < 0.39:
        raise AssertionError("mask percentage did not interpolate to Per_max at the max loss")
    if mk[: int(0.4 * H)].min() < 1:
        raise AssertionError("upper 40% of the image was masked")
    if mk[21:27, 11:21].mean() > 0.2:
        raise AssertionError("transient block not masked")
    return {"calls": results, "block_masked_fraction": float(1 - mk[21:27, 11:21].mean())}


def masked_view_padding(method, device):
    cfg = {**PHOTOMETRIC_DEFAULTS, **_spec(method, "sfw-photometric").config, "robust_mask": True}
    from splatstudio.losses import ssim

    g = torch.Generator().manual_seed(1)
    H, W = 24, 40
    gt = torch.rand(H, W, 3, generator=g).to(device)
    pred = torch.rand(H, W, 3, generator=g).to(device)
    m = torch.zeros(H, W, 1, device=device)
    m[:, : W // 2] = 1
    out = {}
    for k in (100, 7000):
        ra, rb = RobustMask(cfg), RobustMask(cfg)
        ra.max_loss = rb.max_loss = 0.5
        ra.min_loss = rb.min_loss = 0.01
        # official code: full image, gt*mask / pred*mask, then robust mask on |difference|
        g2, p2 = gt * m, pred * m
        err = (g2 - p2).abs()
        if k >= cfg["start_robust_mask_at"]:
            rmask, _ = rb(err, k)
            l1 = (err * rmask).mean()
            g2, p2 = g2 * rmask, p2 * rmask
        else:
            l1 = err.mean()
        ref = 0.8 * l1 + 0.2 * (1 - ssim(p2, g2))
        mine, _, _, _ = photometric_terms(pred[:, : W // 2], gt[:, : W // 2], k, ra, cfg, True)
        if abs(float(mine) - float(ref)) > 1e-6:
            raise AssertionError(f"crop+pad loss {float(mine)} != masked full image {float(ref)}")
        unpadded, _, _, _ = photometric_terms(pred[:, : W // 2], gt[:, : W // 2], k,
                                              RobustMask(cfg), cfg, False)
        out[str(k)] = {"padded": float(mine), "reference": float(ref),
                       "unpadded": float(unpadded)}
    if abs(out["100"]["padded"] - out["100"]["unpadded"]) < 1e-4:
        raise AssertionError("padding has no effect: probe cannot detect a missing mask")
    return out


def alpha_loss(method, device):
    cfg = {**ALPHA_DEFAULTS, **_spec(method, "sfw-alpha").config}
    g = torch.Generator().manual_seed(2)
    H, W = 20, 24
    bg = torch.rand(H, W, 3, generator=g).to(device)
    gt = torch.rand(H, W, 3, generator=g).to(device)
    gt[5:15, 4:14] = bg[5:15, 4:14] + 0.001  # well explained by the background
    gt[16, 20] = bg[16, 20]                   # isolated pixel: removed by the 3x3 box
    alpha = torch.rand(H, W, 1, generator=g).to(device).requires_grad_(True)
    val = alpha_loss_value(alpha, gt, bg, cfg)
    near = ((gt - bg).abs().mean(-1) < cfg["threshold"]).float().cpu().numpy()
    sel = np.zeros_like(near, dtype=bool)
    for i in range(H):
        for j in range(W):
            s = near[max(i - 1, 0): i + 2, max(j - 1, 0): j + 2].sum() / 9.0
            sel[i, j] = s > cfg["select"]
    a = alpha.detach()[..., 0].cpu().numpy()
    ref = cfg["weight"] * a[sel].mean()
    if abs(float(val) - ref) > 1e-6:
        raise AssertionError(f"alpha loss {float(val)} != reference {ref}")
    if sel[16, 20]:
        raise AssertionError("isolated pixel selected")
    val.backward()
    gmask = alpha.grad[..., 0].cpu().numpy() != 0
    if not (gmask == sel).all():
        raise AssertionError("gradient does not reach exactly the selected pixels")
    zero = alpha_loss_value(alpha, gt + 0.5, bg, cfg)
    if float(zero) != 0.0 or not zero.requires_grad:
        raise AssertionError("empty selection must give an attached zero")
    return {"value": float(val), "reference": float(ref), "selected": int(sel.sum())}


def splatfacto_schedule(method, device):
    cfg = _spec(method, "sfw-density").config
    rcfg = _spec(method, "sfw-renderer").config
    resets = [k for k in range(60000) if is_reset_step(k, cfg)]
    if resets != [2600, 5100, 7600, 10100, 12600, 15100, 17600]:
        raise AssertionError(f"opacity reset steps {resets}")
    n = 773
    dens = [k for k in range(60000) if is_densify_step(k, cfg, n)]
    ref = [k for k in range(1100, 20000, 100) if k % 2500 > n + 100]
    if dens != ref or 1000 in dens or 20000 in dens or 2600 in dens:
        raise AssertionError("densification windows differ from the code")
    ds = [downscale_factor(k, rcfg["num_downscales"], rcfg["resolution_schedule"])
          for k in (0, 2999, 3000, 5999, 6000, 50000)]
    if ds != [4, 4, 2, 2, 1, 1] or downscale_factor(0, 2, 3000, training=False) != 1:
        raise AssertionError(f"resolution schedule {ds}")
    degs = [sh_degree(k, 2000, 3) for k in (0, 1999, 2000, 4000, 6000, 50000)]
    bdegs = [sh_degree(k, 1000, 4) for k in (0, 999, 1000, 4000, 9000)]
    if degs != [0, 0, 1, 2, 3, 3] or bdegs != [0, 0, 1, 4, 4]:
        raise AssertionError("SH schedules")
    checks = {
        "means@15000": (lr_at("means", 15000), math.sqrt(1.6e-5 * 1.6e-7)),
        "means@45000": (lr_at("means", 45000), 1.6e-7),
        "features@20000": (lr_at("appearance_features", 20000), math.sqrt(0.02 * 1e-3)),
        "embed@40000": (lr_at("appearance_embed", 40000), 3e-4),
        "app_rest@0": (lr_at("appearance_model_rest", 0), 1e-4),
        "bg_base@30000": (lr_at("field_background_base", 30000), 2e-4),
        "opacities@9999": (lr_at("opacities", 9999), 0.03),
    }
    for name, (got, want) in checks.items():
        if abs(got - want) > 1e-9 * max(1.0, want) + 1e-12 and abs(got / want - 1) > 1e-6:
            raise AssertionError(f"LR {name}: {got} != {want}")
    return {"resets": resets, "densify_steps": len(dens), "downscale": ds,
            "lr": {k: v[0] for k, v in checks.items()}}


def splatfacto_refine(method, device):
    _, density, _ = build(method)
    params, opts = sample_scene(method, device)
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    n = len(params["means"])
    density.s = 1.0
    density.n_train = 0
    with torch.no_grad():
        params["opacities"].fill_(2.0)
        params["scales"].fill_(math.log(0.005))         # small -> duplicate
        params["scales"][[1, 2]] = math.log(0.05)        # large -> split
        params["opacities"][3] = -5.0                     # transparent -> cull
        params["scales"][4] = math.log(0.6)               # too big -> cull (k > 2500)
        params["scales"][6] = math.log(0.014)             # split; 0.014/1.6 <= 0.01 -> also dup
    grad = torch.zeros(n, device=device)
    grad[[0, 1, 2, 5, 6]] = 1.0                           # high gradients
    density.grad_norm = grad
    density.vis_counts = torch.full((n,), 2.0, device=device)
    density.max_2d = torch.zeros(n, device=device)
    density.max_2d[5] = 0.1                               # small but large on screen -> dup+split
    density.last_size = (100, 100)
    density.refine(params, opts, 3100)
    check_state(method, params, opts)
    # Official order (shrink split scales by 1.6 in place, then pick dups from the shrunk scales):
    # splits 1, 2, 5, 6; dups 0, 5, 6 -> n + 3 dups + 4 splits - 1 transparent - 1 big
    expect = n + 3 + 4 - 2
    if len(params["means"]) != expect:
        raise AssertionError(f"refinement gave {len(params['means'])} Gaussians, expected {expect}")
    s_all = params["scales"][:, 0]
    child6 = int((s_all - math.log(0.014 / 1.6)).abs().lt(1e-5).sum())
    if child6 != 3:
        raise AssertionError(f"Gaussian at scale 0.014 has {child6} children with scale/1.6, "
                             "expected 3 (2 split samples + shrunk duplicate)")
    child5 = int((s_all - math.log(0.005 / 1.6)).abs().lt(1e-5).sum())
    if child5 != 3:
        raise AssertionError(f"small screen-large Gaussian: {child5} shrunk children, expected 3")
    # scale 0.005 kept by: 0, 7..11 (untouched) and the plain duplicate of 0
    if int((s_all - math.log(0.005)).abs().lt(1e-5).sum()) != (n - 7) + 1 + 1:
        raise AssertionError("unsplit small Gaussians or the plain duplicate lost their scale")
    if density.grad_norm is not None:
        raise AssertionError("accumulators were not reset after refinement")
    density.refine(params, opts, 2600)  # opacity reset step (no densification inside the pause)
    if float(params["opacities"].sigmoid().max()) > 0.2 + 1e-6:
        raise AssertionError("opacity reset did not clamp to 0.2")
    if torch.count_nonzero(opts["opacities"].state[params["opacities"]]["exp_avg"]):
        raise AssertionError("opacity reset kept Adam moments")
    check_state(method, params, opts)
    return {"initial": n, "after_refine": expect, "children_of_0.014": child6, "reset_max_opacity":
            float(params["opacities"].sigmoid().max())}


def ns_normalisation(method, device):
    rng = np.random.default_rng(4)
    vms = []
    for _ in range(9):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        ang = rng.uniform(0, 0.6)
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Rw = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
        c2w = np.eye(4)
        c2w[:3, :3] = Rw
        c2w[:3, 3] = rng.normal(size=3) * 4 + np.array([10.0, -3.0, 2.0])
        vms.append(torch.tensor(np.linalg.inv(c2w), dtype=torch.float32))
    R, centre, s = nerfstudio_normalisation(vms, 3.0)
    c2ws = np.stack([np.linalg.inv(v.numpy().astype(np.float64)) for v in vms])
    o = c2ws[:, :3, 3]
    up = (-c2ws[:, :3, 1]).mean(0)
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rref = np.eye(3) + vx + vx @ vx / (1 + c)
    cref = o.mean(0)
    sref = 3.0 / np.abs((o - cref) @ Rref.T).max()
    errs = {"R": float(np.abs(R.numpy() - Rref).max()),
            "centre": float(np.abs(centre.numpy() - cref).max()), "scale": abs(s / sref - 1)}
    if errs["R"] > 1e-4 or errs["centre"] > 1e-4 or errs["scale"] > 1e-5:
        raise AssertionError(f"normalisation differs: {errs}")
    if abs(float((Rref @ up)[2]) - 1) > 1e-9:
        raise AssertionError("reference rotation broken")
    return {**errs, "scale_factor": s}


def nerfw_right_half(method, device):
    _cuda()
    from .evaluate import metrics, right_half

    g = torch.Generator().manual_seed(6)
    gt = torch.rand(32, 64, 3, generator=g).to(device)
    pred = torch.rand(32, 64, 3, generator=g).to(device)
    a = metrics(right_half(pred), right_half(gt), device)
    pred2 = pred.clone()
    pred2[:, :32] = 0.0
    b = metrics(right_half(pred2), right_half(gt), device)
    if any(abs(a[k] - b[k]) > 1e-6 for k in a):
        raise AssertionError("right-half metrics depend on the left half")
    if right_half(gt).shape[1] != 32:
        raise AssertionError("right half is not columns >= W//2")
    renderer, _, _ = build(method)
    cams = _cameras(device)
    params, _ = sample_scene(method, device)
    bank = _bank(method, device, cams)
    bank.eval()
    bank.step = 60000
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        with torch.no_grad():
            test_view = {**cams[2], "width": 64}  # full-width test frame, same pose as crop view 2
            own = renderer.render(params, test_view)["rgb"]
            bank.set_override(bank.appearance_embeds.weight[2])
            forced = renderer.render(params, test_view)["rgb"]
            bank.set_override(bank.appearance_embeds.weight[0])
            zero = renderer.render(params, test_view)["rgb"]
            bank.set_override(None)
    finally:
        bank_mod.ACTIVE = prev
    if float((own - forced).abs().max()) > 1e-6:
        raise AssertionError("test view does not use the embedding trained on its left half")
    if float((own - zero).abs().max()) < 1e-4:
        raise AssertionError("fixture cannot distinguish embeddings")
    return {"right_half_invariance": True, "psnr": a["psnr"], "embedding_reused": True}
