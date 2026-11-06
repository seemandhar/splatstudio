"""Mechanism probes for the WildGaussians port. Each raises if its mechanism is removed/wrong."""

import math
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from splatstudio.state import initialize
from splatstudio.verification import ProbeUnavailable

from . import bank as bank_mod
from .appearance import C0, AppearanceMLP, fourier_features
from .density import SKY_COLOR, Density, densify_and_prune
from .geometry import compute_filter_3d, fibonacci_sphere, filtered_scales_opacities, reset_logits
from .losses import color_loss, create_color, create_uncertainty
from .renderer import Renderer
from .uncertainty import (
    UncertaintyHead,
    dino_cosine,
    dino_downsample,
    protected,
    uncertainty_terms,
    warmup_mask,
)


def _close(a, b, tol, what):
    err = float((a - b).abs().max())
    if not err <= tol:
        raise AssertionError(f"{what}: max error {err:.3e} > {tol:.1e}")
    return err


def _component(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _cameras(device, n=3, size=(48, 64), dist=4.0):
    h, w = size
    cams = []
    for i in range(n):
        a = 2 * math.pi * i / n
        eye = torch.tensor([dist * math.sin(a), 0.3 * i, -dist * math.cos(a)])
        fwd = -eye / eye.norm()
        right = torch.linalg.cross(torch.tensor([0.0, -1.0, 0.0]), fwd)
        right = right / right.norm()
        down = torch.linalg.cross(fwd, right)
        R = torch.stack([right, down, fwd])  # OpenCV rows x right, y down, z forward (det +1)
        vm = torch.eye(4)
        vm[:3, :3] = R
        vm[:3, 3] = -R @ eye
        K = torch.tensor([[60.0, 0, w / 2 + 1.5], [0, 60.0, h / 2 - 0.5], [0, 0, 1]])
        cams.append({"viewmat": vm.to(device), "K": K.to(device), "width": w, "height": h})
    return cams


def _scene(method, device, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * 1.5
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    return initialize(method, pts.to(device), cols.to(device))


def _density(method, device, **overrides):
    spec = _component(method, "density")
    cfg = {**spec.config, "num_sky_gaussians": 0, "bank_uncertainty_enabled": False, **overrides}
    return Density(replace(spec, config=cfg))


# ---------------------------------------------------------------------------------------------
def appearance_toning(method, device):
    torch.manual_seed(0)
    mlp = AppearanceMLP().to(device)
    feats = (torch.rand(50, 48, device=device) - 0.5).clamp_max(1.0)
    g = torch.randn(50, 24, device=device)
    e = torch.randn(32, device=device)
    out = mlp(g, e[None], feats)
    # independent: manual layers
    lin = [m for m in mlp.mlp if isinstance(m, torch.nn.Linear)]
    x = torch.cat((feats[:, :3], g, e.expand(50, -1)), 1)
    for i, layer in enumerate(lin):
        x = x @ layer.weight.T + layer.bias
        if i < 2:
            x = x.clamp_min(0)
    off, mul = 0.01 * x[:, :3], 0.01 * x[:, 3:]
    exp = feats.reshape(50, 16, 3) * mul[:, None, :]
    exp[:, 0, :] += off / C0
    err = _close(out, exp.reshape(50, 48), 1e-6, "toned SH")
    e2 = mlp(g, (e + 1.0)[None], feats)
    if float((e2 - out).abs().max()) < 1e-6:
        raise AssertionError("image embedding does not change the toned colours")
    # Fourier features vs the paper formula sin/cos(pi p 2^m), m = 1..4
    xyz = torch.randn(200, 3, device=device) * torch.tensor([1.0, 3.0, 0.5], device=device)
    ff = fourier_features(xyz, 4)
    c = xyz - xyz.mean(0)
    p = c / torch.quantile(c.abs(), 0.97, dim=0) * 0.5 + 0.5
    cols = []
    for k in range(3):
        for m in range(1, 5):
            cols += [torch.sin(math.pi * p[:, k] * 2 ** m), torch.cos(math.pi * p[:, k] * 2 ** m)]
    ferr = _close(ff, torch.stack(cols, 1), 2e-5, "Fourier features")
    return {"toned_err": err, "fourier_err": ferr, "embedding_sensitivity":
            float((e2 - out).abs().max())}


def _ssim_independent(a, b):
    """Separable-filter SSIM map (zero padding), mean over channels; [3,H,W]."""
    x = torch.arange(11, dtype=a.dtype, device=a.device) - 5
    k = torch.exp(-x ** 2 / 4.5)
    k = k / k.sum()

    def blur(t):
        t = F.conv2d(t[:, None], k.view(1, 1, 1, 11), padding=(0, 5))
        return F.conv2d(t, k.view(1, 1, 11, 1), padding=(5, 0))[:, 0]

    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    s = ((2 * ma * mb + 1e-4) * (2 * cov + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4))
    return s.mean(0)


def split_objective(method, device):
    torch.manual_seed(1)
    h, w = 24, 30
    gt = torch.rand(3, h, w, device=device)
    toned = torch.rand(3, h, w, device=device, requires_grad=True)
    raw = torch.rand(3, h, w, device=device, requires_grad=True)
    mask = (torch.rand(1, h, w, device=device) > 0.4).float()
    loss = color_loss(toned, raw, gt, mask, 0.2)
    ref = 0.8 * ((toned - gt).abs() * mask).mean() + 0.2 * ((1 - _ssim_independent(raw, gt))
                                                            * mask).mean()
    err = _close(loss, ref, 1e-5, "colour loss")
    loss.backward()
    exp_toned_grad = 0.8 * torch.sign(toned - gt) * mask / toned.numel()
    _close(toned.grad, exp_toned_grad, 1e-7, "L1 gradient on the toned image")
    if float(toned.grad[:, mask[0] == 0].abs().max()) != 0.0:
        raise AssertionError("masked pixels receive a gradient")
    toned.grad = raw.grad = None
    color_loss(toned, raw, gt, mask, 0.0).backward()
    if raw.grad is not None and float(raw.grad.abs().max()) > 0:
        raise AssertionError("L1 term reaches the raw image")
    toned.grad = raw.grad = None
    color_loss(toned, raw, gt, mask, 1.0).backward()
    if toned.grad is not None and float(toned.grad.abs().max()) > 0:
        raise AssertionError("SSIM term reaches the toned image")
    if raw.grad is None or float(raw.grad.abs().max()) == 0:
        raise AssertionError("SSIM term does not reach the raw image")
    # the component reads the mask published by wg-uncertainty
    comp = create_color(next(c for c in method.components if c.name == "wg-color"))
    hwc = {"rgb": toned.permute(1, 2, 0), "rgb_raw": raw.permute(1, 2, 0), "wg_mask": mask}
    v = comp(None, hwc, gt.permute(1, 2, 0))
    _close(v, ref, 1e-5, "wg-color component")
    try:
        comp(None, {"rgb": hwc["rgb"], "rgb_raw": hwc["rgb_raw"]}, gt.permute(1, 2, 0))
        raise AssertionError("wg-color ran without the uncertainty mask")
    except RuntimeError:
        pass
    return {"loss_err": err, "masked_fraction": float(1 - mask.mean())}


def dino_backbone(method, device):
    from .dino import DEFAULT_WEIGHTS, load_backbone

    try:
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
    except ImportError as exc:
        raise ProbeUnavailable(f"timm missing: {exc}") from exc
    try:
        ours = load_backbone(DEFAULT_WEIGHTS, device)
    except FileNotFoundError as exc:
        raise ProbeUnavailable(str(exc)) from exc
    from pathlib import Path

    sd = torch.load(Path(DEFAULT_WEIGHTS).expanduser(), map_location="cpu", weights_only=True)
    ref = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False, img_size=518,
                            dynamic_img_size=True)
    ref.load_state_dict(checkpoint_filter_fn(sd, ref))
    ref = ref.eval().to(device)
    out = {}
    g = torch.Generator().manual_seed(0)
    for hw in ((518, 518), (252, 350)):
        x = torch.randn(1, 3, *hw, generator=g).to(device)
        with torch.no_grad():
            a = ours.intermediate(x)
            b = ref.forward_intermediates(x, indices=[5], norm=True, output_fmt="NCHW",
                                          intermediates_only=True)[0]
        out[f"err_{hw[0]}x{hw[1]}"] = _close(a, b, 2e-3 * float(b.abs().max()),
                                             f"DINO features {hw}")
        if hw == (518, 518):
            with torch.no_grad():
                c = ours.intermediate(x, layer=11)
            if float((c - a).abs().max()) < 1e-3:
                raise AssertionError("layer index has no effect")
    out["shape"] = list(a.shape)
    return out


class _FakeFeatures:
    """Deterministic stand-in for DINO: fixed random patchify conv, 14x14 stride."""

    def __init__(self, device, dim=384):
        g = torch.Generator().manual_seed(3)
        self.w = (torch.randn(dim, 3, 14, 14, generator=g) * 0.05).to(device)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, x, key=None):
        self.calls += 1
        return F.conv2d(x, self.w, stride=14)


def uncertainty_loss(method, device):
    torch.manual_seed(2)
    feats = _FakeFeatures(device)
    head = UncertaintyHead(generator=torch.Generator().manual_seed(0)).to(device)
    with torch.no_grad():
        head.conv_seg.weight.normal_(0, 0.3)
    h, w = 61, 87  # not multiples of 14: exercises pad / crop
    gt = torch.rand(1, 3, h, w, device=device)
    pred = (gt + 0.3 * torch.rand_like(gt)).clamp(0, 1).requires_grad_(True)
    head.eval()  # deterministic (no dropout / batch stats) for the reference comparison
    loss, mult, sigma = uncertainty_terms(head, feats, gt, pred, prior_weight=0.5, dino_max=40)
    if sigma.shape[2:] != gt.shape[2:] or float(sigma.min()) < 0.1 - 1e-7:
        raise AssertionError("sigma not at image resolution or not clipped at 0.1")
    cos = dino_cosine(feats, dino_downsample(gt, 40), dino_downsample(pred.detach(), 40))
    part = torch.clamp(2 - 2 * cos, 0, 1)  # Eq. 8: min(1, 2 - 2 cos)
    lm = dino_downsample(1 / (2 * sigma ** 2), 40)
    ref = (part[:, None] * lm).mean() + 0.5 * torch.log(sigma).mean()
    err = _close(loss, ref, 1e-5, "uncertainty loss")
    m = (mult > 1).float()
    if not torch.equal(m.bool(), (sigma < 1 / math.sqrt(2)) & ((1 / (2 * sigma ** 2)) > 1)):
        raise AssertionError("mask != 1(1/(2 sigma^2) > 1)")
    loss.backward()
    if pred.grad is not None:
        raise AssertionError("uncertainty loss reaches the rendering")
    if head.conv_seg.weight.grad is None or float(head.conv_seg.weight.grad.abs().sum()) == 0:
        raise AssertionError("uncertainty head gets no gradient")
    # schedules (official train_iteration)
    sched = {"p3100": protected(3100, 15000, 3000, 500), "p3600": protected(3600, 15000, 3000, 500),
             "p400": protected(400, 15000, 3000, 500), "p16000": protected(16000, 15000, 3000, 500)}
    if sched != {"p3100": True, "p3600": False, "p400": True, "p16000": False}:
        raise AssertionError(f"protected window wrong: {sched}")
    mm = torch.tensor([0.0, 1.0])
    ramp = [warmup_mask(mm, it, 35000, 5000)[0].item() for it in (34999, 37500, 40000)]
    if ramp != [1.0, 0.5, 0.0]:
        raise AssertionError(f"warm-up ramp wrong: {ramp}")
    # component: protected window => head gradient None (Adam skipped), otherwise present
    spec = next(c for c in method.components if c.name == "wg-uncertainty")
    comp = create_uncertainty(spec)
    head.train()
    old = bank_mod.ACTIVE
    bank_mod.ACTIVE = SimpleNamespace(head=head, features=feats, uncertainty=True)
    grads = {}
    try:
        for it in (100, 2600):
            head.zero_grad(set_to_none=True)
            rgb = pred.detach()[0].permute(1, 2, 0).clone().requires_grad_(True)
            outs = {"step": it, "rgb": rgb, "view_index": None}
            comp(None, outs, gt[0].permute(1, 2, 0)).backward()
            grads[it] = head.conv_seg.weight.grad is not None
            if "wg_mask" not in outs:
                raise AssertionError("mask not published")
            if it < spec.config["warmup_start"] and float(outs["wg_mask"].min()) != 1.0:
                raise AssertionError("mask active before warm-up start")
    finally:
        bank_mod.ACTIVE = old
    if grads != {100: False, 2600: True}:
        raise AssertionError(f"protected-window gradient gating wrong: {grads}")
    return {"loss_err": err, "mask_fraction": float(m.mean()), **sched}


def sky_init(method, device):
    sph = fibonacci_sphere(2001)
    _close(sph.norm(dim=-1), torch.ones(2001), 1e-5, "Fibonacci norm")
    dz = sph[1:, 2] - sph[:-1, 2]
    _close(dz, torch.full_like(dz, 2 / 2001), 1e-4, "Fibonacci z spacing")
    params, opts = _scene(method, device, n=300)
    rgb0 = params["sh0"][:, 0].detach() * C0 + 0.5
    pts0 = params["means"].detach().clone()
    cams = _cameras(device)
    d = _density(method, device, num_sky_gaussians=3000)
    d.initialize(params, opts, 3.0)
    d.set_training_cameras(cams)
    try:
        n0 = len(pts0)
        sky = params["means"][n0:].detach()
        mean = pts0.mean(0)
        radius = 10 * torch.quantile((pts0 - mean).norm(dim=-1), 0.97)
        _close((sky - mean).norm(dim=-1), radius.expand(len(sky)), 1e-3 * float(radius), "radius")
        full = fibonacci_sphere(3000).to(device) * radius + mean
        keep = torch.zeros(3000, dtype=torch.bool, device=device)
        for c in cams:  # independent: any camera, v < 2H/3, z <= eps -> (cx, cy)
            xc = full @ c["viewmat"][:3, :3].T + c["viewmat"][:3, 3]
            z = xc[:, 2]
            v = torch.where(z > 1.2e-7, xc[:, 1] / z, torch.zeros_like(z)) * c["K"][1, 1] \
                + c["K"][1, 2]
            keep |= v < 2 / 3 * c["height"]
        if int(keep.sum()) != len(sky):
            raise AssertionError(f"kept sky points {len(sky)} != {int(keep.sum())}")
        if len(sky) == 3000:
            raise AssertionError("visibility rule removed nothing (probe scene not informative)")
        op = params["opacities"].sigmoid()
        if not bool((op[n0:] == 1.0).all()):
            raise AssertionError("sky opacity is not 1.0")
        _close(op[:n0], torch.full_like(op[:n0], 0.1), 1e-6, "SfM opacity 0.1")
        _close(params["sh0"][:n0, 0], rgb0, 1e-5, "sh0 = RGB (official features_dc)")
        _close(params["sh0"][n0:, 0], torch.tensor(SKY_COLOR, device=device).expand(len(sky), 3),
               1e-6, "sky colour")
        if "wg_embedding" in params:
            _close(params["wg_embedding"], fourier_features(params["means"].detach()), 1e-3,
                   "Fourier init")
        if float(params["filter_3d"].min()) <= 0:
            raise AssertionError("3D filter not computed at init")
    finally:
        bank_mod.ACTIVE = None
    return {"sky_points": len(sky), "removed": 3000 - len(sky), "radius": float(radius)}


def densify_official(method, device):
    params, opts = _scene(method, device, n=6)
    with torch.no_grad():
        params["scales"].copy_(torch.log(torch.tensor(
            [[0.01] * 3, [0.01] * 3, [0.01] * 3, [0.5] * 3, [5.0] * 3, [0.5] * 3], device=device)))
        params["opacities"].copy_(torch.tensor([0.0, 0, -6, 0, 0, 0], device=device))
    # Adam moments so the ops have state to realign
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    cfg = {**_component(method, "density").config}
    G = torch.tensor([3e-4, 1e-4, 1e-4, 3e-4, 0.0, 1e-4], device=device)
    A = torch.tensor([0.1, 0.9, 0.2, 0.1, 0.0, 0.3], device=device)
    denom = torch.tensor([2.0, 4, 1, 1, 1, 1], device=device)
    stats = {"grad": G * denom, "abs": A * denom, "denom": denom.clone(),
             "id": torch.arange(6, device=device, dtype=torch.float32)}
    ids_before = params["means"].detach().clone()
    nc, ns, npr = densify_and_prune(params, opts, stats, 2e-4, 10.0, cfg, size_prune=False)
    # Q = quantile(A, 1 - 2/6) = 0.2333 -> abs selects 1, 5; grad selects 0, 3
    if (nc, ns) != (2, 2):
        raise AssertionError(f"clone/split counts {(nc, ns)} != (2, 2) (abs-grad union)")
    if npr != 1:
        raise AssertionError(f"pruned {npr} != 1 (opacity only; size prune disabled)")
    n = len(params["means"])
    if n != 6 + 2 - 2 + 4 - 1:
        raise AssertionError(f"population {n}")
    # originals 0, 1 (cloned) keep their accumulated statistics; all new Gaussians start at zero
    kept = {int(i): float(dn) for i, dn in zip(stats["id"], stats["denom"]) if dn > 0}
    if kept != {0: 2.0, 1: 4.0, 4: 1.0}:
        raise AssertionError(f"statistics not carried / not zeroed correctly: {kept}")
    if int((stats["denom"] == 0).sum()) != 2 + 4:
        raise AssertionError("new Gaussians do not start with zero statistics")
    # size prune only when enabled
    stats2 = {k: torch.zeros(n, device=device) for k in ("grad", "abs")}
    stats2["denom"] = torch.ones(n, device=device)
    _, _, np2 = densify_and_prune(params, opts, stats2, 2e-4, 10.0, cfg, size_prune=True)
    if np2 < 1:
        raise AssertionError("world-size prune (> 0.1 extent) did not fire")
    return {"clone": nc, "split": ns, "prune": npr, "size_prune": np2,
            "moved": float((ids_before[0] - params["means"][0]).abs().max())}


def filter_3d(method, device):
    g = torch.Generator().manual_seed(4)
    pts = ((torch.rand(300, 3, generator=g) - 0.5) * 6).to(device)
    cams = _cameras(device)
    f = compute_filter_3d(pts, cams)
    focal = max(float(c["K"][0, 0]) for c in cams)
    exp = torch.empty(300, device=device)
    valid_any = torch.zeros(300, dtype=torch.bool, device=device)
    best = torch.full((300,), 1e5, device=device)
    for i in range(300):
        for c in cams:
            p = c["viewmat"][:3, :3] @ pts[i] + c["viewmat"][:3, 3]
            z = float(p[2])
            zc = max(z, 0.001)
            x = float(p[0]) / zc * float(c["K"][0, 0]) + c["width"] / 2
            y = float(p[1]) / zc * float(c["K"][1, 1]) + c["height"] / 2
            if (z > 0.2 and -0.15 * c["width"] <= x <= 1.15 * c["width"]
                    and -0.15 * c["height"] <= y <= 1.15 * c["height"]):
                valid_any[i] = True
                best[i] = min(float(best[i]), z)
    best[~valid_any] = best[valid_any].max()
    exp = best / focal * math.sqrt(0.2)
    err = _close(f, exp, 1e-5, "3D filter")
    if int((~valid_any).sum()) == 0:
        raise AssertionError("probe scene has no invisible points")
    # scales spanning sub-filter to large, so the doubled-filter quirk matters
    ls = torch.log(f[:, None].double().cpu() * torch.exp(torch.randn(300, 3, generator=g, dtype=torch.float64) * 1.5)).float().to(device)
    lo = torch.randn(300, generator=g).to(device) * 3
    new = reset_logits(ls, lo, f)
    # independent re-derivation of official GaussianModel.reset_opacity (float64, direct form):
    # get_gaussians() -> filtered scales/opacity; the filter is then added a second time
    s, fd, o = ls.double().exp(), f.double()[:, None], lo.double()
    s2 = s ** 2
    coef = torch.sqrt(s2.prod(1) / (s2 + fd ** 2).prod(1))
    cur = torch.sigmoid(o) * coef
    sf2 = s2 + fd ** 2
    coef2 = torch.sqrt(sf2.prod(1) / (sf2 + fd ** 2).prod(1))
    ref = (torch.minimum(cur, torch.full_like(cur, 0.01)) / coef2).clamp(1e-6, 1 - 1e-6)
    rerr = _close(new.double().sigmoid(), ref, 1e-6 + 1e-4 * float(ref.max()), "official reset")
    _, op_new = filtered_scales_opacities(ls, new, f)
    small = (s.max(1).values <= fd[:, 0]) & (ref < 1 - 1e-6)
    if int(small.sum()) == 0 or not bool((op_new[small] < 0.0099).all()):
        raise AssertionError("sub-filter Gaussians must come out of the reset below 0.01 "
                             "(official doubled-filter quirk)")
    return {"filter_err": err, "reset_err": rerr, "invisible": int((~valid_any).sum()),
            "sub_filter_checked": int(small.sum()),
            "sub_filter_max_after_reset": float(op_new[small].max())}


def _live_bank(method, params, cams, device):
    d = _density(method, device)
    from .bank import WildBank

    bank = WildBank([c["viewmat"] for c in cams], {**d.bank_cfg, "uncertainty_enabled": False},
                    device)
    return d, bank


def render_two_pass(method, device):
    from gsplat import rasterization

    params, opts = _scene(method, device, n=500)
    cams = _cameras(device)
    with torch.no_grad():
        params["scales"].fill_(math.log(0.05))
        params["shN"].normal_(0, 0.05)
        params["filter_3d"].copy_(compute_filter_3d(params["means"], cams))
        if "wg_embedding" in params:
            params["wg_embedding"].copy_(fourier_features(params["means"]))
    rspec = _component(method, "renderer")
    r = Renderer(rspec)
    d, bank = _live_bank(method, params, cams, device)
    appearance = bool(rspec.config["appearance"])
    if appearance:
        with torch.no_grad():
            for m in bank.mlp.mlp:
                if isinstance(m, torch.nn.Linear):
                    m.weight.mul_(30.0)
    old = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        bank.step = 2999  # rendering step 3000 -> SH degree 3
        out = r.render(params, cams[1], sh_degree=0)
        if out["view_index"] != 1:
            raise AssertionError("training view not found in the bank")
        scales, opac = filtered_scales_opacities(params["scales"], params["opacities"],
                                                 params["filter_3d"])
        ref, _, _ = rasterization(params["means"], params["quats"], scales, opac,
                                  torch.cat((params["sh0"], params["shN"]), 1).clamp_max(1.0),
                                  cams[1]["viewmat"][None], cams[1]["K"][None], 64, 48,
                                  near_plane=0.2, eps2d=0.1, sh_degree=3,
                                  rasterize_mode="antialiased")
        err = _close(out["rgb_raw"], ref[0], 1e-4, "raw pass vs gsplat SH render")
        res = {"raw_err": err}
        loss = out["rgb"].sum() + out["rgb_raw"].sum()
        # density statistics sum both passes' gradients
        d.params, d.bank = params, bank
        d._reset_stats()
        d.before_backward(params, opts, 1, out)
        loss.backward()
        d.accumulate(out)
        info = out["infos"][0]
        scale = torch.tensor([32.0, 24.0], device=device)
        g = sum(i["means2d"].grad[0] for i in out["infos"]) * scale
        vis = (info["radii"][0] > 0).all(-1)
        _close(d.stats["grad"][vis], g[vis].norm(dim=-1), 1e-6, "grad statistics")
        if appearance:
            g1 = (out["infos"][0]["means2d"].grad[0] * scale)[vis].norm(dim=-1)
            if float((g1 - d.stats["grad"][vis]).abs().max()) < 1e-9:
                raise AssertionError("toned-pass gradient missing from the statistics")
        a = sum(i["means2d"].absgrad[0] for i in out["infos"]) * scale
        _close(d.stats["abs"][vis], a[vis].sum(dim=-1), 1e-6, "abs statistics (L1, official)")
        if float(d.stats["abs"].max()) <= 0:
            raise AssertionError("absolute gradients missing")
        if appearance:
            if len(out["infos"]) != 2:
                raise AssertionError("appearance render must use two passes")
            diff = float((out["rgb"] - out["rgb_raw"]).abs().max())
            with torch.no_grad():
                bank.image_embeddings[1] += 1.0
            out2 = r.render(params, cams[1], sh_degree=0)
            d_toned = float((out2["rgb"] - out["rgb"]).abs().max())
            d_raw = float((out2["rgb_raw"] - out["rgb_raw"]).abs().max())
            if diff < 1e-3 or d_toned < 1e-4 or d_raw > 1e-6:
                raise AssertionError(f"toned pass not driven by the embedding ({diff}, {d_toned}, "
                                     f"{d_raw})")
            res.update(toned_minus_raw=diff, embedding_effect=d_toned)
        res["visible"] = int(vis.sum())
        return res
    finally:
        bank_mod.ACTIVE = old


def nerfw_protocol(method, device):
    from .evaluate import official_metrics, optimize_embedding

    params, _ = _scene(method, device, n=800, seed=5)
    cams = _cameras(device, size=(48, 65))
    with torch.no_grad():
        params["scales"].fill_(math.log(0.08))
        params["filter_3d"].copy_(compute_filter_3d(params["means"], cams))
        params["wg_embedding"].copy_(fourier_features(params["means"]))
    r = Renderer(_component(method, "renderer"))
    _, bank = _live_bank(method, params, cams, device)
    with torch.no_grad():
        for m in bank.mlp.mlp:
            if isinstance(m, torch.nn.Linear):
                m.weight.mul_(20.0)
    bank.eval()
    old = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        target_emb = torch.randn(32, device=device) * 0.5
        bank.set_override(target_emb)
        with torch.no_grad():
            target = r.render(params, cams[0], 3)["rgb"].clamp(0, 1)
        bank.set_override(None)
        frame = {**cams[0], "image": target}
        emb, hist = optimize_embedding(r, params, bank, frame, 3, {"iters": 128})
        if not hist[-1] < 0.7 * hist[0]:
            raise AssertionError(f"embedding optimisation did not reduce the loss {hist[0]} -> "
                                 f"{hist[-1]}")
        with torch.no_grad():
            zero = r.render(params, cams[0], 3)["rgb"].clamp(0, 1)
            bank.set_override(emb)
            opt = r.render(params, cams[0], 3)["rgb"].clamp(0, 1)
            bank.set_override(None)
        w = 65
        m0 = official_metrics(zero[:, w // 2:], target[:, w // 2:], device)
        m1 = official_metrics(opt[:, w // 2:], target[:, w // 2:], device)
        if not m1["psnr"] > m0["psnr"] + 1.0:
            raise AssertionError(f"right-half PSNR did not improve: {m0['psnr']} -> {m1['psnr']}")
        tampered = target.clone()
        tampered[:, : w // 2] = 0
        m2 = official_metrics(opt[:, w // 2:], tampered[:, w // 2:], device)
        if abs(m2["psnr"] - m1["psnr"]) > 1e-9:
            raise AssertionError("right-half metric depends on the left half")
        return {"loss_first": hist[0], "loss_last": hist[-1], "psnr_zero": m0["psnr"],
                "psnr_opt": m1["psnr"]}
    finally:
        bank_mod.ACTIVE = old
        bank.set_override(None)


