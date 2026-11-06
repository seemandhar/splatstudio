"""Mechanism probes for RobustSplat. Each compares an implemented mechanism to an independently
written reference or to the paper/official schedule, so it fails if the mechanism is removed or
altered."""

import math
from pathlib import Path

import numpy as np
import torch
from splatstudio.verification import ProbeUnavailable
from torch import nn

from . import dino
from .context import Shared, ViewTable
from .density import Density
from .losses import MaskedPhotometric, MaskMLPLoss, mlp_active
from .masks import (
    MaskMLP,
    cosine_target,
    histogram_threshold,
    loss_mask,
    reg_weight,
    residual_hinge,
    residual_masks,
)
from .renderer import Renderer, rasterize


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _params(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = nn.ParameterDict({
        "means": nn.Parameter((torch.rand(n, 3, generator=g) - 0.5).to(device)),
        "scales": nn.Parameter(torch.full((n, 3), math.log(0.02)).to(device)),
        "quats": nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1).to(device)),
        "opacities": nn.Parameter(torch.full((n,), 2.0).to(device)),
        "sh0": nn.Parameter((torch.rand(n, 1, 3, generator=g) - 0.5).to(device)),
        "shN": nn.Parameter(torch.zeros(n, 15, 3).to(device)),
    })
    opts = {k: torch.optim.Adam([v], lr=1e-3, eps=1e-15) for k, v in p.items()}
    return p, opts


# ----------------------------------------------------------------------------- C1
def delayed_growth_probe(method, device):
    """Forced large screen gradients on every Gaussian at selected steps; record when growth and
    opacity resets happen and that a huge invisible Gaussian is never size-pruned."""
    spec = _spec(method, "rs-density")
    dens = Density(spec)
    n = 20
    params, opts = _params(n, "cpu")
    with torch.no_grad():
        params["scales"][0] = math.log(50.0)       # world-size >> 0.1 extent: Inria would prune
    for o in opts.values():                        # allocate Adam moments
        for g in o.param_groups:
            for q in g["params"]:
                q.grad = torch.zeros_like(q)
        o.step()
    dens.initialize(params, opts, scene_scale=1.0)
    if abs(opts["opacities"].param_groups[0]["lr"] - 0.025) > 1e-12:
        raise AssertionError("opacity lr must be the official 0.025")
    steps = [1, 500, 600, 3000, 6000, 9000, 9900, 10000, 10100, 12000, 14900, 15000, 15100,
             18000, 19900, 20000, 20100, 21000]
    grown, resets = [], []
    for s in steps:
        N = len(params["means"])
        means2d = torch.zeros(1, N, 2, requires_grad=True)
        means2d.grad = torch.full((1, N, 2), 0.01)
        radii = torch.full((1, N), 5, dtype=torch.int32)
        radii[0, 0] = 0                            # the huge Gaussian is invisible
        info = {"means2d": means2d, "width": 64, "height": 48, "radii": radii}
        before = len(params["means"])
        with torch.no_grad():
            op_before = torch.sigmoid(params["opacities"]).max().item()
        dens.after_backward(params, opts, s, {"info": info})
        if len(params["means"]) > before:
            grown.append(s)
        if ("reset", s) in dens.events:
            resets.append(s)
            if torch.sigmoid(params["opacities"]).max().item() > 0.01 + 1e-6 or op_before < 0.02:
                raise AssertionError("opacity reset must clamp opacities to 0.01")
            with torch.no_grad():
                params["opacities"].fill_(2.0)
    if method.name == "robustsplat-nodg":
        exp_grow = [600, 3000, 6000, 9000, 9900, 10000, 10100, 12000, 14900]
        exp_reset = [3000, 6000, 9000, 12000]
    else:
        exp_grow = [10100, 12000, 14900, 15000, 15100, 18000, 19900]
        exp_reset = [15000, 18000]
    if grown != exp_grow:
        raise AssertionError(f"growth steps {grown} != expected {exp_grow}")
    if resets != exp_reset:
        raise AssertionError(f"opacity resets {resets} != expected {exp_reset}")
    if float(torch.exp(params["scales"]).max()) < 49.0:
        raise AssertionError("huge Gaussian was pruned: official code has no size pruning")
    return {"growth_steps": grown, "reset_steps": resets, "final_gaussians": len(params["means"])}


# ----------------------------------------------------------------------------- helpers
def _np_generate_mask(res, thr):
    inl = (res < thr).astype(np.float64)
    pad = np.pad(inl, 1)
    box = sum(pad[1 + a: 1 + a + res.shape[0], 1 + b: 1 + b + res.shape[1]]
              for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9.0
    return np.logical_or(box > 0.5, inl > 0).astype(np.float32)


def _np_erode(m, k):
    r = k // 2
    pad = np.pad(m, r, constant_values=np.inf)
    out = np.full_like(m, np.inf)
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            out = np.minimum(out, pad[r + a: r + a + m.shape[0], r + b: r + b + m.shape[1]])
    return out


# ----------------------------------------------------------------------------- C3 / C4
def mask_supervision_probe(method, device):
    cfg = _spec(method, "rs-mask-mlp").config
    rng = np.random.default_rng(0)
    # residual masks + discounted histogram
    hist_ref = np.zeros(10000)
    hist = torch.zeros(10000)
    for _ in range(4):
        gt = rng.random((3, 30, 26)).astype(np.float32)
        ren = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, 1).astype(np.float32)
        ren[:, 5:15, 5:12] = 1.0 - gt[:, 5:15, 5:12]           # distractor blob
        lo, up, hist = residual_masks(torch.from_numpy(gt), torch.from_numpy(ren), hist,
                                      cfg["lower_bound"], cfg["upper_bound"], cfg["hist_decay"])
        res = np.abs(gt - ren).mean(0)
        hist_ref = 0.95 * hist_ref + np.histogram(res, bins=10000, range=(0, 1))[0]
        edges = np.linspace(0, 1, 10001)
        cum = np.cumsum(hist_ref)
        for frac, ours in ((0.6, lo), (0.8, up)):
            thr = edges[int(np.nonzero(cum >= hist_ref.sum() * frac)[0][0])]
            ref = _np_generate_mask(res, thr)
            if (ours[0].numpy() != ref).mean() > 0.005:            # bin-edge ties only
                raise AssertionError(f"residual mask at tau={frac} differs from NumPy")
    if cfg["lower_bound"] != 0.6 or cfg["upper_bound"] != 0.8:
        raise AssertionError("tau must be 0.6 / 0.8 (supp.)")
    if float(lo.mean()) > float(up.mean()):
        raise AssertionError("lower (0.6) mask must be a subset of the upper (0.8) mask")
    t = histogram_threshold(torch.tensor([0.0, 2.0, 2.0, 0.0]), 0.5)
    if abs(float(t) - 0.25) > 1e-7:
        raise AssertionError("histogram threshold must be the left bin edge")
    # Eq. 4 at grid resolution (upsample is identity)
    f1 = torch.randn(384, 9, 9, dtype=torch.float64)
    f2 = f1 + 1.5 * torch.randn(384, 9, 9, dtype=torch.float64)
    ours = cosine_target(f1, f2, 9, 9)[0].numpy()
    a, b = f1.numpy(), f2.numpy()
    cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    ref = np.maximum(2 * cos - 1, 0)
    if np.abs(ours - ref).max() > 1e-6:
        raise AssertionError("cosine target differs from Eq. 4")
    # hinge
    m = torch.rand(500, dtype=torch.float64)
    lo_t = (torch.rand(500) > 0.6).double()
    up_t = torch.maximum(lo_t, (torch.rand(500) > 0.3).double())
    h = float(residual_hinge(m, lo_t, up_t))
    mn, lon, upn = m.numpy(), lo_t.numpy(), up_t.numpy()
    href = float(np.mean(np.maximum(mn - upn, 0) + np.maximum(lon - mn, 0)))
    if abs(h - href) > 1e-12:
        raise AssertionError("residual hinge differs from supp. Eq. S1 / code")
    # Eq. 7-8 composition at a known step
    loss = MaskMLPLoss(_spec(method, "rs-mask-mlp"))
    H, W = 8, 10
    mask = torch.rand(1, H, W, dtype=torch.float64, requires_grad=True)
    sup = {"lower": (torch.rand(1, H, W) > 0.5).double(), "upper": torch.ones(1, H, W).double(),
           "cos_target": torch.rand(1, H, W, dtype=torch.float64), "coarse": True,
           "residual_size": (H, W), "grid": 16}
    loss.supervision = lambda params, outputs, shared: sup
    step = 3000
    out = loss({}, {"step": step, "rgb": torch.zeros(H, W, 3, requires_grad=True),
                    "robustsplat": {"step": step, "mask": mask}}, None)
    mm = mask.detach().numpy()
    ref = (0.5 * np.abs(mm - sup["cos_target"].numpy()).mean()
           + 0.5 * np.mean(np.maximum(mm - 1, 0) + np.maximum(sup["lower"].numpy() - mm, 0))
           + 2.0 * np.exp(-step / 2000.0) * (1 - mm).mean())
    if abs(float(out) - ref) > 1e-9:
        raise AssertionError(f"L_MLP {float(out)} != Eq. 8 reference {ref}")
    if abs(reg_weight(2000, 2000) - math.exp(-1)) > 1e-12:
        raise AssertionError("reg decay must be e^{-i/beta}")
    Shared.reset()
    return {"l_mlp": float(out), "reference": float(ref), "lower_frac": float(lo.mean()),
            "upper_frac": float(up.mean())}


def _blob_mlp(device):
    """MLP whose output is ~1 where feature channel 0 is +1 and ~0.007 where it is -1."""
    mlp = MaskMLP().to(device)
    with torch.no_grad():
        mlp.mlp[0].weight.zero_()
        mlp.mlp[0].weight[:, 0] = 10.0
        mlp.mlp[0].bias.zero_()
        mlp.mlp[2].weight.fill_(1.0)
        mlp.mlp[2].bias.fill_(-5.0)
    return mlp


def masked_photometric_probe(method, device):
    torch.manual_seed(0)
    # MLP low-res-first equivalence
    mlp = MaskMLP().double()
    feat = torch.randn(384, 7, 9, dtype=torch.float64)
    a, b = mlp(feat, 23, 31), mlp.forward_naive(feat, 23, 31)
    if (a - b).abs().max() > 1e-10:
        raise AssertionError("MaskMLP.forward differs from upsample-then-MLP")
    # threshold + erosion vs NumPy
    m = torch.rand(1, 40, 50)
    ours = loss_mask(m, 0.25, 7)[0].numpy()
    ref = _np_erode((m[0].numpy() > 0.25).astype(np.float32), 7)
    if not np.array_equal(ours, ref):
        raise AssertionError("loss mask differs from threshold + 7x7 erosion")
    # end to end through the component: transient blob gets no gradient
    spec = _spec(method, "rs-masked-photometric")
    comp = MaskedPhotometric(spec)
    H, W = 40, 48
    f = torch.ones(384, 10, 12)
    f[0, 3:7, 4:8] = -1.0                                      # transient in feature space
    vm = torch.eye(4)
    Shared.reset()
    Shared.mlp = _blob_mlp("cpu")
    Shared.optimizer = torch.optim.Adam(Shared.mlp.parameters(), lr=1e-3)
    Shared.hist = torch.zeros(10000)
    Shared.table = ViewTable([vm], [{"name": "x", "feat_fine": f, "feat_coarse": f[:, :4, :4],
                                     "coarse_image": torch.zeros(3, 10, 12),
                                     "coarse_camera": {}}])
    rgb = torch.rand(H, W, 3, requires_grad=True)
    gt = torch.rand(H, W, 3)
    orig = dino.load_backbone
    dino.load_backbone = lambda *a, **k: None                   # backbone not needed here
    try:
        out = {"rgb": rgb, "step": 600, "camera": {"viewmat": vm}}
        comp({}, out, gt).backward()
    finally:
        dino.load_backbone = orig
    g = rgb.grad.abs().sum(-1)
    mask = out["robustsplat"]["mask"][0].detach()
    inside = g[16:24, 18:30]                                   # well inside the eroded blob
    if float(inside.max()) != 0.0:
        raise AssertionError("transient pixels must receive no photometric gradient")
    if float(g[:5, :5].min()) <= 0.0:
        raise AssertionError("static pixels must receive gradient")
    rgb.grad = None
    out = {"rgb": rgb, "step": 499, "camera": {"viewmat": vm}}
    comp({}, out, gt).backward()
    if float(rgb.grad.abs().sum(-1)[16:24, 18:30].min()) <= 0.0:
        raise AssertionError("before mask_start the plain L1 + D-SSIM loss must cover all pixels")
    Shared.reset()
    return {"masked_fraction": float((mask < 0.25).float().mean()),
            "static_grad_min": float(g[:5, :5].min())}


def mlp_schedule_probe(method, device):
    cfg = _spec(method, "rs-mask-mlp").config
    on = [500, 501, 3000, 12001, 14999, 15000, 15300, 18000, 18300, 20000]
    off = [1, 499, 15001, 15150, 15299, 18001, 18299, 21001, 24299]
    if method.name == "robustsplat-nodg":
        on, off = [500, 3000, 3300], [499, 3001, 3299, 15100]
    bad = [s for s in on if not mlp_active(s, cfg)] + [s for s in off if mlp_active(s, cfg)]
    if bad:
        raise AssertionError(f"MLP schedule wrong at steps {bad}")
    # inactive step: zero attached loss and no Adam step
    Shared.reset()
    Shared.pending_step = False
    loss = MaskMLPLoss(_spec(method, "rs-mask-mlp"))
    rgb = torch.rand(4, 4, 3, requires_grad=True)
    v = loss({}, {"step": off[-1], "rgb": rgb,
                  "robustsplat": {"step": off[-1], "mask": torch.rand(1, 4, 4)}}, None)
    if float(v) != 0.0 or Shared.pending_step or not v.requires_grad:
        raise AssertionError("inactive MLP step must return an attached zero without a step")
    return {"active_checked": on, "inactive_checked": off}


# ----------------------------------------------------------------------------- C2
def _weights():
    path = Path(dino.DEFAULT_WEIGHTS).expanduser()
    if not path.exists():
        raise ProbeUnavailable(f"DINOv2 weights missing: {path}")
    return path


def _synthetic_entry(params, device, H, W, h, w, model):
    vm = torch.eye(4, device=device)
    vm[2, 3] = 2.0
    K = torch.tensor([[W * 0.9, 0, W / 2], [0, W * 0.9, H / 2], [0, 0, 1.0]], device=device)
    Kc = torch.tensor([[w * 0.9, 0, w / 2], [0, w * 0.9, h / 2], [0, 0, 1.0]], device=device)
    cam = {"viewmat": vm, "K": K, "width": W, "height": H}
    ccam = {"viewmat": vm, "K": Kc, "width": w, "height": h}
    with torch.no_grad():
        gt = rasterize(params, cam, 0)[0].permute(2, 0, 1).clamp(0, 1)
        gt[:, H // 4: H // 2, W // 4: W // 2] = torch.tensor([1.0, 0.0, 1.0], device=device)[
            :, None, None]
        cgt = torch.nn.functional.interpolate(gt[None], size=(h, w), mode="area")[0]
    entry = {"name": "synthetic", "feat_fine": dino.extract(model, gt, 36),
             "feat_coarse": dino.extract(model, gt, 16), "coarse_image": cgt,
             "coarse_camera": ccam}
    return cam, entry, gt


def bootstrap_probe(method, device):
    if not str(device).startswith("cuda"):
        raise ProbeUnavailable("needs CUDA")
    model = dino.load_backbone(_weights(), device=device)
    params, _ = _params(3000, device)
    with torch.no_grad():
        params["means"][:, 2] *= 0.5
    H, W, h, w = 96, 128, 24, 32
    cam, entry, gt = _synthetic_entry(params, device, H, W, h, w, model)
    loss = MaskMLPLoss(_spec(method, "rs-mask-mlp"))
    Shared.reset()
    Shared.hist = torch.zeros(10000, device=device)
    out = {}
    with torch.no_grad():
        image = rasterize(params, cam, 0)[0].permute(2, 0, 1)
    for step in (19999, 20000):
        shared = {"step": step, "image": image, "gt": gt, "entry": entry, "dino": model,
                  "grids": (36, 16)}
        hist0 = Shared.hist.clone()
        sup = loss.supervision(params, {"step": step, "sh_degree": 0}, shared)
        out[step] = sup
        if step == 19999 and bool(_spec(method, "rs-mask-mlp").config.get("bootstrap", True)):
            # independent coarse residual: render the coarse camera, same histogram update
            with torch.no_grad():
                c = rasterize(params, entry["coarse_camera"], 0)[0].permute(2, 0, 1)
            lo, _, _ = residual_masks(entry["coarse_image"], c, hist0)
            lo = torch.nn.functional.interpolate(lo[None], size=(H, W), mode="bilinear")[0]
            if (lo - sup["lower"]).abs().max() > 1e-5:
                raise AssertionError("coarse residual mask is not from the coarse render")
            feat = dino.extract(model, image, 16)
            tgt = cosine_target(entry["feat_coarse"], feat, H, W)
            if (tgt - sup["cos_target"]).abs().max() > 1e-5:
                raise AssertionError("coarse cosine target is not from 16x16 features")
    boot = bool(_spec(method, "rs-mask-mlp").config.get("bootstrap", True))
    exp = {19999: (16, (h, w)) if boot else (36, (H, W)), 20000: (36, (H, W))}
    for step, (grid, size) in exp.items():
        if out[step]["grid"] != grid or tuple(out[step]["residual_size"]) != size:
            raise AssertionError(f"step {step}: grid {out[step]['grid']} residual "
                                 f"{out[step]['residual_size']} != expected {grid} {size}")
    diff = float((out[19999]["cos_target"] - out[20000]["cos_target"]).abs().mean())
    if boot and diff == 0.0:
        raise AssertionError("coarse and fine supervision must differ")
    Shared.reset()
    return {"grid": {k: v["grid"] for k, v in out.items()},
            "residual_size": {k: list(v["residual_size"]) for k, v in out.items()},
            "cos_target_mean": {k: float(v["cos_target"].mean()) for k, v in out.items()},
            "coarse_vs_fine_mean_abs_diff": diff}


def dino_parity_probe(method, device):
    if not str(device).startswith("cuda"):
        raise ProbeUnavailable("needs CUDA")
    try:
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
    except ImportError as exc:
        raise ProbeUnavailable("timm not installed") from exc
    path = _weights()
    ours = dino.load_backbone(path, device=device)
    ref = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False, img_size=518,
                            dynamic_img_size=True)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ref.load_state_dict(checkpoint_filter_fn(sd, ref), strict=True)
    ref = ref.to(device).eval()
    torch.manual_seed(0)
    img = torch.rand(3, 120, 160, device=device)
    out = {}
    for g in (16, 36):
        a = dino.extract(ours, img, g)
        x = torch.nn.functional.interpolate(img[None], size=(14 * g, 14 * g), mode="bilinear",
                                            align_corners=False)
        with torch.no_grad():
            b = ref.forward_features(x)[0, 5:].reshape(g, g, -1).permute(2, 0, 1)
        err = float((a - b).abs().max())
        if err > 1e-3:
            raise AssertionError(f"DINOv2 features differ from timm at grid {g}: {err}")
        out[f"max_abs_err_{g}"] = err
    return out


def pipeline_probe(method, device):
    if not str(device).startswith("cuda"):
        raise ProbeUnavailable("needs CUDA")
    model = dino.load_backbone(_weights(), device=device)
    params, opts = _params(3000, device, seed=1)
    with torch.no_grad():
        params["means"][:, 2] *= 0.5
    H, W, h, w = 96, 128, 24, 32
    cam, entry, gt = _synthetic_entry(params, device, H, W, h, w, model)
    Shared.reset()
    Shared.table = ViewTable([cam["viewmat"].cpu()], [entry])
    renderer = Renderer(_spec(method, "rs-renderer"))
    dens = Density(_spec(method, "rs-density"))
    dens.initialize(params, opts, scene_scale=1.0)
    photo = MaskedPhotometric(_spec(method, "rs-masked-photometric"))
    mlp_loss = MaskMLPLoss(_spec(method, "rs-mask-mlp"))
    target = gt.permute(1, 2, 0).contiguous()
    log = []
    snapshot = None
    for step in (1, 499, 500, 501, 502, 10100, 20001, 20002):
        outputs = renderer.render(params, cam, sh_degree=0)
        outputs["step"] = step
        dens.before_backward(params, opts, step, outputs)
        lp = photo(params, outputs, target)
        lm = mlp_loss(params, outputs, target)
        for v in (lp, lm):
            if not torch.isfinite(v) or not v.requires_grad:
                raise AssertionError(f"step {step}: loss not finite/attached")
        (lp + lm).backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        dens.after_backward(params, opts, step, outputs)
        if step == 500:
            snapshot = [p.detach().clone() for p in Shared.mlp.parameters()]
        log.append({"step": step, "photo": float(lp), "mlp": float(lm),
                    "mlp_active": mlp_loss.last.get("active", False),
                    "coarse": mlp_loss.last.get("coarse"), "gaussians": len(params["means"])})
    moved = sum(float((a - b).abs().sum()) for a, b in zip(snapshot, Shared.mlp.parameters()))
    if moved == 0.0:
        raise AssertionError("mask MLP was never updated")
    boot = bool(_spec(method, "rs-mask-mlp").config.get("bootstrap", True))
    expected = [boot] * 4 + [False, False]
    if [r["coarse"] for r in log if r["mlp_active"]] != expected:
        raise AssertionError(f"unexpected supervision scales: {log}")
    Shared.reset()
    return {"log": log, "mlp_param_change": moved}
