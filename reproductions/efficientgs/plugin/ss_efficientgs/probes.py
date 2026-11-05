"""EfficientGS mechanism probes. Each one fails if its mechanism is removed or wrong.

References are written independently of the CUDA kernels: a brute-force per-pixel blend in
torch (autograd for the per-pixel gradients), gsplat's own SH evaluation and backward pass.
"""

import torch
from torch import nn

from . import core, kernels
from .density import DEFAULTS, EfficientGSDensity
from .renderer import EfficientGSRenderer


class _Spec:
    def __init__(self, config):
        self.config = config


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _scene(means, scales, opacities, quats=None, colors=None, device="cuda", shN=None,
           order=None):
    n = len(means)
    if quats is None:
        quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    if colors is None:
        colors = torch.full((n, 3), 0.5)
    values = {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
        "quats": torch.as_tensor(quats, dtype=torch.float32),
        "opacities": torch.logit(torch.as_tensor(opacities, dtype=torch.float32)),
        "sh0": ((torch.as_tensor(colors, dtype=torch.float32) - 0.5) / core.SH_C0)[:, None, :],
        "shN": torch.zeros(n, 15, 3) if shN is None else shN.float(),
        "sh_order": torch.zeros(n, 1) if order is None else order.float().reshape(n, 1),
    }
    params = nn.ParameterDict({k: nn.Parameter(v.to(device)) for k, v in values.items()})
    rates = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 0.05, "sh0": 2.5e-3,
             "shN": 1.25e-4, "sh_order": 1e-12}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    return params, opts


def _camera(width, height, f, device="cuda", image=None):
    cam = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device),
        "width": width,
        "height": height,
    }
    if image is not None:
        cam["image"] = image.to(device)
    return cam


def _renderer():
    return EfficientGSRenderer(None)


def _density(**overrides):
    return EfficientGSDensity(_Spec({**DEFAULTS, **overrides}))


def _random_scene(seed, n, device):
    g = torch.Generator().manual_seed(seed)
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1,
                       torch.rand(n, 1, generator=g) * 3 + 2], 1)
    means[:, :2] *= means[:, 2:3] * 0.45
    scales = torch.rand(n, 3, generator=g) * 0.25 + 0.05
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.rand(n, generator=g) * 0.8 + 0.15
    colors = torch.rand(n, 3, generator=g)
    return _scene(means, scales, opac, quats, colors, device=device), g


def _forward_backward(params, cam, target):
    out = _renderer().render(params, cam)
    out["info"]["means2d"].retain_grad()
    out["raw_rgb"].retain_grad()
    out["raw_alpha"].retain_grad()
    loss = (out["rgb"] - target).abs().mean() + 0.3 * (out["rgb"] - target).square().mean()
    loss.backward()
    return out


def _symmetric_fixture(device, size=32):
    """One isotropic Gaussian centred on a symmetric pixel grid, uniform wrong target: every
    pixel pulls the centre, the pulls cancel (Fig. 3, left)."""
    params, opts = _scene([[0.0, 0.0, 4.0]], [[0.6, 0.6, 0.6]], [0.8],
                          colors=[[0.2, 0.4, 0.6]], device=device)
    cam = _camera(size, size, 40.0, device=device)
    target = torch.full((size, size, 3), 0.9, device=device)
    return params, opts, cam, target


def gradnorm_kernel(method, device):
    _require_cuda()
    (params, _), g = _random_scene(5, 8, device)
    w, h = 20, 16
    cam = _camera(w, h, 22.0, device=device)
    target = torch.rand(h, w, 3, generator=g).to(device)
    out = _forward_backward(params, cam, target)
    info = out["info"]
    norm, vec = kernels.gradnorm(info, out["colors"], out["raw_rgb"].grad[0],
                                 out["raw_alpha"].grad[0])
    gs = info["means2d"].grad[0]
    scale_vec = gs.abs().max().item()
    err_vec = (vec - gs).abs().max().item() / max(scale_vec, 1e-12)
    ref_norm, ref_vec = core.gradnorm_reference(
        info["means2d"][0].detach().cpu().double(), info["conics"][0].detach().cpu().double(),
        info["opacities"][0].detach().cpu().double(), out["colors"].cpu().double(),
        info["depths"][0].detach().cpu(), w, h, out["raw_rgb"].grad[0].cpu(),
        out["raw_alpha"].grad[0, ..., 0].cpu(),
    )
    err_norm = ((norm.cpu().double() - ref_norm).abs().max() / ref_norm.abs().max()).item()
    err_ref_vec = ((ref_vec - gs.cpu().double()).abs().max() / max(scale_vec, 1e-12)).item()
    if err_vec > 1e-3 or err_norm > 1e-3 or err_ref_vec > 1e-3:
        raise AssertionError(f"Eq. 6 kernel mismatch: vec {err_vec}, norm {err_norm}, "
                             f"reference vs gsplat {err_ref_vec}")
    # cancellation case: the Eq. 6 statistic must not collapse to |sum of gradients|
    p2, _, cam2, t2 = _symmetric_fixture(device)
    out2 = _forward_backward(p2, cam2, t2)
    n2, v2 = kernels.gradnorm(out2["info"], out2["colors"], out2["raw_rgb"].grad[0],
                              out2["raw_alpha"].grad[0])
    summed = (v2[0] * torch.tensor([cam2["width"] / 2, cam2["height"] / 2],
                                   device=device)).norm().item()
    if not n2[0].item() > 50 * summed or n2[0].item() <= 0:
        raise AssertionError(f"Cancellation fixture: S_g {n2[0].item()} vs |sum| {summed}")
    return {"rel_err_vs_gsplat_grad": err_vec, "rel_err_vs_autograd_norm": err_norm,
            "reference_vs_gsplat": err_ref_vec, "cancel_S_g": n2[0].item(),
            "cancel_E_g": summed}


def selective_densify(method, device):
    _require_cuda()
    values = {}
    for selective in (True, False):
        params, opts, cam, target = _symmetric_fixture(device)
        d = _density(selective=selective)
        d.initialize(params, opts, scene_scale=100.0)  # small Gaussian -> clone branch
        out = _renderer().render(params, cam)
        d.before_backward(params, opts, 600, out)
        loss = (out["rgb"] - target).abs().mean()
        loss.backward()
        d._accumulate(out)
        values[selective] = float(d.stats["grad"][0] / d.stats["count"][0])
    s_g, e_g = values[True], values[False]
    if not s_g > 50 * e_g:
        raise AssertionError(f"S_g {s_g} not >> E_g {e_g}")
    threshold = s_g / 7.0  # e_g < s_g / 50 < threshold <= s_g
    counts = {}
    for selective in (True, False):
        params, opts, cam, target = _symmetric_fixture(device)
        d = _density(selective=selective, grad_threshold=threshold)
        d.initialize(params, opts, scene_scale=100.0)
        out = _renderer().render(params, cam)
        d.before_backward(params, opts, 600, out)
        (out["rgb"] - target).abs().mean().backward()
        d.after_backward(params, opts, 600, out)
        counts[selective] = len(params["means"])
    if counts[True] != 2 or counts[False] != 1:
        raise AssertionError(f"Densification counts selective/3DGS {counts}")
    return {"S_g": s_g, "E_g": e_g, "threshold": threshold, "n_after_selective": counts[True],
            "n_after_3dgs_statistic": counts[False]}


def stats_kernel(method, device):
    _require_cuda()
    (params, _), g = _random_scene(3, 20, device)
    w, h = 36, 28
    cam = _camera(w, h, 36.0, device=device)
    with torch.no_grad():
        out = _renderer().render(params, cam)
    info = out["info"]
    pix_loss = torch.rand(h, w, 3, generator=g).to(device)
    evidence = {}
    for after in (True, False):
        for k in (1, 2):
            got = kernels.blend_stats(info, k, after, pix_loss=pix_loss)
            ref = core.blend_reference(
                info["means2d"][0].cpu().double(), info["conics"][0].cpu().double(),
                info["opacities"][0].cpu().double(), out["colors"].cpu().double(),
                info["depths"][0].cpu(), w, h, top_k=k, weight_after_T=after,
                pix_loss=pix_loss.cpu(),
            )
            flags = int((got["dominant"].cpu().long() != ref["dominant"]).sum())
            err = ((got["loss_weight"].cpu().double() - ref["loss_weight"]).abs().max()
                   / ref["loss_weight"].abs().max()).item()
            if flags > 0 or err > 1e-4:
                raise AssertionError(f"K={k} after_T={after}: flags {flags}, lw err {err}")
            evidence[f"k{k}_afterT{int(after)}_dominant"] = int(got["dominant"].sum())
            evidence[f"k{k}_afterT{int(after)}_lw_rel_err"] = err
        rgb_err = (ref["rgb"] - (out["raw_rgb"][0].cpu().double())).abs().max().item()
        if rgb_err > 1e-4:
            raise AssertionError(f"Reference blend differs from gsplat render: {rgb_err}")
    return evidence


def topk_prune(method, device):
    _require_cuda()
    # A: large, opaque, in front. B: small, behind A, inside A's footprint. C: in front, off-centre.
    means = [[0.0, 0.0, 3.0], [0.0, 0.0, 3.5], [0.75, 0.5, 2.5]]
    scales = [[0.5, 0.5, 0.5], [0.08, 0.08, 0.08], [0.1, 0.1, 0.1]]
    kept = {}
    for k in (1, 2):
        params, opts = _scene(means, scales, [0.95, 0.9, 0.8], device=device)
        d = _density(top_k=k)
        d.initialize(params, opts, scene_scale=1.0)
        d.set_training_cameras([_camera(48, 40, 40.0, device=device)])
        d._prune(params, opts, 15000)
        kept[k] = sorted(round(float(z), 2) for z in params["means"][:, 2])
    if kept[1] != [2.5, 3.0] or kept[2] != [2.5, 3.0, 3.5]:
        raise AssertionError(f"Top-K pruning kept depths {kept}")
    return {"kept_depths_k1": kept[1], "kept_depths_k2": kept[2]}


def sparse_sh(method, device):
    _require_cuda()
    from gsplat import rasterization, spherical_harmonics

    g = torch.Generator().manual_seed(11)
    n = 16
    (params, _), _ = _random_scene(7, n, device)
    shN = (torch.randn(n, 15, 3, generator=g) * 0.3).to(device)
    order = torch.arange(n) % 4
    params["shN"].data.copy_(shN)
    params["sh_order"].data.copy_(order.float()[:, None].to(device))
    cam = _camera(32, 24, 30.0, device=device)
    r = _renderer()
    colors, degree = r.colors(params, cam, sh_degree=3)
    dirs = params["means"] - torch.linalg.inv(cam["viewmat"])[:3, 3]
    coeffs = torch.cat((params["sh0"], params["shN"]), 1)
    worst = 0.0
    for d in range(4):
        sel = order.to(device) == d
        ref = torch.clamp_min(spherical_harmonics(d, dirs[sel], coeffs[sel][:, : (d + 1) ** 2])
                              + 0.5, 0)
        worst = max(worst, (colors[sel] - ref).abs().max().item())
    if worst > 1e-5 or degree != 3:
        raise AssertionError(f"Per-Gaussian SH colours differ: {worst}")
    # whole render at uniform degree 1 equals gsplat's own SH path
    params["sh_order"].data.fill_(1)
    out = r.render(params, cam)
    rgb, alpha, _ = rasterization(
        params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        coeffs[:, :4], cam["viewmat"][None], cam["K"][None], 32, 24, sh_degree=1, packed=False)
    render_err = (out["raw_rgb"] - rgb).abs().max().item()
    out["rgb"].sum().backward()
    leak = params["shN"].grad[:, 3:].abs().max().item()
    if render_err > 1e-5 or leak != 0.0 or params["shN"].grad[:, :3].abs().max().item() == 0:
        raise AssertionError(f"degree-1 render err {render_err}, higher-band gradient {leak}")
    # Eq. 9 selection: the wrongly coloured Gaussian gets the extra degree, capped at 3
    means = [[-0.6, 0.0, 3.0], [0.0, 0.0, 3.0], [0.6, 0.0, 3.0], [0.0, 0.5, 3.0], [0.0, -0.5, 3.0]]
    cols = [[0.5, 0.5, 0.5] for _ in range(5)]
    cols[2] = [0.05, 0.9, 0.1]
    p5, o5 = _scene(means, [[0.12, 0.12, 0.12]] * 5, [0.9] * 5, colors=cols, device=device)
    cam5 = _camera(48, 40, 40.0, device=device)
    # target: the same scene with the correct (grey) colour -> only Gaussian 2 is wrong
    grey, _ = _scene(means, [[0.12, 0.12, 0.12]] * 5, [0.9] * 5, device=device)
    with torch.no_grad():
        target = _renderer().render(grey, cam5)["rgb"].clamp(0, 1)
    d = _density(sparse_rate=0.0)
    d.initialize(p5, o5, 1.0)
    d.set_training_cameras([cam5])
    d.images[0] = (target * 255).round().to(torch.uint8)
    d._cached = 1
    for _ in range(5):
        d._sparse_increment(p5, 15999)
    got = p5["sh_order"][:, 0].tolist()
    if got != [0, 0, 3, 0, 0]:
        raise AssertionError(f"Sparse increments gave degrees {got}")
    return {"max_colour_err": worst, "degree1_render_err": render_err,
            "masked_band_grad": leak, "degrees_after_5_increments": got}


def schedule(method, device):
    d = _density()
    if d.sparse_steps != (16000, 17000, 18000):
        raise AssertionError(f"sparse steps {d.sparse_steps}")
    if list(d.cfg["prune_steps"]) != [15000, 18000, 21000] or d.cfg["top_k"] != 1:
        raise AssertionError("prune schedule differs from official prune_way p4 / K=1")
    params, opts = _scene([[0, 0, 3.0], [0.1, 0, 3.0]], [[0.1] * 3] * 2, [0.5, 0.5],
                          device="cpu")
    d.initialize(params, opts, 1.0)
    lr = opts["shN"].param_groups[0]["lr"]
    if abs(lr - 5e-4) > 1e-12:
        raise AssertionError(f"shN lr {lr}")
    params["shN"].data.fill_(0.3)
    params["sh_order"].data.fill_(2)
    opts["shN"].state[params["shN"]] = {"exp_avg": torch.ones_like(params["shN"])}
    d.after_backward(params, opts, 15499, {})
    if params["shN"].abs().max() != 0 or opts["shN"].state or params["sh_order"].abs().max() != 0:
        raise AssertionError("SH allocation at 15500 did not reset shN / Adam / degrees")
    if d.cfg["grad_threshold"] != 0.0007 or d.cfg["sparse_rate"] != 0.2:
        raise AssertionError("paper hyper-parameters changed")
    return {"sparse_steps": list(d.sparse_steps), "prune_steps": list(d.cfg["prune_steps"]),
            "shN_lr": lr, "events": d.state["events"]}
