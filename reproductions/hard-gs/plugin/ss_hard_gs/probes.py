"""Mechanism probes for the HGS native port (each fails if its mechanism is removed or wrong)."""

import io
import math
from types import SimpleNamespace

import torch

from . import kernels, model
from .density import DEFAULTS, HGSDensity, new_anchor_fields, push_topk, rehgs_hard, ssim_map
from .renderer import BLOB, HGSRenderer, visible_anchors
from .renderer import neural_gaussians as decode_neural


def _need_cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("probe needs CUDA")


def _spec(config):
    return SimpleNamespace(config=config)


def _camera(device, w=64, h=64, f=60.0):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], device=device),
            "width": w, "height": h}


def _fixture(device, n=1200, config=None, seed=4):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dens = HGSDensity(_spec(dict(config or {})))
    dens.c["voxel_size"] = 0.01
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    pts = (pts + torch.tensor([0, 0, 3.0])).to(device)
    c = dens.c
    params = torch.nn.ParameterDict(new_anchor_fields(
        pts, torch.zeros(n, 6, device=device), torch.zeros(n, c["feat_dim"], device=device),
        c["n_offsets"]))
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    dens.initialize(params, opts, 1.0)
    renderer = HGSRenderer(_spec({"feat_dim": c["feat_dim"], "n_offsets": c["n_offsets"]}))
    renderer.bank = dens.bank
    return dens, params, opts, renderer


def _check(params, opts):
    n = len(params["means"])
    for k, p in params.items():
        if p.shape[0] != n:
            raise AssertionError(f"{k} misaligned")
        if opts[k].param_groups[0]["params"][0] is not p:
            raise AssertionError(f"{k} optimizer references a stale parameter")
        if not (p.requires_grad and p.is_leaf):
            raise AssertionError(f"{k} is not a trainable leaf")
        st = opts[k].state.get(p, {})
        if "exp_avg" in st and st["exp_avg"].shape != p.shape:
            raise AssertionError(f"{k} Adam moment misaligned")


def _prime(params, opts):
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)


def _isolate(dens, params, anchor=0):
    """Anchor `anchor` has offsets pointing 2 units behind the cloud; no OG candidate anywhere; nothing is
    pruned. Returns the world positions of its offsets."""
    k = dens.c["n_offsets"]
    with torch.no_grad():
        params["hgs_scaling"].data[:] = math.log(0.05)
        params["hgs_offset"].data[:] = 0
        params["hgs_offset"].data[anchor] = torch.tensor([0, 0, 40.0])  # 2 units: empty space
    dens.offset_gradient_accum.zero_()
    dens.offset_denom[:] = 100
    dens.opacity_accum[:] = 1.0
    dens.anchor_demon[:] = 100
    rows = slice(anchor * k, (anchor + 1) * k)
    return rows, (params["means"][anchor] + params["hgs_offset"][anchor]
                  * params["hgs_scaling"][anchor, :3].exp()).detach()


def _grown_at(params, n_old, targets, dens):
    c = dens.c
    sizes = [c["voxel_size"] * (c["update_init_factor"] // c["update_hierachy_factor"] ** i)
             for i in range(c["update_depth"])]
    new = params["means"][n_old:].detach()
    for a in new:
        if not any(torch.allclose(a, torch.round(t / s) * s, atol=1e-5)
                   for s in sizes for t in targets):
            raise AssertionError("grown anchor is not at the voxel of a hard offset")
    return len(new)


# ------------------------------------------------------------------------------- backbone
def neural_gaussians(method, device):
    g = torch.Generator(device="cpu").manual_seed(3)
    k, f = 4, 8
    torch.manual_seed(3)
    mlps = model.scaffold_mlps(f, k).to(device)
    n = 6
    anchor = torch.randn(n, 3, generator=g).to(device) + torch.tensor([0, 0, 4.0], device=device)
    feat = torch.randn(n, f, generator=g).to(device)
    offsets = torch.randn(n, k, 3, generator=g).to(device)
    scaling = torch.rand(n, 6, generator=g).to(device) + 0.1
    center = torch.zeros(3, device=device)
    xyz, color, opa, scales, quats, _, _ = decode_neural(mlps, anchor, feat, offsets, scaling,
                                                         center, k)
    ref = []
    with torch.no_grad():
        for i in range(n):
            v = anchor[i] - center
            inp = torch.cat([feat[i], v / v.norm()])[None]  # add_*_dist False: no distance
            o = mlps["mlp_opacity"](inp)[0]
            c = mlps["mlp_color"](inp)[0].view(k, 3)
            sr = mlps["mlp_cov"](inp)[0].view(k, 7)
            for j in range(k):
                if o[j] > 0:
                    ref.append(torch.cat([anchor[i] + offsets[i, j] * scaling[i, :3], c[j],
                                          o[j][None], scaling[i, 3:] * torch.sigmoid(sr[j, :3]),
                                          sr[j, 3:] / sr[j, 3:].norm()]))
    ref = torch.stack(ref)
    got = torch.cat([xyz, color, opa[:, None], scales, quats], dim=1)
    if got.shape != ref.shape or (got - ref).abs().max() > 1e-5:
        raise AssertionError("neural Gaussian decode differs from the per-anchor reference")
    c2 = decode_neural(mlps, anchor, feat, offsets, scaling, torch.tensor([3.0, 0, 0],
                                                                           device=device), k)[1]
    if c2.shape == color.shape and torch.allclose(c2, color):
        raise AssertionError("decode is not view dependent")
    return {"gaussians": len(ref), "max_error": float((got - ref).abs().max())}


def visibility_filter(method, device):
    cam = _camera(device)
    anchors = torch.tensor([[0.0, 0, 3], [0, 0, -3], [40.0, 0, 3], [2.4, 0, 3], [0, 0, 0.1]],
                           device=device)
    scales = torch.full((5, 3), 0.01, device=device)
    scales[3] = 0.5  # centre off-screen but footprint reaches the image
    got = visible_anchors(anchors, scales, cam).tolist()
    want = [True, False, False, True, False]
    if got != want:
        raise AssertionError(f"visible filter {got} != {want}")
    return {"visible": got}


def anchor_density(method, device):
    _need_cuda(device)
    dens, params, opts, _ = _fixture(device)
    k = dens.c["n_offsets"]
    _prime(params, opts)
    rows, targets = _isolate(dens, params)
    dens.offset_gradient_accum[rows] = 1.0 * 100  # OG: average gradient 1 >> threshold
    dens.opacity_accum[5] = 0.0  # anchor 5: low opacity, visible often -> pruned
    means_before = params["means"].detach().clone()
    n0 = len(params["means"])
    torch.manual_seed(0)
    dens.adjust_anchor(params, opts, 2000)
    _check(params, opts)
    kept_old = torch.cat([means_before[:5], means_before[6:]])
    if not torch.equal(params["means"][: n0 - 1].detach(), kept_old):
        raise AssertionError("pruning did not remove exactly the low-opacity anchor")
    grown = _grown_at(params, n0 - 1, targets, dens)
    if grown == 0:
        raise AssertionError("no anchor grown at the high-gradient offset")
    n1 = len(params["means"])
    if len(dens.offset_denom) != n1 * k or len(dens.grad_topk) != n1 * k:
        raise AssertionError("density statistics misaligned with the population")
    return {"anchors_before": n0, "anchors_after": n1, "grown": grown}


# ---------------------------------------------------------------------------------- PGHGS
def pghgs_criterion(method, device):
    _need_cuda(device)
    # 1) running top-k equals a sort of all per-view gradients (Eq. 5)
    g = torch.Generator(device="cpu").manual_seed(0)
    buf = torch.zeros(20, 3, device=device)
    hist = [[] for _ in range(20)]
    for _ in range(30):
        ids = torch.randperm(20, generator=g)[:7].to(device)
        vals = torch.rand(7, generator=g).to(device)
        push_topk(buf, ids, vals)
        for i, v in zip(ids.tolist(), vals.tolist()):
            hist[i].append(v)
    ref = torch.tensor([sorted(h + [0, 0, 0], reverse=True)[:3] for h in hist], device=device)
    if (buf - ref).abs().max() > 1e-6:
        raise AssertionError("top-k buffer differs from the sorted per-view gradients")

    # 2) average below tau, 3rd-largest above lambda*tau -> grown only with PGHGS (Eq. 6)
    out = {}
    tau = DEFAULTS["densify_grad_threshold"]
    for enabled in (True, False):
        dens, params, opts, _ = _fixture(device, config={"pghgs": {"enabled": enabled}})
        _prime(params, opts)
        rows, targets = _isolate(dens, params)
        dens.offset_gradient_accum[rows] = 0.5 * tau * 100  # avg 0.5 tau: OG never fires
        dens.grad_topk[rows] = torch.tensor([9 * tau, 6 * tau, 5 * tau], device=device)
        n0 = len(params["means"])
        torch.manual_seed(0)
        dens.adjust_anchor(params, opts, 2000)
        _check(params, opts)
        grown = _grown_at(params, n0, targets, dens)
        out[f"grown_{'on' if enabled else 'off'}"] = grown
        if enabled and grown == 0:
            raise AssertionError("PGHGS did not grow a Gaussian with 3 large per-view gradients")
        if not enabled and grown != 0:
            raise AssertionError("growth without PGHGS from a low average gradient")
        if float(dens.grad_topk.abs().sum()) != 0:
            raise AssertionError("PGHGS statistics not reset after the growth interval")
    # 3) k-th (not max) is the statistic: only 2 large views -> not hard
    dens, params, opts, _ = _fixture(device)
    _prime(params, opts)
    rows, _ = _isolate(dens, params)
    dens.offset_gradient_accum[rows] = 0.5 * tau * 100
    dens.grad_topk[rows] = torch.tensor([9 * tau, 6 * tau, 0.5 * tau], device=device)
    n0 = len(params["means"])
    dens.adjust_anchor(params, opts, 2000)
    if len(params["means"]) != n0:
        raise AssertionError("PGHGS grew with only 2 views above the threshold (k=3)")
    out["topk_max_error"] = float((buf - ref).abs().max())
    return out


# ---------------------------------------------------------------------------------- REHGS
def _area_max_reference(means2d, conics, opacities, depths, width, height):
    """Brute-force argmax_i w_i per pixel with gsplat numerics -> |S_i| (int64 [M])."""
    n = means2d.shape[0]
    order = torch.argsort(depths).tolist()
    area = torch.zeros(n, dtype=torch.int64)
    for i in range(height):
        for j in range(width):
            px, py = j + 0.5, i + 0.5
            T, wmax, imax = 1.0, 0.0, -1
            for gi in order:
                dx, dy = float(means2d[gi, 0]) - px, float(means2d[gi, 1]) - py
                a, b, c = (float(v) for v in conics[gi])
                sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                alpha = min(0.999, float(opacities[gi]) * math.exp(-sigma))
                if sigma < 0 or alpha < 1 / 255:
                    continue
                nxt = T * (1 - alpha)
                if nxt <= 1e-4:
                    break
                if alpha * T > wmax:
                    wmax, imax = alpha * T, gi
                T = nxt
            if imax >= 0:
                area[imax] += 1
    return area


def rehgs_criterion(method, device):
    _need_cuda(device)
    from gsplat import rasterization

    # 1) |S_i| from the CUDA kernel equals a brute-force argmax reference (Eq. 7 input)
    g = torch.Generator(device="cpu").manual_seed(1)
    w = h = 32
    n = 12
    means = torch.cat([(torch.rand(n, 2, generator=g) - 0.5) * 1.5,
                       2.5 + torch.rand(n, 1, generator=g)], 1).to(device)
    scales = (torch.rand(n, 3, generator=g) * 0.3 + 0.05).to(device)
    scales[0] = 0.8  # one over-large Gaussian
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device)
    opac = (torch.rand(n, generator=g) * 0.6 + 0.3).to(device)
    cam = _camera(device, w, h, 30.0)
    _, _, info = rasterization(means, quats, scales, opac, torch.rand(n, 3, device=device),
                               cam["viewmat"][None], cam["K"][None], w, h, packed=False)
    area, _ = kernels.max_contrib_counts(info)
    ref = _area_max_reference(info["means2d"][0].cpu().double(), info["conics"][0].cpu().double(),
                              info["opacities"][0].cpu().double(), info["depths"][0].cpu(), w, h)
    diff = int((area.cpu().long() - ref).abs().sum())
    if diff > 2 or int(ref.sum()) < w * h // 2:
        raise AssertionError(f"|S_i| kernel differs from the reference by {diff} pixels")

    # 2) Eq. 7 + Eq. 8 on constructed per-Gaussian values
    m2d = torch.tensor([[5.0, 5.0], [5.0, 5.0], [20.0, 20.0], [40.0, 5.0]], device=device)
    am = torch.tensor([500, 500, 3, 500], device=device, dtype=torch.int32)
    smap = torch.ones(32, 32, device=device)
    smap[5, 5] = 0.3
    smap[20, 20] = 0.3
    ok = torch.ones(4, dtype=torch.bool, device=device)
    hard = rehgs_hard(am, m2d, ok, smap, 32, 32, 0.2, 0.7).tolist()
    # large+low SSIM -> hard; small (Eq. 7 fails) -> not; projected outside -> not
    smap_hi = torch.ones(32, 32, device=device)
    hard_hi = rehgs_hard(am, m2d, ok, smap_hi, 32, 32, 0.2, 0.7).tolist()
    if hard != [True, True, False, False] or any(hard_hi):
        raise AssertionError(f"REHGS selection wrong: {hard} / {hard_hi}")
    # SSIM map sanity: identical images -> 1, noise -> < 0.7
    img = torch.rand(32, 32, 3, generator=g).to(device)
    if (ssim_map(img, img) - 1).abs().max() > 1e-4 or ssim_map(img, torch.rand_like(img)).mean() > 0.7:
        raise AssertionError("ssim_map is not the 3DGS SSIM map")

    # 3) density: hard in >= 2 views grows, 1 view does not, disabled does not
    out = {"area_max_abs_diff": diff}
    for count, enabled, expect in ((2, True, True), (1, True, False), (2, False, False)):
        dens, params, opts, _ = _fixture(device, config={"rehgs": {"enabled": enabled},
                                                         "pghgs": {"enabled": False}})
        _prime(params, opts)
        rows, targets = _isolate(dens, params)
        dens.re_count[rows] = count
        n0 = len(params["means"])
        torch.manual_seed(0)
        dens.adjust_anchor(params, opts, 2000)
        _check(params, opts)
        grown = _grown_at(params, n0, targets, dens)
        out[f"grown_views{count}_{'on' if enabled else 'off'}"] = grown
        if (grown > 0) != expect:
            raise AssertionError(f"REHGS growth wrong for {count} views, enabled={enabled}")
    # 4) the training-time statistic: a real render with the tapped target counts detections
    dens, params, opts, renderer = _fixture(device, n=1200)
    with torch.no_grad():
        params["hgs_scaling"].data[:] = math.log(0.3)  # big neural Gaussians -> large |S_i|
        params["hgs_feat"].data.normal_()
    cam = _camera(device)
    outputs = renderer.render(params, cam)
    outputs["hgs_target"] = torch.rand(64, 64, 3, device=device)  # unrelated image: low SSIM
    dens.before_backward(params, opts, 600, outputs)
    outputs["rgb"].sum().backward()
    dens.training_statis(outputs)
    out["detections_one_view"] = int(dens.re_count.sum())
    if out["detections_one_view"] == 0:
        raise AssertionError("REHGS found no over-large low-SSIM Gaussian in a blurry render")
    return out


# ------------------------------------------------------------------------------- eval path
def blob_eval(method, device):
    _need_cuda(device)
    dens, params, opts, renderer = _fixture(device, n=2000)
    with torch.no_grad():
        params["hgs_feat"].data.normal_()
        params["hgs_offset"].data.normal_(0, 0.5)
        params["hgs_scaling"].data[:] = math.log(0.05)
        for p in dens.bank.mlps.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    dens.pack(params)
    cam = _camera(device)
    with torch.no_grad():
        live = renderer.render(params, cam)["rgb"]
    buf = io.BytesIO()
    torch.save({k: v.detach() for k, v in params.items()}, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    saved = model.ACTIVE["bank"]
    model.ACTIVE["bank"] = None
    try:
        fresh = HGSRenderer(_spec({}))
        with torch.no_grad():
            again = fresh.render(loaded, cam)["rgb"]
        empty = dict(loaded)
        empty[BLOB] = torch.zeros_like(loaded[BLOB])
        try:
            HGSRenderer(_spec({})).render(empty, cam)
            raise AssertionError("an untrained (empty-blob) state rendered")
        except ValueError:
            pass
    finally:
        model.ACTIVE["bank"] = saved
    err = float((live - again).abs().max())
    if err > 1e-6:
        raise AssertionError(f"fresh-process render differs from the live render ({err})")
    return {"max_abs_diff": err}
