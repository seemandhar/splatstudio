"""Mechanism probes for the Scaffold-GS native port (each fails if its mechanism is removed or wrong).
Backbone probes adapted from the HGS native plugin (reproductions/hard-gs/plugin/ss_hard_gs/probes.py)."""

import io
import math
from types import SimpleNamespace

import torch

from . import model
from .density import ScaffoldDensity, new_anchor_fields
from .renderer import BLOB, ScaffoldRenderer, visible_anchors
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
    dens = ScaffoldDensity(_spec(dict(config or {})))
    dens.c["voxel_size"] = 0.01
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    pts = (pts + torch.tensor([0, 0, 3.0])).to(device)
    c = dens.c
    params = torch.nn.ParameterDict(new_anchor_fields(
        pts, torch.zeros(n, 6, device=device), torch.zeros(n, c["feat_dim"], device=device), c["n_offsets"]))
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    dens.initialize(params, opts, 1.0)
    renderer = ScaffoldRenderer(_spec({"feat_dim": c["feat_dim"], "n_offsets": c["n_offsets"]}))
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
    """Anchor `anchor` has offsets pointing 2 units into empty space; no candidate anywhere; nothing is pruned.
    Returns its offset rows and their world positions."""
    k = dens.c["n_offsets"]
    with torch.no_grad():
        params["sgs_scaling"].data[:] = math.log(0.05)
        params["sgs_offset"].data[:] = 0
        params["sgs_offset"].data[anchor] = torch.tensor([0, 0, 40.0])
    dens.offset_gradient_accum.zero_()
    dens.offset_denom[:] = 100
    dens.opacity_accum[:] = 1.0
    dens.anchor_demon[:] = 100
    rows = slice(anchor * k, (anchor + 1) * k)
    return rows, (params["means"][anchor] + params["sgs_offset"][anchor]
                  * params["sgs_scaling"][anchor, :3].exp()).detach()


def _grown_at(params, n_old, targets, dens):
    c = dens.c
    sizes = [c["voxel_size"] * (c["update_init_factor"] // c["update_hierachy_factor"] ** i)
             for i in range(c["update_depth"])]
    new = params["means"][n_old:].detach()
    for a in new:
        if not any(torch.allclose(a, torch.round(t / s) * s, atol=1e-5) for s in sizes for t in targets):
            raise AssertionError("grown anchor is not at the voxel of a high-gradient offset")
    return len(new)


# ------------------------------------------------------------------------------------ C1
def anchor_init(method, device):
    _need_cuda(device)
    g = torch.Generator(device="cpu").manual_seed(7)
    pts = (torch.rand(5000, 3, generator=g) * 0.5).to(device)
    dens = ScaffoldDensity(_spec({"voxel_size": 0.02}))
    params = torch.nn.ParameterDict({"means": torch.nn.Parameter(pts.clone())})
    opts = {"means": torch.optim.Adam([params["means"]], lr=1.0)}
    try:
        dens.initialize(params, opts, 1.0)
    finally:
        model.ACTIVE["bank"] = None
    want = torch.unique(torch.round(pts / 0.02), dim=0) * 0.02  # Eq. 2
    got = params["means"].detach()
    if got.shape != want.shape or not torch.allclose(torch.unique(got, dim=0), want, atol=1e-6):
        raise AssertionError("anchors are not the de-duplicated voxel centres of the SfM cloud")
    if len(got) >= len(pts):
        raise AssertionError("voxelization did not reduce the point count")
    d2 = torch.cdist(got, got).square()
    d2.fill_diagonal_(float("inf"))
    ref = torch.log(torch.sqrt(d2.topk(3, dim=1, largest=False).values.mean(1).clamp_min(1e-7)))
    err = float((params["sgs_scaling"][:, 0].detach() - ref).abs().max())
    if err > 1e-3 or not torch.equal(params["sgs_scaling"][:, :1].expand(-1, 6), params["sgs_scaling"].detach()):
        raise AssertionError(f"initial anchor scaling differs from the 3-NN reference ({err})")
    if params["sgs_offset"].abs().sum() != 0 or params["sgs_feat"].abs().sum() != 0:
        raise AssertionError("offsets / features must start at zero")
    return {"points": len(pts), "anchors": len(got), "scaling_max_error": err}


# ------------------------------------------------------------------------------------ C2
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
    xyz, color, opa, scales, quats, _, _ = decode_neural(mlps, anchor, feat, offsets, scaling, center, k)
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
                    ref.append(torch.cat([anchor[i] + offsets[i, j] * scaling[i, :3], c[j], o[j][None],
                                          scaling[i, 3:] * torch.sigmoid(sr[j, :3]), sr[j, 3:] / sr[j, 3:].norm()]))
    ref = torch.stack(ref)
    got = torch.cat([xyz, color, opa[:, None], scales, quats], dim=1)
    if got.shape != ref.shape or (got - ref).abs().max() > 1e-5:
        raise AssertionError("neural Gaussian decode differs from the per-anchor reference")
    c2 = decode_neural(mlps, anchor, feat, offsets, scaling, torch.tensor([3.0, 0, 0], device=device), k)[1]
    if c2.shape == color.shape and torch.allclose(c2, color):
        raise AssertionError("decode is not view dependent")
    return {"gaussians": len(ref), "max_error": float((got - ref).abs().max())}


def visibility_filter(method, device):
    cam = _camera(device)
    anchors = torch.tensor([[0.0, 0, 3], [0, 0, -3], [40.0, 0, 3], [2.4, 0, 3], [0, 0, 0.1]], device=device)
    scales = torch.full((5, 3), 0.01, device=device)
    scales[3] = 0.5  # centre off-screen but footprint reaches the image
    got = visible_anchors(anchors, scales, cam).tolist()
    want = [True, False, False, True, False]
    if got != want:
        raise AssertionError(f"visible filter {got} != {want}")
    return {"visible": got}


def opacity_selection(method, device):
    _need_cuda(device)
    dens, params, opts, renderer = _fixture(device, n=1200)
    try:
        with torch.no_grad():
            params["sgs_feat"].data.normal_()
            params["sgs_offset"].data.normal_(0, 0.5)
            params["sgs_scaling"].data[:] = math.log(0.05)
        out = renderer.render(params, _camera(device))
        sgs, info = out["info"]["sgs"], out["info"]
        raw = sgs["neural_opacity"].view(-1)
        sel = sgs["selection"]
        n_vis = int(sgs["visible"].sum())
        if not torch.equal(sel, raw > 0):
            raise AssertionError("selection is not opacity > 0")
        if int(sel.sum()) == 0 or int(sel.sum()) == len(sel):
            raise AssertionError("fixture does not exercise the selection")
        if info["means2d"].shape[1] != int(sel.sum()) or len(sel) != n_vis * dens.c["n_offsets"]:
            raise AssertionError("rasterized Gaussian count != selected neural Gaussians")
    finally:
        model.ACTIVE["bank"] = None
    # straight-through cap: forward value capped, gradient of the uncapped value
    x = torch.tensor([0.5, 0.9995, 1.0], device=device, requires_grad=True)
    y = x - (x - 0.999).clamp(min=0).detach()
    y.sum().backward()
    if y.max() > 0.999 + 1e-7 or not torch.allclose(x.grad, torch.ones_like(x)):
        raise AssertionError("opacity cap is not straight-through")
    if renderer.opacity_cap != 0.999:
        raise AssertionError("renderer opacity cap not configured")
    return {"visible_anchors": n_vis, "selected": int(sel.sum()), "spawned": len(sel)}


# ------------------------------------------------------------------------------------ C3
def anchor_density(method, device):
    _need_cuda(device)
    out = {}
    try:
        for grad, expect in ((1.0, True), (0.5 * 2e-4, False)):
            dens, params, opts, _ = _fixture(device)
            _prime(params, opts)
            rows, targets = _isolate(dens, params)
            dens.offset_gradient_accum[rows] = grad * 100  # average = grad (denom 100)
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
            if (grown > 0) != expect:
                raise AssertionError(f"growth {grown} for average gradient {grad} (expected {expect})")
            n1 = len(params["means"])
            if len(dens.offset_denom) != n1 * dens.c["n_offsets"] or len(dens.opacity_accum) != n1:
                raise AssertionError("density statistics misaligned with the population")
            out[f"grown_grad_{grad:g}"] = grown
            out["anchors_after"] = n1
    finally:
        model.ACTIVE["bank"] = None
    return out


def blob_eval(method, device):
    _need_cuda(device)
    dens, params, opts, renderer = _fixture(device, n=2000)
    with torch.no_grad():
        params["sgs_feat"].data.normal_()
        params["sgs_offset"].data.normal_(0, 0.5)
        params["sgs_scaling"].data[:] = math.log(0.05)
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
    model.ACTIVE["bank"] = None
    fresh = ScaffoldRenderer(_spec({}))
    with torch.no_grad():
        again = fresh.render(loaded, cam)["rgb"]
    empty = dict(loaded)
    empty[BLOB] = torch.zeros_like(loaded[BLOB])
    try:
        ScaffoldRenderer(_spec({})).render(empty, cam)
        raise AssertionError("an untrained (empty-blob) state rendered")
    except ValueError:
        pass
    err = float((live - again).abs().max())
    if err > 1e-6:
        raise AssertionError(f"fresh-process render differs from the live render ({err})")
    return {"max_abs_diff": err}

