"""Mechanism probes for HDGS. Each raises AssertionError when its mechanism is removed/wrong."""

import math
from types import SimpleNamespace

import torch

from . import fisher, reference
from . import texture as tex
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import Density, expon_lr
from .losses import create_geometry
from .raster import activated, raster_settings, rasterize

W_IMG, H_IMG = 32, 24


def camera(device, W=W_IMG, H=H_IMG):
    K = torch.tensor([[28.0, 0, W / 2 + 0.3], [0, 28.0, H / 2 - 0.2], [0, 0, 1]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": W, "height": H}


def scene(device, n=10, seed=3, opacity=(0.35, 0.85), spread=1.2):
    """Surfels 2-4 units in front of the camera, roughly facing it, overlapping."""
    g = torch.Generator().manual_seed(seed)
    means = torch.cat([(torch.rand(n, 2, generator=g) - 0.5) * spread,
                       2.0 + 2.0 * torch.rand(n, 1, generator=g)], 1)
    axis = torch.randn(n, 3, generator=g) * 0.5
    axis[:, 2] = 0
    ang = torch.rand(n, generator=g) * 1.2
    quats = torch.cat([torch.cos(ang / 2)[:, None],
                       torch.nn.functional.normalize(axis + 1e-6, dim=-1) * torch.sin(ang / 2)[:, None]], 1)
    scales = 0.08 + 0.25 * torch.rand(n, 2, generator=g)
    lo, hi = opacity
    opac = lo + (hi - lo) * torch.rand(n, 1, generator=g)
    sh = torch.randn(n, 16, 3, generator=g) * 0.3
    sh[:, 0] = (torch.rand(n, 3, generator=g) - 0.5) / reference.SH_C0
    return {k: v.to(device) for k, v in dict(means=means, quats=quats, scales=scales,
                                             opac=opac, sh=sh).items()}


def _cpu_settings(s):
    return {k: (v.detach().double().cpu() if torch.is_tensor(v) else v) for k, v in s.items()}


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _require_cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        from splatstudio.probes import ProbeUnavailable

        raise ProbeUnavailable("CUDA required")


def _cuda_render(sc, s, texture=None, index=None):
    leaves = {k: sc[k].clone().requires_grad_(True) for k in ("means", "scales", "quats",
                                                             "opac", "sh")}
    rots = torch.nn.functional.normalize(leaves["quats"], dim=-1)
    tx = texture.clone().requires_grad_(True) if texture is not None else None
    color, radii, allmap, _ = rasterize(leaves["means"], leaves["sh"].contiguous(),
                                        leaves["opac"].contiguous(), leaves["scales"].contiguous(),
                                        rots.contiguous(), s, tx, index)
    return color, leaves, tx


def _ref_render(sc, s, variant="stage1", texture=None, index=None, **kw):
    leaves = {k: sc[k].detach().double().cpu().clone().requires_grad_(True)
              for k in ("means", "scales", "quats", "opac", "sh")}
    tx = texture.detach().double().cpu().clone().requires_grad_(True) if texture is not None else None
    img = reference.render_reference(leaves["means"], leaves["scales"], leaves["quats"],
                                     leaves["opac"], leaves["sh"], _cpu_settings(s), variant,
                                     tx, index.cpu() if index is not None else None, **kw)
    return img, leaves, tx


def raster_vs_reference(method, device):
    """Stage-1 kernel (k-buffer sort, tile-centre keys, frustum) vs the dense reference:
    forward image and gradients of a random projection w.r.t. every surfel parameter.
    Also shows the scene is sensitive to the per-ray sort and the frustum average."""
    _require_cuda(device)
    out = {}
    refs = {}
    dense = scene(device, n=24, seed=4, opacity=(0.1, 0.4), spread=0.5)
    # k = 16 (paper, real scenes) on the sparse scene; k = 4 and 16 on a dense scene (k = 4
    # overflows the buffer: blend-on-full path changes the image)
    for tag, sc, k_buf in (("k16", scene(device), 16), ("dense_k16", dense, 16),
                           ("dense_k4", dense, 4)):
        s = raster_settings(camera(device), torch.zeros(3, device=device), 3, k_buf)
        color, leaves, _ = _cuda_render(sc, s)
        ref, rleaves, _ = _ref_render(sc, s)
        r = torch.randn(color.shape, generator=torch.Generator().manual_seed(1))
        (color * r.to(device)).sum().backward()
        (ref * r.double()).sum().backward()
        err = float((color.detach().cpu().double() - ref.detach()).abs().max())
        out[f"{tag}_forward_max_abs"] = err
        refs[tag] = ref.detach()
        assert err < 1e-4, out
        for k in ("means", "scales", "quats", "opac", "sh"):
            out[f"{tag}_grad_rel_{k}"] = _rel(leaves[k].grad.cpu().double(), rleaves[k].grad)
            assert out[f"{tag}_grad_rel_{k}"] < 2e-3, out
    out["dense_k4_vs_k16"] = float((refs["dense_k4"] - refs["dense_k16"]).abs().max())
    assert out["dense_k4_vs_k16"] > 1e-3, out
    sc = scene(device)
    s = raster_settings(camera(device), torch.zeros(3, device=device), 3, 16)
    ref = refs["k16"]
    with torch.no_grad():
        unsorted = _ref_render(sc, s, sort=False)[0]
        centre = _ref_render(sc, s, frustum=False)[0]
    out["sort_effect"] = float((unsorted - ref.detach()).abs().max())
    out["frustum_effect"] = float((centre - ref.detach()).abs().max())
    assert out["sort_effect"] > 1e-2 and out["frustum_effect"] > 1e-2, out
    return out


def _texture_scene(sc, device, tex_res=60.0):
    index, total = tex.texture_index(sc["scales"], tex_res, 500)
    g = torch.Generator().manual_seed(5)
    texture = (torch.rand(total, 3, generator=g) - 0.5).to(device) / reference.SH_C0
    return texture, index.to(device), total


def texture_vs_reference(method, device):
    """Textured kernel vs reference (bilinear texel lookup at (u/r+1)/2 (W-1), SH at the exact
    intersection direction): forward + gradients w.r.t. texture and SH rest. A constant texture
    equal to the DC must not equal the textured image (texture is used)."""
    _require_cuda(device)
    sc = scene(device, seed=7)
    s = raster_settings(camera(device), torch.zeros(3, device=device), 3, 16)
    texture, index, total = _texture_scene(sc, device)
    color, leaves, tx = _cuda_render(sc, s, texture, index)
    ref, rleaves, rtx = _ref_render(sc, s, "texture", texture, index)
    r = torch.randn(color.shape, generator=torch.Generator().manual_seed(2))
    (color * r.to(device)).sum().backward()
    (ref * r.double()).sum().backward()
    out = {"texels": total,
           "forward_max_abs": float((color.detach().cpu().double() - ref.detach()).abs().max()),
           "grad_rel_texture": _rel(tx.grad.cpu().double(), rtx.grad),
           "grad_rel_shN": _rel(leaves["sh"].grad[:, 1:].cpu().double(), rleaves["sh"].grad[:, 1:])}
    assert out["forward_max_abs"] < 1e-4 and out["grad_rel_texture"] < 2e-3, out
    assert out["grad_rel_shN"] < 2e-3, out
    with torch.no_grad():
        flat = tex.initial_texture(sc["sh"][:, :1], index, total)
        c_flat, _, _ = _cuda_render(sc, s, flat, index)
        c_stage1, _, _ = _cuda_render(sc, s)
    out["texture_effect"] = float((c_flat - color.detach()).abs().max())
    # constant texture = DC: equals stage 1 up to the SH direction (intersection vs centre)
    out["flat_vs_stage1"] = float((c_flat - c_stage1).abs().max())
    assert out["texture_effect"] > 1e-2, out
    return out


def fisher_estimator(method, device):
    """Hutchinson diag-F vs the exact sum of squared per-pixel Jacobians (torch.func-free:
    one backward per pixel/channel on a 16x12 image); 10 % lowest scores are removed."""
    _require_cuda(device)
    sc = scene(device, n=8, seed=11)
    cam = camera(device, 16, 12)
    cam["K"] = cam["K"] * torch.tensor([[0.5], [0.5], [1.0]], device=device)
    params = {"means": sc["means"], "scales": sc["scales"].log(), "quats": sc["quats"],
              "opacities": torch.logit(sc["opac"][:, 0]), "sh0": sc["sh"][:, :1],
              "shN": sc["sh"][:, 1:]}
    est = fisher.fisher_diagonal(params, [cam], 16, sh_degree=3, probes=400, seed=0)
    means, sh, opac, scales, rots = (t.detach() for t in activated(params))
    means = means.clone().requires_grad_(True)
    scales = scales.clone().requires_grad_(True)
    s = raster_settings(cam, torch.zeros(3, device=device), 3, 16)
    color, _, _, _ = rasterize(means, sh, opac, scales, rots, s)
    exact = torch.zeros(len(means), 5, device=device, dtype=torch.float64)
    flat = color.reshape(-1)
    for j in range(flat.numel()):
        gm, gs = torch.autograd.grad(flat[j], [means, scales], retain_graph=True)
        exact[:, :3] += gm.double().square()
        exact[:, 3:] += gs.double().square()
    rel = float(((est - exact).norm() / exact.norm()))
    lo, le = fisher.sensitivity(est), fisher.sensitivity(exact)
    corr = float(torch.corrcoef(torch.stack([lo, le]))[0, 1])
    mask = fisher.prune_mask(torch.arange(50.0), 0.1)
    out = {"diag_rel_error": rel, "score_corr": corr, "pruned_of_50": int(mask.sum()),
           "pruned_are_lowest": bool(mask[:5].all())}
    assert rel < 0.25 and corr > 0.95 and out["pruned_of_50"] == 5 and out["pruned_are_lowest"], out
    return out


def stage_schedule(method, device):
    """Two-stage optimisation: xyz expon LR (30000), stage switch at 40000 freezes geometry,
    opacity and DC, delays SH rest by 1000 its, texture StepLR(8000, 0.1); geometry losses off
    in stage 2; texture index sizes = ceil(T s) with the 500 cap."""
    c = DENSITY_DEFAULTS
    out = {}
    lr0 = expon_lr(0, 1.6e-4, 1.6e-6, 30000)
    lr30 = expon_lr(30000, 1.6e-4, 1.6e-6, 30000)
    lr39 = expon_lr(39000, 1.6e-4, 1.6e-6, 30000)
    assert math.isclose(lr0, 1.6e-4) and math.isclose(lr30, 1.6e-6) and lr39 == lr30
    lrs = [tex.tex_lr(t, c["tex_lr"], c["tex_step_size"], c["tex_gamma"]) for t in
           (1, 8000, 8001, 16001, 20000)]
    assert [round(v, 7) for v in lrs] == [0.008, 0.008, 0.0008, 0.00008, 0.00008], lrs
    out["tex_lrs"] = lrs
    d = Density(SimpleNamespace(config={}))
    opts = {k: SimpleNamespace(param_groups=[{"lr": 1.0}]) for k in
            ("means", "scales", "quats", "opacities", "sh0", "shN")}
    tex.STORE["optimizer"] = None
    d._stage2_lrs(opts, 1)
    assert all(opts[k].param_groups[0]["lr"] == 0.0 for k in
               ("means", "scales", "quats", "opacities", "sh0", "shN"))
    d._stage2_lrs(opts, 1002)
    assert opts["shN"].param_groups[0]["lr"] == c["shn_lr"] and opts["means"].param_groups[0]["lr"] == 0
    idx, total = tex.texture_index(torch.tensor([[0.0101, 0.0199], [3.0, 0.5], [0.001, 0.001]]),
                                   200, 500)
    assert idx[:, 1:].tolist() == [[3, 4], [500, 83], [1, 1]] and total == 12 + 41500 + 1, idx
    loss = create_geometry(SimpleNamespace(config={}))
    rgb = torch.ones(2, 2, 3, requires_grad=True)
    z = loss(None, {"step": 50000, "textured": True, "rgb": rgb}, None)
    assert float(z) == 0.0 and z.requires_grad
    out["texture_index"] = idx.tolist()
    return out


def density_recipe(method, device):
    """Official 2DGS densify_and_prune on a 6-surfel state: clone (small, grad >= 2e-4), split
    (big: two children, scales / 1.6, in-plane samples), opacity cull < 0.01, world-size prune
    (> 0.1 extent, after it 3000), statistics reset; Adam moments stay aligned."""
    from splatstudio.state import check_state

    dev = "cpu"
    n = 6
    extent = 10.0
    logs = torch.log(torch.tensor([[0.05, 0.05], [0.05, 0.05], [0.5, 0.2], [0.05, 0.05],
                                   [2.0, 0.1], [0.05, 0.05]]))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(n, 3)),
        "scales": torch.nn.Parameter(torch.cat([logs, torch.full((n, 1), -30.0)], 1)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * n)),
        "opacities": torch.nn.Parameter(torch.logit(torch.tensor([0.5, 0.5, 0.5, 0.005, 0.5, 0.5]))),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3)),
        tex.FLAG: torch.nn.Parameter(torch.zeros(n, 1)),
    })
    optimizers = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    loss = sum((p * p).sum() for p in params.values())
    loss.backward()
    for o in optimizers.values():
        o.step()
        o.zero_grad()
    d = Density(SimpleNamespace(config={}))
    d.extent = extent
    d._zero_stats(n, dev)
    d.accum[:] = torch.tensor([3e-4, 1e-4, 3e-4, 1e-4, 1e-4, 1e-4])
    d.denom[:] = 1.0
    with torch.no_grad():
        means_before = params["means"].clone()
        parent = torch.exp(params["scales"][2, :2]).clone()
        d._densify_and_prune(params, optimizers, 3150)
    # 6 + 1 clone (idx 0) + 1 extra split child (idx 2) - 1 cull (idx 3) - 1 world prune (idx 4)
    counts = d.events[-1][2:]
    assert counts[:2] == (1, 1) and len(params["means"]) == 6, d.events
    child = torch.exp(params["scales"][-2:, :2])
    assert torch.allclose(child, parent.expand(2, 2) / 1.6, rtol=1e-5), (child, parent)
    assert (params["means"][-2:, 2].abs() < 1e-9).all() and means_before.shape[0] == n
    assert float(d.accum.abs().sum()) == 0.0 and len(d.accum) == 6
    big = torch.exp(params["scales"][:, :2]).max(1).values > 0.1 * extent
    assert not big.any()
    method_like = SimpleNamespace(fields=[SimpleNamespace(name=tex.FLAG, shape=(1,))])
    check_state(method_like, params, optimizers)
    return {"events": d.events, "children_scales": child.tolist()}
