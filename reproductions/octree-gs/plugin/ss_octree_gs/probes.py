"""Mechanism probes for the Octree-GS port. Each compares against an independent formula, a brute-force reference
or a constructed case and raises when the mechanism is removed or wrong.

neural-gaussians, visibility-filter and blob-eval are adapted from the ContextGS / HAC native ports
(reproductions/contextgs/plugin/ss_contextgs/probes.py); the others are Octree-GS specific.
"""

import io
import math
from types import SimpleNamespace

import numpy as np
import torch

from . import model, octree
from .density import OctreeGSDensity, new_anchor_fields
from .renderer import BLOB, OctreeGSRenderer, blob_floats, visible_anchors
from .renderer import neural_gaussians as decode_neural


def _need_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is required")


def _spec(config):
    return SimpleNamespace(config=config)


CAM = {"K": [[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], "width": 64, "height": 64}


def _cam(device):
    return {"viewmat": torch.eye(4, device=device), "K": torch.tensor(CAM["K"], device=device),
            "width": CAM["width"], "height": CAM["height"]}


def _fixture(device, n=400, seed=0, levels=3, init_level=1, std=8.0, voxel=0.4, init_pos=-10.0, **cfg):
    """Density + bank + anchors on their level grids (z in [2, 4]) in front of an identity camera; two training
    cameras at the origin. The octree scalars are set directly (no SfM build)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dens = OctreeGSDensity(_spec({"visible_threshold": 0.0, **cfg}))
    c = dens.c
    lev = torch.randint(0, levels, (n,), generator=g).float()
    size = voxel / 2.0**lev
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 2.0]) + torch.tensor([0, 0, 3.0])
    pos = torch.round((pts - init_pos) / size[:, None]) * size[:, None] + init_pos
    key = torch.cat([pos, lev[:, None]], 1)
    key = torch.unique(key, dim=0)
    pos, lev = key[:, :3].to(device), key[:, 3].to(device)
    size = (voxel / 2.0**lev)
    log_s = torch.log(size)[:, None].repeat(1, 6)
    params = torch.nn.ParameterDict(new_anchor_fields(pos, log_s, torch.zeros(len(pos), c["feat_dim"], device=device),
                                                      lev, c["n_offsets"]))
    dens.spatial_lr_scale = 1.0
    opts = {k: torch.optim.Adam([p], lr=v, eps=1e-15) for (k, p), v in
            zip(params.items(), [dens.lr_table()[k] for k in params.keys()])}
    torch.manual_seed(seed)
    b = model.OctreeBank(c, device)
    b.levels, b.init_level, b.standard_dist = levels, init_level, std
    b.voxel_size = torch.tensor(voxel, dtype=torch.float32, device=device)
    b.init_pos = torch.full((3,), init_pos, dtype=torch.float32, device=device)
    b.cam_centers = torch.tensor([[0.0, 0, 0], [0.1, 0, 0]], device=device)
    b.visible_threshold = 0.0
    b.coarse_intervals = octree.coarse_intervals(levels, init_level, c["coarse_iter"], c["coarse_factor"])
    dens.bank = b
    dens.offset_sched = model.expon_lr(0.01, 0.0001, c["total_steps"])
    dens.reset_stats(len(pos), device)
    if len(pos) * 8 >= blob_floats(c["feat_dim"], c["n_offsets"]):
        dens.pack(params, 0)
    renderer = OctreeGSRenderer(_spec({"feat_dim": c["feat_dim"], "n_offsets": c["n_offsets"]}))
    renderer.bank = b
    return dens, params, opts, renderer, _cam(device)


def _check(params, opts):
    n = len(params["means"])
    for k, p in params.items():
        if p.shape[0] != n:
            raise AssertionError(f"{k} misaligned")
        if opts[k].param_groups[0]["params"][0] is not p:
            raise AssertionError(f"{k} optimizer references a stale parameter")
        st = opts[k].state.get(p, {})
        if "exp_avg" in st and st["exp_avg"].shape != p.shape:
            raise AssertionError(f"{k} Adam moment misaligned")


# --------------------------------------------------------------------------- C1 octree
def octree_init(method, device):
    _need_cuda(device)
    g = np.random.default_rng(0)
    # SfM-like cloud: dense near region + sparse far points; cameras on a ring
    near = g.normal(0, 1.0, (3000, 3))
    far = g.normal(0, 1.0, (300, 3)) * 15
    pts_np = np.concatenate([near, far]).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cams_np = np.stack([4 * np.cos(ang), 4 * np.sin(ang), np.full_like(ang, 0.5)], 1).astype(np.float32)
    pts = torch.tensor(pts_np, device=device)
    cams = torch.tensor(cams_np, device=device)
    dens = OctreeGSDensity(_spec({"base_layer": 6}))
    params = torch.nn.ParameterDict(new_anchor_fields(pts, torch.zeros(len(pts), 6, device=device),
                                                      torch.zeros(len(pts), 32, device=device),
                                                      torch.zeros(len(pts), device=device), 10))
    opts = {k: torch.optim.Adam([p], lr=0.0) for k, p in params.items()}
    dens.initialize(params, opts, 1.0)
    model.ACTIVE["bank"] = None
    dens.build(params, opts, cams)
    st = dens.state["octree_gs"]
    # independent K / standard distance (numpy linear quantiles, two stages)
    r = 0.999
    pairs = []
    for c in cams_np:
        d = np.sqrt(((pts_np.astype(np.float64) - c) ** 2).sum(1))
        pairs += [np.quantile(d, 1 - r), np.quantile(d, r)]
    dmax, dmin = np.quantile(pairs, r), np.quantile(pairs, 1 - r)
    k_ref = int(np.round(np.log2(dmax / dmin))) + 1
    if st["levels"] != k_ref or abs(st["standard_dist"] - dmax) > 1e-3 * dmax:
        raise AssertionError(f"K/std {st['levels']}/{st['standard_dist']} != {k_ref}/{dmax}")
    if st["init_level"] != k_ref // 2:
        raise AssertionError("init_level != floor(K/2)")
    # independent voxelization + view frequency of every sampled anchor
    box_min = float(pts_np.min()) * 1.1
    box_d = float(pts_np.max()) * 1.1 - box_min
    voxel = box_d / 2**6
    if abs(st["voxel_size"] - voxel) > 1e-5 * voxel or abs(st["init_pos"] - box_min) > 1e-5 * abs(box_min):
        raise AssertionError("voxel size / origin != box_d / 2^base_layer, 1.1 min(P)")
    v32, o32 = np.float32(st["voxel_size"]), np.float32(st["init_pos"])
    cand, freqs = [], []
    for L in range(k_ref):
        size = np.float32(v32 / np.float32(2.0**L))
        cells = {tuple(v) for v in np.round((pts_np - o32) / size).astype(np.int64).tolist()}
        for cell in cells:
            p = np.array(cell, dtype=np.float64) * float(size) + float(o32)
            cnt = 0
            for c in cams_np:
                pred = math.log2(dmax / np.linalg.norm(p - c))
                cnt += L <= min(max(round(pred), 0), k_ref - 1)
            cand.append((cell, L))
            freqs.append(cnt / len(cams_np))
    freqs = np.array(freqs)
    if st["octree_sampled"] != len(cand):
        raise AssertionError(f"octree sampled {st['octree_sampled']} anchors, brute force {len(cand)}")
    if abs(st["visible_threshold"] - freqs.mean()) > 2e-3:
        raise AssertionError(f"adaptive threshold {st['visible_threshold']} != mean view frequency {freqs.mean()}")
    want = int((freqs > st["visible_threshold"]).sum())
    have = len(params["means"])
    if abs(have - want) > max(3, 0.01 * want):
        raise AssertionError(f"weed_out kept {have} anchors, expected {want}")
    if have >= len(cand):
        raise AssertionError("weed_out removed nothing")
    # vendored simple-knn initial scaling vs exact 3-NN
    pos = params["means"].detach()
    d2 = torch.cdist(pos, pos).square()
    d2.fill_diagonal_(float("inf"))
    ref = d2.topk(3, largest=False).values.mean(1).clamp_min(1e-7)
    got = torch.exp(params["ogs_scaling"][:, 0].detach()) ** 2
    rel = float(((got - ref).abs() / ref).median())
    if rel > 1e-3:
        raise AssertionError(f"initial scaling differs from 3-NN (median rel. error {rel})")
    return {"levels": st["levels"], "standard_dist": st["standard_dist"], "sampled": len(cand), "kept": have,
            "visible_threshold": st["visible_threshold"], "per_level": st["initial_per_level"],
            "knn_median_rel_error": rel}


# --------------------------------------------------------------------------- C2 LOD selection
def lod_selection(method, device):
    g = torch.Generator(device="cpu").manual_seed(1)
    n, K, fork, voxel, std = 300, 5, 2, 0.8, 60.0
    anchor = (torch.rand(n, 3, generator=g) - 0.5) * 30
    level = torch.randint(0, K, (n, 1), generator=g).float()
    extra = torch.rand(n, 1, generator=g) * 0.6
    cam = torch.tensor([1.0, -2.0, 0.5])
    got = octree.anchor_mask(anchor, level, extra, cam, voxel, std, fork, K - 1)
    ref = []
    for i in range(n):
        L = float(level[i])
        p = anchor[i].double() + (voxel / 2) / fork**L
        d = float(torch.linalg.norm(p - cam.double()))
        pred = math.log2(std / d) / math.log2(fork) + float(extra[i])
        ref.append(L <= min(max(round(pred), 0), K - 1))
    ref = torch.tensor(ref)
    mism = int((got != ref).sum())
    if mism > 1:  # float32 vs float64 at an exact .5 boundary at most once
        raise AssertionError(f"LOD mask differs from Eq. 3 reference on {mism} anchors")
    raised = octree.anchor_mask(anchor, level, extra + 1.0, cam, voxel, std, fork, K - 1)
    if not bool((raised | ~got).all()) or int(raised.sum()) <= int(got.sum()):
        raise AssertionError("raising Delta L does not select a superset with more anchors")
    if not bool((level[:, 0][got] <= K - 1).all()):
        raise AssertionError("selected level above K-1")
    limited = octree.anchor_mask(anchor, level, extra, cam, voxel, std, fork, 1)
    if bool((level[:, 0][limited] > 1).any()) or torch.equal(limited, got):
        raise AssertionError("max_level does not cap the selected levels")
    iv = octree.coarse_intervals(K, 2)
    if octree.coarse_index(iv, 1, 2, K, True, training=False) != K:
        raise AssertionError("evaluation must use all K levels")
    if octree.coarse_index(iv, 1, 2, K, False, training=True) != K:
        raise AssertionError("progressive=False must use all K levels")
    return {"selected": int(got.sum()), "selected_bias+1": int(raised.sum()), "mismatch": mism}


# --------------------------------------------------------------------------- C4 progressive
def progressive_training(method, device):
    K, init = 7, 3
    iv = octree.coarse_intervals(K, init, 10000, 1.5)
    steps = np.diff([0.0] + iv)
    if len(iv) != K - 1 - init or abs(iv[-1] - 10000) > 1e-6:
        raise AssertionError("coarse intervals do not cover levels init+1..K-1 within 10000 iterations")
    if not np.allclose(steps[1:] / steps[:-1], 1 / 1.5):
        raise AssertionError("coarse intervals are not geometric with N_{i-1} = 1.5 N_i")
    for s, want in ((1, init + 1), (int(iv[0]), init + 1), (int(iv[0]) + 1, init + 2), (10001, K)):
        if octree.coarse_index(iv, s, init, K) != want:
            raise AssertionError(f"coarse index at step {s} != {want}")
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=600, levels=3, init_level=0)
    out = {}
    for step, want in ((1, 0), (10001, 2)):
        dens.bank.step = step
        with torch.enable_grad():
            r = renderer.render(params, cam)
        info = r["info"]["ogs"]
        lv = params["ogs_level"][:, 0][info["visible"]]
        if info["max_level"] != want or float(lv.max()) > want:
            raise AssertionError(f"training render at step {step} uses levels up to {float(lv.max())}, want {want}")
        out[step] = int(info["visible"].sum())
    if not out[10001] > out[1]:
        raise AssertionError("unlocking levels does not add anchors")
    return {"intervals": [round(v, 1) for v in iv], "visible_step1": out[1], "visible_step10001": out[10001]}


# --------------------------------------------------------------------------- backbone
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
            inp = torch.cat([feat[i], v / v.norm()])[None]
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
    c3 = decode_neural(mlps, anchor * 2, feat, offsets, scaling, center, k)[1]
    if not (c3.shape == color.shape and torch.allclose(c3, color, atol=1e-6)):
        raise AssertionError("decode depends on the camera distance (Octree-GS input is [f, view dir] only)")
    return {"gaussians": len(ref), "max_error": float((got - ref).abs().max())}


def visibility_filter(method, device):
    """Constructed case for the Scaffold fork's visible_filter (64x64 image, f=60, 16-px tiles)."""
    _need_cuda(device)
    cam = _cam(device)
    anchors = torch.tensor([
        [0.0, 0.0, 3.0], [5.0, 0.0, 3.0], [5.0, 0.0, 3.0], [0.0, 0.0, -3.0], [0.0, 0.0, 0.1],
        [0.0, -1.95, 3.0], [-3.0, 0.0, 3.0]], device=device)
    scales = torch.full((7, 3), 0.01, device=device)
    scales[2] = 3.0
    scales[5] = 0.2
    got = visible_anchors(anchors, scales, cam).tolist()
    want = [True, False, True, False, False, True, False]
    if got != want:
        raise AssertionError(f"visible_filter mismatch: {got} != {want}")
    if (anchors[:, 2] > 0.2).tolist() == want:
        raise AssertionError("constructed case does not distinguish the z-only filter")
    dens, params, opts, renderer, cam2 = _fixture(device, n=400)
    with torch.no_grad():
        params["means"].data[:200, 0] += 50.0
    dens.bank.step = 20000
    with torch.enable_grad():
        out = renderer.render(params, cam2)
    info = out["info"]["ogs"]
    ref = info["lod"].clone()
    idx = torch.where(ref)[0]
    ref[idx] = visible_anchors(params["means"][idx].detach(), params["ogs_scaling"][idx, :3].exp().detach(), cam2)
    if not torch.equal(info["visible"], ref) or bool(info["visible"][:200].any()):
        raise AssertionError("renderer visibility differs from LOD mask & visible_filter")
    return {"case": got, "renderer_visible": int(info["visible"].sum())}


# --------------------------------------------------------------------------- C3 growing / pruning
def anchor_growing(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=1, levels=3, init_level=1, std=8.0, voxel=0.4)
    k, f = 10, 32
    vox, ip = 0.4, -10.0

    def grid(p, size):
        return [round((v - ip) / size) * size + ip for v in p]

    a0 = grid([0.0, 0.0, 3.0], vox)
    b1 = grid([0.6, 0.6, 3.0], vox / 2)
    c0 = grid([-0.8, 0.4, 3.2], vox)
    xyz = torch.tensor([a0, b1, c0], device=device)
    lev = torch.tensor([0.0, 1.0, 0.0], device=device)
    feat = torch.randn(3, f, device=device)
    vals = new_anchor_fields(xyz, torch.zeros(3, 6, device=device), feat, lev, k)  # exp(scaling) = 1
    vals["ogs_offset"][0, 0] = torch.tensor([2.0, 0.0, 0.0])  # mid gradient -> same level (0)
    vals["ogs_offset"][0, 1] = torch.tensor([-2.0, 0.0, 0.0])  # high gradient -> next level (1)
    vals["ogs_offset"][0, 2] = torch.tensor([0.0, 0.0, 1000.0])  # high gradient, far -> weeded
    for key, v in vals.items():
        params[key] = torch.nn.Parameter(v)
        opts[key] = torch.optim.Adam([params[key]], lr=0.0)
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    dens.reset_stats(3, device)
    th = 0.0002
    uv = 2**0.2
    dens.offset_denom[:] = 100
    dens.offset_gradient_accum[0] = 0.5 * (th + th * uv) * 100
    dens.offset_gradient_accum[1] = 5 * th * 100
    dens.offset_gradient_accum[2] = 5 * th * 100
    dens.opacity_accum[:] = 1.0
    dens.anchor_demon[:] = 100
    dens.opacity_accum[2] = 0.0  # anchor c0: low opacity, visible often -> pruned
    feat0 = params["ogs_feat"][0].detach().clone()
    stats = dens.adjust_anchor(params, opts, 2000)
    _check(params, opts)
    if len(dens.offset_denom) != len(params["means"]) * k or len(dens.opacity_accum) != len(params["means"]):
        raise AssertionError("density statistics misaligned with the population")
    pos = params["means"].detach().cpu().numpy()
    lv = params["ogs_level"][:, 0].detach().cpu().numpy()
    if np.isclose(pos, np.array(c0), atol=1e-5).all(1).any():
        raise AssertionError("low-opacity anchor not pruned")
    same = grid([a0[0] + 2.0, a0[1], a0[2]], vox)
    nxt = grid([a0[0] - 2.0, a0[1], a0[2]], vox / 2)
    i_same = np.where(np.isclose(pos, np.array(same), atol=1e-5).all(1) & (lv == 0))[0]
    i_next = np.where(np.isclose(pos, np.array(nxt), atol=1e-5).all(1) & (lv == 1))[0]
    if len(i_same) != 1 or len(i_next) != 1:
        raise AssertionError(f"expected one same-level and one next-level anchor, got {len(i_same)}/{len(i_next)}")
    if not torch.equal(params["ogs_feat"][int(i_same[0])].detach(), feat0):
        raise AssertionError("same-level anchor does not inherit the parent feature (scatter_max)")
    if float(params["ogs_feat"][int(i_next[0])].abs().max()) != 0.0:
        raise AssertionError("next-level anchor feature must be zero")
    sc = params["ogs_scaling"].detach()
    if not (torch.allclose(sc[int(i_same[0])], torch.full((6,), math.log(vox), device=device))
            and torch.allclose(sc[int(i_next[0])], torch.full((6,), min(math.log(vox / 2), 0.05), device=device))):
        raise AssertionError("grown anchor scaling != log(voxel size)")
    if stats["weeded"] < 1 or len(pos) != 4:
        raise AssertionError(f"far next-level candidate not weeded out (stats {stats}, {len(pos)} anchors)")
    i_a0 = np.where(np.isclose(pos, np.array(a0), atol=1e-5).all(1))[0][0]
    dl = float(params["ogs_extra_level"][int(i_a0), 0])
    # anchor gradient (0.5(th+th uv) + 10 th)/10 exceeds 0.25 th uv^L at levels 0 and 1 (level 2 has no anchors)
    if abs(dl - 0.02) > 1e-6:
        raise AssertionError(f"Delta L of the high-gradient anchor is {dl}, expected 0.02 (two level passes)")
    if float(params["ogs_extra_level"][int(i_same[0]), 0]) != 0.0:
        raise AssertionError("new anchors must start with Delta L = 0")
    return {"stats": stats, "anchors_after": int(len(pos)), "delta_L": dl}


# --------------------------------------------------------------------------- eval transport
def blob_eval(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=3000, seed=5)
    g = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        params["ogs_feat"].data.copy_(torch.randn(params["ogs_feat"].shape, generator=g))
        params["ogs_offset"].data.copy_(torch.randn(params["ogs_offset"].shape, generator=g) * 0.5)
        params["ogs_scaling"].data.copy_(torch.randn(params["ogs_scaling"].shape, generator=g) * 0.3 - 3)
        params["ogs_extra_level"].data.copy_(torch.rand(params["ogs_extra_level"].shape, generator=g))
        for p in dens.bank.mlps.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(device) * 0.05)
    dens.pack(params, 123)
    dens.bank.step = 20000  # past the progressive phase: training render uses all K levels
    with torch.enable_grad():
        live = renderer.render(params, cam)["rgb"].detach()
    buf = io.BytesIO()
    torch.save({"params": dict(params)}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location=device, weights_only=True)
    saved = model.ACTIVE["bank"]
    model.ACTIVE["bank"] = None
    try:
        fresh = OctreeGSRenderer(_spec({"feat_dim": 32, "n_offsets": 10}))
        with torch.no_grad():
            again = fresh.render(ck["params"], cam)["rgb"]
        mlps, oct_ = fresh.decoded(ck["params"])
        empty = {k: v.detach().clone() for k, v in ck["params"].items()}
        empty[BLOB].zero_()
        try:
            OctreeGSRenderer(_spec({"feat_dim": 32, "n_offsets": 10})).render(empty, cam)
        except ValueError:
            pass
        else:
            raise AssertionError("renderer must refuse a state without a packed blob")
    finally:
        model.ACTIVE["bank"] = saved
    err = float((live - again).abs().max())
    if err > 1e-5:
        raise AssertionError(f"fresh-process render differs from the live render by {err}")
    for kname in model.RENDER_MLPS:
        for a, b in zip(mlps[kname].parameters(), dens.bank.mlps[kname].parameters()):
            if not torch.equal(a, b):
                raise AssertionError("blob does not restore the trained MLP weights")
    b = dens.bank
    if (oct_["levels"], oct_["init_level"]) != (b.levels, b.init_level) or float(oct_["voxel_size"]) != float(
            b.voxel_size) or abs(oct_["standard_dist"] - b.standard_dist) > 1e-6:
        raise AssertionError("blob does not restore the octree scalars")
    if float(again.min()) > 0.999:
        raise AssertionError("fixture renders nothing")
    return {"max_error": err, "anchors": len(params["means"])}
