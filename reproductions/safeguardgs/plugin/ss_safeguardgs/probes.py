"""SafeguardGS mechanism probes. Each one fails if its mechanism is removed or wrong.

References are written independently of the CUDA kernel: a sequential per-pixel blend in torch
over gsplat's tile lists with `core.score` (a separate port of the official compute_score), the
official slot rule re-derived in Python, and analytic two-Gaussian cases.
"""

import torch
from splatstudio import ComponentSpec
from torch import nn

from . import core, kernels
from .density import DEFAULTS, SafeguardDensity, runtime_sh_degree
from .renderer import TargetPassRenderer

SH_C0 = 0.28209479177387814


class _Spec:
    def __init__(self, config):
        self.config = config


def _require_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def _scene(means, scales, opacities, colors, device="cuda", shN=None):
    n = len(means)
    values = {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "scales": torch.as_tensor(scales, dtype=torch.float32).log(),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.logit(torch.as_tensor(opacities, dtype=torch.float32)),
        "sh0": ((torch.as_tensor(colors, dtype=torch.float32) - 0.5) / SH_C0)[:, None, :],
        "shN": torch.zeros(n, 15, 3) if shN is None else shN.float(),
    }
    params = nn.ParameterDict({k: nn.Parameter(v.to(device)) for k, v in values.items()})
    rates = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 0.05, "sh0": 2.5e-3,
             "shN": 1.25e-4}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    return params, opts


def _camera(width, height, f, device="cuda", z=0.0, image=None):
    view = torch.eye(4, device=device)
    view[2, 3] = z
    cam = {
        "viewmat": view,
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device),
        "width": width,
        "height": height,
    }
    if image is not None:
        cam["image"] = image.to(device)
    return cam


def _random_scene(n, seed, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    means = torch.stack(
        [(torch.rand(n, generator=g) - 0.5) * 3.0, (torch.rand(n, generator=g) - 0.5) * 2.2,
         2.0 + torch.rand(n, generator=g) * 3.0], dim=1)
    scales = 0.08 + 0.4 * torch.rand(n, 3, generator=g)
    opac = 0.05 + 0.9 * torch.rand(n, generator=g)
    colors = torch.rand(n, 3, generator=g) * 1.2  # some colours above 1, as SH colours can be
    shN = 0.2 * torch.randn(n, 15, 3, generator=g)
    params, opts = _scene(means, scales, opac, colors, device, shN)
    image = torch.rand(48, 64, 3, generator=g)
    return params, opts, _camera(64, 48, 40.0, device, image=image)


@torch.no_grad()
def _reference(meta, colors, image, func_id, ks=(1, 2)):
    """Sequential per-pixel blend over gsplat's tile lists (CPU float32). Returns the official
    slot keys (q, id) per tile-private slot and the paper-text per-pixel top-k keep mask."""
    W, H, ts = int(meta["width"]), int(meta["height"]), int(meta["tile_size"])
    offs = meta["isect_offsets"][0].flatten().long().cpu()
    flat = meta["flatten_ids"].long().cpu()
    m2d = meta["means2d"][0].float().cpu()
    con = meta["conics"][0].float().cpu()
    opa = meta["opacities"][0].float().cpu()
    col = colors.float().cpu()
    img = image.float().cpu()
    n = m2d.shape[0]
    pad, slots, n_slots = core.padded_offsets(offs, flat.numel(), ts * ts)
    q_ref = torch.full((n_slots,), -1, dtype=torch.int64)
    id_ref = torch.full((n_slots,), -1, dtype=torch.int64)
    keep_px = {k: torch.zeros(n, dtype=torch.bool) for k in ks}
    lengths = core.tile_lengths(offs, flat.numel())
    tile_w = (W + ts - 1) // ts
    for tile in range(len(offs)):
        ty, tx = divmod(tile, tile_w)
        ys, xs = torch.meshgrid(torch.arange(ts), torch.arange(ts), indexing="ij")
        pi, pj = (ty * ts + ys).flatten(), (tx * ts + xs).flatten()  # thread rank = y*ts + x
        inside = (pi < H) & (pj < W)
        px, py = pj.float() + 0.5, pi.float() + 0.5
        gt = img[pi.clamp(max=H - 1), pj.clamp(max=W - 1)]
        T = torch.ones(ts * ts)
        done = ~inside
        best = [[] for _ in range(ts * ts)]
        start, L = int(offs[tile]), int(lengths[tile])
        for idx in range(L):
            b = idx // (ts * ts)
            gidx = int(flat[start + idx])
            dx, dy = m2d[gidx, 0] - px, m2d[gidx, 1] - py
            c = con[gidx]
            sigma = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
            alpha = torch.clamp(opa[gidx] * torch.exp(-sigma), max=0.999)
            ok = (~done) & (sigma >= 0) & (alpha >= 1.0 / 255.0)
            nT = T * (1 - alpha)
            stop = ok & (nT <= 1e-4)
            done = done | stop
            ok = ok & ~stop
            s = core.score(func_id, opa[gidx], alpha, T, dx, dy, gt, col[gidx][None].expand(
                ts * ts, 3))
            for t in ok.nonzero().flatten().tolist():
                slot = int(pad[tile]) + b * ts * ts + t
                q_ref[slot] = int(core.quantize(s[t]))
                id_ref[slot] = gidx
                best[t].append((float(s[t]), gidx))
            T = torch.where(ok, nT, T)
        for t in range(ts * ts):
            ranked = sorted(best[t], key=lambda x: -x[0])
            for k in ks:
                for _, gidx in ranked[:k]:
                    keep_px[k][gidx] = True
    return q_ref, id_ref, slots, keep_px


def topk_kernel(method=None, device="cuda"):
    """CUDA slots / keep masks equal the torch reference (k = 1, 2; SG_36 and SG_4)."""
    _require_cuda()
    params, _, cam = _random_scene(1200, 7, device)
    meta, colors = kernels.project(params, cam, 3)
    lengths = core.tile_lengths(meta["isect_offsets"][0].flatten().long(),
                                meta["flatten_ids"].numel())
    if int(lengths.max()) <= 256:
        raise AssertionError("fixture must have a tile with > 256 intersections (2+ batches)")
    out = {"max_tile_isects": int(lengths.max()), "n": len(params["means"])}
    for func_id in (core.SG_36, core.SG_4):
        q_ref, id_ref, slots, keep_px_refs = _reference(meta, colors, cam["image"], func_id)
        pad, _, n_slots = core.padded_offsets(meta["isect_offsets"][0].flatten().int(),
                                              meta["flatten_ids"].numel())
        q, ids = kernels.ext().sg_topk(
            meta["means2d"][0].float().contiguous(), meta["conics"][0].float().contiguous(),
            meta["opacities"][0].float().contiguous(), colors.float().contiguous(),
            meta["isect_offsets"][0].int().contiguous(), meta["flatten_ids"].int().contiguous(),
            int(meta["width"]), int(meta["height"]), int(meta["tile_size"]),
            cam["image"].float().reshape(-1, 3).contiguous(), func_id, 1.0, 1.0, 1, True,
            pad.contiguous(), n_slots)
        id_mismatch = int((ids.cpu().long() != id_ref).sum())
        q_err = int((q.cpu().long() - q_ref).abs().max())
        if id_mismatch or q_err > 1:
            raise AssertionError(f"{func_id:#x}: slot ids differ at {id_mismatch}, q err {q_err}")
        for k in (1, 2):
            keep_px_ref = keep_px_refs[k]
            keep_tile_ref = core.tile_topk_keep(q_ref, id_ref, slots, k, len(params["means"]))
            keep_tile = kernels.view_keep(meta, colors, cam["image"], func_id, k, "tile")
            keep_px = kernels.view_keep(meta, colors, cam["image"], func_id, k, "pixel")
            dt = int((keep_tile.cpu() != keep_tile_ref).sum())
            dp = int((keep_px.cpu() != keep_px_ref).sum())
            if dt or dp:
                raise AssertionError(f"{func_id:#x} k={k}: tile mask diff {dt}, pixel diff {dp}")
            out[f"{func_id:#x}_k{k}_tile_kept"] = int(keep_tile.sum())
            out[f"{func_id:#x}_k{k}_pixel_kept"] = int(keep_px.sum())
    if out["0x24_k1_tile_kept"] >= out["0x24_k1_pixel_kept"]:
        raise AssertionError("official per-tile selection should keep fewer than per-pixel top-1")
    if out["0x24_k2_tile_kept"] <= out["0x24_k1_tile_kept"]:
        raise AssertionError("top-2 must keep more Gaussians than top-1")
    return out


def color_score(method=None, device="cuda"):
    """Red ground truth. (a) Two disjoint Gaussians in one tile, blue with the larger alpha*T and
    red: per-tile top-1 keeps blue under SG_4 and red under SG_36. (b) Blue in front of red on
    the same rays: per-pixel top-1 keeps the front one under SG_4 and the back one under SG_36;
    the official per-tile rule only ever sees the last contributor of a batch (back, both
    functions). (c) Safeguard: after a pixel-level prune every previously covered pixel is still
    covered."""
    _require_cuda()
    red = torch.zeros(16, 16, 3)
    red[..., 0] = 1.0
    cam = _camera(16, 16, 16.0, device, image=red)
    blue_red = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    res = {}
    # (a) side by side (x = -/+0.5 at z = 2 -> pixel columns 4 and 12), sigma ~0.7 px
    params, _ = _scene([[-0.5, 0, 2.0], [0.5, 0, 2.0]], [[0.05] * 3] * 2, [0.9, 0.6],
                       blue_red, device)
    meta, colors = kernels.project(params, cam, 0)
    for name, fid in (("sg4", core.SG_4), ("sg36", core.SG_36)):
        res[f"disjoint_{name}_tile"] = kernels.view_keep(
            meta, colors, cam["image"], fid, 1, "tile").cpu().tolist()
    if res["disjoint_sg4_tile"] != [True, False] or res["disjoint_sg36_tile"] != [False, True]:
        raise AssertionError(f"colour term not effective (tile): {res}")
    # (b) front / back on the same rays
    params, _ = _scene([[0, 0, 2.0], [0, 0, 3.0]], [[2.0] * 3, [3.0] * 3], [0.6, 0.9],
                       blue_red, device)
    meta, colors = kernels.project(params, cam, 0)
    for name, fid in (("sg4", core.SG_4), ("sg36", core.SG_36)):
        for sel in ("tile", "pixel"):
            res[f"stacked_{name}_{sel}"] = kernels.view_keep(
                meta, colors, cam["image"], fid, 1, sel).cpu().tolist()
    if res["stacked_sg4_pixel"] != [True, False] or res["stacked_sg36_pixel"] != [False, True]:
        raise AssertionError(f"colour term not effective (pixel): {res}")
    if res["stacked_sg4_tile"] != [False, True] or res["stacked_sg36_tile"] != [False, True]:
        raise AssertionError(f"official last-contributor-per-batch rule not reproduced: {res}")
    # safeguard on a random scene: pixel-level pruning never empties a covered ray
    params, opts, cam = _random_scene(700, 3, device)
    rend = TargetPassRenderer(_Spec({}))
    a0 = rend.render(params, cam, 3)["alpha"][..., 0]
    meta, colors = kernels.project(params, cam, 3)
    keep = kernels.view_keep(meta, colors, cam["image"], core.SG_36, 1, "pixel")
    sub = nn.ParameterDict({k: nn.Parameter(v[keep]) for k, v in params.items()})
    a1 = rend.render(sub, cam, 3)["alpha"][..., 0]
    covered = a0 > 1.0 / 255.0
    lost = int((covered & (a1 < 1.0 / 255.0)).sum())
    if lost:
        raise AssertionError(f"{lost} covered pixels lost all primitives after pixel-level prune")
    res["safeguard_covered_pixels"] = int(covered.sum())
    res["safeguard_kept"] = int(keep.sum())
    return res


def prune_schedule(method=None, device="cuda"):
    """Paper schedule (prune once at 20000, top-1, SG_36); cached GT required for every view;
    survivors = union of per-view masks; params, Adam moments and strategy state aligned."""
    _require_cuda()
    if (DEFAULTS["prune_steps"], DEFAULTS["top_k"], DEFAULTS["score_function"],
            DEFAULTS["selection"]) != ([20000], 1, 0x24, "tile"):
        raise AssertionError("defaults differ from the paper / official sweep settings")
    if method is not None:
        cfg = next(c for c in method.components if c.kind == "density").config
        if cfg["prune_steps"] != [20000] or cfg["top_k"] != 1 or cfg["score_function"] != 0x24:
            raise AssertionError("installed method does not use the paper schedule")
    if runtime_sh_degree(20000) != 3:
        raise AssertionError("SH degree at the prune step must be 3")
    params, opts, cam0 = _random_scene(700, 11, device)
    cam1 = dict(cam0)
    cam1["viewmat"] = cam0["viewmat"].clone()
    cam1["viewmat"][0, 3] = 0.4
    cam1["image"] = cam0["image"].flip(1).contiguous()
    cams = [cam0, cam1]
    dens = SafeguardDensity(ComponentSpec(
        "sg-density", "density", "ss_safeguardgs.density:create_density",
        ("gradient.screen.signed", "training.target_image"), (), dict(DEFAULTS)))
    dens.set_training_cameras([{k: c[k] for k in ("K", "viewmat", "width", "height")}
                               for c in cams])
    dens.initialize(params, opts, 1.0)
    rend = TargetPassRenderer(_Spec({}))
    # Adam state for every Gaussian
    out = rend.render(params, cam0, 3)
    out["rgb"].sum().backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    n0 = len(params["means"])
    dens.after_backward(params, opts, 19999, rend.render(params, cam0, 3))
    if len(params["means"]) != n0:
        raise AssertionError("pruned before the prune step")
    try:
        dens.after_backward(params, opts, 20000, rend.render(params, cam0, 3))
    except RuntimeError as exc:
        if "never rendered" not in str(exc):
            raise
    else:
        raise AssertionError("prune ran without the ground truth of every training view")
    dens.images[1] = (cam1["image"].clamp(0, 1) * 255).round().to(torch.uint8)
    expect = torch.zeros(n0, dtype=torch.bool, device=device)
    for c in cams:
        meta, colors = kernels.project(params, c, 3)
        gt = (c["image"].clamp(0, 1) * 255).round() / 255
        expect |= kernels.view_keep(meta, colors, gt, 0x24, 1, "tile")
    m_before = opts["means"].state[params["means"]]["exp_avg"][expect].clone()
    means_before = params["means"][expect].detach().clone()
    dens.after_backward(params, opts, 20000, rend.render(params, cam1, 3))
    n1 = len(params["means"])
    if n1 != int(expect.sum()) or n1 >= n0 or n1 == 0:
        raise AssertionError(f"survivors {n1} != union of per-view masks {int(expect.sum())}")
    if not torch.equal(params["means"].detach(), means_before):
        raise AssertionError("survivors are not the union of the per-view keep masks")
    st = opts["means"].state[params["means"]]
    if not torch.equal(st["exp_avg"], m_before):
        raise AssertionError("Adam moments not aligned with survivors")
    for k, v in dens.base.state.items():
        if isinstance(v, torch.Tensor) and v.shape[0] not in (n1,):
            raise AssertionError(f"strategy state {k} misaligned")
    dens.after_backward(params, opts, 20001, rend.render(params, cam0, 3))
    if len(params["means"]) != n1 or len(dens.history) != 1:
        raise AssertionError("pruned more than once")
    return {"before": n0, "after": n1, "events": dens.history}
