"""OMG mechanism probes. Each one fails if its mechanism is removed or wrong.

The references here are written independently of core.py: Morton codes by per-bit loops, the
tcnn frequency encoding by explicit sin/cos loops, the MLPs by plain matmuls, and the Huffman
cost by a textbook heap.
"""

import copy
import heapq
import math
from dataclasses import replace

import numpy as np
import torch
from gsplat import rasterization
from splatstudio.components import build
from splatstudio.registry import resolve

from . import codec, core, model, ms_core
from .ms_probes import _camera, _require_cuda, _scene
from .renderer import STATIC, VIEW, live_attributes


def _density(method, **overrides):
    spec = next(c for c in method.components if c.kind == "density")
    if overrides:
        spec = replace(spec, config={**spec.config, **overrides})
    return resolve(spec.factory)(spec)


def _restore_active(fn):
    def wrapped(method, device):
        saved = model.ACTIVE["bank"]
        try:
            return fn(method, device)
        finally:
            model.ACTIVE["bank"] = saved
    wrapped.__name__ = fn.__name__
    return wrapped


def _random_scene(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.cat([torch.rand(n, 2, generator=g) * 1.6 - 0.8,
                       torch.rand(n, 1, generator=g) * 2 + 3], 1)
    scales = torch.rand(n, 3, generator=g) * 0.12 + 0.04
    opac = torch.rand(n, generator=g) * 0.6 + 0.3
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    colors = torch.rand(n, 3, generator=g)
    params, opts = _scene(means, scales, opac, quats, colors, device=device)
    with torch.no_grad():
        params[STATIC].copy_(torch.randn(n, 3, generator=g).to(device) * 0.5)
        params[VIEW].copy_(torch.randn(n, 3, generator=g).to(device) * 0.3)
    return params, opts


# ----------------------------------------------------------------------------- C1
def _morton_ref(q):
    out = []
    for x, y, z in q.tolist():
        code = 0
        for b in range(21):
            code |= ((x >> b) & 1) << (3 * b) | ((y >> b) & 1) << (3 * b + 1) \
                | ((z >> b) & 1) << (3 * b + 2)
        out.append(code)
    return out


@_restore_active
def ld_scoring(method, device):
    _require_cuda()
    density = _density(method)
    n = 60
    params, opts = _random_scene(n, device, seed=5)
    density.initialize(params, opts, scene_scale=1.0)
    cams = [_camera(64, 48, 50.0, device=device)]
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.15, -0.05, 0.0])
    cams.append(_camera(64, 48, 50.0, c2w, device=device))
    density.set_training_cameras(cams)
    base = density._importance(params).cpu().double()
    # independent LD reference (official neighbour construction, Morton by bit loops)
    means = params["means"].detach().cpu().double()
    lo, hi = means.min(0).values, means.max(0).values
    q = ((2**21 - 1) * (means - lo) / (hi - lo)).long()
    codes = _morton_ref(q)
    order = sorted(range(n), key=lambda i: codes[i])
    feats = params["sh0"].detach().cpu().double()[:, 0]  # net not enabled: SH DC
    lam = density.omg["lambda_ld"]
    ref = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        left = order[max(order[i] - 1, 0)]
        right = order[min(order[i] + 1, max(order))]
        res = ((feats[left] - feats[i]).abs() + (feats[right] - feats[i]).abs()).mean()
        ref[i] = base[i] * res**lam
    tau = density.omg["importance_thresh"]
    vals = torch.sort(ref + 1e-6).values
    cum = torch.cumsum(vals, 0) / vals.sum()
    split = int((cum > 1 - tau).nonzero().min())
    keep_ref = ref > vals[split]
    keep_plain = ms_core.cdf_keep_mask(base, tau)
    density._simplify_prune(params, opts, density.cfg["simp_iteration2"])
    kept = params["means"].detach().cpu().double()
    want = means[keep_ref]
    if len(kept) != len(want) or not torch.allclose(kept, want, atol=1e-6):
        raise AssertionError(f"LD prune kept {len(kept)} vs reference {len(want)}")
    if torch.equal(keep_ref, keep_plain):
        raise AssertionError("LD scoring did not change the kept set on the fixture")
    if any(len(o.state) for o in opts.values()):
        raise AssertionError("LD prune must reset the Gaussian optimiser (training_setup)")
    return {"gaussians": n, "kept_ld": int(keep_ref.sum()), "kept_without_ld": int(keep_plain.sum()),
            "differs_from_plain_cdf": int((keep_ref != keep_plain).sum()), "tau": tau}


# ----------------------------------------------------------------------------- C2
def _freq_ref(x, nf):
    cols = []
    for d in range(x.shape[1]):
        for f in range(nf):
            arg = x[:, d] * (2.0**f) * math.pi
            cols += [torch.sin(arg), torch.cos(arg)]
    return torch.stack(cols, 1)


def _mlp_ref(ws, x, leaky):
    h = x
    for w in ws[:-1]:
        h = h @ w.T
        h = torch.where(h > 0, h, 0.01 * h) if leaky else h.clamp_min(0)
    return h @ ws[-1].T


def _appearance_ref(bank, means, static, view):
    w = {k: [p.detach().double() for p in v] for k, v in bank.mlps.items()}
    x = means.double()
    r = x.norm(dim=-1, keepdim=True)
    xc = torch.where(r > 1, (2 - 1 / r) * x / r, x) / 4 + 0.5
    f = _mlp_ref(w["cont"], _freq_ref(xc, 16), False)[:, :13]
    sp = torch.cat([f, static.double()], 1)
    vw = torch.cat([f, view.double()], 1)
    return (_mlp_ref(w["dc"], sp, True)[:, :3], _mlp_ref(w["view"], vw, True)[:, :45],
            _mlp_ref(w["opacity"], sp, True)[:, 0])


def space_feature(method, device):
    # float64: sin(2^15 pi x) is ill-conditioned, so float32 runs of two correct implementations
    # differ by ~1e-3; in float64 they agree to ~1e-9
    bank = model.OMGBank().double()
    g = torch.Generator().manual_seed(1)
    means = torch.randn(200, 3, generator=g, dtype=torch.float64) * 1.5  # in and out of |x|<1
    static = torch.randn(200, 3, generator=g, dtype=torch.float64)
    view = torch.randn(200, 3, generator=g, dtype=torch.float64)
    dc, shn, opa = bank.appearance(means, static, view)
    rdc, rsh, ropa = _appearance_ref(bank, means, static, view)
    err = max((dc[:, 0].double() - rdc).abs().max().item(),
              (shn.reshape(200, 45).double() - rsh).abs().max().item(),
              (opa.double() - ropa).abs().max().item())
    shapes = {k: [tuple(w.shape) for w in v] for k, v in bank.mlps.items()}
    want = {"cont": [(64, 96), (16, 64)], "view": [(64, 16), (48, 64)],
            "dc": [(64, 16), (16, 64)], "opacity": [(64, 16), (16, 64)]}
    if shapes != want or bank.param_count() != 15360:
        raise AssertionError(f"MLP layout differs from tcnn's: {shapes}")
    if err > 1e-7:
        raise AssertionError(f"appearance differs from the reference ({err})")
    # the space feature carries position information; T only drives DC/opacity, V only SH rest
    dc2, shn2, opa2 = bank.appearance(means + 0.05, static, view)
    dc3, shn3, _ = bank.appearance(means, static, view + 1.0)
    if (dc2 - dc).abs().max() < 1e-4 or (shn3 - shn).abs().max() < 1e-4 \
            or (dc3 - dc).abs().max() > 0:
        raise AssertionError("space feature / feature routing is wrong")
    return {"max_error": err, "mlp_params_tcnn": bank.param_count()}


@_restore_active
def train_render(method, device):
    """Live OMG render equals a direct gsplat rasterisation of the bank attributes, and
    gradients reach the MLPs, T, V, scale, rotation and position but not the unused base
    appearance fields."""
    _require_cuda()
    renderer, _, _ = build(method)
    params, _ = _random_scene(80, device, seed=2)
    bank = model.OMGBank().to(device)
    bank.net_enabled = True
    bank.step = 18001  # SH degree 3
    renderer.bank = bank
    cam = _camera(64, 48, 50.0, device=device)
    out = renderer.render(params, cam, sh_degree=0)
    with torch.no_grad():  # attribute formulas are checked independently by space_feature
        rdc, rsh, ropa = bank.appearance(params["means"], params[STATIC], params[VIEW])
    colors = torch.cat([rdc, rsh], 1)
    ref, alpha, _ = rasterization(params["means"].detach(), params["quats"].detach(),
                                  params["scales"].detach().exp(), ropa.sigmoid(), colors,
                                  cam["viewmat"][None], cam["K"][None], 64, 48, sh_degree=3)
    ref = ref[0] + (1 - alpha[0])
    err = (out["rgb"].detach() - ref).abs().max().item()
    if err > 1e-4 or out["info"]["omg"]["sh_degree"] != 3:
        raise AssertionError(f"live render differs from the reference ({err})")
    out["rgb"].square().sum().backward()
    grads = {k: (p.grad is not None and p.grad.abs().sum().item() > 0) for k, p in params.items()}
    mlp = all(w.grad is not None and w.grad.abs().sum() > 0 for v in bank.mlps.values()
              for w in v)
    need = ("means", "scales", "quats", STATIC, VIEW)
    if not mlp or not all(grads[k] for k in need) or any(
            grads[k] for k in ("opacities", "sh0", "shN")):
        raise AssertionError(f"wrong gradient flow: {grads}, mlp {mlp}")
    if core.sh_degree(15500) != 0 or core.sh_degree(16000) != 1 or core.sh_degree(18000) != 3:
        raise AssertionError("SH schedule differs from the official one")
    return {"max_render_error": err, "gradients": grads, "mlp_gradients": mlp}


def net_schedule(method, device):
    density = _density(method)
    o = density.omg
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w], lr=o["net_lr"])
    sched = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000, 3500, 6000], gamma=0.33)])
    worst = 0.0
    for epoch in range(0, 15001):
        if epoch:
            opt.step()
            sched.step()
        got = density.net_lr(epoch)
        worst = max(worst, abs(got - opt.param_groups[0]["lr"]) / opt.param_groups[0]["lr"])
    if worst > 1e-9:
        raise AssertionError(f"net lr differs from the torch scheduler ({worst})")
    want = {"net_itr": 15000, "svq_itr": 29000, "total_steps": 30000, "lambda_ld": 2.0}
    if any(o[k] != v for k, v in want.items()) or density.cfg["simp_iteration2"] != 20000:
        raise AssertionError("OMG event steps differ from the official arguments")
    return {"max_rel_lr_error": worst, **want}


# ----------------------------------------------------------------------------- C3
@_restore_active
def svq(method, device):
    _require_cuda()
    density = _density(method)
    o = density.omg
    params, _ = _random_scene(3000, device, seed=3)
    bank = model.OMGBank().to(device)
    bank.net_enabled = True
    bank.step = o["svq_itr"]
    data = {"scale": params["scales"].detach(), "rotation": params["quats"].detach(),
            "app": torch.cat([params[STATIC].detach(), params[VIEW].detach()], -1)}
    bank.apply_svq(data, o["svq_slices"], o["svq_clusters"], o["codebook_lr"], 50, 1e-4, 0)
    layout = {k: [tuple(c.shape) for c in v] for k, v in bank.codes.items()}
    if layout != {"scale": [(64, 1)] * 3, "rotation": [(512, 2)] * 2, "app": [(1024, 2)] * 3}:
        raise AssertionError(f"SVQ layout differs: {layout}")
    for attr, x in data.items():
        length = o["svq_slices"][attr]
        for m, (c, i) in enumerate(zip(bank.codes[attr], bank.indices[attr])):
            sub = x[:, m * length:(m + 1) * length]
            d = (sub.double()[:, None, :] - c.detach().double()[None]).square().sum(-1)
            if not torch.allclose(d.gather(1, i[:, None])[:, 0], d.min(1).values, atol=1e-9):
                raise AssertionError(f"{attr} indices are not nearest codes")
    renderer, _, _ = build(method)
    renderer.bank = bank
    cam = _camera(64, 48, 50.0, device=device)
    out = renderer.render(params, cam)
    out["rgb"].square().sum().backward()
    code_grad = all(c.grad is not None and c.grad.abs().sum() > 0
                    for v in bank.codes.values() for c in v)
    leaked = [k for k in ("scales", "quats", STATIC, VIEW) if params[k].grad is not None]
    if not code_grad or leaked or params["means"].grad is None:
        raise AssertionError(f"SVQ gradient flow wrong: codes {code_grad}, leaked {leaked}")
    before = [i.clone() for i in bank.indices["scale"]]
    bank.code_optimizer.step()
    if any(not torch.equal(a, b) for a, b in zip(before, bank.indices["scale"])):
        raise AssertionError("indices must stay frozen")
    with torch.no_grad():
        params["scales"].mul_(3.0)  # per-Gaussian scale is unused once SVQ is on
    out2 = renderer.render(params, cam)
    if (out2["rgb"] - out["rgb"].detach()).abs().max() > 1e-3:
        raise AssertionError("render still depends on the per-Gaussian scale after SVQ")
    return {"layout": layout, "codebook_gradients": code_grad}


# ----------------------------------------------------------------------------- C4
def _textbook_huffman_bits(counts):
    heap = list(counts)
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        total += a + b
        heapq.heappush(heap, a + b)
    return total


@_restore_active
def codec_roundtrip(method, device):
    _require_cuda()
    from splatstudio.verification import ProbeUnavailable

    rng = np.random.default_rng(0)
    sym = rng.zipf(1.6, 20000) % 700
    h = codec.huffman_encode(sym)
    dec = codec.huffman_decode(h)
    _, cnt = np.unique(sym, return_counts=True)
    # dahuffman adds an EOF symbol of frequency 1 to the tree
    opt_bits = int(_textbook_huffman_bits([int(c) for c in cnt] + [1]))
    if not np.array_equal(dec, sym):
        raise AssertionError("Huffman round trip is lossy")
    eof_bits = 8 * h["official_bytes"] - h["bits"]
    if not (h["bits"] + 1 <= opt_bits <= h["bits"] + max(eof_bits, 1)):
        raise AssertionError(f"Huffman cost {h['bits']} not optimal ({opt_bits} with EOF)")
    x = torch.tensor([[-1.5, 0.0, 3.25], [65504.0, -0.0001, 2.0]])
    u = core.float16_to_uint16(x)
    if not torch.equal(core.uint16_to_float16(u).float(), x.half().float()) or u.max() > 65535:
        raise AssertionError("fp16 <-> uint16 mapping is not lossless")
    if codec.tmc3_path() is None:
        raise ProbeUnavailable("tmc3 (MPEG G-PCC) not installed")
    # full encode -> decode -> bake on a fixture
    density = _density(method, net_itr=15000, svq_itr=29000, total_steps=30000)
    params, opts = _random_scene(2500, device, seed=4)
    density.initialize(params, opts, scene_scale=1.0)
    bank = density.bank
    bank.net_enabled = True
    bank.step = 29000
    density._apply_svq(params, 28999)
    renderer, _, _ = build(method)
    renderer.bank = bank
    cam = _camera(64, 48, 50.0, device=device)
    rounded = copy.deepcopy(bank)
    rparams = {k: v.detach().clone() for k, v in params.items()}
    with torch.no_grad():
        for p in list(rounded.mlps.parameters()) + [c for v in rounded.codes.values() for c in v]:
            p.copy_(p.half().float())
        rparams["means"] = rparams["means"].half().float()
        expected = live_attributes(rounded, rparams, True)
        live = live_attributes(bank, params, True)
    density._encode(params, opts, 30000)
    rep = density.state["omg"]["encoded"]
    # the decoded state is in the coder's (sorted) order; compare per Gaussian
    order = core.position_order(core.float16_to_uint16(rparams["means"]))
    with torch.no_grad():
        got = {"means": params["means"], "scales": params["scales"].exp(),
               "quats": torch.nn.functional.normalize(params["quats"], dim=-1),
               "opacities": params["opacities"].sigmoid(),
               "colors": torch.cat([params["sh0"], params["shN"]], 1)}
        expected["quats"] = torch.nn.functional.normalize(expected["quats"], dim=-1)
        err = max((got[k] - expected[k][order]).abs().max().item() for k in got)
        drift = (got["colors"] - live["colors"][order]).abs().mean().item()
        renderer.bank = None
        model.ACTIVE["bank"] = None  # fresh-process path (ss-eval) renders the baked state
        out = renderer.render(params, cam, sh_degree=3)
    if not bank.baked or err > 1e-5 or not torch.isfinite(out["rgb"]).all():
        raise AssertionError(f"decoded model differs from the fp16-rounded live model ({err})")
    if rep["container_bytes"] <= 0 or rep["official_bytes"]["xyz"] <= 0:
        raise AssertionError("missing storage report")
    return {"huffman_bits": h["bits"], "optimal_bits_with_eof": opt_bits,
            "decoded_vs_fp16_live_max_abs": err, "decoded_vs_fp32_live_colour_mean_abs": drift,
            "storage": rep}
