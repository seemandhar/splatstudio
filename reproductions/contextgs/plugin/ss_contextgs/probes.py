"""Mechanism probes for the ContextGS port. Each compares against an independent formula, a brute-force
reference or a constructed case and raises when the mechanism is removed or wrong.

entropy-model, codec-roundtrip, neural-gaussians, visibility-filter, anchor-density, training-phases and
bake-eval are adapted from the HAC native port (reproductions/hac/plugin/ss_hac/probes.py); factorized-prior,
level-partition and context-model are ContextGS-specific.
"""

import copy
import io
import math
from types import SimpleNamespace

import numpy as np
import torch

from . import codec, levels, model
from .density import DEFAULTS, ContextGSDensity, merge, new_anchor_fields
from .renderer import BLOB, BLOB_WIDTH, ContextGSRenderer, visible_anchors
from .renderer import neural_gaussians as decode_neural


def _need_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is required")


def _spec(config):
    return SimpleNamespace(config=config)


def _bank(device, seed=0, **kw):
    torch.manual_seed(seed)
    c = merge(DEFAULTS, {"voxel_size": 0.01, **kw})
    return model.ContextBank(c, device)


def _cloud(n, seed, device, voxel=0.01):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    pts = torch.unique(torch.round(pts / voxel), dim=0) * voxel
    return pts[torch.randperm(len(pts), generator=g)].to(device)


# --------------------------------------------------------------------------- entropy
def entropy_model(method, device):
    g = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(1000, generator=g, device=device) * 3
    mean = torch.randn(1000, generator=g, device=device)
    scale = torch.rand(1000, generator=g, device=device) * 2 + 0.1
    q = torch.rand(1000, generator=g, device=device) + 0.2
    bits = model.gaussian_bits(x, mean, scale, q, x.mean())

    def phi(t):
        return 0.5 * (1 + torch.erf(t / math.sqrt(2)))

    p = (phi((x + q / 2 - mean) / scale) - phi((x - q / 2 - mean) / scale)).clamp_min(1e-6)
    err = (bits - (-torch.log2(p))).abs().max().item()
    if err > 1e-3:
        raise AssertionError(f"Gaussian bit estimate differs from the erf formula by {err}")
    bank = _bank(device)
    pred = bank.predict(0, torch.randn(50, 71, generator=g, device=device) * 30)
    for key, q0 in (("feat", 1.0), ("scaling", 0.001), ("offsets", 0.2)):
        qq = pred[key][2]
        if not (qq.min() > 0 and qq.max() <= 2 * q0 + 1e-9):
            raise AssertionError(f"adaptive Q of {key} outside (0, 2 Q0)")
    v = torch.tensor([0.26, -0.74, 1.01, 1e6], device=device)
    qv = model.STEMultistep.apply(v, torch.tensor(0.5, device=device))
    if not torch.allclose(qv, torch.tensor([0.5, -0.5, 1.0, 7500.0], device=device)):
        raise AssertionError("STE multistep does not round to Q with the +-15000 Q clamp")
    b = torch.tensor([1.0, 0, 0, 1, 1, 1, 1, 0], device=device)
    ref_b = -(5 * math.log2(5 / 8) + 3 * math.log2(3 / 8)) + 32
    if abs(float(model.binary_bits(b)) - ref_b) > 1e-3:
        raise AssertionError("binary (mask) entropy wrong")
    return {"gaussian_bits_error": err}


# --------------------------------------------------------------------------- hyperprior
def factorized_prior(method, device):
    torch.manual_seed(7)
    prior = model.FactorizedPrior(12).to(device)
    n_params = sum(p.numel() for p in prior.parameters()) + prior.quantiles.numel()
    # compressai EntropyBottleneck(12), filters (3,3,3,3): per channel 33 matrix + 13 bias + 12 factor, +3
    if n_params != 12 * (33 + 13 + 12) + 12 * 3:
        raise AssertionError(f"factorized prior has {n_params} parameters, compressai has 732")
    init = np.log(np.expm1(1 / 10 ** (1 / 5) / 3))
    if abs(float(prior.matrices[1][0, 0, 0].detach()) - init) > 1e-6 or float(prior.medians().abs().max()) != 0:
        raise AssertionError("factorized prior initial values differ from compressai")
    g = torch.Generator(device="cpu").manual_seed(8)
    with torch.no_grad():  # move away from the initialisation
        for p in prior.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(device) * 0.3)
    x = (torch.randn(200, 12, generator=g) * 4).to(device)

    def ref_cum(v):  # independent per-channel scalar recursion, float64
        out = np.zeros(v.shape)
        mats = [m.detach().double().cpu().numpy() for m in prior.matrices]
        bias = [b.detach().double().cpu().numpy() for b in prior.biases]
        fac = [f.detach().double().cpu().numpy() for f in prior.factors]
        vv = v.detach().double().cpu().numpy()
        for c in range(v.shape[1]):
            h = vv[:, c][None, :]  # [1, M]
            for i in range(len(mats)):
                h = np.log1p(np.exp(mats[i][c])) @ h + bias[i][c]
                if i < len(fac):
                    h = h + np.tanh(fac[i][c]) * np.tanh(h)
            out[:, c] = h[0]
        return out

    def sig(t):
        return 1 / (1 + np.exp(-t))

    out, lik = prior(x, training=False)
    xr = torch.round(x)
    if not torch.equal(out, xr):
        raise AssertionError("dequantize mode is not round(x - median) + median")
    ref = np.maximum(sig(ref_cum(xr + 0.5)) - sig(ref_cum(xr - 0.5)), 1e-9)
    err = float(np.abs(lik.detach().double().cpu().numpy() - ref).max())
    if err > 1e-5:
        raise AssertionError(f"factorized likelihood differs from the reference by {err}")
    noisy, _ = prior(x, training=True)
    d = noisy - x
    if d.abs().max() > 0.5 or torch.equal(noisy, torch.round(noisy)):
        raise AssertionError("training mode is not additive U(-1/2, 1/2) noise")
    far = torch.full((4, 12), 1e4, device=device)
    if float(prior(far, training=False)[1].min()) < 1e-9 - 1e-15:
        raise AssertionError("likelihood lower bound 1e-9 missing")
    # gradient flows to the latent and the density parameters
    xg = x.clone().requires_grad_(True)
    (-torch.log2(prior(xg, training=True)[1])).sum().backward()
    if xg.grad is None or prior.matrices[0].grad is None or prior.factors[0].grad is None:
        raise AssertionError("rate does not reach the hyper latent and the prior")
    return {"parameters": n_params, "likelihood_error": err}


# --------------------------------------------------------------------------- partition
def _brute_reps(points, cell):
    """Minimum index per voxel (Eq. 5 M) by a Python dictionary."""
    keys = torch.round(points / cell).long().cpu().tolist()
    first = {}
    for i, kk in enumerate(map(tuple, keys)):
        first.setdefault(kk, i)
    return sorted(first.values())


def level_partition(method, device):
    bank = _bank(device)
    anchor = _cloud(6000, 11, device)
    bank.update_anchor_bound(anchor)
    v = bank.c["voxel_size"]
    scales = levels.find_divide_scale(anchor, v, bank.x_bound_min, bank.x_bound_max, 0.2, 3)
    lv = levels.divide_levels(anchor, v, scales, 3)
    hal, inv, mapping, _ = lv
    n = len(anchor)
    hat1 = mapping[0].cpu().tolist()
    if sorted(hat1) != _brute_reps(anchor, v * scales[0]):
        raise AssertionError("level-1 representatives are not the minimum index of each voxel")
    lvl1_pos = hal[1]
    hat2_in_l1 = mapping[1].cpu().tolist()
    if sorted(hat2_in_l1) != _brute_reps(lvl1_pos, v * scales[1]):
        raise AssertionError("level-2 representatives are not the minimum index of each voxel")
    v2 = set(levels.mapping_to_orign(mapping, 2).cpu().tolist())
    v1 = set(hat1) - v2
    v0 = set(range(n)) - set(hat1)
    if v0 & v1 or v1 & v2 or v0 & v2 or len(v0) + len(v1) + len(v2) != n:
        raise AssertionError("levels are not a disjoint cover (Eq. 6)")
    bank.level_scale = scales
    z = torch.zeros(n, 12, device=device)
    with torch.no_grad():
        _, _, _, info = levels.multi_scale(bank, anchor, z, torch.zeros(n, 50, device=device),
                                           torch.zeros(n, 10, 3, device=device),
                                           torch.ones(n, 6, device=device) * 0.01,
                                           torch.ones(n, 10, 1, device=device))
    coded = [set(o.cpu().tolist()) for o in info["order"]]
    if coded != [v2, v1, v0]:
        raise AssertionError("coding order does not visit V^2, V^1, V^0")
    r1, r2 = len(hat1) / n, len(v2) / len(hat1)
    if abs(r1 - 0.2) > 0.03 or abs(r2 - 0.2) > 0.03:
        raise AssertionError(f"binary-searched scales give ratios {r1:.3f}, {r2:.3f}, not ~0.2")
    # masked anchors are moved to the origin before the level-1 voxelization (training path)
    mask = torch.rand(n, generator=torch.Generator(device="cpu").manual_seed(3)).to(device) > 0.3
    hal_m, _, mapping_m, _ = levels.divide_levels(anchor, v, scales, 3, mask)
    ref = _brute_reps(anchor * mask[:, None], v * scales[0])
    if sorted(mapping_m[0].cpu().tolist()) != ref:
        raise AssertionError("masked anchors are not moved to the origin before level 1")
    return {"scales": scales, "ratios": [r1, r2], "levels": [len(v2), len(v1), len(v0)]}


# --------------------------------------------------------------------------- context model
def _recording(bank):
    rec = {}
    orig_predict = bank.predict

    def predict(level, feat_in):
        out = orig_predict(level, feat_in)
        rec[level] = torch.cat([out["feat"][0], out["scaling"][0], out["offsets"][0],
                                out["feat"][2]], dim=1).detach().clone()
        return out

    bank.predict = predict
    return rec


def context_model(method, device):
    _need_cuda(device)
    bank = _bank(device, seed=2)
    with torch.no_grad():
        for p in bank.mlp_grid.parameters():
            p.mul_(3.0)
    anchor = _cloud(5000, 12, device)
    bank.update_anchor_bound(anchor)
    n = len(anchor)
    g = torch.Generator(device="cpu").manual_seed(13)
    feat = (torch.randn(n, 50, generator=g) * 3).to(device)
    scal = (torch.rand(n, 6, generator=g) * 0.05 + 0.01).to(device)
    offs = torch.randn(n, 10, 3, generator=g).to(device)
    hyper = (torch.randn(n, 12, generator=g) * 2).to(device)
    masks = torch.ones(n, 10, 1, device=device)

    def run(f, s, h, use_hyper=True):
        rec = _recording(bank)
        with torch.no_grad():
            _, _, _, info = levels.multi_scale(bank, anchor, h, f, offs, s, masks, None,
                                               training=False, use_hyper=use_hyper)
        bank.predict = type(bank).predict.__get__(bank)
        return rec, info

    base, info = run(feat, scal, hyper)
    v2, v1, v0 = info["order"]
    # finer-level change must not affect coarser predictions (decodability / autoregressive order)
    f0 = feat.clone()
    f0[v0] += 5.0
    r0, _ = run(f0, scal, hyper)
    if not (torch.equal(r0[2], base[2]) and torch.equal(r0[1], base[1])):
        raise AssertionError("level-0 attributes leak into level-2/1 predictions")
    f2 = feat.clone()
    f2[v2] += 5.0
    r2, _ = run(f2, scal, hyper)
    if not torch.equal(r2[2], base[2]):
        raise AssertionError("level-2 prediction depends on its own attributes")
    if torch.allclose(r2[1], base[1]) or torch.allclose(r2[0], base[0]):
        raise AssertionError("level-2 attributes do not condition finer levels (no context)")
    rh, _ = run(feat, scal, hyper, use_hyper=False)
    if any(torch.allclose(rh[i], base[i]) for i in range(3)):
        raise AssertionError("hyperprior does not enter every level's prior")
    # context rows equal an independent gather in the official order
    lv = info["levels"]
    hal, inv, mapping, _ = lv
    fq, sq, _, _ = levels.multi_scale(bank, anchor, hyper, feat, offs, scal, masks, None)
    content = torch.cat([anchor, fq, sq], dim=1)
    v = bank.c["voxel_size"]
    s1, s2 = bank.level_scale
    key1 = {tuple(kk): i for i, kk in reversed(list(enumerate(
        torch.round(anchor / v / s1).long().cpu().tolist())))}
    parent1 = [key1[tuple(kk)] for kk in torch.round(anchor / v / s1).long().cpu().tolist()]
    done = torch.zeros(n, dtype=torch.bool, device=device)
    done[torch.cat([v2, v1])] = True
    got0 = levels.extract_context_feat(anchor, fq, sq, done, inv, mapping, 1)
    want0 = content[torch.tensor([parent1[a] for a in sorted(v0.cpu().tolist())], device=device)]
    if not torch.equal(got0, want0):
        raise AssertionError("level-0 context is not the [x, f, l] of the level-1 parent")
    hat1 = mapping[0]
    lvl1_key = torch.round(anchor[hat1] / v / s2).long().cpu().tolist()
    key2 = {}
    for j, kk in enumerate(lvl1_key):
        key2.setdefault(tuple(kk), int(hat1[j]))
    parent2 = {int(hat1[j]): key2[tuple(kk)] for j, kk in enumerate(lvl1_key)}
    done2 = torch.zeros(n, dtype=torch.bool, device=device)
    done2[v2] = True
    got1 = levels.extract_context_feat(anchor, fq, sq, done2, inv, mapping, 2)
    rows = sorted(v1.cpu().tolist())  # official: increasing original index
    want1 = content[torch.tensor([parent2[a] for a in rows], device=device)]
    if not torch.equal(got1, want1):
        raise AssertionError("level-1 context differs from the official gather order")
    mismatch = float(np.mean([a != b for a, b in zip(rows, v1.cpu().tolist())]))
    return {"levels": [len(v2), len(v1), len(v0)], "level_scale": [s1, s2],
            "level1_context_order_mismatch_fraction": mismatch}


# --------------------------------------------------------------------------- codec
def codec_roundtrip(method, device):
    _need_cuda(device)
    g = torch.Generator(device=device).manual_seed(2)
    n = 60000
    mean = torch.randn(n, generator=g, device=device) * 4
    scale = torch.rand(n, generator=g, device=device) * 3 + 0.3
    q = 0.2 + 0.5 * (1 + torch.tanh(torch.randn(n, generator=g, device=device)))
    x = torch.round((mean + scale * torch.randn(n, generator=g, device=device)) / q) * q
    payload, bits = codec.encode_gaussian(x, mean, scale, q)
    dec = codec.decode_gaussian(payload, mean, scale, q)
    if (dec - x).abs().max() > 1e-4:
        raise AssertionError("Gaussian arithmetic coding is not lossless")
    est = float(model.gaussian_bits(x, mean, scale, q, x.mean()).sum())
    if abs(bits - est) / est > 0.02:
        raise AssertionError(f"coded size {bits} vs entropy {est}")
    b = (torch.rand(n, generator=g, device=device) < 0.2).float()
    bp, bb = codec.encode_binary(b)
    if not torch.equal(codec.decode_binary(bp, n, device).float(), b):
        raise AssertionError("binary arithmetic coding is not lossless")
    be = float(model.binary_bits(b))
    if abs(bb - be) / be > 0.02:
        raise AssertionError(f"binary coded size {bb} vs entropy {be}")
    torch.manual_seed(4)
    prior = model.FactorizedPrior(12).to(device)
    z = torch.randn(20000, 12, generator=g, device=device) * 2.5
    z[0, 0] = 40.0  # far outside the initial quantile range
    sym = prior.symbols(z)
    fp, fb = codec.encode_factorized(sym, prior)
    fd = codec.decode_factorized(fp, len(z), prior, device)
    if not torch.equal(fd, sym):
        raise AssertionError("factorized-prior coding is not lossless")
    fe = float((-torch.log2(prior(z, training=False)[1])).sum())
    if abs(fb - fe) / fe > 0.02:
        raise AssertionError(f"hyperprior coded size {fb} vs entropy {fe}")
    return {"gaussian_bits": bits, "gaussian_entropy": est, "binary_bits": bb,
            "binary_entropy": be, "hyper_bits": fb, "hyper_entropy": fe}


# --------------------------------------------------------------------------- decode
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
    masks = (torch.rand(n, k, 1, generator=g) > 0.3).float().to(device)
    center = torch.zeros(3, device=device)
    xyz, color, opa, scales, quats, _, _ = decode_neural(mlps, anchor, feat, offsets, scaling, masks,
                                                         center, k)
    ref = []
    with torch.no_grad():
        for i in range(n):
            v = anchor[i] - center
            d = v.norm()
            inp = torch.cat([feat[i], v / d, d[None]])[None]
            o = mlps["mlp_opacity"](inp)[0] * masks[i, :, 0]
            c = mlps["mlp_color"](inp)[0].view(k, 3)
            sr = mlps["mlp_cov"](inp)[0].view(k, 7)
            for j in range(k):
                if o[j] > 0:
                    ref.append(torch.cat([anchor[i] + offsets[i, j] * scaling[i, :3], c[j], o[j][None],
                                          scaling[i, 3:] * torch.sigmoid(sr[j, :3]),
                                          sr[j, 3:] / sr[j, 3:].norm()]))
    ref = torch.stack(ref)
    got = torch.cat([xyz, color, opa[:, None], scales, quats], dim=1)
    if got.shape != ref.shape or (got - ref).abs().max() > 1e-5:
        raise AssertionError("neural Gaussian decode differs from the per-anchor reference")
    c2 = decode_neural(mlps, anchor, feat, offsets, scaling, masks,
                       torch.tensor([3.0, 0, 0], device=device), k)[1]
    if c2.shape == color.shape and torch.allclose(c2, color):
        raise AssertionError("decode is not view dependent")
    return {"gaussians": len(ref), "max_error": float((got - ref).abs().max())}


def _fixture(device, n=3000, seed=4):
    """A density + bank on n anchors in front of an identity camera."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = merge(DEFAULTS, {})
    dens = ContextGSDensity(_spec({}))
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    pts = (pts + torch.tensor([0, 0, 3.0])).to(device)
    hyper = torch.zeros(n, c["feat_dim"] // c["hyper_divisor"], device=device)
    params = torch.nn.ParameterDict(new_anchor_fields(
        pts, torch.zeros(n, 6, device=device), torch.zeros(n, c["feat_dim"], device=device),
        c["n_offsets"], BLOB_WIDTH, hyper))
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    dens.c["voxel_size"] = 0.01
    dens.initialize(params, opts, 1.0)
    renderer = ContextGSRenderer(_spec({"feat_dim": c["feat_dim"], "n_offsets": c["n_offsets"]}))
    renderer.bank = dens.bank
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device),
           "width": 64, "height": 64}
    return dens, params, opts, renderer, cam


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


def anchor_density(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=50)
    model.ACTIVE["bank"] = None
    k = dens.c["n_offsets"]
    with torch.no_grad():
        params["cgs_scaling"].data[:] = math.log(0.05)
        params["cgs_offset"].data[:] = 0
        params["cgs_offset"].data[0, :] = torch.tensor([10.0, 0, 0])
        params["cgs_hyper"].data.normal_()
        params["cgs_feat"].data.normal_()
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    n0 = len(params["means"])
    dens.offset_gradient_accum.zero_()
    dens.offset_denom[:] = 100
    dens.offset_gradient_accum[:k] = 1.0 * 100  # offsets of anchor 0: grad 1 >> threshold
    dens.opacity_accum[:] = 1.0
    dens.anchor_demon[:] = 100
    dens.opacity_accum[5] = 0.0  # anchor 5: low opacity, visible often -> pruned
    torch.manual_seed(0)
    anchor0 = dens.bank.quantize_anchor(params["means"])[0][0].clone()
    with torch.no_grad():
        targets = anchor0 + params["cgs_offset"][0] * params["cgs_scaling"][0, :3].exp()
    means_before = params["means"].detach().clone()
    hyper0 = params["cgs_hyper"][0].detach().clone()
    feat0 = params["cgs_feat"][0].detach().clone()
    dens.adjust_anchor(params, opts, 2000)
    _check(params, opts)
    n1 = len(params["means"])
    kept_old = torch.cat([means_before[:5], means_before[6:]])
    if not torch.equal(params["means"][: n0 - 1].detach(), kept_old):
        raise AssertionError("pruning did not remove exactly the low-opacity anchor")
    new = params["means"][n0 - 1:].detach()
    if len(new) == 0:
        raise AssertionError("no anchor grown at the high-gradient offset")
    c = dens.c
    sizes = [c["voxel_size"] * (c["update_init_factor"] // c["update_hierachy_factor"] ** i)
             for i in range(c["update_depth"])]
    for a in new:
        if not any(torch.allclose(a, torch.round(t / s) * s, atol=1e-5) for s in sizes for t in targets):
            raise AssertionError("grown anchor is not at the voxel of the offset")
    if not (torch.equal(params["cgs_hyper"][n0 - 1:].detach(), hyper0.expand(len(new), -1))
            and torch.equal(params["cgs_feat"][n0 - 1:].detach(), feat0.expand(len(new), -1))):
        raise AssertionError("grown anchors do not inherit the parent's feature / hyper latent")
    if len(dens.offset_denom) != n1 * k or len(dens.opacity_accum) != n1:
        raise AssertionError("density statistics misaligned with the population")
    return {"anchors_before": n0, "anchors_after": n1, "grown": n1 - (n0 - 1)}


def training_phases(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=3000)
    bank = dens.bank
    with torch.no_grad():
        params["cgs_feat"].data.normal_()
        params["cgs_offset"].data.normal_(0, 0.5)
        params["cgs_scaling"].data[:] = math.log(0.05)
        params["cgs_hyper"].data.normal_()
    out = {}

    def twice(step):
        bank.step = step
        torch.manual_seed(0)
        a = renderer.render(params, cam)
        torch.manual_seed(1)
        b = renderer.render(params, cam)
        return float((a["rgb"] - b["rgb"]).abs().max()), a["info"]["cgs"]["bits"]

    for step, noisy, rate in ((3000, False, False), (3001, True, False), (10000, True, False),
                              (10001, True, True)):
        if step == 10000:
            with torch.no_grad():
                params["means"].data[0, 0] += 7.0  # bounds must be recomputed at 10000
        if bank.level_scale is not None:
            raise AssertionError(f"level scales searched before the entropy phase (step {step})")
        diff, bits = twice(step)
        if (diff > 0) != noisy or (bits is not None) != rate:
            raise AssertionError(f"phase wrong at step {step}: diff {diff}, bits {bits is not None}")
        if step == 10000 and float(bank.x_bound_max[0, 0]) < float(params["means"][0, 0]):
            raise AssertionError("anchor bounds not updated at step 10000")
        out[f"diff_{step}"] = diff
    if bank.level_scale is None or len(bank.level_scale) != 2:
        raise AssertionError("level scales not searched at the first entropy step")
    bank.step = 20000
    torch.manual_seed(2)
    res = renderer.render(params, cam)
    bits = res["info"]["cgs"]["bits"]
    frac = bits["sampled"] / max(bits["eligible"], 1)
    if not 0.12 < frac < 0.18:
        raise AssertionError(f"rate sample fraction {frac} is not ~15%")
    out["sample_fraction"] = frac
    from .losses import create_rate

    lmbda, lm = 0.004, 5e-4
    rate = create_rate(_spec({"lmbda": lmbda, "mask_weight": lm}))
    fixed = {"bit_per_param": torch.tensor(2.5, device=device, requires_grad=True)}
    r = rate(params, {"info": {"cgs": {"bits": fixed}}}, None)
    want = lmbda * 2.5 + lm * torch.sigmoid(params["cgs_mask"]).mean()
    if abs(float(r) - float(want)) > 1e-7:
        raise AssertionError("rate loss value differs from lambda_e * bpp + lambda_m * mean sigmoid(mask)")
    r = rate(params, res, None)
    r.backward()
    reach = {"feat": params["cgs_feat"].grad, "hyper": params["cgs_hyper"].grad,
             "mask": params["cgs_mask"].grad, "mlp_grid": bank.mlp_grid[0][0].weight.grad,
             "mlp_grid_top": bank.mlp_grid[2][0].weight.grad, "prior": bank.prior.matrices[0].grad}
    missing = [k for k, v in reach.items() if v is None or float(v.abs().sum()) == 0]
    if missing:
        raise AssertionError(f"rate loss does not reach {missing}")
    lr = dens.c["lr"]
    ref = math.exp(math.log(0.005) * 0.5 + math.log(0.00001) * 0.5)
    dens.update_learning_rate(params, opts, 15000)
    got = {g["name"]: g["lr"] for g in bank.optimizer.param_groups}
    if abs(got["latent_codec"] - ref) > 1e-12 or abs(got["mlp_grid"] - ref) > 1e-12:
        raise AssertionError("latent codec / context MLP learning rates not scheduled")
    ref_c = math.exp(math.log(lr["mlp_color"][0]) * 0.5 + math.log(lr["mlp_color"][1]) * 0.5)
    if abs(got["mlp_color"] - ref_c) > 1e-12 or opts["means"].param_groups[0]["lr"] != 0:
        raise AssertionError("render MLP / anchor learning rates not scheduled")
    out["rate"] = float(r)
    out["bit_per_param"] = float(bits["bit_per_param"])
    return out


def bake_eval(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=3000)
    g = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        params["cgs_feat"].data.copy_(torch.randn(params["cgs_feat"].shape, generator=g) * 3)
        params["cgs_offset"].data.copy_(torch.randn(params["cgs_offset"].shape, generator=g))
        params["cgs_scaling"].data.copy_(torch.randn(params["cgs_scaling"].shape, generator=g) * 0.3 - 3)
        params["cgs_mask"].data.copy_(torch.randn(params["cgs_mask"].shape, generator=g) * 4)
        params["cgs_hyper"].data.copy_(torch.randn(params["cgs_hyper"].shape, generator=g) * 2)
        dens.bank.update_anchor_bound(params["means"])
    bank_ref = copy.deepcopy(dens.bank.mlps)
    model.ACTIVE["bank"] = dens.bank
    try:
        dens.bake(params, opts)
        with torch.no_grad():
            baked = renderer.render(params, cam)["rgb"]
        enc = dens.state["contextgs"]["encoded"]
        buf = io.BytesIO()
        torch.save({"params": dict(params), "density_state": dens.state}, buf)
    finally:
        model.ACTIVE["bank"] = None
    if enc["bits"]["hyper"] <= 0 or min(enc["level_counts"].values()) <= 0:
        raise AssertionError("hyperprior stream or a level is empty")
    buf.seek(0)
    ck = torch.load(buf, map_location=device, weights_only=True)
    fresh = ContextGSRenderer(_spec({"feat_dim": 50, "n_offsets": 10}))
    with torch.no_grad():
        again = fresh.render(ck["params"], cam)["rgb"]
    if not torch.equal(baked, again):
        raise AssertionError("fresh-process render of the checkpoint differs from the bake")
    restored = fresh.decoded_mlps(ck["params"])
    for k in model.RENDER_MLPS:
        for a, b in zip(restored[k].parameters(), bank_ref[k].parameters()):
            if not torch.equal(a, b):
                raise AssertionError("blob does not restore the trained MLP weights")
    if float(again.min()) > 0.999:
        raise AssertionError("baked fixture renders nothing")
    empty = {k: v.detach().clone() for k, v in ck["params"].items()}
    empty[BLOB].zero_()
    try:
        ContextGSRenderer(_spec({"feat_dim": 50, "n_offsets": 10})).render(empty, cam)
    except ValueError:
        pass
    else:
        raise AssertionError("renderer must refuse an un-baked state")
    return {"anchors_coded": enc["anchors_coded"], "level_counts": enc["level_counts"],
            "size_mb": enc["size_mb"]["total"], "roundtrip_max_error": max(enc["roundtrip_max_error"].values())}


def visibility_filter(method, device):
    """Constructed case for the Scaffold fork's visible_filter (64x64 image, f=60, 16-px tiles).
    A z-only filter keeps the off-screen anchors and fails this probe."""
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device),
           "width": 64, "height": 64}
    anchors = torch.tensor([
        [0.0, 0.0, 3.0],    # centre, small: visible
        [5.0, 0.0, 3.0],    # off-screen right (u ~ 132 px), small: dropped
        [5.0, 0.0, 3.0],    # same position but huge footprint reaching the image: kept
        [0.0, 0.0, -3.0],   # behind the camera: dropped
        [0.0, 0.0, 0.1],    # in front but z <= 0.2: dropped
        [0.0, -1.95, 3.0],  # just above the top edge (v ~ -7 px) but radius > 7: kept
        [-3.0, 0.0, 3.0],   # off-screen left (u ~ -28 px), small: dropped
    ], device=device)
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
        params["means"].data[:200, 0] += 50.0  # push half the anchors far off-screen
        dens.bank.update_anchor_bound(params["means"])
    out = renderer.render(params, cam2)
    vis = out["info"]["cgs"]["visible"]
    ref = visible_anchors(dens.bank.quantize_anchor(params["means"])[0].detach(),
                          params["cgs_scaling"][:, :3].exp().detach(), cam2)
    if not torch.equal(vis, ref) or bool(vis[:200].any()):
        raise AssertionError("renderer visibility differs from visible_filter")
    return {"case": got, "renderer_visible": int(vis.sum())}
