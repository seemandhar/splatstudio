"""Mechanism probes for the HAC port. Each compares against an independent formula or a
constructed case and raises when the mechanism is removed or wrong."""

import copy
import io
import math
from types import SimpleNamespace

import torch

from . import codec, model
from .density import DEFAULTS, HACDensity, merge, new_anchor_fields
from .renderer import BLOB, BLOB_WIDTH, HACRenderer, visible_anchors
from .renderer import neural_gaussians as decode_neural


def _need_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is required")


def _spec(config):
    return SimpleNamespace(config=config)


def _small_config(**kw):
    c = merge(DEFAULTS, {"log2": 10, "log2_2d": 10, "resolutions": [6, 9, 14],
                         "resolutions_2d": [8, 17], **kw})
    return c


# --------------------------------------------------------------------------- hash grid
def _ref_grid(x, emb, resolutions, offsets):
    """Independent reference of HAC's grid: pos = x (R-2) + 0.5, multilinear over the 2^d
    corners (upper corner clipped to R-1); corners on the border (0 or R-1 in any dim) are
    dropped and the remaining weights renormalised; dense index sum_d p_d R^d while R^d fits
    the level table, else XOR prime hash; mod table size."""
    primes = [1, 2654435761, 805459861]
    n, dim = x.shape
    outs = []
    for lv, r in enumerate(resolutions):
        size = int(offsets[lv + 1] - offsets[lv])
        table = emb[int(offsets[lv]) : int(offsets[lv + 1])]
        pos = x.double() * (r - 2) + 0.5
        base = torch.floor(pos).long()
        frac = pos - base
        acc = torch.zeros(n, emb.shape[1], dtype=torch.float64, device=x.device)
        wsum = torch.zeros(n, dtype=torch.float64, device=x.device)
        for corner in range(1 << dim):
            w = torch.ones(n, dtype=torch.float64, device=x.device)
            idx_grid = []
            for d in range(dim):
                bit = (corner >> d) & 1
                w = w * (frac[:, d] if bit else 1 - frac[:, d])
                idx_grid.append((base[:, d] + bit).clamp(max=r - 1))
            valid = torch.ones(n, dtype=torch.bool, device=x.device)
            for d in range(dim):
                valid &= (idx_grid[d] != 0) & (idx_grid[d] != r - 1)
            w = w * valid
            stride, index, dense = 1, torch.zeros(n, dtype=torch.long, device=x.device), True
            for d in range(dim):
                if stride > size:
                    break
                index = index + idx_grid[d] * stride
                stride *= r
            if stride > size:
                dense = False
            if not dense:
                h = torch.zeros(n, dtype=torch.long, device=x.device)
                for d in range(dim):
                    h = h ^ ((idx_grid[d] * primes[d]) & 0xFFFFFFFF)
                index = h
            index = (index & 0xFFFFFFFF) % size
            acc = acc + w[:, None] * table[index].double()
            wsum = wsum + w
        wsum = torch.where(wsum == 0, torch.full_like(wsum, 1e-9), wsum)
        outs.append(acc / wsum[:, None])
    return torch.cat(outs, dim=1)


def hash_grid(method, device):
    _need_cuda(device)
    g = torch.Generator(device=device).manual_seed(0)
    evidence = {}
    for dim, res, log2 in ((3, [6, 9, 40], 10), (2, [8, 70], 9)):
        enc = model.GridEncoder(dim, 4, res, log2).to(device)
        enc.params.data.uniform_(-1, 1, generator=g)
        x = torch.rand(500, dim, generator=g, device=device) * 0.98 + 0.01
        out = enc(x)
        emb = model.STEBinary.apply(enc.params.detach())
        if not torch.all((emb == 1) | (emb == -1)):
            raise AssertionError("hash embeddings are not binary")
        ref = _ref_grid(x, emb, res, enc.offsets_list.tolist())
        err = (out.double() - ref).abs().max().item()
        if err > 1e-4:
            raise AssertionError(f"{dim}D hash grid differs from the reference by {err}")
        w = torch.randn_like(out)
        (out * w).sum().backward()
        theta = enc.params.detach().clone().requires_grad_(True)
        ref2 = _ref_grid(x, model.STEBinary.apply(theta), res, enc.offsets_list.tolist())
        (ref2 * w.double()).sum().backward()
        gerr = (enc.params.grad - theta.grad).abs().max().item()
        if gerr > 1e-3:
            raise AssertionError(f"{dim}D hash grid gradient differs by {gerr}")
        evidence[f"{dim}d_max_error"] = err
        evidence[f"{dim}d_grad_error"] = gerr
    mix = model.Mix3D2DEncoding(4, (6, 9), 10, (8,), 10).to(device)
    out = mix(torch.rand(10, 3, generator=g, device=device))
    if out.shape != (10, 4 * 2 + 3 * 4):
        raise AssertionError("3D + 2D context feature has the wrong width")
    evidence["mix_dim"] = out.shape[1]
    return evidence


# --------------------------------------------------------------------------- entropy
def entropy_model(method, device):
    g = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(1000, generator=g, device=device) * 3
    mean = torch.randn(1000, generator=g, device=device)
    scale = torch.rand(1000, generator=g, device=device) * 2 + 0.1
    q = torch.rand(1000, generator=g, device=device) + 0.2
    bits = model.gaussian_bits(x, mean, scale, q)

    def phi(t):
        return 0.5 * (1 + torch.erf(t / math.sqrt(2)))

    p = (phi((x + q / 2 - mean) / scale) - phi((x - q / 2 - mean) / scale)).clamp_min(1e-6)
    ref = -torch.log2(p)
    err = (bits - ref).abs().max().item()
    if err > 1e-3:
        raise AssertionError(f"Gaussian bit estimate differs from the erf formula by {err}")
    r = torch.linspace(-20, 20, 101, device=device)
    q_adj = 0.2 * (1 + torch.tanh(r))
    if not (q_adj.min() >= 0 and q_adj.max() <= 0.4):
        raise AssertionError("adaptive Q outside (0, 2 Q0)")
    v = torch.tensor([0.26, -0.74, 1.01], device=device)
    qv = model.STEMultistep.apply(v, torch.tensor(0.5, device=device), torch.tensor(0.0))
    if not torch.allclose(qv, torch.tensor([0.5, -0.5, 1.0], device=device)):
        raise AssertionError("STE multistep does not round to the step")
    b = torch.tensor([1.0, 0, 0, 1, 1, 1, 1, 0], device=device)
    hb = float(model.binary_bits(b))
    p1 = 5 / 8
    ref_b = -(5 * math.log2(p1) + 3 * math.log2(1 - p1)) + 32
    if abs(hb - ref_b) > 1e-3:
        raise AssertionError("binary entropy (L_hash / masks) wrong")
    return {"gaussian_bits_error": err, "binary_bits": hb}


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
    bd = codec.decode_binary(bp, n, device).float()
    if not torch.equal(bd, b):
        raise AssertionError("binary arithmetic coding is not lossless")
    be = float(model.binary_bits(b))
    if abs(bb - be) / be > 0.02:
        raise AssertionError(f"binary coded size {bb} vs entropy {be}")
    return {"gaussian_bits": bits, "gaussian_entropy": est, "binary_bits": bb,
            "binary_entropy": be}


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
    xyz, color, opa, scales, quats, _, _ = decode_neural(mlps, anchor, feat, offsets,
                                                            scaling, masks, center, k)
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
                    ref.append(torch.cat([anchor[i] + offsets[i, j] * scaling[i, :3], c[j],
                                          o[j][None], scaling[i, 3:] * torch.sigmoid(sr[j, :3]),
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



def _fixture(device, n=3000, cfg=None, seed=4):
    """A density + bank on n anchors in front of an identity camera."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = cfg or _small_config()
    dens = HACDensity(_spec({k: v for k, v in c.items() if k in DEFAULTS}))
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 2.0, 1.0])
    pts = (pts + torch.tensor([0, 0, 3.0])).to(device)
    params = torch.nn.ParameterDict(new_anchor_fields(
        pts, torch.zeros(n, 6, device=device), torch.zeros(n, c["feat_dim"], device=device),
        c["n_offsets"], BLOB_WIDTH))
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    dens.c["voxel_size"] = 0.01
    dens.initialize(params, opts, 1.0)
    renderer = HACRenderer(_spec({"feat_dim": c["feat_dim"], "n_offsets": c["n_offsets"]}))
    renderer.bank = dens.bank
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device),
           "width": 64, "height": 64}
    return dens, params, opts, renderer, cam


def _check(params, opts, extra=None):
    n = len(params["means"])
    for k, p in params.items():
        if p.shape[0] != n:
            raise AssertionError(f"{k} misaligned")
        owned = opts[k].param_groups[0]["params"][0]
        if owned is not p:
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
        params["hac_scaling"].data[:] = math.log(0.05)
        params["hac_offset"].data[:] = 0
        # all offsets of anchor 0 point 0.5 world units away (each passes the 50% random gate
        # of level 0 independently; the official loop skips deeper levels if level 0 adds none)
        params["hac_offset"].data[0, :] = torch.tensor([10.0, 0, 0])
    # allocate Adam moments
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
    with torch.no_grad():  # offset world positions after the Adam step above
        targets = anchor0 + params["hac_offset"][0] * params["hac_scaling"][0, :3].exp()
    means_before = params["means"].detach().clone()
    dens.adjust_anchor(params, opts, 2000)
    _check(params, opts)
    n1 = len(params["means"])
    kept_old = torch.cat([means_before[:5], means_before[6:]])
    if not torch.equal(params["means"][: n0 - 1].detach(), kept_old):
        raise AssertionError("pruning did not remove exactly the low-opacity anchor")
    new = params["means"][n0 - 1 :].detach()
    if len(new) == 0:
        raise AssertionError("no anchor grown at the high-gradient offset")
    c = dens.c
    sizes = [c["voxel_size"] * (c["update_init_factor"] // c["update_hierachy_factor"] ** i)
             for i in range(c["update_depth"])]
    for a in new:
        if not any(torch.allclose(a, torch.round(t / s) * s, atol=1e-5)
                   for s in sizes for t in targets):
            raise AssertionError("grown anchor is not at the voxel of the offset")
    if len(dens.offset_denom) != n1 * k or len(dens.opacity_accum) != n1:
        raise AssertionError("density statistics misaligned with the population")
    return {"anchors_before": n0, "anchors_after": n1, "grown": n1 - (n0 - 1)}


def training_phases(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=3000)
    bank = dens.bank
    with torch.no_grad():
        params["hac_feat"].data.normal_()
        params["hac_offset"].data.normal_(0, 0.5)
        params["hac_scaling"].data[:] = math.log(0.05)
    out = {}

    def twice(step):
        bank.step = step
        torch.manual_seed(0)
        a = renderer.render(params, cam)
        torch.manual_seed(1)
        b = renderer.render(params, cam)
        return float((a["rgb"] - b["rgb"]).abs().max()), a["info"]["hac"]["bits"]

    # exact boundaries: <=3000 deterministic, 3001..10000 noisy w/o rate, >10000 rate
    for step, noisy, rate in ((3000, False, False), (3001, True, False),
                              (10000, True, False), (10001, True, True)):
        if step == 10000:
            with torch.no_grad():
                params["means"].data[0, 0] += 7.0  # bounds must be recomputed at 10000
        diff, bits = twice(step)
        if (diff > 0) != noisy or (bits is not None) != rate:
            raise AssertionError(f"phase wrong at step {step}: diff {diff}, bits {bits is not None}")
        if step == 10000 and float(bank.x_bound_max[0, 0]) < float(params["means"][0, 0]):
            raise AssertionError("anchor bounds not updated at step 10000")
        out[f"diff_{step}"] = diff
    bank.step = 20000
    torch.manual_seed(2)
    res = renderer.render(params, cam)
    bits = res["info"]["hac"]["bits"]
    frac = bits["sampled"] / max(bits["eligible"], 1)
    if not 0.035 < frac < 0.065:
        raise AssertionError(f"rate sample fraction {frac} is not ~5%")
    out["sample_fraction"] = frac
    from .losses import create_rate

    lmbda, lm = 0.004, 5e-4
    rate = create_rate(_spec({"lmbda": lmbda, "mask_weight": lm, "feat_dim": 50,
                              "n_offsets": 10}))
    fixed = {"bit_per_param": torch.tensor(2.5, device=device, requires_grad=True),
             "hash_bits": torch.tensor(1.0e5, device=device)}
    r = rate(params, {"info": {"hac": {"bits": fixed}}}, None)
    denom = len(params["means"]) * (50 + 6 + 30)
    want = lmbda * (2.5 + 1.0e5 / denom) + lm * torch.sigmoid(params["hac_mask"]).mean()
    if abs(float(r) - float(want)) > 1e-7:
        raise AssertionError("rate loss value differs from lambda_e(b + hash/denom) + lambda_m m")
    r = rate(params, res, None)
    r.backward()
    if params["hac_feat"].grad is None or bank.mlp_grid[0].weight.grad is None:
        raise AssertionError("rate loss does not reach anchor features and the context MLP")
    lr = dens.c["lr"]
    f = model.expon_lr(lr["mlp_color"][0], lr["mlp_color"][1], 30000)
    ref = math.exp(math.log(0.008) * 0.5 + math.log(0.00005) * 0.5)
    if abs(f(15000) - ref) > 1e-12:
        raise AssertionError("expon lr schedule wrong")
    dens.update_learning_rate(params, opts, 15000)
    got = {g["name"]: g["lr"] for g in bank.optimizer.param_groups}
    if abs(got["mlp_color"] - ref) > 1e-12 or opts["means"].param_groups[0]["lr"] != 0:
        raise AssertionError("bank / anchor learning rates not scheduled")
    out["rate"] = float(r)
    return out


def bake_eval(method, device):
    _need_cuda(device)
    dens, params, opts, renderer, cam = _fixture(device, n=3000)
    g = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        params["hac_feat"].data.copy_(torch.randn(params["hac_feat"].shape, generator=g) * 3)
        params["hac_offset"].data.copy_(torch.randn(params["hac_offset"].shape, generator=g))
        params["hac_scaling"].data.copy_(torch.randn(params["hac_scaling"].shape, generator=g)
                                         * 0.3 - 3)
        params["hac_mask"].data.copy_(torch.randn(params["hac_mask"].shape, generator=g) * 4)
        dens.bank.update_anchor_bound(params["means"])
    bank_ref = copy.deepcopy(dens.bank.mlps)
    model.ACTIVE["bank"] = dens.bank
    try:
        dens.bake(params, opts)
        with torch.no_grad():
            baked = renderer.render(params, cam)["rgb"]
        enc = dens.state["hac"]["encoded"]
        buf = io.BytesIO()
        torch.save({"params": dict(params), "density_state": dens.state}, buf)
    finally:
        model.ACTIVE["bank"] = None
    buf.seek(0)
    ck = torch.load(buf, map_location=device, weights_only=True)
    fresh = HACRenderer(_spec({"feat_dim": 50, "n_offsets": 10}))
    with torch.no_grad():
        again = fresh.render(ck["params"], cam)["rgb"]
    if not torch.equal(baked, again):
        raise AssertionError("fresh-process render of the checkpoint differs from the bake")
    # decoded MLP weights equal the trained ones
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
        HACRenderer(_spec({"feat_dim": 50, "n_offsets": 10})).render(empty, cam)
    except ValueError:
        pass
    else:
        raise AssertionError("renderer must refuse an un-baked state")
    return {"anchors_coded": enc["anchors_coded"], "size_mb": enc["size_mb"]["total"],
            "roundtrip_max_error": max(enc["roundtrip_max_error"].values())}


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
    zonly = (anchors[:, 2] > 0.2).tolist()
    if zonly == want:
        raise AssertionError("constructed case does not distinguish the z-only filter")
    # the renderer uses this filter for rendering and for the density statistics
    dens, params, opts, renderer, cam2 = _fixture(device, n=400)
    with torch.no_grad():
        params["means"].data[:200, 0] += 50.0  # push half the anchors far off-screen
        dens.bank.update_anchor_bound(params["means"])
    out = renderer.render(params, cam2)
    vis = out["info"]["hac"]["visible"]
    ref = visible_anchors(dens.bank.quantize_anchor(params["means"])[0].detach(),
                          params["hac_scaling"][:, :3].exp().detach(), cam2)
    if not torch.equal(vis, ref) or bool(vis[:200].any()):
        raise AssertionError("renderer visibility differs from visible_filter")
    return {"case": got, "renderer_visible": int(vis.sum())}
