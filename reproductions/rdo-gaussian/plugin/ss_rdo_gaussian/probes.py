"""Mechanism probes for the RDO-Gaussian port. Each raises if its mechanism is removed or wrong."""

import io
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from splatstudio.verification import ProbeUnavailable

from . import bank as bank_mod
from . import density, losses, rans
from .bank import KEYS, SH_KEYS, RDOBank
from .renderer import GS_MASK, SH_MASK, RDORenderer, rasterize, rdo_attributes

SMALL_CB = {k: 32 for k in KEYS}


def _cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA device required")
    return torch.device(device)


def _component(method, name):
    return next(c for c in method.components if c.name == name)


def _cfg(method, **overrides):
    cfg = density.merged(density.DEFAULTS, _component(method, "rdo-density").config)
    cfg["codebook_size"] = dict(SMALL_CB)
    cfg.update(overrides)
    return cfg


def _camera(dev, w=64, h=48, f=60.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=dev)
    return {"viewmat": torch.eye(4, device=dev), "K": K, "width": w, "height": h}


def _params(n, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) - 0.5
    means[:, 2] = means[:, 2] + 3.0
    p = {
        "means": means,
        "scales": torch.rand(n, 3, generator=g) * 0.8 - 3.5,
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g) + 1.0,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
        GS_MASK: torch.randn(n, generator=g) * 2.0,
        SH_MASK: torch.randn(n, 3, generator=g) * 2.5,
    }
    return {k: torch.nn.Parameter(v.to(dev)) for k, v in p.items()}


def _bank(cfg, dev, gs=False, sh=False, vq=False, seed=0):
    torch.manual_seed(seed)
    b = RDOBank(cfg).to(dev)
    b.build_optimizer()
    b.gs_active, b.sh_active, b.vq_active = gs, sh, vq
    return b


def _renderer(b):
    r = RDORenderer(replace(_component_spec_stub(), config={}))
    r.bank = b
    return r


def _component_spec_stub():
    from splatstudio.specs import ComponentSpec

    return ComponentSpec("rdo-renderer", "renderer", "ss_rdo_gaussian.renderer:create_renderer")


# ------------------------------------------------------------------------------------ masks
def gs_mask_ste(method, device):
    dev = _cuda(device)
    cfg = _cfg(method)
    thr = cfg["gs_threshold"]
    p = _params(60, dev, 1)
    with torch.no_grad():
        p[GS_MASK][::2] = -4.0  # sigmoid 0.018 <= 0.1 -> pruned
        p[GS_MASK][1::2] = 1.5
    b = _bank(cfg, dev, gs=True)
    r = _renderer(b)
    cam = _camera(dev)
    out = r.render(p, cam, 3)
    keep = torch.sigmoid(p[GS_MASK].detach()) > thr
    with torch.no_grad():
        ref = rasterize(p["means"][keep], p["quats"][keep], p["scales"][keep].exp(),
                        torch.sigmoid(p["opacities"][keep]), p["sh0"][keep], p["shN"][keep],
                        cam, 3)["rgb"]
        full = rasterize(p["means"], p["quats"], p["scales"].exp(), torch.sigmoid(p["opacities"]),
                         p["sh0"], p["shN"], cam, 3)["rgb"]
    err = float((out["rgb"].detach() - ref).abs().max())
    if err > 1e-5:
        raise AssertionError(f"masked Gaussians still contribute: {err}")
    if float((full - ref).abs().max()) < 1e-3:
        raise AssertionError("fixture: pruned Gaussians are invisible anyway")
    w = torch.randn_like(out["rgb"])
    (out["rgb"] * w).sum().backward()
    grad = p[GS_MASK].grad.clone()
    # independent chain: dL/dphi = (dL/ds_hat . s + dL/dalpha_hat alpha) sigmoid'(phi)
    s = p["scales"].detach().exp()
    a = torch.sigmoid(p["opacities"].detach())
    hard = keep.float()
    s_hat = (s * hard[:, None]).requires_grad_(True)
    a_hat = (a * hard).requires_grad_(True)
    o2 = rasterize(p["means"].detach(), p["quats"].detach(), s_hat, a_hat, p["sh0"].detach(),
                   p["shN"].detach(), cam, 3)
    (o2["rgb"] * w).sum().backward()
    sig = torch.sigmoid(p[GS_MASK].detach())
    expect = ((s_hat.grad * s).sum(-1) + a_hat.grad * a) * sig * (1 - sig)
    gerr = float((grad - expect).abs().max())
    scale = float(expect.abs().max())
    if scale == 0 or gerr > 1e-4 * max(1.0, scale):
        raise AssertionError(f"STE mask gradient mismatch {gerr} (scale {scale})")
    lerr = abs(float(out["rdo"]["gs_prune"]) - float(sig.mean()))
    if lerr > 1e-6:
        raise AssertionError(f"L_GS != mean sigmoid: {lerr}")
    return {"render_err": err, "grad_err": gerr, "grad_scale": scale, "loss_err": lerr,
            "pruned": int((~keep).sum())}


def sh_mask_ste(method, device):
    dev = _cuda(device)
    cfg = _cfg(method)
    p = _params(80, dev, 2)
    b = _bank(cfg, dev, gs=True, sh=True)
    _, _, _, _, shn, rdo = rdo_attributes(b, p, True)
    sig = torch.sigmoid(p[SH_MASK].detach())
    degree = [1] * 3 + [2] * 5 + [3] * 7  # degree of shN coefficient j, written independently
    expect = p["shN"].detach().clone()
    for i in range(len(expect)):
        for j in range(15):
            if not sig[i, degree[j] - 1] > cfg["sh_threshold"]:
                expect[i, j] = 0.0
    err = float((shn.detach() - expect).abs().max())
    if err > 0:
        raise AssertionError(f"SH mask applied to the wrong coefficients: {err}")
    kept = torch.sigmoid(p[GS_MASK].detach()) > cfg["gs_threshold"]
    weights = torch.tensor([(2 * deg + 1) / ((3 + 1) ** 2 - 1) for deg in (1, 2, 3)], device=dev)
    ref = float((sig[kept] * weights).sum(-1).mean())  # Eq. 8 over GS-kept Gaussians
    lerr = abs(float(rdo["sh_prune"]) - ref)
    if lerr > 1e-6:
        raise AssertionError(f"L_SH mismatch {lerr}")
    rdo["sh_prune"].backward()
    g = p[SH_MASK].grad
    eg = torch.zeros_like(g)
    eg[kept] = weights * sig[kept] * (1 - sig[kept]) / int(kept.sum())
    gerr = float((g - eg).abs().max())
    if gerr > 1e-7:
        raise AssertionError(f"L_SH gradient mismatch {gerr}")
    if float(expect.abs().sum()) == float(p["shN"].detach().abs().sum()):
        raise AssertionError("fixture prunes no SH coefficient")
    return {"mask_err": err, "loss_err": lerr, "grad_err": gerr, "gs_kept": int(kept.sum())}


# -------------------------------------------------------------------------------------- ECVQ
def _nearest(x, cb, bits=None, lmbda=None):
    """Independent double-precision loop: argmin_m ||x - c_m||_2 (+ bits_m / lambda)."""
    xs = x.detach().double().cpu().numpy()
    cs = cb.detach().double().cpu().numpy()
    bs = None if bits is None else bits.detach().double().cpu().numpy()
    out = []
    for row in xs:
        best, arg = None, -1
        for m in range(len(cs)):
            v = math.sqrt(float(((row - cs[m]) ** 2).sum()))
            if bs is not None:
                v += float(bs[m]) / lmbda
            if best is None or v < best:
                best, arg = v, m
        out.append(arg)
    return torch.tensor(out, device=x.device)


def ecvq(method, device):
    dev = _cuda(device)
    cfg = _cfg(method)
    cfg["lmbda"] = {k: 0.5 for k in KEYS}  # small lambda so the rate term changes choices
    n = 40
    p = _params(n, dev, 3)
    b = _bank(cfg, dev, vq=True, seed=3)
    with torch.no_grad():
        for k in KEYS:
            b.logits[k].normal_(0, 2.0)
            b.codebooks[k].mul_(0.5)
    results = {}
    vec = {"scale": p["scales"].detach().exp(), "rot": p["quats"].detach(),
           "dc": p["sh0"].detach().reshape(-1, 3)}
    for k in SH_KEYS:
        a, c = {"sh1": (0, 3), "sh2": (3, 8), "sh3": (8, 15)}[k]
        vec[k] = p["shN"].detach()[:, a:c].reshape(n, -1)
    changed = 0
    for rate in (False, True):
        b.rate_constrain = rate
        b.render_step = 1
        scales, _, quats, sh0, shn, rdo = rdo_attributes(b, p, True)
        rate_ref, vq_ref = 0.0, 0.0
        for k in KEYS:
            cb = b.codebooks[k].detach()
            bits = -torch.log2(torch.softmax(b.logits[k].detach(), -1)).clamp_max(
                -math.log2(1e-9))
            idx = _nearest(vec[k], cb, bits if rate else None, cfg["lmbda"][k])
            if not torch.equal(idx, b.index_cache[k]):
                raise AssertionError(f"{k}: ECVQ assignment differs (rate={rate})")
            rate_ref += float(bits[idx].sum()) / cfg["lmbda"][k]
            vq_ref += float(torch.linalg.vector_norm(cb[idx] - vec[k], dim=-1).sum())
            if rate:
                changed += int((idx != _nearest(vec[k], cb)).sum())
        for name, ref, val in (("rate", rate_ref / n / 55, rdo["rate"]),
                               ("vq", vq_ref / n / 55, rdo["vq"])):
            if abs(float(val) - ref) > 1e-5 * max(1.0, abs(ref)):
                raise AssertionError(f"{name} loss {float(val)} != {ref} (rate={rate})")
        results[f"rate_loss_rate{int(rate)}"] = float(rdo["rate"])
    if changed == 0:
        raise AssertionError("rate term never changed an assignment")
    results["assignments_changed_by_rate"] = changed
    if not torch.allclose(scales.detach(), b.codebooks["scale"].detach()[b.index_cache["scale"]]):
        raise AssertionError("rendered scale is not the codeword")
    # gradients: render -> x only (STE); L_VQ -> codebook; L_rate -> logits only
    b.render_step = 1
    r = _renderer(b)
    out = r.render(p, _camera(dev), 3)
    out["rgb"].sum().backward(retain_graph=True)
    cb_g = b.codebooks["dc"].grad
    if cb_g is not None and float(cb_g.abs().sum()) != 0:
        raise AssertionError("render gradient reaches the codebook (STE broken)")
    if p["sh0"].grad is None or float(p["sh0"].grad.abs().sum()) == 0:
        raise AssertionError("render gradient does not reach the Gaussian attribute")
    b.optimizer.zero_grad(set_to_none=True)
    out["rdo"]["vq"].backward(retain_graph=True)
    if b.codebooks["dc"].grad is None or b.logits["dc"].grad is not None:
        raise AssertionError("L_VQ must train codebooks only")
    b.optimizer.zero_grad(set_to_none=True)
    out["rdo"]["rate"].backward()
    if b.logits["dc"].grad is None or b.codebooks["dc"].grad is not None:
        raise AssertionError("L_rate must train logits only")
    # index cache: non-update steps reuse the cached indices
    cached = {k: v.clone() for k, v in b.index_cache.items()}
    with torch.no_grad():
        for k in KEYS:
            b.codebooks[k].copy_(torch.flip(b.codebooks[k], [0]))
    b.render_step = 2
    rdo_attributes(b, p, True)
    if not all(torch.equal(cached[k], b.index_cache[k]) for k in KEYS):
        raise AssertionError("indices recomputed on a cached step")
    b.render_step = 11
    rdo_attributes(b, p, True)
    if all(torch.equal(cached[k], b.index_cache[k]) for k in KEYS):
        raise AssertionError("indices not recomputed on an update step ((step-1) % 10 == 0)")
    return results


def kmeans_reactivate(method, device):
    dev = _cuda(device)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(600, 3, generator=g).to(dev)
    torch.manual_seed(11)
    means, bins = bank_mod.kmeans(x, 16, 10, 128)
    torch.manual_seed(11)
    init = torch.randperm(600, device=dev)[:16]
    m = x[init].clone()
    for _ in range(10):
        a = ((x[:, None, :] - m[None]) ** 2).sum(-1).argmin(-1)
        cnt = torch.stack([(a == j).sum() for j in range(16)])
        m = torch.stack([x[a == j].mean(0) if cnt[j] > 0 else m[j] for j in range(16)])
    if not torch.allclose(means, m, atol=1e-5) or not torch.equal(bins, cnt):
        raise AssertionError("k-means differs from the independent Lloyd loop")
    cfg = _cfg(method)
    cfg["codebook_size"] = {k: 16 for k in KEYS}
    b = _bank(cfg, dev)
    torch.manual_seed(11)
    b.kmeans_initialize("scale", x)
    ref_logits = torch.log(cnt.float() / torch.linalg.vector_norm(cnt.float()))
    if not torch.allclose(b.logits["scale"].detach(), ref_logits):
        raise AssertionError("logits != log(bins / ||bins||_2)")
    with torch.no_grad():
        b.logits["rot"].copy_(torch.linspace(0, 1, 16, device=dev))
        b.logits["rot"][:5] = -40.0  # p ~ 1e-18: dead
    before_cb = b.codebooks["rot"].detach().clone()
    before_lg = b.logits["rot"].detach().clone()
    n = b.reactivate("rot", cfg["reactivate_prob"])
    lg, cb = b.logits["rot"].detach(), b.codebooks["rot"].detach()
    if n != 5 or not torch.equal(lg[5:], before_lg[5:]) or not torch.equal(cb[5:], before_cb[5:]):
        raise AssertionError("reactivation touched live codewords or missed dead ones")
    for i in range(5):
        j = int((before_lg[5:] == lg[i]).nonzero()[0]) + 5
        if float((cb[i] - before_cb[j]).abs().max()) > 1e-3:
            raise AssertionError("revived codeword is not a perturbed live codeword")
    return {"reactivated": n, "kmeans_bins": cnt.tolist()}


# ----------------------------------------------------------------------------------- codec
def _quantized_cdf_reference(pmf, precision):
    """Independent Python port of CompressAI pmf_to_quantized_cdf."""
    cdf = [0] + [int(round(p * (1 << precision))) for p in pmf]
    total = sum(cdf)
    cdf = [((1 << precision) * c) // total for c in cdf]
    for i in range(1, len(cdf)):
        cdf[i] += cdf[i - 1]
    cdf[-1] = 1 << precision
    for i in range(len(cdf) - 1):
        if cdf[i] == cdf[i + 1]:
            best_freq, best = None, -1
            for j in range(len(cdf) - 1):
                f = cdf[j + 1] - cdf[j]
                if f > 1 and (best_freq is None or f < best_freq):
                    best_freq, best = f, j
            if best < i:
                for j in range(best + 1, i + 1):
                    cdf[j] -= 1
            else:
                for j in range(i + 1, best + 1):
                    cdf[j] += 1
    return cdf


def rans_codec(method, device):
    g = torch.Generator().manual_seed(7)
    worst = 0
    for trial in range(4):
        logits = torch.randn(1, 50 + 40 * trial, generator=g) * (2 + 3 * trial)
        cdf, length = rans.quantized_cdf(logits)
        pmf = torch.softmax(logits, -1)[0]
        p = torch.cat([pmf, (1 - pmf.sum(0, keepdim=True)).clamp_min(0)]).tolist()
        p32 = np.asarray(p, dtype=np.float32).tolist()  # the C++ takes float
        ref = _quantized_cdf_reference(p32, 16)
        if cdf != ref or length != logits.shape[1] + 2:
            raise AssertionError(f"quantized CDF differs from the reference (trial {trial})")
        worst = max(worst, sum(1 for a, b in zip(ref, ref[1:]) if b - a == 1))
    logits = torch.randn(1, 300, generator=g) * 2
    probs = torch.softmax(logits, -1)[0]
    sym = torch.multinomial(probs, 50000, replacement=True, generator=g)
    s = rans.encode(sym, logits)
    back = rans.decode(s, len(sym), logits)
    if not torch.equal(back, sym):
        raise AssertionError("rANS round trip is lossy")
    cdf, _ = rans.quantized_cdf(logits)
    freq = np.diff(np.asarray(cdf, dtype=np.float64))
    ideal = float(-np.log2(freq[sym.numpy()] / 65536).sum() / 8)
    if not len(s) <= ideal * 1.01 + 16:
        raise AssertionError(f"rANS bytes {len(s)} far above ideal {ideal:.0f}")
    return {"bytes": len(s), "ideal_bytes": ideal, "min_freq_bins": worst}


def _density_fixture(method, dev, n=400, seed=0):
    spec = _component(method, "rdo-density")
    spec = replace(spec, config={**spec.config, "codebook_size": dict(SMALL_CB)})
    from splatstudio.state import initialize

    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) - 0.5
    pts[:, 2] += 3.0
    params, optimizers = initialize(method, pts.to(dev), torch.rand(n, 3, generator=g).to(dev))
    with torch.no_grad():
        src = _params(n, dev, seed + 1)
        for k in params:
            params[k].copy_(src[k])
    d = density.RDODensity(spec)
    d.initialize(params, optimizers, 1.0)
    return d, params, optimizers


def encode_decode(method, device):
    dev = _cuda(device)
    saved = bank_mod.ACTIVE["bank"]
    try:
        d, params, optimizers = _density_fixture(method, dev)
        cfg = d.cfg
        b = d.bank
        torch.manual_seed(0)
        d._kmeans(params)
        with torch.no_grad():
            b.codebooks["scale"][:3] = -0.01  # remove_negatives must clip these
            for k in KEYS:
                b.logits[k].normal_(0, 1)
        b.gs_active = b.sh_active = b.vq_active = b.rate_constrain = True
        orig = {k: v.detach().clone() for k, v in params.items()}
        keep = torch.sigmoid(orig[GS_MASK]) > cfg["gs_threshold"]
        bits3 = torch.sigmoid(orig[SH_MASK][keep]) > cfg["sh_threshold"]
        d._encode(params, optimizers)
        enc = d.encoded
        n = int(keep.sum())
        if len(params["means"]) != n or enc["gaussians"] != n:
            raise AssertionError("masked Gaussians were not removed")
        # independent expected sort / boundaries
        value = bits3[:, 0].long() * 4 + bits3[:, 1].long() * 2 + bits3[:, 2].long()
        bnd = [int((value <= j).sum()) for j in range(7)] + [n]
        if enc["boundaries"] != bnd:
            raise AssertionError(f"boundaries {enc['boundaries']} != {bnd}")
        order = torch.sort(value, stable=True).indices
        cb_scale = b.codebooks["scale"].detach().clamp_min(1e-8)  # negatives -> 1e-8
        if float(b.codebooks["scale"].detach().min()) < 0:
            raise AssertionError("negative scale codewords survived")
        x = {"scale": orig["scales"][keep].exp(), "rot": orig["quats"][keep],
             "dc": orig["sh0"][keep].reshape(-1, 3)}
        exp_params = {}
        for k in ("scale", "rot", "dc"):
            cb = cb_scale if k == "scale" else b.codebooks[k].detach()
            bits = -torch.log2(torch.softmax(b.logits[k].detach(), -1))
            idx = _nearest(x[k], cb, bits, cfg["lmbda"][k])
            exp_params[k] = cb[idx][order]
        if not torch.allclose(params["scales"].detach(), exp_params["scale"].log(), atol=1e-6):
            raise AssertionError("decoded scales != log(rate-constrained scale codewords)")
        if not torch.allclose(params["quats"].detach(), exp_params["rot"], atol=1e-6):
            raise AssertionError("decoded rotations != raw rotation codewords")
        if not torch.allclose(params["sh0"].detach().reshape(-1, 3), exp_params["dc"],
                              atol=1e-6):
            raise AssertionError("decoded DC != DC codewords")
        shn = params["shN"].detach()
        bs = bits3[order]
        for i, (a, c) in enumerate(((0, 3), (3, 8), (8, 15))):
            if float(shn[~bs[:, i], a:c].abs().sum()) != 0:
                raise AssertionError("pruned SH degree not zero after decode")
            if bool(bs[:, i].any()) and float(shn[bs[:, i], a:c].abs().sum()) == 0:
                raise AssertionError("kept SH degree lost")
        xyz_err = float((params["means"].detach() - orig["means"][keep][order]).abs().max())
        if xyz_err > 2e-3:
            raise AssertionError(f"decoded positions differ beyond fp16: {xyz_err}")
        o = orig["opacities"][keep][order]
        step = float((o.max() - o.min()) / 255)
        op_err = float((params["opacities"].detach() - o).abs().max())
        if op_err > step / 2 + 1e-5:
            raise AssertionError(f"opacity quantisation error {op_err} > step/2 {step / 2}")
        # sizes: the official files written by np.savez on disk
        files = {k: bytes(v.numpy().tobytes()) for k, v in d.stream.items()}
        with tempfile.TemporaryDirectory() as tmp:
            sizes = {}
            for name in ("codebook.npz", "logits.npz", "position.npz"):
                arrays = dict(np.load(io.BytesIO(files[name])))
                np.savez(Path(tmp) / name, **arrays)
                sizes[name] = (Path(tmp) / name).stat().st_size
        for name, s in sizes.items():
            if s != len(files[name]):
                raise AssertionError(f"{name}: in-memory {len(files[name])} != on-disk {s}")
        if len(files["header.bin"]) != 40:
            raise AssertionError("header must be 8 uint32 + 2 float32")
        total = sum(len(v) for v in files.values())
        if enc["total_bytes"] != total or abs(enc["size_MB"] - total / 1e6) > 1e-12:
            raise AssertionError("reported size differs from the bitstream bytes")
        # fresh process view: no bank -> stored fields rendered as they are
        cam = _camera(dev)
        with torch.no_grad():
            baked = RDORenderer(_component_spec_stub()).render(params, cam, 3)["rgb"]
            bank_mod.ACTIVE["bank"] = None
            fresh = RDORenderer(_component_spec_stub()).render(params, cam, 3)["rgb"]
        if not torch.equal(baked, fresh):
            raise AssertionError("baked render differs from a fresh renderer")
        return {"gaussians": n, "size_bytes": total, "boundaries": bnd, "xyz_err": xyz_err,
                "opacity_err": op_err, "components_MB": enc["components_MB"]}
    finally:
        bank_mod.ACTIVE["bank"] = saved


def schedule(method, device):
    spec = _component(method, "rdo-density")
    spec = replace(spec, config={**spec.config, "codebook_size": {k: 4 for k in KEYS}})
    saved = bank_mod.ACTIVE["bank"]
    try:
        d = density.RDODensity(spec)
        cfg = d.cfg
        d.bank = RDOBank(cfg)
        d.bank.build_optimizer()
        d.base.after_backward = lambda *a, **k: None
        calls = {"kmeans": [], "encode": []}
        d._kmeans = lambda params: calls["kmeans"].append(len(calls["kmeans"])) or {}
        d._encode = lambda params, opt: calls["encode"].append(1) or {}
        lrs = {}
        step_box = {}
        real_step = d.bank.optimizer.step

        def record(*a, **k):
            lrs[step_box["s"]] = d.bank.optimizer.param_groups[0]["lr"]
            return real_step(*a, **k)

        d.bank.optimizer.step = record
        rate_from, active_from = None, None
        params = {"means": torch.zeros(1, 3)}
        total = cfg["total_steps"]
        for s in range(1, total + 1):
            step_box["s"] = s
            if active_from is None and d.bank.vq_active:
                active_from = s
            if rate_from is None and d.bank.rate_constrain:
                rate_from = s
            d.after_backward(params, {"means": d.bank.optimizer}, s, {})
            if d.bank.render_step != s + 1:
                raise AssertionError("render_step not advanced")
        events = {}
        for e in d.history:
            events.setdefault(e["rdo"], []).append(e["step"])
        expect = {"vq_on": [15000], "gs_on": [15000], "sh_on": [15000], "rate_on": [20000],
                  "reactivate": list(range(16000, 20001, 1000)), "encode": [total]}
        if events != expect:
            raise AssertionError(f"events {events} != {expect}")
        if active_from != 15001 or rate_from != 20001:
            raise AssertionError(f"VQ active from {active_from}, rate from {rate_from}")
        steps = sorted(lrs)
        if steps != list(range(15001, total)):
            raise AssertionError("bank Adam must step on 15001..29999 only")
        base = cfg["codebook_lr"]
        bad = [s for s in steps if not math.isclose(lrs[s], base * (0.1 if s >= 27001 else 1.0))]
        if bad:
            raise AssertionError(f"bank LR schedule wrong at {bad[:3]}")
        if len(calls["kmeans"]) != 1 or len(calls["encode"]) != 1:
            raise AssertionError("k-means / encode must run exactly once")
        return {"events": {k: v[:3] for k, v in events.items()}, "bank_steps": len(steps)}
    finally:
        bank_mod.ACTIVE["bank"] = saved


def rd_losses(method, device):
    dev = _cuda(device)
    cfg = _cfg(method)
    p = _params(30, dev, 9)
    fns = {c.name: (losses_factory(c), float(c.config["weight"])) for c in method.components
           if c.name.startswith("rdo-") and c.kind == "loss"}
    b = _bank(cfg, dev)
    r = _renderer(b)
    out = r.render(p, _camera(dev), 3)
    if "rdo" in out:
        raise AssertionError("inactive bank produced RD terms")
    for name, (fn, _) in fns.items():
        v = fn(p, out, None)
        if float(v) != 0 or not v.requires_grad:
            raise AssertionError(f"{name}: must be an attached zero before activation")
    b.gs_active = b.sh_active = b.vq_active = True
    out = r.render(p, _camera(dev), 3)
    term = {"rdo-rate": "rate", "rdo-vq": "vq", "rdo-gs-prune": "gs_prune",
            "rdo-sh-prune": "sh_prune"}
    res = {}
    for name, (fn, w) in fns.items():
        v = float(fn(p, out, None))
        ref = w * float(out["rdo"][term[name]])
        if ref == 0 or abs(v - ref) > 1e-9:
            raise AssertionError(f"{name}: {v} != weight x term {ref}")
        res[name] = v
    return res


def losses_factory(component):
    fn = getattr(losses, component.factory.split(":")[1])
    return fn(component)

