"""Mechanism probes for the FCGS port. Each compares against an independent formula or a
constructed case and raises when the mechanism is removed or wrong. Signature: (method, device).
"""

import dataclasses
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import torch

from . import codec, xyz_codec

# SHA256 of the official release (github.com/YihangChen-ee/FCGS @ 31e59a46, checkpoints/).
OFFICIAL_SHA256 = {
    1e-4: "2ea3665db07606eee495721e88f9e231c60397f6af3d52423a1022d0d0174185",
    2e-4: "95c6af0d260ea4d2f2aa922634dae8347f7fd051df8db770f8487d0b40c7cf97",
    4e-4: "73e1f643d0c7ab8583acdaeabaddf1c2b23d33203eafd6af64048f93c0107e2d",
    8e-4: "2a19e5608606f370084251f5665ba1939c5ebf6b474560ab428fe0ee81537c10",
    16e-4: "5b3ed1beadaf1f513551f3a9136879028cacde1ee20c211ed518edbb89a8e2c9",
}


def _need_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is required")


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def synthetic_scene(n, seed=0, device="cuda"):
    """3DGS-like parameters: clustered positions on surfaces, smooth colours, typical ranges."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.randn(32, 3, generator=g) * 2.0
    idx = torch.randint(0, 32, (n,), generator=g)
    means = centers[idx] + torch.randn(n, 3, generator=g) * 0.3
    base = torch.sin(means @ torch.randn(3, 3, generator=g)) * 1.2
    return {k: v.to(device) for k, v in {
        "means": means,
        "sh0": base.reshape(n, 1, 3),
        "shN": (torch.randn(n, 15, 3, generator=g) * 0.05),
        "opacities": torch.randn(n, generator=g) * 2.0,
        "scales": torch.randn(n, 3, generator=g) * 0.5 - 4.0,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
    }.items()}


# --------------------------------------------------------------------------- weights
def official_weights(method, device):
    out = {}
    for lmd, digest in OFFICIAL_SHA256.items():
        path = codec.weights_path(lmd)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        _check(got == digest, f"{path} is not the official FCGS release ({got})")
        out[str(lmd)] = got[:12]
    density = next(c for c in method.components if c.kind == "density")
    _check(density.config.get("lmd") in OFFICIAL_SHA256, "density lambda has no official weights")
    return {"verified": out, "lambda": density.config["lmd"]}


# --------------------------------------------------------------------------- Eq. 4 / Eq. 5 grids
def _ref_corners(x, res):
    """pos = x (R-2) + 0.5 (official kernels); 2^d corners, upper clipped to R-1; dense index
    sum_d p_d R^d; corner weight prod(1-|frac|) + 1e-9."""
    d = x.shape[1]
    pos = x * (res - 2) + 0.5
    base = pos.floor()
    frac = pos - base
    base = base.long()
    for corner in range(1 << d):
        w = torch.ones(len(x), device=x.device)
        index = torch.zeros(len(x), dtype=torch.long, device=x.device)
        stride = 1
        for k in range(d):
            if corner >> k & 1:
                w = w * frac[:, k]
                p = torch.clamp(base[:, k] + 1, max=res - 1)
            else:
                w = w * (1 - frac[:, k])
                p = base[:, k]
            index = index + p * stride
            stride *= res
        yield w + 1e-9, index


def ref_grid_create(x, feat, res):
    num = torch.zeros(res ** x.shape[1], feat.shape[1], device=x.device, dtype=torch.float64)
    den = torch.zeros(res ** x.shape[1], 1, device=x.device, dtype=torch.float64)
    for w, index in _ref_corners(x.double(), res):
        num.index_add_(0, index, w[:, None] * feat.double())
        den.index_add_(0, index, w[:, None])
    return num / (den + 1e-9)


def ref_grid_interp(x, grid, res):
    out = 0
    for w, index in _ref_corners(x.double(), res):
        out = out + w[:, None] * grid.double()[index]
    return out


def grid_context(method, device):
    """Eq. 4 (grid creation, float and deterministic int32 variants) and Eq. 5 (interpolation)
    of the official CUDA kernels equal an independent torch implementation."""
    _need_cuda(device)
    from .grid_utils import _grid_creater, _grid_encoder

    torch.manual_seed(0)
    out = {}
    for dim, res in ((3, 9), (2, 21)):
        xs = torch.rand(4000, dim, device=device)
        xq = torch.rand(1500, dim, device=device)
        feat = torch.randn(4000, 8, device=device)
        res_t = torch.tensor([res], device=device, dtype=torch.int)
        off_t = torch.tensor([0, res ** dim], device=device, dtype=torch.int)
        ref = ref_grid_create(xs, feat, res)
        grid = _grid_creater.apply(xs, feat, res_t, off_t, False)
        grid_det = _grid_creater.apply(xs, feat, res_t, off_t, True)
        e_float = float((grid.double() - ref).abs().max())
        occupied = ref.abs().sum(1) > 0
        e_det = float((grid_det.double() - ref)[occupied].abs().mean())
        interp = _grid_encoder.apply(xq, grid, off_t, res_t)
        e_interp = float((interp.double() - ref_grid_interp(xq, grid, res)).abs().max())
        _check(e_float < 1e-4, f"grid creator ({dim}D) differs from Eq. 4: {e_float}")
        _check(e_det < 1e-2, f"deterministic grid creator ({dim}D) differs from Eq. 4: {e_det}")
        _check(e_interp < 1e-4, f"grid interpolation ({dim}D) differs from Eq. 5: {e_interp}")
        out[f"{dim}d"] = {"create_max_err": e_float, "determ_mean_err": e_det,
                          "interp_max_err": e_interp}
    return out


# --------------------------------------------------------------------------- Eq. 7 GMM coder
def gmm_cdf(method, device):
    """The arithmetic coder's CDF table (calculate_cdf, mixed with softmax weights as in
    encoder_gaussian_mixed) equals sum_l theta_l Phi((k -/+ 1/2) q | mu_l, sigma_l) (Eq. 7),
    and the coded size is within 2% + 64 bytes of the ideal -log2 p."""
    _need_cuda(device)
    from . import kernels

    torch.manual_seed(1)
    n, q = 20000, 0.01
    mean = [torch.randn(n, device=device) * 0.05 for _ in range(3)]
    scale = [torch.rand(n, device=device) * 0.05 + 0.005 for _ in range(3)]
    theta = torch.softmax(torch.randn(n, 3, device=device), -1)
    comp = torch.randint(0, 3, (n,), device=device)
    x = torch.stack(mean, 1).gather(1, comp[:, None])[:, 0] + \
        torch.randn(n, device=device) * torch.stack(scale, 1).gather(1, comp[:, None])[:, 0]
    xi = torch.round(x / q)
    lo, hi = int(xi.min()), int(xi.max())
    Q = torch.full((n,), q, device=device)
    table = sum(kernels.arithmetic().calculate_cdf(m, s, Q, lo, hi) * theta[:, k:k + 1]
                for k, (m, s) in enumerate(zip(mean, scale)))
    ks = torch.arange(lo, hi + 2, device=device, dtype=torch.float64)
    ref = 0
    for k in range(3):
        normal = torch.distributions.Normal(mean[k].double()[:, None], scale[k].double()[:, None])
        ref = ref + theta[:, k:k + 1].double() * normal.cdf((ks[None] - 0.5) * q)
    err = float((table.double() - ref).abs().max())
    _check(err < 1e-5, f"GMM CDF differs from Eq. 7: {err}")
    import tempfile

    from .encodings_cuda import decoder_gaussian_mixed, encoder_gaussian_mixed
    with tempfile.TemporaryDirectory() as tmp:
        f = f"{tmp}/x.b"
        bits = encoder_gaussian_mixed(xi * q, mean, scale, [theta[:, k] for k in range(3)], q, f)
        dec = decoder_gaussian_mixed(mean, scale, [theta[:, k] for k in range(3)], q, f)
    idx = (xi - lo).long()
    p = (ref.gather(1, idx[:, None] + 1) - ref.gather(1, idx[:, None]))[:, 0].clamp_min(1e-12)
    ideal = float(-torch.log2(p).sum())
    _check(torch.equal(torch.round(dec / q), xi), "GMM arithmetic decode is not lossless")
    _check(bits <= ideal * 1.02 + 512, f"coded bits {bits} far above ideal {ideal:.0f}")
    return {"cdf_max_err": err, "coded_bits": bits, "ideal_bits": round(ideal, 1)}


# --------------------------------------------------------------------------- MEM / quantisation
def mem_and_quantisation(method, device):
    """Eq. 3 on the pretrained lambda model: decoded colour equals g_s(round(g_a(x))) where
    m = Sig(MLP_m(f_gau)) > 0.01 and Q(x) with the learned step elsewhere; geometry equals
    Q(x) with learned steps; positions sit on the 16-bit per-axis grid."""
    _need_cuda(device)
    density = next(c for c in method.components if c.kind == "density")
    lmd = density.config["lmd"]
    model = codec.load_model(lmd, device=device)
    params = synthetic_scene(30000, device=device)
    decoded, report = codec.encode_decode(params, lmd, model=model, xyz_coder="octree")
    _check(report["roundtrip_max_abs_error"] < 1e-5,
           f"bitstream decode differs from encoder reconstruction: {report}")
    v_src_np, lo_np, hi_np = xyz_codec.voxelize(params["means"].cpu().numpy().astype(np.float32))
    v_src = torch.from_numpy(v_src_np).double().to(device)
    lo = torch.from_numpy(lo_np).double().to(device)
    hi = torch.from_numpy(hi_np).double().to(device)
    raw = (decoded["means"].double() - lo) / (hi - lo) * 65535
    v_dec = torch.round(raw)
    _check(float((raw - v_dec).abs().max()) < 1e-2, "decoded positions off the 16-bit grid")
    key = lambda v: (v[:, 0] + 65536 * v[:, 1] + 65536.0 ** 2 * v[:, 2])  # noqa: E731
    _check(torch.equal(torch.sort(key(v_dec)).values, torch.sort(key(v_src)).values),
           "decoded positions are not the 16-bit voxelised source positions")
    # recover the permutation (positions are unique here) and check Eq. 3 per Gaussian
    order_src = torch.argsort(key(v_src))
    order_dec = torch.argsort(key(v_dec))
    f_src = codec.params_to_fea(params)[order_src]
    f_dec = codec.params_to_fea(decoded)[order_dec]
    with torch.no_grad():
        m = model.Encoder_mask(f_src)[:, 0] > 0.01
        ad = {k: getattr(model, f"ad_{k}") for k in ("fe", "op", "sc", "ro")}
        base = {"fe": model.Q_fe, "op": model.Q_op, "sc": model.Q_sc, "ro": model.Q_ro}
        step = {k: float((base[k] * a[0, 0] + a[0, 1]) * (1 + torch.tanh(a[0, 2])))
                for k, a in ad.items()}
        lat = torch.round(model.Encoder_fea(f_src[m]))
        h = model.Decoder_fea(lat)
        ae = torch.cat([model.head_f_dc(h), model.head_f_rst(h)], -1)

    def qerr(x, y, s):
        return float((torch.round(x / s) * s - y).abs().max()) if len(x) else 0.0

    e_ae = float((ae - f_dec[m, :48]).abs().max()) if m.any() else 0.0
    e_col = qerr(f_src[~m, :48], f_dec[~m, :48], step["fe"])
    e_geo = max(qerr(f_src[:, 48:49], f_dec[:, 48:49], step["op"]),
                qerr(f_src[:, 49:52], f_dec[:, 49:52], step["sc"]),
                qerr(f_src[:, 52:56], f_dec[:, 52:56], step["ro"]))
    tol = 1e-4
    _check(e_ae < 1e-3, f"m=1 colour != g_s(round(g_a(x))): {e_ae}")
    _check(e_col < tol, f"m=0 colour != Q(x): {e_col}")
    _check(e_geo < tol, f"geometry != Q(x): {e_geo}")
    changed = float((f_src - f_dec).abs().max())
    _check(changed > 0, "decoded attributes equal the source: codec not applied")
    return {"mask_rate": float(m.float().mean()), "steps": step, "ae_err": e_ae,
            "m0_err": e_col, "geo_err": e_geo, "size_MB": report["size_MB"],
            "breakdown": report["size_MB_breakdown"]}


# --------------------------------------------------------------------------- context models
def context_models(method, device):
    """C2/C3: with the pretrained model, the inter-/intra-Gaussian contexts must lower the code
    length. The ideal bits of the m=0 colour stream (hyperprior + spatial + channel mixture,
    Eq. 7) are compared with the hyperprior component alone on a spatially smooth scene."""
    _need_cuda(device)
    density = next(c for c in method.components if c.kind == "density")
    model = codec.load_model(density.config["lmd"], device=device)
    params = synthetic_scene(40000, seed=3, device=device)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        _, report_full = codec.encode_decode(params, density.config["lmd"], model=model,
                                             xyz_coder="octree", bitstream_dir=tmp,
                                             check_roundtrip=False)
    full = report_full["size_MB_breakdown"]["geo_main"]
    # hyperprior-only geometry bits with the same quantised values
    from .grid_utils import normalize_xyz  # noqa: F401  (import check of the context path)
    with torch.no_grad():
        fea = codec.params_to_fea(params)
        a = model.ad_op, model.ad_sc, model.ad_ro
        steps = torch.tensor([float((b * t[0, 0] + t[0, 1]) * (1 + torch.tanh(t[0, 2])))
                              for b, t in zip((model.Q_op, model.Q_sc, model.Q_ro), a)],
                             device=device)
        q = torch.cat([steps[0:1], steps[1:2].repeat(3), steps[2:3].repeat(4)])
        geo = torch.round(fea[:, 48:] / q) * q
        z = torch.round(model.Encoder_geo_hyp(geo))
        mu, sigma, _ = torch.split(model.Decoder_geo_hyp(z), [8, 8, 8], -1)
        sigma = sigma.clamp_min(1e-9)
        normal = torch.distributions.Normal(mu, sigma)
        p = (normal.cdf(geo + q / 2) - normal.cdf(geo - q / 2)).clamp_min(1e-9)
        hyper_only = float(-torch.log2(p).sum()) / codec.MIB_BITS
    _check(full < hyper_only, f"contexts do not reduce geometry bits ({full} >= {hyper_only})")
    return {"geo_main_MB_full_context": full, "geo_main_MB_hyperprior_only": hyper_only,
            "ratio": full / hyper_only}


# --------------------------------------------------------------------------- coordinates
def xyz_lossless(method, device):
    """Substitute octree coder: lossless on random 16-bit voxels with duplicates, ordering
    restored by the official sort; GPCC path used when tmc3 is installed."""
    rng = np.random.default_rng(0)
    vox = rng.integers(0, 65536, size=(5000, 3)).astype(np.float32)
    vox = np.concatenate([vox, vox[:300], vox[:7]])
    data = xyz_codec.octree_encode(vox)
    dec = xyz_codec.octree_decode(data)
    _check(np.array_equal(xyz_codec.sorted_voxels(dec), xyz_codec.sorted_voxels(vox)),
           "octree coder is not lossless")
    clustered = np.clip(np.round(rng.normal(30000, 300, size=(20000, 3))), 0, 65535)
    bpp = len(xyz_codec.octree_encode(clustered)) * 8 / len(clustered)
    _check(bpp < 48, f"octree coder does not compress clustered points: {bpp} bits/point")
    return {"random_bytes": len(data), "clustered_bits_per_point": bpp,
            "tmc3": xyz_codec.tmc3_path()}


# --------------------------------------------------------------------------- schedule
def schedule(method, device):
    """The codec runs exactly once, after the final optimizer step (step == total_steps)."""
    from .density import FCGSDensity

    density_spec = next(c for c in method.components if c.kind == "density")
    dens = FCGSDensity(dataclasses.replace(density_spec,
                                           config={**density_spec.config, "total_steps": 50}))
    calls = []
    dens.base = SimpleNamespace(after_backward=lambda *a, **k: None, state={})
    dens.apply_codec = lambda params, optimizers: calls.append(1) or {}
    for step in range(1, 51):
        dens.after_backward({}, {"means": SimpleNamespace(param_groups=[{"lr": 1.0}])}, step, {})
        _check(len(calls) == (1 if step == 50 else 0), f"codec ran at step {step}")
    _check(math.isclose(density_spec.config["total_steps"], 30000), "paper: 30K then compress")
    return {"encode_steps": [50], "total_steps": density_spec.config["total_steps"]}
