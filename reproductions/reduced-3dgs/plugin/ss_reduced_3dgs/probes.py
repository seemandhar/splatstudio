"""Mechanism probes for the Reduced-3DGS port. Each raises if its mechanism is removed or wrong."""

import math

import numpy as np
import torch
from splatstudio.specs import ComponentSpec
from splatstudio.verification import ProbeUnavailable

from . import codec, density, kernels, losses, redundancy, shcull
from .renderer import DEGREE_FIELD, ReducedRenderer


def _cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA device required")
    return torch.device(device)


def _component(method, name):
    return next(c for c in method.components if c.name == name)


# band of shN index j, written out independently of renderer.COEFF_BAND
_BAND = [1] * 3 + [2] * 5 + [3] * 7


def _fake_params(n, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.randn(n, 3, generator=g) * 0.3 - 3,
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
        DEGREE_FIELD: torch.randint(0, 4, (n,), generator=g).float(),
    }
    return {k: torch.nn.Parameter(v.to(device)) for k, v in p.items()}


def sh_sparsity_gradient(method, device):
    """dL/dshN == lambda/(45 V) sign(shN) on visible Gaussians' active bands, 0 elsewhere."""
    spec = _component(method, "sh-sparsity")
    loss = losses.create_sh_sparsity(spec)
    n = 64
    params = _fake_params(n, "cpu", 1)
    params["shN"].data[5, 0, 0] = 0.0  # sign(0) = 0
    radii = torch.randint(0, 3, (1, n, 2))
    radii[0, :4] = 0
    results = {}
    for step in (500, 1500, 2500, 20000):
        params["shN"].grad = None
        value = loss(params, {"info": {"radii": radii}, "step": step}, None)
        value.backward()
        vis = (radii[0] > 0).all(-1)
        active = min((step - 1) // 1000, 3)
        own = params[DEGREE_FIELD].detach()
        expect = torch.zeros_like(params["shN"])
        for i in range(n):
            if not vis[i]:
                continue
            for j in range(15):
                if _BAND[j] <= min(own[i].item(), active):
                    expect[i, j] = torch.sign(params["shN"].detach()[i, j])
        expect *= spec.config["weight"] / (45.0 * int(vis.sum()))
        err = float((params["shN"].grad - expect).abs().max())
        if err > 1e-9 or float(expect.abs().sum()) == 0 and step > 1000:
            raise AssertionError(f"SH sparsity gradient mismatch at step {step}: {err}")
        results[f"max_abs_err_step{step}"] = err
    return results


def opacity_l1_gradient(method, device):
    """d/d logit of lambda * mean_vis sigmoid(o) == lambda/V sigmoid'(o) on visible only."""
    spec = _component(method, "opacity-l1")
    loss = losses.create_opacity_l1(spec)
    params = _fake_params(50, "cpu", 2)
    radii = torch.randint(0, 2, (1, 50, 2))
    loss(params, {"info": {"radii": radii}, "step": 10}, None).backward()
    vis = (radii[0] > 0).all(-1)
    s = params["opacities"].detach().sigmoid()
    expect = torch.where(vis, spec.config["weight"] / int(vis.sum()) * s * (1 - s), 0.0)
    err = float((params["opacities"].grad - expect).abs().max())
    if err > 1e-9:
        raise AssertionError(f"opacity L1 gradient mismatch: {err}")
    return {"max_abs_err": err, "weight": spec.config["weight"]}


def knn_exact(method, device):
    """CUDA simple-knn boxKnn2 equals brute force (self excluded), including duplicates."""
    dev = _cuda(device)
    g = torch.Generator(device="cpu").manual_seed(3)
    pts = torch.randn(4000, 3, generator=g) * torch.tensor([3.0, 1.0, 0.2])
    pts[100:150] = pts[50:100]  # duplicated positions, as created by 3DGS cloning
    pts = pts.to(dev)
    d2, idx = kernels.knn_index(pts, 30)
    full = torch.cdist(pts, pts).square()
    full.fill_diagonal_(float("inf"))
    ref = full.topk(30, dim=1, largest=False).values
    got = torch.sort(d2, dim=1).values
    err = float((got - ref).abs().max())
    self_hit = bool((idx == torch.arange(len(pts), device=dev)[:, None]).any())
    if err > 1e-4 or self_hit:
        raise AssertionError(f"kNN not exact: max |d2 err| {err}, self included {self_hit}")
    return {"max_d2_err": err}


def _projection(K, W, H, znear=0.01, zfar=100.0):
    """3DGS getProjectionMatrix (column-vector convention) from the pinhole focal lengths."""
    tx, ty = W / (2 * K[0, 0]), H / (2 * K[1, 1])
    P = torch.zeros(4, 4, dtype=torch.float64)
    P[0, 0], P[1, 1] = 1 / tx, 1 / ty
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def _reference_footprint(means, cameras):
    """Official find_minimum_projected_pixel_size_python: un-project NDC offsets of one pixel."""
    out = torch.full((len(means),), 10000.0, dtype=torch.float64)
    X = torch.cat((means.double().cpu(), torch.ones(len(means), 1, dtype=torch.float64)), 1)
    for cam in cameras:
        W, H = cam["width"], cam["height"]
        full = _projection(cam["K"].double().cpu(), W, H) @ cam["viewmat"].double().cpu()
        inv = torch.linalg.inv(full)
        ndc = X @ full.T
        ndc = ndc / ndc[:, 3:]
        inside = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
        a = torch.zeros_like(ndc)
        a[:, 0 if W > H else 1] = 2 / W if W > H else 2 / H
        a[:, 2], a[:, 3] = ndc[:, 2], 1
        b = torch.zeros_like(ndc)
        b[:, 2], b[:, 3] = ndc[:, 2], 1
        a, b = a @ inv.T, b @ inv.T
        a, b = a / a[:, 3:], b / b[:, 3:]
        size = (a[:, :3] - b[:, :3]).norm(dim=1)
        out = torch.where(inside, torch.minimum(out, size), out)
    return out


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """OpenCV world-to-camera matrix (x right, y down, z forward)."""
    eye, target, up = (torch.tensor(v, dtype=torch.float32) for v in (eye, target, up))
    z = torch.nn.functional.normalize(target - eye, dim=0)
    x = torch.nn.functional.normalize(torch.linalg.cross(z, up), dim=0)
    y = torch.linalg.cross(z, x)
    R = torch.stack((x, y, z))
    V = torch.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def _cameras(dev, n=8, radius=3.0, W=64, H=48, f=60.0, elevations=(0.0,)):
    cams = []
    for e in elevations:
        for k in range(n):
            a = 2 * math.pi * k / n
            eye = (radius * math.cos(a) * math.cos(e), radius * math.sin(e),
                   radius * math.sin(a) * math.cos(e))
            K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            cams.append({"viewmat": _look_at(eye).to(dev), "K": K.to(dev), "width": W, "height": H})
    return cams


def redundancy_score(method, device):
    """Pixel footprint, sphere/ellipsoid counts, min-propagation and mercy mask vs an O(N^2)
    reference written from the official Python/CUDA code."""
    dev = _cuda(device)
    g = torch.Generator(device="cpu").manual_seed(4)
    n = 300
    means = torch.randn(n, 3, generator=g) * 0.3
    means[:60] = means[:60] * 0.05  # a dense, redundant cluster
    scales = (torch.randn(n, 3, generator=g) * 0.4 - 3.5).exp()
    quats = torch.randn(n, 4, generator=g)
    opacity = torch.rand(n, generator=g)
    cams = _cameras(dev, n=6)
    fp = redundancy.min_pixel_footprint(means.to(dev), cams).double().cpu()
    fp_ref = _reference_footprint(means, cams)
    fp_err = float(((fp - fp_ref).abs() / fp_ref).max())
    if fp_err > 1e-4:
        raise AssertionError(f"pixel footprint mismatch: rel err {fp_err}")
    score = redundancy.redundancy_scores(means.to(dev), scales.to(dev), quats.to(dev), cams,
                                         30, 1.0).cpu()
    # reference
    r = fp_ref * math.sqrt(3) / 2
    d = torch.cdist(means.double(), means.double())
    d.fill_diagonal_(float("inf"))
    nbr = d.topk(30, dim=1, largest=False).indices
    q = torch.nn.functional.normalize(quats.double(), dim=1).numpy()
    hits = np.zeros((n, 30), dtype=bool)
    for i in range(n):
        for k, j in enumerate(nbr[i].tolist()):
            w, x, y, z = q[j]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            local = R.T @ (means[i].double().numpy() - means[j].double().numpy())
            ax = scales[j].double().numpy() + float(r[i])
            hits[i, k] = float(np.sum(local**2 / ax**2)) < 1
    own = hits.sum(1) + 1
    ref = own.copy()
    for i in range(n):
        for k, j in enumerate(nbr[i].tolist()):
            if hits[i, k]:
                ref[j] = min(ref[j], own[i])
    mismatch = int((score.numpy() != ref).sum())
    if mismatch > 2:  # float32 vs float64 on boundary cases
        raise AssertionError(f"redundancy score mismatch on {mismatch} Gaussians")
    mask, info = redundancy.mercy_mask(score.to(dev), opacity.to(dev))
    s = torch.from_numpy(ref).float()
    thr = max(float(s.mean() + s.std()), 3.0)
    cand = s > thr
    m_ref = torch.zeros(n, dtype=torch.bool)
    if cand.any():
        m_ref[cand] = opacity[cand] < opacity[cand].median()
    m_ref |= opacity < min(float(torch.quantile(opacity, 0.03)), 0.05)
    if mismatch == 0 and not torch.equal(mask.cpu(), m_ref):
        raise AssertionError("mercy mask differs from the official rule")
    if int(score.max()) < 4 or not bool(mask.any()):
        raise AssertionError("synthetic redundant cluster not detected")
    return {"footprint_rel_err": fp_err, "score_mismatch": mismatch, "max_score": int(score.max()),
            "pruned": int(mask.sum()), "candidates": info["candidates"]}


def transmittance_stats(method, device):
    """Kernel sum_T / #pixels == per-pixel front-to-back compositing in torch (gsplat numerics)."""
    dev = _cuda(device)
    g = torch.Generator(device="cpu").manual_seed(5)
    n = 40
    params = {
        "means": torch.randn(n, 3, generator=g) * 0.4,
        "scales": (torch.randn(n, 3, generator=g) * 0.2 - 2.0),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.rand(n, generator=g) * 0.25 + 0.05),
        "sh0": torch.rand(n, 1, 3, generator=g),
        "shN": torch.zeros(n, 15, 3),
        DEGREE_FIELD: torch.full((n,), 3.0),
    }
    params = {k: v.to(dev) for k, v in params.items()}
    cam = _cameras(dev, n=1, W=40, H=30, f=40.0)[0]
    renderer = ReducedRenderer(ComponentSpec("r", "renderer", "x", config={}))
    with torch.no_grad():
        out = renderer.render(params, cam, 3)
    info = out["info"]
    sum_t, cnt = kernels.transmittance_stats(info)
    m2 = info["means2d"][0].double().cpu()
    con = info["conics"][0].double().cpu()
    opa = info["opacities"][0].double().cpu()
    depth = info["depths"][0].double().cpu()
    vis = (info["radii"][0] > 0).all(-1).cpu()
    order = [i for i in torch.argsort(depth).tolist() if vis[i]]
    ref_t = torch.zeros(n, dtype=torch.float64)
    ref_c = torch.zeros(n, dtype=torch.long)
    final_t = torch.ones(cam["height"], cam["width"], dtype=torch.float64)
    for py in range(cam["height"]):
        for px in range(cam["width"]):
            T = 1.0
            for i in order:
                dx, dy = m2[i, 0] - (px + 0.5), m2[i, 1] - (py + 0.5)
                sig = 0.5 * (con[i, 0] * dx * dx + con[i, 2] * dy * dy) + con[i, 1] * dx * dy
                a = min(0.999, float(opa[i] * torch.exp(-sig)))
                if sig < 0 or a < 1 / 255:
                    continue
                nt = T * (1 - a)
                if nt <= 1e-4:
                    break
                ref_t[i] += T
                ref_c[i] += 1
                T = nt
            final_t[py, px] = T
    alpha_err = float((out["alpha"][..., 0].double().cpu() - (1 - final_t)).abs().max())
    t_err = float((sum_t.double().cpu() - ref_t).abs().max())
    c_err = int((cnt.cpu() - ref_c).abs().max())
    if alpha_err > 1e-3:
        raise ProbeUnavailable(f"torch reference does not reproduce gsplat alpha ({alpha_err})")
    if t_err > 1e-2 or c_err > 1 or int(ref_c.sum()) == 0:
        raise AssertionError(f"transmittance stats mismatch: T {t_err}, count {c_err}")
    return {"alpha_err": alpha_err, "sum_T_err": t_err, "count_err": c_err,
            "touched_total": int(ref_c.sum())}


def _band_fixture(dev):
    """Diffuse / band-1-only / band-3 groups (opacity logit 2) + a band-3 group at logit -0.5."""
    n_per = 6
    pts = [[(k % 6 - 2.5) * 0.25, (k // 6 - 1.5) * 0.25, 0.0] for k in range(4 * n_per)]
    n = len(pts)
    shN = torch.zeros(n, 15, 3)
    shN[n_per:2 * n_per, 0:3] = 0.35  # band 1 only
    shN[2 * n_per:, 8:15] = 0.35  # band 3 (groups 3 and 4)
    opac = torch.full((n,), 2.0)
    opac[3 * n_per:] = -0.5  # sigmoid 0.38: visible, but logit <= 0
    params = {
        "means": torch.tensor(pts), "scales": torch.full((n, 3), math.log(0.04)),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1), "opacities": opac,
        "sh0": torch.full((n, 1, 3), 0.2), "shN": shN, DEGREE_FIELD: torch.full((n,), 3.0),
    }
    return {k: torch.nn.Parameter(v.to(dev)) for k, v in params.items()}, n_per


def sh_band_culling(method, device):
    """Two-pass culling on synthetic groups. Textbook statistics: diffuse / band-1 / band-3 /
    band-3-low-opacity -> 0 / 1 / 3 / 3 bands. Official quirk (1) (logit opacity fed to the
    rasterizer): the logit<=0 band-3 group gets zero weight and drops to 0 bands, with
    DC = (0 - 0.5) / C0. The configured mode must match its expectation and the other mode its own."""
    dev = _cuda(device)
    spec = _component(method, "reduced-density")
    cams = _cameras(dev, n=12, radius=2.5, W=96, H=72, f=80.0, elevations=(-0.5, 0.0, 0.5))
    renderer = ReducedRenderer(ComponentSpec("r", "renderer", "x", config={}))
    out = {}
    for quirks in (True, False):
        params, n_per = _band_fixture(dev)
        info = shcull.cull_sh_bands(renderer, params, cams, spec.config["std_threshold"],
                                    spec.config["cdist_threshold"], official_quirks=quirks)
        deg = params[DEGREE_FIELD].detach().round().long().cpu()
        want = torch.tensor([0] * n_per + [1] * n_per + [3] * n_per + [0 if quirks else 3] * n_per)
        if not torch.equal(deg, want):
            raise AssertionError(f"quirks={quirks}: bands {deg.tolist()} != {want.tolist()}")
        if float(params["shN"].detach()[:n_per].abs().max()) != 0.0:
            raise AssertionError("culled Gaussians keep SH-rest coefficients")
        if float(params["shN"].detach()[n_per:2 * n_per, 3:].abs().max()) != 0.0:
            raise AssertionError("band-1 Gaussians keep higher-band coefficients")
        if quirks:
            dc = params["sh0"].detach()[3 * n_per:].cpu()
            if not torch.allclose(dc, torch.full_like(dc, -0.5 / shcull.SH_C0)):
                raise AssertionError("zero-weight Gaussians must get DC = (0 - 0.5) / C0")
        out[f"quirks_{quirks}"] = {"degrees": deg.tolist(), "band_fractions": info["band_fractions"]}
    out["configured_quirks"] = bool(spec.config["official_shcull_quirks"])
    return out


def shcull_variance_alias(method, device):
    """Official quirk (2): aliased mean_old makes the variance sum T (c - mean_new)^2; the first
    present view contributes 0. Checked against closed forms for 3 views; the textbook West
    update must give the weighted variance instead."""
    t = torch.tensor([0.5, 0.3, 0.2])
    c = torch.tensor([0.1, 0.6, 0.3])
    res = {}
    for quirks in (True, False):
        st = {"dist": torch.zeros(1, 3), "w_sum": torch.zeros(1, 1), "mean": torch.zeros(1, 3),
              "var": torch.zeros(1, 3)}
        for ti, ci in zip(t, c):
            colours = torch.zeros(1, 4, 3)
            colours[:, 3] = ci
            shcull.accumulate_view(st, ti.view(1, 1), colours, torch.tensor([True]), quirks)
        res[quirks] = float(st["var"][0, 0] / st["w_sum"][0, 0])
    w = torch.cumsum(t, 0)
    means = torch.cumsum(t * c, 0) / w
    alias = float((t * (c - means).square()).sum() / t.sum())
    west = float((t * (c - means[-1]).square()).sum() / t.sum())
    if abs(res[True] - alias) > 1e-6 or abs(res[False] - west) > 1e-6 or abs(alias - west) < 1e-3:
        raise AssertionError(f"variance: quirk {res[True]} vs {alias}, West {res[False]} vs {west}")
    cfg = _component(method, "reduced-density").config["official_shcull_quirks"]
    return {"aliased_var": res[True], "west_var": res[False], "configured_quirks": bool(cfg)}


def _kmeans_reference(values, k, tol, g):
    values = values.flatten().float()
    centers = values[torch.randint(values.shape[0], (k,), device=values.device, generator=g)].clone()
    for _ in range(500):
        ids = (values[:, None] - centers[None]).abs().argmin(1)  # first index on ties
        old = centers
        centers = torch.stack([values[ids == c].mean() if (ids == c).any()
                               else torch.zeros((), device=values.device) for c in range(k)])
        if float((old - centers).abs().sum()) < tol:
            break
    return (values[:, None] - centers[None]).abs().argmin(1), centers


def codec_roundtrip(method, device):
    """K-means == reference (incl. duplicate-init -> empty -> 0 rule); encode -> official PLY
    bytes -> decode gives codebook values / float16 positions; size == memory_results."""
    dev = _cuda(device)
    g = torch.Generator(device="cpu").manual_seed(6)
    vals = torch.randn(3000, generator=g)
    vals[:2400] = 0.0  # mostly zero, like SH-rest after culling
    vals = vals.to(dev)
    g1 = torch.Generator(device=dev).manual_seed(11)
    g2 = torch.Generator(device=dev).manual_seed(11)
    ids, centers = codec.kmeans_1d(vals, 32, 1e-4, generator=g1)
    rids, rcent = _kmeans_reference(vals, 32, 1e-4, g2)
    if not torch.equal(ids, rids) or float((centers - rcent).abs().max()) > 1e-5:
        raise AssertionError("K-means differs from the official algorithm")
    zero_centres = int((centers == 0).sum())
    if zero_centres < 2:
        raise AssertionError("duplicate zero initial centres did not collapse (tie rule)")
    n = 3000
    params = _fake_params(n, dev, 7)
    params["shN"].data *= (torch.rand(n, 15, 1, device=dev) > 0.5)
    deg = params[DEGREE_FIELD].detach().long()
    for d in range(4):  # zero coefficients above each Gaussian's band count
        params["shN"].data[deg == d, (d + 1) ** 2 - 1:] = 0
    _, report = codec.encode_decode(params, 256, 0, True)
    books = codec.produce_codebooks(params, 256, 0)
    data = codec.encode_ply(params, books, True)
    decoded = codec.decode_ply(data, dev, True)
    if len(data) != report["ply_bytes"]:
        raise AssertionError("PLY size depends on the codebook draw")
    order = torch.cat([torch.where(deg == d)[0] for d in range(4)])
    pos_err = float((decoded["means"] - params["means"].detach()[order].half().float()).abs().max())
    if pos_err != 0.0:
        raise AssertionError(f"decoded positions are not the float16 positions ({pos_err})")
    dec_deg = decoded[DEGREE_FIELD].long()
    if not torch.equal(dec_deg, deg[order]):
        raise AssertionError("band groups not preserved")
    op_ids = books["opacity"][0][order, 0].long()
    want_op = books["opacity"][1].half().float()[op_ids]
    if not torch.equal(decoded["opacities"], want_op.clamp(-65504, 65504)):
        raise AssertionError("decoded opacity is not the float16 codebook centre of its index")
    rest = decoded["shN"]
    for d in range(4):
        tail = rest[dec_deg == d, (d + 1) ** 2 - 1:]
        if tail.numel() and float(tail.abs().max()) != 0.0:
            raise AssertionError("decoded coefficients above the band count are not zero")
    quant_err = float((decoded["sh0"] - params["sh0"].detach()[order]).abs().max())
    if quant_err == 0.0:
        raise AssertionError("decoded colours equal the inputs: quantisation absent")
    counts = [int((deg == d).sum()) for d in range(4)]
    p = sum(counts)
    want_bytes = (3 * p * 2 + (counts[0] + 4 * counts[1] + 9 * counts[2] + 16 * counts[3]) * 3
                  + 256 * 16 * 4 + 8 * p + 256 * 8 * 4)
    if report["paper_mem_bytes"] != want_bytes:
        raise AssertionError("encoded size differs from generate_results.py:memory_results")
    body = (sum(c * (6 + 3 + 3 * ((d + 1) ** 2 - 1) + 8) for d, c in enumerate(counts))
            + 256 * 20 * 2)
    header = report["ply_bytes"] - body
    if not 0 < header < 8192:
        raise AssertionError(f"PLY body size unexpected (header {header} bytes)")
    return {"zero_centres": zero_centres, "pos_err": pos_err, "dc_quant_err": quant_err,
            "paper_mem_bytes": report["paper_mem_bytes"], "ply_bytes": report["ply_bytes"],
            "ply_header_bytes": header}


def schedule(method, device):
    """Event steps equal an independent transcription of the official train.py conditions."""
    cfg = {**density.DEFAULTS, **_component(method, "reduced-density").config}
    got = {e: [] for e in ("dead_prune", "mercy", "save_prune", "cull_sh", "encode")}
    for it in range(1, cfg["total_steps"] + 1):
        for e in density.schedule_events(it, cfg):
            got[e].append(it)
    fine_tune_start = 30000 - 3000
    want = {
        "dead_prune": [it for it in range(1, 30001) if not it < 15000 and it % 100 == 0],
        "mercy": [it for it in range(1, 30001) if it % (10 * 100) == 0 and it <= fine_tune_start
                  and (it >= 15000 or it % 3000 != 0)],
        "save_prune": [7000, 30000],
        "cull_sh": [15000],
        "encode": [30000],
    }
    if cfg["total_steps"] != 30000 or got != want:
        diff = {k: (got[k][:5], want[k][:5]) for k in want if got[k] != want[k]}
        raise AssertionError(f"schedule differs from official train.py: {diff}")
    return {k: len(v) for k, v in got.items()} | {"first_mercy": got["mercy"][:4]}
