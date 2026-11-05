"""Mechanism probes for the Speedy-Splat port. Each raises on failure and returns measurements."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
TILE = 16


# ---------------------------------------------------------------------------- helpers
def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _density(method, **config):
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")
    return create_density(replace(spec, config={**spec.config, **config}))


def _params(means, log_scales, quats, opac, sh0, device):
    vals = {"means": means, "scales": log_scales, "quats": quats,
            "opacities": torch.logit(opac), "sh0": sh0,
            "shN": torch.zeros(len(means), 15, 3)}
    return {k: v.float().to(device) for k, v in vals.items()}


def _optimizable(params):
    params = {k: torch.nn.Parameter(v.clone()) for k, v in params.items()}
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for k, p in params.items():  # allocate Adam moments, then restore the values
        before = p.detach().clone()
        p.grad = torch.ones_like(p)
        opts[k].step()
        opts[k].zero_grad(set_to_none=True)
        p.data.copy_(before)
    return params, opts


def _fixture(device, n=48, seed=3):
    """Random anisotropic Gaussians in front of a 96x80 camera; opacities span 0.005..0.99 so
    the alpha=1/255 ellipse is both tighter than and wider than the 3-sigma circle."""
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([2.4, 2.0, 2.0]) + torch.tensor(
        [-1.2, -1.0, 3.0])
    scales = torch.rand(n, 3, generator=g) * torch.tensor([0.25, 0.12, 0.05]) + 0.01
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = torch.cat([torch.rand(n // 2, generator=g) * 0.09 + 0.9,  # high: beyond 3 sigma
                      torch.rand(n - n // 2, generator=g) ** 3 * 0.6 + 0.005])
    sh0 = ((torch.rand(n, 1, 3, generator=g) - 0.5) / C0 * 0.8)
    W, H = 96, 80
    camera = {"viewmat": torch.eye(4, device=device), "width": W, "height": H,
              "K": torch.tensor([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]], device=device)}
    return _params(means, scales.log(), quats, opac, sh0, device), camera


def _rot(q):
    """Rotation matrices by rotating basis vectors (v' = v + 2w(u x v) + 2u x (u x v))."""
    q = F.normalize(q.double(), dim=-1)
    w, u = q[:, :1], q[:, 1:]
    cols = []
    for e in torch.eye(3, dtype=torch.float64):
        v = e.expand_as(u)
        t = 2 * torch.cross(u, v, dim=-1)
        cols.append(v + w * t + torch.cross(u, t, dim=-1))
    return torch.stack(cols, dim=-1)


def _project(params, camera):
    """Independent float64 EWA projection with the Inria conventions (forward.cu)."""
    dd = torch.float64
    p = {k: v.detach().cpu().to(dd) for k, v in params.items()}
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].cpu().to(dd)
    view = camera["viewmat"].cpu().to(dd)
    t = p["means"] @ view[:3, :3].T + view[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tz = t[:, 2]
    tx = (t[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
    J = torch.zeros(len(t), 2, 3, dtype=dd)
    J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
    R = _rot(p["quats"])
    cov3 = R @ torch.diag_embed(p["scales"].exp() ** 2) @ R.transpose(1, 2)
    Wr = view[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2) + 0.3 * torch.eye(2, dtype=dd)
    conic = torch.linalg.inv(cov2)
    ndc = torch.stack([fx * t[:, 0] / tz * 2 / W + 2 * cx / W - 1,
                       fy * t[:, 1] / tz * 2 / H + 2 * cy / H - 1], -1)
    xy = ((ndc + 1) * torch.tensor([W, H], dtype=dd) - 1) * 0.5  # ndc2Pix
    opac = p["opacities"].sigmoid()
    col = (C0 * p["sh0"][:, 0] + 0.5).clamp_min(0)
    return {"xy": xy, "conic": conic, "cov2": cov2, "opac": opac, "col": col, "depth": tz,
            "W": W, "H": H}


def _dense_reference(proj, bg, weights=None):
    """Unculled float64 compositor: every Gaussian evaluated at every pixel in depth order with
    forward.cu's rules (power > 0 skipped, alpha = min(0.99, o G), alpha < 1/255 skipped, stop
    when T < 1e-4). Returns (image HxWx3, dL/dalpha [pixels, N]) for L = sum_p w_p . C_p."""
    dd = torch.float64
    W, H = proj["W"], proj["H"]
    n = len(proj["xy"])
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dd), torch.arange(W, dtype=dd), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    e = torch.zeros(len(pix), n, dtype=dd, requires_grad=True)
    T = torch.ones(len(pix), dtype=dd)
    C = torch.zeros(len(pix), 3, dtype=dd)
    alive = torch.ones(len(pix), dtype=torch.bool)
    for i in torch.argsort(proj["depth"]).tolist():
        if proj["depth"][i] <= 0.2:
            continue
        d = proj["xy"][i] - pix
        a, b, c = proj["conic"][i, 0, 0], proj["conic"][i, 0, 1], proj["conic"][i, 1, 1]
        power = -0.5 * (a * d[:, 0] ** 2 + c * d[:, 1] ** 2) - b * d[:, 0] * d[:, 1]
        raw = proj["opac"][i] * power.exp()
        alpha = raw - (raw - 0.99).clamp_min(0).detach() + e[:, i]  # clamp, straight-through
        valid = alive & (power <= 0) & (alpha.detach() >= 1 / 255)
        test_T = T * (1 - alpha)
        stop = valid & (test_T.detach() < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        C = C + torch.where(use[:, None], proj["col"][i] * (alpha * T)[:, None], 0)
        T = torch.where(use, test_T, T)
    C = C + T[:, None] * bg.cpu().to(dd)
    w = torch.ones_like(C) if weights is None else weights.reshape(-1, 3).to(dd)
    (w * C).sum().backward()
    return C.detach().reshape(H, W, 3), e.grad


def _box_min(conic, mu, x0, x1, y0, y1):
    """min over the box [x0,x1]x[y0,y1] of q(p) = (p-mu)^T A (p-mu), A positive definite."""
    if x0 <= mu[0] <= x1 and y0 <= mu[1] <= y1:
        return 0.0
    a, b, c = conic[0][0], conic[0][1], conic[1][1]
    best = math.inf
    for x in (x0, x1):  # vertical edges: minimise over y
        dx = x - mu[0]
        y = min(max(mu[1] - b * dx / c, y0), y1)
        dy = y - mu[1]
        best = min(best, a * dx * dx + 2 * b * dx * dy + c * dy * dy)
    for y in (y0, y1):  # horizontal edges: minimise over x
        dy = y - mu[1]
        x = min(max(mu[0] - b * dy / a, x0), x1)
        dx = x - mu[0]
        best = min(best, a * dx * dx + 2 * b * dx * dy + c * dy * dy)
    return best


def _exact_tile_pairs(proj):
    """Independent count of (tile, Gaussian) pairs whose 16x16 tile square [16i, 16i+16) meets
    the ellipse (p-mu)^T Sigma^-1 (p-mu) <= 2 log(255 o) (paper Eqs. 10-15)."""
    gx, gy = (proj["W"] + TILE - 1) // TILE, (proj["H"] + TILE - 1) // TILE
    total = 0
    for i in range(len(proj["xy"])):
        if proj["depth"][i] <= 0.2 or proj["opac"][i] * 255 <= 1:
            continue
        t = 2 * math.log(255 * float(proj["opac"][i]))
        A = proj["conic"][i].tolist()
        mu = proj["xy"][i].tolist()
        for ty in range(gy):
            for tx in range(gx):
                q = _box_min(A, mu, tx * TILE, (tx + 1) * TILE, ty * TILE, (ty + 1) * TILE)
                total += q <= t
    return total


def _square_tile_pairs(proj):
    """3D-GS getRect count: square of radius ceil(3 sqrt(lambda_max)) (Eq. 9)."""
    gx, gy = (proj["W"] + TILE - 1) // TILE, (proj["H"] + TILE - 1) // TILE
    total = 0
    for i in range(len(proj["xy"])):
        if proj["depth"][i] <= 0.2:
            continue
        cov = proj["cov2"][i]
        mid = 0.5 * (cov[0, 0] + cov[1, 1])
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
        lam = mid + math.sqrt(max(0.1, float(mid * mid - det)))
        r = math.ceil(3 * math.sqrt(lam))
        x, y = proj["xy"][i].tolist()
        x0, x1 = min(gx, max(0, int((x - r) / TILE))), min(gx, max(0, int((x + r + TILE - 1)
                                                                         / TILE)))
        y0, y1 = min(gy, max(0, int((y - r) / TILE))), min(gy, max(0, int((y + r + TILE - 1)
                                                                         / TILE)))
        total += (x1 - x0) * (y1 - y0)
    return total


# ---------------------------------------------------------------------------- C1/C2 (CUDA)
def culling_exact(method, device):
    _cuda(device)
    from .raster import render

    params, camera = _fixture(device)
    bg = torch.tensor([0.2, 0.3, 0.1], device=device)
    with torch.no_grad():
        img, _, radii, stats = render(params, camera, 0, bg, "speedy")
        img_stock, _, _, stats_stock = render(params, camera, 0, bg, "stock")
    proj = _project(params, camera)
    ref, _ = _dense_reference(proj, bg)
    got = img.permute(1, 2, 0).double().cpu()
    stock = img_stock.permute(1, 2, 0).double().cpu()
    err = (got - ref).abs().max().item()
    err_stock = (stock - ref).abs().max().item()
    exact = _exact_tile_pairs(proj)
    square = _square_tile_pairs(proj)
    pairs, pairs_stock = stats["num_rendered"], stats_stock["num_rendered"]
    out = {"max_abs_err_speedy_vs_dense": err, "max_abs_err_stock_vs_dense": err_stock,
           "pairs_speedy": pairs, "pairs_exact_reference": exact, "pairs_stock": pairs_stock,
           "pairs_3sigma_reference": square, "visible": int((radii > 0).sum())}
    if err > 1e-4:
        raise AssertionError(f"culled render differs from the unculled reference: {out}")
    if err_stock < 1e-3:
        raise AssertionError(f"degenerate fixture: 3-sigma truncation not visible: {out}")
    if abs(pairs - exact) > max(2, 0.01 * exact):
        raise AssertionError(f"AccuTile pairs != exact ellipse-tile pairs: {out}")
    if pairs_stock != square or not pairs < 0.8 * pairs_stock:
        raise AssertionError(f"tile-pair reduction vs 3-sigma squares missing: {out}")
    return out


# ---------------------------------------------------------------------------- C3 (CUDA)
def score_kernel(method, device):
    _cuda(device)
    from .density import create_density

    params, cam0 = _fixture(device, n=32, seed=11)
    cam1 = {**cam0, "viewmat": torch.eye(4, device=device)}
    cam1["viewmat"][:3, 3] = torch.tensor([0.25, -0.1, 0.3], device=device)
    bg = torch.tensor([0.2, 0.3, 0.1], device=device)
    spec = next(c for c in method.components if c.kind == "density")
    dens = create_density(replace(spec, config={**spec.config,
                                                "score_background": bg.tolist()}))
    dens.set_training_cameras([cam0, cam1])
    got = dens.scores(params, 1).double().cpu()  # step 1 -> SH degree 0
    ref = torch.zeros(len(got), dtype=torch.float64)
    for cam in (cam0, cam1):
        proj = _project(params, cam)
        _, dalpha = _dense_reference(proj, bg)
        ref += ((proj["opac"][None, :] * dalpha) ** 2).sum(0)  # sum_p (o dL/dalpha)^2
    linear = torch.zeros_like(ref)  # the un-squared aggregate, to show the probe can fail
    for cam in (cam0, cam1):
        proj = _project(params, cam)
        _, dalpha = _dense_reference(proj, bg)
        linear += (proj["opac"][None, :] * dalpha).sum(0) ** 2
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    rel_linear = ((linear - ref).abs().max() / ref.abs().max()).item()
    out = {"rel_err": rel, "rel_diff_if_squared_after_sum": rel_linear,
           "nonzero": int((ref > 0).sum()), "n": len(ref)}
    if rel > 2e-3:
        raise AssertionError(f"kernel pruning score differs from the reference: {out}")
    if rel_linear < 1e-2 or out["nonzero"] < len(ref) // 2:
        raise AssertionError(f"degenerate score fixture: {out}")
    return out


# ---------------------------------------------------------------------------- schedules (CPU)
def _drive(dens, params, opts, steps, record=None):
    for step in range(1, steps + 1):
        n = len(params["means"])
        m2 = torch.zeros(n, 3, requires_grad=True)
        m2.grad = torch.zeros(n, 3)
        out = {"info": {"means2d": m2, "radii": torch.ones(n, dtype=torch.int32)},
               "step": step}
        dens.before_backward(params, opts, step, out)
        dens.after_backward(params, opts, step, out)
        if record is not None:
            record(step, params)


def _small_params(n, big=None, opac=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    scales = torch.full((n, 3), 0.001)
    if big is not None:
        scales[big] = 0.2
    raw = {"means": torch.rand(n, 3, generator=g), "scales": scales.log(),
           "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
           "opacities": torch.logit(torch.full((n,), opac)), "sh0": torch.rand(n, 1, 3),
           "shN": torch.zeros(n, 15, 3)}
    return _optimizable(raw)


def adc_schedule(method, device):
    dens = _density(method, prune=False)
    params, opts = _small_params(8, big=0)
    dens.initialize(params, opts, 1.0)
    seen = {}

    def rec(step, p):
        if step in (3000, 3100):
            seen[step] = (len(p["means"]), float(p["opacities"].sigmoid().max()))

    _drive(dens, params, opts, 15100, rec)
    ev = [e["step"] for e in dens.state["events"]]
    out = {"first": ev[0], "last": ev[-1], "count": len(ev), "n_3000": seen[3000][0],
           "n_3100": seen[3100][0], "max_opacity_after_reset": seen[3000][1]}
    if ev[0] != 600 or ev[-1] != 14900 or len(ev) != 144:
        raise AssertionError(f"densify cadence wrong: {out}")
    if seen[3000][0] != 8 or seen[3100][0] != 7:
        raise AssertionError(f"world-size prune must start after step 3000: {out}")
    if abs(seen[3000][1] - 0.01) > 1e-4:
        raise AssertionError(f"opacity reset to 0.01 at 3000 missing: {out}")
    return out


def prune_schedule(method, device):
    n = 2000
    params, opts = _small_params(n, opac=0.5, seed=1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(2)).float()
    with torch.no_grad():  # tag rows: score = sh0[:, 0, 0] (a permutation), Adam state = tag
        params["sh0"][:, 0, 0] = perm
        opts["sh0"].state[params["sh0"]]["exp_avg"][:, 0, 0] = perm
    calls = []

    def stub(p, cameras, degree, bg):
        calls.append({"n": len(p["means"]), "opacity": float(p["opacities"].sigmoid().max()),
                      "degree": degree, "tags": p["sh0"][:, 0, 0].clone()})
        return p["sh0"][:, 0, 0].detach().clone()

    dens = _density(method)
    dens.initialize(params, opts, 1.0)
    dens.set_training_cameras([{}])
    dens.score_fn = stub

    def recover(step, p):  # training would raise opacities again after each reset
        if step + 1 in (6000, 9000, 12000):
            with torch.no_grad():
                p["opacities"].fill_(0.0)  # sigmoid 0.5

    _drive(dens, params, opts, 30000, recover)
    from .density import prune_mask

    rows = dens.state["prunes"]
    steps = [r["step"] for r in rows]
    kinds = [r["kind"] for r in rows]
    expect = [6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000]
    out = {"steps": steps, "kinds": kinds, "counts": [(r["n_before"], r["n_after"])
                                                      for r in rows]}
    if steps != expect or kinds != ["soft"] * 3 + ["hard"] * 5:
        raise AssertionError(f"prune schedule wrong: {out}")
    if any(abs(c["opacity"] - 0.5) > 1e-4 for c in calls[:3]):
        raise AssertionError(f"soft prune must run before the opacity reset: {out}")
    for r, c in zip(rows, calls):
        ratio = 0.8 if r["kind"] == "soft" else 0.3
        if r["n_after"] != int((~prune_mask(c["tags"], ratio)).sum()):
            raise AssertionError(f"percentile rule violated at {r['step']}: {out}")
        if r["n_after"] != r["n_before"] - (int(ratio * (r["n_before"] - 1)) + 1):
            raise AssertionError(f"wrong number pruned at {r['step']}: {out}")
    survivors = params["sh0"][:, 0, 0]
    if not torch.equal(survivors.sort().values, perm.sort().values[-len(survivors):]):
        raise AssertionError("pruning did not keep the highest-score Gaussians")
    if not torch.equal(opts["sh0"].state[params["sh0"]]["exp_avg"][:, 0, 0], survivors):
        raise AssertionError("optimizer state misaligned after pruning")
    zero = prune_mask(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]), 0.2)
    if zero.tolist() != [True, True, True, False, False]:
        raise AssertionError("ties at the percentile value must all be pruned")
    out["final"] = len(survivors)
    return out
