"""Mechanism probes for DET-GS; each raises if its mechanism is removed or wrong."""

import numpy as np
import torch

from . import losses, priors


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _ref_norm_loss(depth, target, p, margin, use_image_std):
    """Independent loop over patches: (x - mu_P) / (sigma + 1e-2 * std_all), masked L2."""
    H, W = depth.shape
    out = []
    for full in (depth, target):
        rows = [full[i * p:(i + 1) * p, j * p:(j + 1) * p].reshape(-1)
                for i in range(H // p) for j in range(W // p)]
        m = np.stack(rows).astype(np.float64)
        delta = 1e-2 * m.reshape(-1).std(ddof=1)
        sig = full.std(ddof=1) if use_image_std else m.std(axis=1, ddof=1, keepdims=True)
        out.append((m - m.mean(axis=1, keepdims=True)) / (sig + delta))
    diff = out[0] - out[1]
    sel = np.abs(diff) > margin
    return float((diff[sel] ** 2).mean())


def depth_formula(method, device):
    cfg = {**losses.DEPTH_DEFAULTS, **_spec(method, "det-gs-depth").config}
    g = np.random.default_rng(0)
    depth = g.random((37, 51)) * 20 + 5
    disp = g.random((37, 51)) * 3 + 0.5
    target = -disp
    m = float(cfg["error_tolerance"])
    for p in (5, 11, 20):
        dt = torch.tensor(depth, dtype=torch.float64)[None, None]
        tt = torch.tensor(target, dtype=torch.float64)[None, None]
        got = float(losses.hierarchical_depth_loss(dt, tt, p, p, cfg["gamma"], cfg["eta"], m))
        ref = (cfg["gamma"] * _ref_norm_loss(depth, target, p, m, False)
               + cfg["eta"] * _ref_norm_loss(depth, target, p, m, True))
        if abs(got - ref) > 1e-9 * max(1.0, abs(ref)):
            raise AssertionError(f"Eq.7 mismatch at p={p}: {got} vs {ref}")
    # Affine invariance of the normalised target (scale/shift of the disparity changes nothing) ...
    dt = torch.tensor(depth)[None, None]
    a = float(losses.hierarchical_depth_loss(dt, losses.mono_target(torch.tensor(disp))[None, None],
                                             7, 9, 0.1, 1.0, m))
    b = float(losses.hierarchical_depth_loss(
        dt, losses.mono_target(torch.tensor(3.0 * disp + 2.0))[None, None], 7, 9, 0.1, 1.0, m))
    if abs(a - b) > 1e-6 * a:
        raise AssertionError("depth loss not affine invariant in the monocular prior")
    # ... and a perfect (affine-related) depth gives zero loss.
    zero = float(losses.hierarchical_depth_loss(
        losses.mono_target(torch.tensor(disp))[None, None] * 4 + 1,
        losses.mono_target(torch.tensor(disp))[None, None], 7, 9, 0.1, 1.0, m))
    # disparity space: an exact inverse-depth relation gives zero loss, a depth-space target does not.
    z = torch.tensor(depth)[None, None]
    dz, tz = losses.to_space(z, 2.0 / z + 0.3, "disparity")
    zd = float(losses.hierarchical_depth_loss(dz, tz, 7, 9, 0.1, 1.0, m))
    dd, td = losses.to_space(z, 2.0 / z + 0.3, "depth")
    zz = float(losses.hierarchical_depth_loss(dd, td, 7, 9, 0.1, 1.0, m))
    if zd > 1e-12 or zz < 1e-3:
        raise AssertionError(f"space option broken (disparity {zd}, depth {zz})")
    if zero > 1e-12 or cfg["gamma"] != 0.1 or cfg["eta"] != 1.0:
        raise AssertionError(f"depth loss config/zero check failed ({zero})")
    return {"eq7_checked_patch_sizes": [5, 11, 20], "affine_gap": abs(a - b), "loss": a}


def depth_schedule(method, device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    dev = torch.device(device)
    g = torch.Generator().manual_seed(0)
    n, H, W = 400, 48, 64
    means = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([3.0, 2.0, 1.0]) + torch.tensor([0, 0, 4.0])
    params = {
        "means": means.to(dev).requires_grad_(),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1).to(dev).requires_grad_(),
        "scales": torch.full((n, 3), -2.5).to(dev).requires_grad_(),
        "opacities": torch.zeros(n).to(dev).requires_grad_(),
    }
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], device=dev)
    cam = {"viewmat": torch.eye(4, device=dev), "K": K, "width": W, "height": H}
    yy = torch.linspace(0, 1, H)[:, None].expand(H, W)
    disp = (1.0 + yy).numpy()[None]
    priors.set_table(priors.PriorTable(cam["viewmat"].cpu().numpy()[None], disp,
                                       np.zeros((1, H, W), np.uint8)))
    fn = losses.create_depth(_spec(method, "det-gs-depth"))
    every = int(_spec(method, "det-gs-depth").config.get("every", 5))
    values = {}
    for step in range(1, 2 * every + 1):
        for p in params.values():
            p.grad = None
        v = fn(params, {"step": step, "camera": cam}, None)
        v.backward()
        gm = float(params["means"].grad.abs().sum()) if params["means"].grad is not None else 0.0
        if step % every == 0:
            if not (float(v) > 0 and gm > 0):
                raise AssertionError(f"depth loss inactive at step {step}")
            for k in ("scales", "quats", "opacities"):
                if params[k].grad is not None and float(params[k].grad.abs().sum()) != 0:
                    raise AssertionError(f"hard-depth gradient leaked into {k}")
        elif float(v) != 0 or gm != 0:
            raise AssertionError(f"depth loss active at step {step}")
        values[step] = float(v)
    priors.set_table(None)
    priors._TABLE.clear()
    return {"every": every, "values": values}


def edge_formula(method, device):
    cfg = {**losses.EDGE_DEFAULTS, **_spec(method, "det-gs-edge").config}
    g = np.random.default_rng(1)
    H, W = 13, 17
    depth = g.random((H, W)) * 4
    edges = g.random((H, W)) < 0.3
    m = (~edges).astype(np.float64)
    num, cnt = 0.0, 0
    for i in range(H):
        for j in range(W):
            if not m[i, j]:
                continue
            s, c = 0.0, 0.0
            for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < H and 0 <= b < W:
                    s += depth[a, b] * m[a, b]
                    c += m[a, b]
            num += (depth[i, j] - s / (c + cfg["eps"])) ** 2
            cnt += 1
    ref = num / cnt
    d = torch.tensor(depth, requires_grad=True)
    e = torch.tensor(edges)
    got = losses.edge_aware_depth_loss(d, e, cfg["eps"])
    if abs(float(got) - ref) > 1e-9:
        raise AssertionError(f"Eq.11 mismatch {float(got)} vs {ref}")
    # A depth step that sits exactly on an edge line is not penalised; without the mask it is.
    step_d = torch.zeros(H, W, dtype=torch.float64)
    step_d[:, 9:] = 5.0
    line = torch.zeros(H, W, dtype=torch.bool)
    line[:, 8:10] = True
    masked = float(losses.edge_aware_depth_loss(step_d, line))
    unmasked = float(losses.edge_aware_depth_loss(step_d, torch.zeros_like(line)))
    if masked > 1e-12 or unmasked < 1e-3:
        raise AssertionError(f"edge mask ineffective: masked {masked}, unmasked {unmasked}")
    if cfg["beta"] != 0.1:
        raise AssertionError("beta differs from the paper (0.1)")
    return {"loss": float(got), "reference": ref, "masked_step": masked, "unmasked_step": unmasked}


def tv_formula(method, device):
    cfg = {**losses.TV_DEFAULTS, **_spec(method, "det-gs-tv").config}
    te, ts = cfg["tau_edge"], cfg["tau_smooth"]
    g = np.random.default_rng(2)
    H, W, C = 9, 11, 3
    gt = np.round(g.random((H, W, C)) * 4) * 0.004  # steps of 0.004 (<te) and >= 0.008
    pred = g.random((H, W, C)) * 0.01
    total = 0.0
    for i in range(H):
        for j in range(W):
            for c in range(C):
                if j + 1 < W and abs(gt[i, j + 1, c] - gt[i, j, c]) < te:
                    total += max(abs(pred[i, j + 1, c] - pred[i, j, c]) - ts, 0)
                if i + 1 < H and abs(gt[i + 1, j, c] - gt[i, j, c]) < te:
                    total += max(abs(pred[i + 1, j, c] - pred[i, j, c]) - ts, 0)
    ref = total / (H * W * C)
    got = float(losses.edge_preserving_tv(torch.tensor(pred), torch.tensor(gt), te, ts))
    if abs(got - ref) > 1e-12:
        raise AssertionError(f"Eq.15 mismatch {got} vs {ref}")
    # Strong GT edges and sub-margin predicted gradients are not penalised.
    edge_gt = torch.zeros(4, 4, 3, dtype=torch.float64)
    edge_gt[:, 2:] = 1.0
    edge_pred = edge_gt.clone()
    tiny = torch.arange(16, dtype=torch.float64).reshape(4, 4, 1).expand(4, 4, 3) * (ts / 10)
    if float(losses.edge_preserving_tv(edge_pred, edge_gt, te, ts)) != 0.0:
        raise AssertionError("TV penalises a ground-truth edge")
    if float(losses.edge_preserving_tv(tiny, torch.zeros_like(tiny), te, ts)) != 0.0:
        raise AssertionError("TV penalises gradients below tau_smooth")
    if (te, ts, cfg["phi"]) != (1e-2, 1e-4, 0.8):
        raise AssertionError("TV hyper-parameters differ from the paper")
    return {"loss": got, "reference": ref}
