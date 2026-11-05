"""Mechanism probes for EDGS. Each raises if its mechanism is removed or wrong."""

import math
from pathlib import Path

import torch

from . import core


def _camera(eye, target, f=300.0, w=320, h=240):
    eye, target = torch.tensor(eye, dtype=torch.float64), torch.tensor(target, dtype=torch.float64)
    z = torch.nn.functional.normalize(target - eye, dim=0)
    x = torch.nn.functional.normalize(torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0],
                                                                      dtype=torch.float64), z), dim=0)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])  # rows: camera axes in world (OpenCV, y down after sign below)
    viewmat = torch.eye(4, dtype=torch.float64)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = -R @ eye
    K = torch.tensor([[f, 0, w / 2 + 3.0], [0, f * 1.01, h / 2 - 2.0], [0, 0, 1]],
                     dtype=torch.float64)
    return {"K": K, "viewmat": viewmat, "width": w, "height": h}


def _pixel_ndc(cam, points):
    """Independent pinhole projection: u = fx x/z + cx (pixel-corner convention), ndc = 2u/W - 1."""
    pc = (cam["viewmat"][:3, :3] @ points.T).T + cam["viewmat"][:3, 3]
    u = cam["K"][0, 0] * pc[:, 0] / pc[:, 2] + cam["K"][0, 2]
    v = cam["K"][1, 1] * pc[:, 1] / pc[:, 2] + cam["K"][1, 2]
    return torch.stack([2 * u / cam["width"] - 1, 2 * v / cam["height"] - 1], dim=1)


def triangulation(method, device):
    """C2: DLT least squares recovers synthetic points; error filter rejects perturbed matches."""
    g = torch.Generator().manual_seed(0)
    a = _camera([0.0, 0.0, -4.0], [0.0, 0.0, 0.0])
    b = _camera([1.0, 0.3, -3.8], [0.0, 0.0, 0.0])
    pts = (torch.rand(500, 3, generator=g, dtype=torch.float64) - 0.5) * 2
    Ma = core.ndc_projection(a["K"], a["viewmat"], a["width"], a["height"])
    Mb = core.ndc_projection(b["K"], b["viewmat"], b["width"], b["height"])
    uva, uvb = _pixel_ndc(a, pts), _pixel_ndc(b, pts)
    x, e1, e2 = core.triangulate(Ma, Mb, uva, uvb)
    err = float((x - pts).abs().max())
    # Reprojection carries the official EPS=1e-4 bias in the w division (~1e-5 NDC here).
    if err > 1e-6 or float(torch.maximum(e1, e2).max()) > 1e-3:
        raise AssertionError(f"Triangulation does not recover clean points (max err {err:.2e})")
    noisy = uvb.clone()
    noisy[:250, 1] += 0.2  # vertical offset: inconsistent with the epipolar geometry
    _, e1, e2 = core.triangulate(Ma, Mb, uva, noisy)
    e = torch.maximum(e1, e2)
    keep = e <= 0.01
    if keep[:250].any() or not keep[250:].all():
        raise AssertionError("tau_proj filter does not separate inconsistent matches")
    return {"max_point_error": err, "rejected_noisy": int((~keep[:250]).sum()),
            "kept_clean": int(keep[250:].sum())}


def sampling(method, device):
    """C3: thresholded weights, max aggregation over neighbours, best-neighbour selection."""
    cert = torch.stack([torch.tensor([0.9, 0.01, 0.2, 0.04]), torch.tensor([0.1, 0.03, 0.0, 0.5])])
    agg = cert.max(dim=0).values
    w = core.sampling_weights(agg, 0.05)
    expected = torch.tensor([1.0, 0.03, 1.0, 1.0])
    torch.testing.assert_close(w, expected)
    g = torch.Generator().manual_seed(0)
    big = torch.cat([torch.full((5000,), 0.5), torch.full((5000,), 0.01)])
    counts = torch.zeros(2)
    for _ in range(20):
        idx = torch.multinomial(core.sampling_weights(big), 1000, replacement=False, generator=g)
        counts += torch.stack([(idx < 5000).sum(), (idx >= 5000).sum()]).float()
    ratio = float(counts[1] / counts.sum())
    if not 0.002 < ratio < 0.03:  # weights 1 vs 0.01 -> ~1% of draws below threshold
        raise AssertionError(f"Below-threshold sampling share {ratio:.4f} not ~0.01")
    pts = torch.arange(2 * 3 * 3, dtype=torch.float64).reshape(2, 3, 3)
    e1 = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.1, 0.4]])
    e2 = torch.tensor([[0.4, 0.1, 0.1], [0.2, 0.2, 0.3]])
    x, best, idx = core.select_best(pts, e1, e2)
    # max errors: nn0 [0.4,0.5,0.2], nn1 [0.3,0.2,0.4] -> argmin [1,1,0]
    if idx.tolist() != [1, 1, 0] or not torch.equal(x[2], pts[0, 2]):
        raise AssertionError("Best neighbour is not argmin of max(eps_i, eps_j)")
    return {"below_threshold_share": ratio, "best_nn": idx.tolist()}


def _literal_shn(dirs, obs):
    """Review-1 defect: min-norm pinv over all 16 coefficients, then DC overwritten."""
    return core.fit_sh(dirs, obs)[:, 1:]


def sh_fit(method, device):
    """C4: composed init (DC = reference colour + residual bands) reproduces the reference colour
    exactly and the neighbour colour up to ridge shrinkage under gsplat's CUDA SH at degree 3;
    coefficients obey the 2|R| bound; the double-counting composition fails the same check."""
    from gsplat import spherical_harmonics

    g = torch.Generator().manual_seed(1)
    n = 256
    dirs = torch.nn.functional.normalize(torch.randn(n, 2, 3, generator=g), dim=-1)
    near = torch.nn.functional.normalize(dirs[:64, 0] + 0.01 * torch.randn(64, 3, generator=g), dim=-1)
    dirs[:64, 1] = near  # nearly parallel pairs (~0.6 deg)
    obs = torch.rand(n, 2, 3, generator=g) * 0.8 + 0.1
    sh0 = core.rgb_to_sh(obs[:, 0])[:, None]

    def render(shn):
        coeffs = torch.cat([sh0, shn], dim=1).repeat_interleave(2, 0).to(device)
        return (spherical_harmonics(3, dirs.reshape(-1, 3).to(device), coeffs) + 0.5).cpu().reshape(n, 2, 3)

    shn = core.fit_sh_residual(dirs, obs)
    out = render(shn)
    ref_err = float((out[:, 0] - obs[:, 0]).abs().max())
    resid = (obs[:, 1] - obs[:, 0]).abs()
    nn_far = float(((out[64:, 1] - obs[64:, 1]).abs() / resid[64:].clamp_min(1e-3)).median())
    bound = float((shn.norm(dim=1) / (obs[:, 1] - obs[:, 0]).abs().clamp_min(1e-9)).max())
    if ref_err > 1e-4:
        raise AssertionError(f"Composed init changes the reference colour by {ref_err:.2e}")
    if nn_far > 0.5:
        raise AssertionError(f"Neighbour residual not fitted (median relative error {nn_far:.2f})")
    if bound > 2.0 + 1e-4 or float(shn[:64].abs().max()) > 2 * 0.8 + 1e-4:
        raise AssertionError(f"SH coefficients exceed the ridge bound 2|R| ({bound:.3f})")
    literal = float((render(_literal_shn(dirs, obs))[:, 0] - obs[:, 0]).abs().median())
    if literal < 0.02:
        raise AssertionError("Probe cannot distinguish the double-counting composition")
    return {"ref_max_err": ref_err, "nn_median_relative_err_far": nn_far,
            "coef_over_residual_max": bound, "literal_ref_median_err": literal}


def _scene_cameras(root):
    """Independent camera parsing of transforms.json (OpenGL c2w -> OpenCV W2C)."""
    import json

    meta = json.loads((root / "transforms.json").read_text())
    frames = {f["file_path"]: f for f in meta["frames"]}
    names = sorted(frames)
    train = [n for n in names if n not in set(names[::8])]  # llffhold8
    cams = []
    for name in train:
        f = {**meta, **frames[name]}
        c2w = torch.tensor(f["transform_matrix"], dtype=torch.float64)
        c2w[:3, 1:3] *= -1
        cams.append({"name": name, "c2w": c2w, "fx": float(f["fl_x"]), "fy": float(f["fl_y"]),
                     "cx": float(f["cx"]), "cy": float(f["cy"])})
    return cams


def sh_composed(method, device):
    """C4 on a real prepared scene: each point's initial SH colour, rendered (gsplat CUDA SH,
    degree 3) toward the cameras of its observing training views, matches the pixels it projects
    to in those images. Fails on the double-counting composition."""
    import numpy as np
    from gsplat import spherical_harmonics
    from PIL import Image
    from splatstudio.verification import ProbeUnavailable

    root = Path.home() / "datasets/edgs-init/mipnerf360/garden"
    try:
        side = np.load(root / "edgs_init.npz", allow_pickle=False)
        need = {k: side[k] for k in ("shN", "ref_view", "obs_view", "obs", "dirs")}
        pts = np.load(root / "points.npz")
        points, colors = pts["points"], pts["colors"]
    except (OSError, KeyError) as exc:
        raise ProbeUnavailable(f"prepared garden scene with cached observations absent: {exc}")
    cams = _scene_cameras(root)
    views = sorted(set(need["ref_view"][:: max(1, len(points) // 40)].tolist()))[:3]
    sel = np.where(np.isin(need["ref_view"], views))[0][::10][:6000]
    x = torch.tensor(points[sel], dtype=torch.float64)
    sh0 = core.rgb_to_sh(torch.tensor(colors[sel]))[:, None]
    images = {}
    result = {}
    for label, shn in (("residual", torch.tensor(need["shN"][sel])),
                       ("literal", _literal_shn(torch.tensor(need["dirs"][sel]),
                                                torch.tensor(need["obs"][sel])))):
        errs = []
        for col, key in ((0, "ref_view"), (1, "obs_view")):
            pred, seen = [], []
            for v in np.unique(need[key][sel]):
                m = need[key][sel] == v
                cam = cams[int(v)]
                if cam["name"] not in images:
                    images[cam["name"]] = np.asarray(Image.open(root / cam["name"]).convert("RGB"),
                                                     dtype=np.float32) / 255
                img = images[cam["name"]]
                centre = cam["c2w"][:3, 3]
                pc = (x[m] - centre) @ cam["c2w"][:3, :3]
                u = cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"]
                w = cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]
                c = np.clip(np.floor(u.numpy()).astype(int), 0, img.shape[1] - 1)
                r = np.clip(np.floor(w.numpy()).astype(int), 0, img.shape[0] - 1)
                seen.append(torch.from_numpy(img[r, c]))
                coeffs = torch.cat([sh0[m], shn[m]], dim=1).to(device)
                pred.append((spherical_harmonics(3, (x[m] - centre).float().to(device), coeffs)
                             + 0.5).cpu())
            errs.append(float((torch.cat(pred) - torch.cat(seen)).abs().mean()))
        result[label] = {"ref_view_mae": errs[0], "obs_view_mae": errs[1]}
    r, lit = result["residual"], result["literal"]
    if r["ref_view_mae"] > 0.01:
        raise AssertionError(f"Initial SH colour differs from the reference pixels: {r}")
    if r["obs_view_mae"] > 0.1:
        raise AssertionError(f"Initial SH colour far from the neighbour observations: {r}")
    if lit["ref_view_mae"] < 2 * r["ref_view_mae"] + 0.02:
        raise AssertionError(f"Probe does not separate the double-counting composition: {result}")
    result["points"] = int(len(sel))
    return result


def no_densify(method, device):
    """C5: only pruning/decay, schedule boundaries, LR floor, opacity LR, sidecar application."""
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene

    _, density, _ = build(method)
    params, optimizers = sample_scene(method, device)
    n = len(params["means"])
    log_scales = torch.linspace(-6, -4, n)
    extras = {"log_scales": log_scales.numpy(), "opacity_logits": torch.zeros(n).numpy(),
              "shN": torch.full((n, 15, 3), 0.07).numpy()}
    density.init_override = extras
    runtime_lr = optimizers["means"].param_groups[0]["lr"]  # runtime: 1.6e-4 * scene_scale(=2)
    density.initialize(params, optimizers, 2.0)
    cfg = density.cfg
    cams = [{"viewmat": torch.eye(4, device=device)} for _ in range(3)]
    for c, x in zip(cams, (0.0, 1.0, 2.0)):
        c["viewmat"][0, 3] = -x  # centres at x = 0, 1, 2 -> extent 1.1
    density.set_training_cameras(cams)
    base_lr = runtime_lr / 2.0 * (1.1 if cfg["camera_extent_lr"] else 2.0)
    torch.testing.assert_close(params["scales"][:, 1].cpu(), log_scales)
    if float(params["opacities"].abs().max()) != 0.0:
        raise AssertionError("Initial opacity logits not applied")
    want_sh = 0.07 if cfg["sh_init"] else 0.0
    if abs(float(params["shN"].mean()) - want_sh) > 1e-6:
        raise AssertionError("SH-rest initialization not applied as configured")
    if optimizers["opacities"].param_groups[0]["lr"] != cfg["opacity_lr"]:
        raise AssertionError("Opacity LR not set")
    lr1 = optimizers["means"].param_groups[0]["lr"]
    expect = base_lr * 0.01 ** (8000 / 30000)
    if not math.isclose(lr1, expect, rel_tol=1e-9):
        raise AssertionError(f"Means LR at step 1 {lr1} != lr(8000) {expect}")
    with torch.no_grad():
        params["opacities"][:3] = -8.0  # sigmoid < 0.005 -> pruned before 15000
    before = params["opacities"][3:].clone()
    density.after_backward(params, optimizers, 10, {})
    check_state(method, params, optimizers)
    if len(params["means"]) != n - 3:
        raise AssertionError("Low-opacity Gaussians not pruned at step 10")
    torch.testing.assert_close(params["opacities"], before + math.log(0.99))
    density.after_backward(params, optimizers, 11, {})
    torch.testing.assert_close(params["opacities"], before + math.log(0.99))  # no decay at 11
    lr = optimizers["means"].param_groups[0]["lr"]
    if not math.isclose(lr, expect, rel_tol=1e-9):
        raise AssertionError("Means LR not floored before step 8000")
    density.after_backward(params, optimizers, 19999, {})
    if not math.isclose(optimizers["means"].param_groups[0]["lr"],
                        base_lr * 0.01 ** (20000 / 30000), rel_tol=1e-9):
        raise AssertionError("Means LR does not follow 3DGS decay after 8000")
    with torch.no_grad():
        params["opacities"][:2] = -8.0
    m = len(params["means"])
    snapshot = params["opacities"].clone()
    density.after_backward(params, optimizers, 15000, {})
    if len(params["means"]) != m or not torch.equal(params["opacities"], snapshot):
        raise AssertionError("Pruning/decay active at step 15000 (official stops before 15000)")
    for step in range(15001, 15050):
        density.after_backward(params, optimizers, step, {})
    if len(params["means"]) > m:
        raise AssertionError("Density control added Gaussians")
    return {"initial": n, "after": len(params["means"]), "means_lr_step1": lr1}


def roma_matching(method, device):
    """C1: RoMa installed with weights; warp convention (x, y) verified with a shifted crop."""
    from splatstudio.verification import ProbeUnavailable

    from .preprocess import DEFAULT_MODELS, load_roma, match_pair

    weights = DEFAULT_MODELS / "roma_outdoor.pth"
    dino = DEFAULT_MODELS / "dinov2_vitl14_pretrain.pth"
    scene = Path.home() / "datasets/mipnerf360/native/garden/images/DSC07956.JPG"
    if not (weights.exists() and dino.exists() and scene.exists()):
        raise ProbeUnavailable("RoMa weights or the garden test image are absent")
    try:
        import romatch  # noqa: F401
    except ImportError as exc:
        raise ProbeUnavailable(f"romatch not installed: {exc}") from exc
    import numpy as np
    from PIL import Image

    image = np.asarray(Image.open(scene).convert("RGB"), dtype=np.float32) / 255
    shift, H, W = 40, 400, 560
    a = torch.from_numpy(image[200 : 200 + H, 300 : 300 + W].copy())
    b = torch.from_numpy(image[200 : 200 + H, 300 + shift : 300 + shift + W].copy())
    roma = load_roma(device, weights, dino)
    cert, flow = match_pair(roma, a, b, H, W, device)
    # Pixel x in A appears at x - shift in B.
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                            indexing="ij")
    ax = (2 * xs + 1) / W - 1
    ay = (2 * ys + 1) / H - 1
    valid = (cert > 0.5) & (xs > shift + 10)
    dx = ((flow[..., 0] - ax)[valid] * W / 2).median().item()
    dy = ((flow[..., 1] - ay)[valid] * H / 2).median().item()
    if abs(dx + shift) > 2 or abs(dy) > 2 or valid.float().mean() < 0.3:
        raise AssertionError(f"RoMa flow convention wrong: dx {dx:.2f} dy {dy:.2f}")
    return {"median_dx_px": dx, "median_dy_px": dy, "valid_fraction": valid.float().mean().item()}
