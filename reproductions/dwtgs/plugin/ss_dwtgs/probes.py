"""Mechanism probes for DWTGS; each raises if its mechanism is removed or wrong."""


import numpy as np
import torch


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _has(method, name):
    return any(c.name == name for c in method.components)


def _haar_reference(x, k):
    """Independent per-block loop with 1D filters lo = [1, 1], hi = [1, -1] and 2D divisor k."""
    lo, hi = (1.0, 1.0), (1.0, -1.0)
    c, h, w = x.shape
    hp, wp = h + h % 2, w + w % 2
    xp = torch.zeros(c, hp, wp, dtype=torch.float64)
    xp[:, :h, :w] = x.double()
    out = {b: torch.zeros(c, hp // 2, wp // 2, dtype=torch.float64) for b in ("LL", "LH", "HL", "HH")}
    filt = {"LL": (lo, lo), "LH": (hi, lo), "HL": (lo, hi), "HH": (hi, hi)}  # (along y, along x)
    for b, (fy, fx) in filt.items():
        for i in range(hp // 2):
            for j in range(wp // 2):
                acc = 0.0
                for dy in range(2):
                    for dx in range(2):
                        acc = acc + fy[dy] * fx[dx] * xp[:, 2 * i + dy, 2 * j + dx]
                out[b][:, i, j] = acc / k
    return out


PAPER_NORM = "average"  # §IV-B: LL "in the same space as the original image, only coarser"


def haar_dwt_probe(method, device):
    """Eq. 2 with Haar filters in the paper's averaging form (LL = 2x2 block mean): subbands match
    an independent loop (odd size zero-padded), LL equals avg-pooling, a = LL + LH + HL + HH
    reconstructs the image, HH is the diagonal subband; the configured norm must be the paper's."""
    from .wavelet import haar_dwt

    for name in ("dwtgs-lf", "dwtgs-hf"):
        if _has(method, name) and _spec(method, name).config.get("haar_norm") != PAPER_NORM:
            raise AssertionError(f"{name} haar_norm is not the paper's averaging form")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 9, 12, generator=g)
    ll, lh, hl, hh = haar_dwt(x, PAPER_NORM)
    ref = _haar_reference(x, 4.0)
    err = max(float((a.double() - ref[b]).abs().max())
              for a, b in zip((ll, lh, hl, hh), ("LL", "LH", "HL", "HH")))
    if err > 1e-5:
        raise AssertionError(f"Haar subbands differ from the averaging reference by {err}")
    xe = torch.rand(3, 8, 10, generator=g)
    a, b_, c, d = haar_dwt(xe, PAPER_NORM)
    pool = torch.nn.functional.avg_pool2d(xe[None], 2)[0]
    if float((a - pool).abs().max()) > 1e-6:
        raise AssertionError("LL is not the 2x2 block mean (not in the image's value space)")
    rec = torch.zeros_like(xe)
    rec[:, 0::2, 0::2] = a + b_ + c + d
    rec[:, 0::2, 1::2] = a + b_ - c - d
    rec[:, 1::2, 0::2] = a - b_ + c - d
    rec[:, 1::2, 1::2] = a - b_ - c + d
    if float((rec - xe).abs().max()) > 1e-5:
        raise AssertionError("averaging Haar subbands do not reconstruct the image")
    yy, xx = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    ramp = haar_dwt(xx[None], PAPER_NORM)[3]
    cb = haar_dwt(((xx + yy) % 2)[None], PAPER_NORM)
    if float(ramp.abs().max()) > 1e-6 or abs(float(cb[3].abs().mean()) - 0.5) > 1e-6:
        raise AssertionError("HH is not the diagonal (high-high) subband")
    return {"max_abs_error": err, "shape_odd": list(ll.shape), "norm": PAPER_NORM}


def lf_loss_probe(method, device):
    """Eq. 5: sum_n lambda_LL L1(LL_n, LL_n_gt) with paper N = 2, lambda = 0.5; blind to a
    level-1 HH checkerboard (no HF supervision), sensitive to a low-frequency change."""
    from .losses import create_lf

    cfg = _spec(method, "dwtgs-lf").config
    if (float(cfg["weight"]), int(cfg["levels"]), cfg["haar_norm"]) != (0.5, 2, PAPER_NORM):
        raise AssertionError(f"LF config differs from the paper: {cfg}")
    loss = create_lf(_spec(method, "dwtgs-lf"))
    g = torch.Generator().manual_seed(1)
    gt = torch.rand(30, 22, 3, generator=g)
    render = (gt + 0.1 * torch.randn(30, 22, 3, generator=g)).requires_grad_(True)
    got = loss(None, {"rgb": render, "step": 1}, gt)
    ref = 0.0
    r, t = render.detach().permute(2, 0, 1).double(), gt.permute(2, 0, 1).double()
    for _ in range(2):  # independent LL: 2x2 block mean, zero-padded
        rr, tt = (_haar_reference(v, 4.0)["LL"] for v in (r, t))
        ref += 0.5 * float((rr - tt).abs().mean())
        r, t = rr, tt
    if abs(float(got) - ref) > 1e-5:
        raise AssertionError(f"LF loss {float(got)} != reference {ref}")
    got.backward()
    if render.grad is None or float(render.grad.abs().sum()) == 0:
        raise AssertionError("LF loss is not attached to the render")
    yy, xx = torch.meshgrid(torch.arange(30.0), torch.arange(22.0), indexing="ij")
    cb = (((xx + yy) % 2) * 2 - 1)[..., None] * 0.2  # pure level-1 HH content
    base = float(loss(None, {"rgb": gt, "step": 1}, gt))
    hf = float(loss(None, {"rgb": gt + cb, "step": 1}, gt))
    lf = float(loss(None, {"rgb": gt + 0.2, "step": 1}, gt))
    if abs(hf - base) > 1e-6:
        raise AssertionError("LF loss reacts to HH-only content (supervises high frequencies)")
    if lf < 0.1:
        raise AssertionError("LF loss does not react to a low-frequency change")
    return {"loss": float(got), "reference": ref, "hh_checkerboard": hf, "dc_shift": lf}


def hf_schedule_probe(method, device):
    """§V-A: novel-view HF loss every 10 iterations, stopped after 5000; zero but attached
    otherwise (and absent from the dwtgs-lf ablation)."""
    from .losses import create_hf, hf_active

    if not _has(method, "dwtgs-hf"):
        raise AssertionError("dwtgs-hf loss missing")
    cfg = _spec(method, "dwtgs-hf").config
    i, s0, s1 = int(cfg["interval"]), int(cfg["start"]), int(cfg["stop"])
    if (i, s1, float(cfg["weight"]), cfg["haar_norm"], cfg["novel_views"]) != (
        10, 5000, 1.0, PAPER_NORM, "interp"
    ):
        raise AssertionError(f"HF schedule differs from the paper: {cfg}")
    on = [t for t in range(1, 10001) if hf_active(t, i, s0, s1)]
    if on[0] != 10 or on[-1] != 5000 or len(on) != 500:
        raise AssertionError(f"wrong active steps: first {on[0]}, last {on[-1]}, n {len(on)}")
    loss = create_hf(_spec(method, "dwtgs-hf"))
    rgb = torch.rand(8, 8, 3, requires_grad=True)
    z = loss(None, {"rgb": rgb, "step": 11}, None)
    if float(z) != 0.0 or not z.requires_grad:
        raise AssertionError("inactive HF loss must be a graph-attached zero")
    return {"active_steps": len(on), "first": on[0], "last": on[-1]}


def _angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _train_cams(n=3, seed=0):
    rng = np.random.default_rng(seed)
    cams = []
    for _ in range(n):
        a = rng.normal(size=3) * 0.15
        th = np.linalg.norm(a)
        kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]) / th
        R = np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx  # Rodrigues
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, rng.normal(size=3)
        cams.append({"viewmat": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32),
                     "K": torch.tensor([[400.0, 0, 252], [0, 400.0, 189], [0, 0, 1]]),
                     "width": 504, "height": 378})
    return cams


def pseudo_poses_probe(method, device):
    """Fig. 2 note 3: every novel view lies on the interpolation path between two training views
    (centre on the segment, rotation on the geodesic at the same fraction t), is not itself a
    training view, and successive draws differ."""
    from .losses import HF_DEFAULTS, InterpViews, _check

    cfg = _check(dict(_spec(method, "dwtgs-hf").config), HF_DEFAULTS, "dwtgs-hf")
    if cfg["novel_views"] != "interp":
        raise AssertionError("novel views are not interpolated between training views")
    cams = _train_cams()
    c2ws = [np.linalg.inv(c["viewmat"].double().numpy()) for c in cams]
    views = InterpViews(cfg)
    views.build(cams, None)
    worst_c, worst_r, ts = 0.0, 0.0, []
    prev = None
    for _ in range(200):
        cam = views.next()
        c2w = np.linalg.inv(cam["viewmat"].double().numpy())
        best = None
        for i in range(len(c2ws)):
            for j in range(len(c2ws)):
                if i == j:
                    continue
                pa, pb, p = c2ws[i][:3, 3], c2ws[j][:3, 3], c2w[:3, 3]
                t = float(np.dot(p - pa, pb - pa) / np.dot(pb - pa, pb - pa))
                dc = float(np.linalg.norm(pa + t * (pb - pa) - p))
                ra, rb, r = c2ws[i][:3, :3], c2ws[j][:3, :3], c2w[:3, :3]
                dr = abs(_angle(ra, r) - t * _angle(ra, rb)) + abs(_angle(r, rb) - (1 - t) * _angle(ra, rb))
                if 0 <= t <= 1 and (best is None or dc + dr < best[0] + best[1]):
                    best = (dc, dr, t)
        if best is None:
            raise AssertionError("novel view is not between any two training views")
        worst_c, worst_r = max(worst_c, best[0]), max(worst_r, best[1])
        ts.append(best[2])
        if prev is not None and torch.equal(prev, cam["viewmat"]):
            raise AssertionError("successive novel views are identical")
        prev = cam["viewmat"]
    if worst_c > 1e-4 or worst_r > 1e-3:
        raise AssertionError(f"novel views leave the interpolation path ({worst_c}, {worst_r})")
    if min(ts) < 0 or max(ts) > 1 or float(np.std(ts)) < 0.15:
        raise AssertionError("interpolation fraction is not spread over (0, 1)")
    return {"n": len(ts), "max_centre_err": worst_c, "max_rot_err": worst_r,
            "t_mean": float(np.mean(ts)), "t_std": float(np.std(ts))}


def hf_novel_view_probe(method, device):
    """The HF term renders a novel camera (not the training one), equals mean|HH_1| of an
    independent all-Gaussian black-background render of that camera, and back-propagates."""
    from gsplat import rasterization
    from splatstudio.verification import ProbeUnavailable
    from ss_dropgaussian.probes import _fixture_scene

    from .density import create_density
    from .losses import create_hf

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    params, camera = _fixture_scene(device)
    loss = create_hf(_spec(method, "dwtgs-hf"))
    density = create_density(_spec(method, "dropgaussian-density"))
    density.initialize(params, {}, 1.0)
    cams = []
    for dx in (-0.3, 0.0, 0.3):
        vm = camera["viewmat"].clone()
        vm[0, 3] = dx
        cams.append({**camera, "viewmat": vm})
    density.set_training_cameras(cams)
    train_rgb = torch.zeros(96, 96, 3, device=device, requires_grad=True)
    outputs = {"rgb": train_rgb, "step": 10, "dwtgs": density}
    value = loss(params, outputs, None)
    cam = loss.stats["last_camera"]
    if min(float((cam["viewmat"] - c["viewmat"]).abs().max()) for c in cams) < 1e-4:
        raise AssertionError("HF loss rendered a training camera, not a novel view")
    with torch.no_grad():
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        ref, _, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                  params["opacities"].sigmoid(), colors, cam["viewmat"][None],
                                  cam["K"][None], cam["width"], cam["height"], sh_degree=0,
                                  packed=False)
        img = ref[0].clamp(0, 1).permute(2, 0, 1)  # independent averaging HH: (a-b-c+d)/4
        hh = (img[:, 0::2, 0::2] - img[:, 0::2, 1::2] - img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) / 4
        expect = float(hh.abs().mean())
    if abs(float(value) - expect) > 1e-4 * max(1.0, expect):
        raise AssertionError(f"HF loss {float(value)} != mean|HH_1| of the novel render {expect}")
    if expect <= 0:
        raise AssertionError("fixture novel view has no HH content")
    value.backward()
    g = float(params["sh0"].grad.abs().sum()) if params["sh0"].grad is not None else 0.0
    gm = float(params["means"].grad.abs().sum()) if params["means"].grad is not None else 0.0
    if g == 0 or gm == 0:
        raise AssertionError("HF loss does not back-propagate into the Gaussians")
    if train_rgb.grad is not None and float(train_rgb.grad.abs().sum()) != 0:
        raise AssertionError("HF loss used the training render")
    return {"loss": float(value), "independent": expect, "grad_sh0": g, "grad_means": gm}
