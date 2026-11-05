"""Mechanism probes for the DN-Splatter port. Each one compares the implementation with an
independently written formula or an analytic case, so removing or altering the mechanism fails."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _component(method, name):
    from splatstudio.registry import resolve

    spec = next(c for c in method.components if c.name == name)
    return spec, resolve(spec.factory)(spec)


# ------------------------------------------------------------------------ losses
def depth_loss(method, device):
    spec, fn = _component(method, "dn-depth")
    g = torch.Generator(device="cpu").manual_seed(0)
    H, W = 9, 11
    pred = (torch.rand(H, W, 1, generator=g) * 3).to(device).requires_grad_()
    gt = (torch.rand(H, W, 1, generator=g) * 3).to(device)
    gt[0, :3] = 0.05  # invalid (< 0.1 m)
    rgb = torch.rand(H, W, 3, generator=g).to(device)
    value = fn({}, {"depth": pred, "dn_view": {"depth": gt}}, rgb)
    # independent loop formula (losses.py:EdgeAwareLogL1 scalar, regularization_strategy 1.2)
    img = rgb.clamp(min=10 / 255).cpu()
    p, d = pred.detach().cpu()[..., 0], gt.cpu()[..., 0]
    lx, ly = [], []
    for i in range(H):
        for j in range(W):
            if d[i, j] <= 0.1:
                continue
            ll = math.log(1 + abs(p[i, j] - d[i, j]))
            if j < W - 1:
                lx.append(math.exp(-float((img[i, j] - img[i, j + 1]).abs().mean())) * ll)
            if i < H - 1:
                ly.append(math.exp(-float((img[i, j] - img[i + 1, j]).abs().mean())) * ll)
    expected = 1.2 * (sum(lx) / len(lx) + sum(ly) / len(ly))
    if abs(value.item() - expected) > 1e-5 * max(1.0, expected):
        raise AssertionError(f"depth loss {value.item()} != independent {expected}")
    # constant image: no edge weighting -> plain masked log-L1 (x and y halves)
    flat = torch.full_like(rgb, 0.5)
    v_flat = fn({}, {"depth": pred, "dn_view": {"depth": gt}}, flat).item()
    m = d > 0.1
    ll = torch.log(1 + (p - d).abs())
    plain = 1.2 * (ll[:, :-1][m[:, :-1]].mean() + ll[:-1][m[:-1]].mean()).item()
    if abs(v_flat - plain) > 1e-5:
        raise AssertionError("constant image must reduce to masked log-L1")
    if not v_flat > value.item():
        raise AssertionError("image edges must down-weight the depth loss")
    grad = torch.autograd.grad(value, pred)[0]
    if not torch.count_nonzero(grad) or grad[0, :3].abs().max() != 0:
        raise AssertionError("depth gradient missing or leaking into invalid pixels")
    return {"loss": value.item(), "independent": expected, "flat_image_loss": v_flat,
            "effective_weight": 1.2}


def normal_loss(method, device):
    spec, fn = _component(method, "dn-normal")
    g = torch.Generator(device="cpu").manual_seed(1)
    pred = torch.rand(7, 8, 3, generator=g).to(device).requires_grad_()
    gt = torch.rand(7, 8, 3, generator=g).to(device)
    value = fn({}, {"normal": pred, "dn_view": {"normal": gt}}, None)
    p, q = pred.detach().cpu().double(), gt.cpu().double()
    l1 = sum(abs(float(a)) for a in (p - q).flatten()) / p.numel()
    dw = [abs(float(p[i, j, c] - p[i, j + 1, c])) for i in range(7) for j in range(7)
          for c in range(3)]
    dh = [abs(float(p[i, j, c] - p[i + 1, j, c])) for i in range(6) for j in range(8)
          for c in range(3)]
    expected = l1 + sum(dw) / len(dw) + sum(dh) / len(dh)
    if abs(value.item() - expected) > 1e-5:
        raise AssertionError(f"normal loss {value.item()} != independent {expected}")
    smooth = fn({}, {"normal": gt.clone().requires_grad_(), "dn_view": {"normal": gt}}, None)
    tv_only = sum(abs(float(q[i, j, c] - q[i, j + 1, c])) for i in range(7) for j in range(7)
                  for c in range(3)) / 147 + sum(abs(float(q[i, j, c] - q[i + 1, j, c]))
                                                 for i in range(6) for j in range(8)
                                                 for c in range(3)) / 144
    if abs(smooth.item() - tv_only) > 1e-5 or tv_only <= 0:
        raise AssertionError("TV smoothing term missing")
    return {"loss": value.item(), "independent": expected, "tv_at_gt": smooth.item()}


def scale_loss(method, device):
    spec, fn = _component(method, "dn-scale")
    s = torch.tensor([[-1.0, -3.0, -2.0], [-0.5, -0.2, -4.0], [-1e4, -1e4, -1e4]], device=device,
                     requires_grad=True)
    value = fn({"scales": s}, {}, None)
    expected = (math.exp(-3.0) + math.exp(-4.0) + 0.0) / 3
    if abs(value.item() - expected) > 1e-6:
        raise AssertionError("scale loss != mean of min activated scale")
    grad = torch.autograd.grad(value, s)[0].cpu()
    want = torch.zeros(3, 3)
    want[0, 1], want[1, 2] = math.exp(-3.0) / 3, math.exp(-4.0) / 3
    torch.testing.assert_close(grad, want, atol=1e-7, rtol=1e-5)
    return {"loss": value.item(), "zero_scale_gradient": float(grad[2].abs().max())}


# ----------------------------------------------------------------------- renderer
def gaussian_normals(method, device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    from gsplat.utils import normalized_quat_to_rotmat

    from .renderer import Renderer

    r = Renderer(SimpleNamespace(config={}))
    r._table = None
    # one flat Gaussian (shortest axis = local y) tilted about x, 2 m in front of the camera
    ang = 0.6
    q = torch.tensor([[math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0]], device=device)
    params = {
        "means": torch.tensor([[0.0, 0.0, 2.0]], device=device, requires_grad=True),
        "quats": q.clone().requires_grad_(),
        "scales": torch.log(torch.tensor([[0.5, 0.01, 0.5]], device=device)).requires_grad_(),
        "opacities": torch.tensor([3.0], device=device, requires_grad=True),
        "sh0": torch.zeros(1, 1, 3, device=device, requires_grad=True),
        "shN": torch.zeros(1, 15, 3, device=device, requires_grad=True),
    }
    K = torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": 32, "height": 32}
    out = r.render(params, cam, sh_degree=0)
    n_world = normalized_quat_to_rotmat(q)[0][:, 1]
    if n_world[2] > 0:  # facing the camera at the origin means n . (0 - mu) > 0
        n_world = -n_world
    alpha = out["alpha"][..., 0].detach()
    D = torch.tensor([1.0, -1.0, -1.0], device=device)  # OpenCV <-> OpenGL camera frame

    def encode(v_cv, a):
        # official: composite in the OpenGL frame over the legacy white background (1, 1, 1),
        # normalise, map back to OpenCV with D, encode (n + 1) / 2
        v_gl = a * (D * v_cv) + (1 - a) * torch.ones_like(v_cv)
        return (D * (v_gl / v_gl.norm()) + 1) / 2

    a = alpha[16, 16]
    expect01 = encode(n_world, a)
    centre = out["normal"][16, 16]
    if (centre - expect01).abs().max() > 2e-3:
        raise AssertionError(f"centre normal {centre.tolist()} != {expect01.tolist()}")
    # partial-alpha pixel: the background colour (and its frame) matters here
    row = alpha[16]
    cand = torch.where((row > 0.2) & (row < 0.8))[0]
    if len(cand) == 0:
        raise AssertionError("probe scene has no partial-alpha pixel")
    j = int(cand[0])
    ap = alpha[16, j]
    got = out["normal"][16, j]
    want = encode(n_world, ap)
    wrong = ap * n_world + (1 - ap)  # white background in the OpenCV frame (old deviation)
    wrong = (wrong / wrong.norm() + 1) / 2
    if (got - want).abs().max() > 2e-3:
        raise AssertionError(f"partial-alpha normal {got.tolist()} != official {want.tolist()}")
    if (want - wrong).abs().max() < 2e-2:
        raise AssertionError("probe pixel does not discriminate the background frame")
    # camera behind the Gaussian: normal must flip to face it (identity rotation, centre z=4)
    vm = torch.eye(4, device=device)
    vm[:3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device))
    vm[2, 3] = 4.0  # camera at world z=4 looking -z
    back = r.render(params, {**cam, "viewmat": vm}, sh_degree=0)["normal"][16, 16]
    n_back = -n_world if (-n_world)[2] > 0 else n_world
    expect_back = encode(vm[:3, :3] @ n_back, a)
    if (back - expect_back).abs().max() > 2e-3:
        raise AssertionError("normal not flipped towards the camera")
    loss = (out["normal"] - torch.tensor([0.5, 0.5, 0.0], device=device)).abs().mean()
    loss.backward()
    if params["quats"].grad is None or not torch.count_nonzero(params["quats"].grad):
        raise AssertionError("normal loss does not reach the quaternions")
    m2 = out["info"]["means2d"]
    if m2.grad is not None and torch.count_nonzero(m2.grad):
        raise AssertionError("normal pass must use detached 2D means")
    return {"centre_normal": centre.tolist(), "expected": expect01.tolist(),
            "partial_alpha": float(ap), "partial_normal": got.tolist(),
            "partial_expected": want.tolist(), "partial_wrong_frame": wrong.tolist(),
            "back_normal": back.tolist(), "quat_grad_norm": params["quats"].grad.norm().item()}


# ------------------------------------------------------------------------ density
def _fake_info(n, device, grads, radii, H=100, W=200):
    m2 = torch.zeros(1, n, 2, device=device)
    m2.absgrad = torch.stack([grads, torch.zeros_like(grads)], -1)[None]
    r = torch.stack([radii, radii], -1)[None].to(torch.int32)
    # conics of an isotropic blob with variance v: (1/v, 0, 1/v); v1 radius = ceil(3 sqrt(v+...))
    v = (radii.float() / 3) ** 2
    conics = torch.stack([1 / v.clamp_min(1e-6), torch.zeros_like(v), 1 / v.clamp_min(1e-6)],
                         -1)[None]
    return {"means2d": m2, "radii": r, "conics": conics, "width": W, "height": H}


def density_recipe(method, device):
    from splatstudio.state import check_state, sample_scene
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    from .density import DEFAULTS, Density

    d = Density(SimpleNamespace(config={**DEFAULTS, "use_sidecar_init": False}))
    params, opts = sample_scene(method, device, n=6)
    with torch.no_grad():
        params["scales"].copy_(torch.log(torch.tensor(
            [[0.02] * 3, [0.012] * 3, [0.005] * 3, [0.005] * 3, [0.6] * 3, [0.005] * 3],
            device=device)))
        params["opacities"].copy_(torch.logit(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.05],
                                                           device=device)))
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    opts["means"].param_groups[0]["lr"] = 123.0
    d.initialize(params, opts, scene_scale=7.0)
    if opts["means"].param_groups[0]["lr"] != 1.6e-4:
        raise AssertionError("means LR must be reset to the unscaled 1.6e-4")
    d.set_training_cameras([None] * 50)
    H, W = 100, 200
    # per-step absgrad g -> avg = g*k/(k+1) * 0.5*max(H,W); threshold 0.0008
    hi, lo = 0.0008 / 100 * 1.5, 0.0008 / 100 * 0.5
    grads = torch.tensor([hi, hi, hi, lo, hi, lo], device=device)
    radii = torch.tensor([2, 2, 2, 2, 2, 2], device=device)
    for step in range(1, 602):  # ns = 600 refines (600 % 3000 = 600 > 50 + 100)
        info = _fake_info(6, device, grads, radii, H, W)
        d.after_backward(params, opts, step, {"info": info})
    check_state(method, params, opts)
    # 0: split (0.02 > 0.01) -> 2 children; 1: split, 0.012/1.6 <= 0.01 -> also duplicated
    # 2: dup; 3: low grad; 4: scale 0.6 -> split (cull by size only after 3000); 5: culled alpha
    if (d.state["last_split"], d.state["last_dup"]) != (3, 2):
        raise AssertionError(f"split/dup {d.state['last_split']}/{d.state['last_dup']} != 3/2")
    # 6 + 6 children + 2 dups - 3 split originals - 1 low-alpha = 10
    if len(params["means"]) != 10:
        raise AssertionError(f"population {len(params['means'])} != 10")
    n_after = len(params["means"])
    # ns 601..3000 with low gradients: nothing densifies, nothing is culled (alpha 0.5, before
    # the size cull); ns 3001..3100 with high gradients: ns = 3100 lies in the post-reset window
    # (3100 % 3000 = 100 <= num_train + 100) -> no densification, but the opacity reset fires.
    for step in range(602, 3102):
        g_ = hi if step > 3001 else lo
        info = _fake_info(n_after, device, torch.full((n_after,), g_, device=device),
                          torch.full((n_after,), 2, device=device), H, W)
        d.after_backward(params, opts, step, {"info": info})
    if len(params["means"]) != n_after:
        raise AssertionError("densification must be suspended right after an opacity reset")
    op = params["opacities"].sigmoid()
    if op.max() > 0.2 + 1e-6:
        raise AssertionError("opacity reset must clamp opacities to 0.2")
    st = opts["opacities"].state[params["opacities"]]
    if torch.count_nonzero(st["exp_avg"]) or torch.count_nonzero(st["exp_avg_sq"]):
        raise AssertionError("opacity reset must zero the Adam moments")
    check_state(method, params, opts)
    return {"split": 3, "dup": 2, "population_after_600": n_after,
            "population_after_3100": len(params["means"]), "max_opacity": float(op.max())}


def sidecar_init(method, device):
    from splatstudio.state import sample_scene

    from . import density as dmod

    params, opts = sample_scene(method, device, n=5)
    pts = params["means"].detach().cpu().clone()
    ls = torch.log(torch.tensor([[0.1, 0.1, 0.01]] * 5))
    ls[2] = -float("inf")
    quats = F.normalize(torch.randn(5, 4, generator=torch.Generator().manual_seed(3)), dim=-1)

    class FakeTable:
        def init(self):
            return {"points": pts, "log_scales": ls, "quats": quats}

    orig = dmod.sidecar.table
    dmod.sidecar.table = lambda root=None: FakeTable()
    try:
        d = dmod.Density(SimpleNamespace(config=dict(dmod.DEFAULTS)))
        d.initialize(params, opts)
        bad = params["means"].detach().clone()
        bad[0, 0] += 1
        p2, o2 = sample_scene(method, device, n=5)
        with torch.no_grad():
            p2["means"].copy_(bad)
        try:
            dmod.Density(SimpleNamespace(config=dict(dmod.DEFAULTS))).initialize(p2, o2)
            raise AssertionError("mismatched sidecar points must be rejected")
        except ValueError:
            pass
    finally:
        dmod.sidecar.table = orig
    if not d.state["init_from_sidecar"]:
        raise AssertionError("sidecar init not applied")
    torch.testing.assert_close(params["quats"].detach().cpu(), quats)
    s = params["scales"].detach().cpu()
    if not torch.isfinite(s).all() or s[2].exp().max() != 0:
        raise AssertionError("-inf log-scale must become a finite exact-zero scale")
    torch.testing.assert_close(s[0], ls[0])
    return {"applied": True, "zero_scale_row": s[2].tolist()}


def isotropic_init(method, device):
    """Vanilla Splatfacto init from the sidecar points (splatfacto.py:214-220)."""
    from splatstudio.state import sample_scene

    from . import density as dmod

    params, opts = sample_scene(method, device, n=6)
    pts = params["means"].detach().cpu().clone()
    avg = torch.tensor([0.1, 0.05, 0.0, 0.2, 0.03, 0.07])
    log_avg = torch.log(avg)  # zero distance -> -inf, as official
    ls = torch.stack([log_avg, log_avg, torch.log(avg / 10)], -1)  # oriented sidecar layout
    oriented_q = torch.tensor([[1.0, 0, 0, 0]]).repeat(6, 1)

    class FakeTable:
        def init(self):
            return {"points": pts, "log_scales": ls, "quats": oriented_q}

    spec = next(c for c in method.components if c.name == "splatfacto-density")
    orig = dmod.sidecar.table
    dmod.sidecar.table = lambda root=None: FakeTable()
    try:
        torch.manual_seed(1234)
        d = dmod.Density(spec)
        d.initialize(params, opts)
    finally:
        dmod.sidecar.table = orig
    if d.cfg["sidecar_init"] != "isotropic" or not d.state["init_from_sidecar"]:
        raise AssertionError("control must use the isotropic sidecar init")
    s = params["scales"].detach().cpu()
    want = torch.where(torch.isfinite(log_avg), log_avg, torch.tensor(d.cfg["zero_log_scale"]))
    torch.testing.assert_close(s, want[:, None].repeat(1, 3))
    if s[2].exp().max() != 0:
        raise AssertionError("zero kNN distance must give an exact-zero scale")
    # independent Shoemake formula on the same CPU RNG stream
    torch.manual_seed(1234)
    u, v, w = torch.rand(6), torch.rand(6), torch.rand(6)
    q = torch.stack([(1 - u).sqrt() * torch.sin(2 * math.pi * v),
                     (1 - u).sqrt() * torch.cos(2 * math.pi * v),
                     u.sqrt() * torch.sin(2 * math.pi * w), u.sqrt() * torch.cos(2 * math.pi * w)],
                    -1)
    got = params["quats"].detach().cpu()
    torch.testing.assert_close(got, q)
    torch.testing.assert_close(got.norm(dim=-1), torch.ones(6))
    if torch.allclose(got, oriented_q):
        raise AssertionError("quaternions must be random, not the oriented sidecar quats")
    return {"log_scales": s[:, 0].tolist(), "quats_unit": True}


def resolution_schedule(method, device):
    """Splatfacto num_downscales = 2 / resolution_schedule = 3000: 1/4 res for steps 0-2999, 1/2
    for 3000-5999, full afterwards (0-based); eval full res; target area-downscaled."""
    from splatstudio.registry import resolve
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    rspec = next(c for c in method.components if c.name == "dn-render")
    r = resolve(rspec.factory)(rspec)
    r._table = None
    if (r.cfg["num_downscales"], r.cfg["resolution_schedule"]) != (2, 3000):
        raise AssertionError("splatfacto control must use num_downscales 2 / schedule 3000")
    params = {
        "means": torch.tensor([[0.1, -0.05, 2.0]], device=device, requires_grad=True),
        "quats": torch.tensor([[1.0, 0, 0, 0]], device=device, requires_grad=True),
        "scales": torch.log(torch.full((1, 3), 0.1, device=device)).requires_grad_(),
        "opacities": torch.tensor([3.0], device=device, requires_grad=True),
        "sh0": torch.zeros(1, 1, 3, device=device, requires_grad=True),
        "shN": torch.zeros(1, 15, 3, device=device, requires_grad=True),
    }
    H, W = 70, 90
    K = torch.tensor([[80.0, 0, 44.0], [0, 80.0, 35.0], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": W, "height": H}
    sizes = {}
    for k, d in ((0, 4), (2999, 4), (3000, 2), (5999, 2), (6000, 1), (20000, 1)):
        r.train_renders = k
        out = r.render(params, cam, sh_degree=0)
        if out["downscale"] != d or out["rgb"].shape[:2] != (H // d, W // d):
            raise AssertionError(f"step {k}: got {out['downscale']} {tuple(out['rgb'].shape)}")
        if r.train_renders != k + 1:
            raise AssertionError("training render counter not advanced")
        # projected centre follows the scaled intrinsics (fx/d, cx/d)
        m2 = out["info"]["means2d"][0, 0].detach()
        want = torch.tensor([(80 * 0.05 + 44.0) / d, (-80 * 0.025 + 35.0) / d], device=device)
        if (m2 - want).abs().max() > 1e-3:
            raise AssertionError(f"step {k}: means2d {m2.tolist()} != {want.tolist()}")
        sizes[k] = list(out["rgb"].shape[:2])
    with torch.no_grad():
        ev = r.render(params, cam, sh_degree=0)
    if ev["downscale"] != 1 or ev["rgb"].shape[:2] != (H, W):
        raise AssertionError("evaluation must render at full resolution")
    # photometric target downscale = OpenCV area (independent: avg_pool2d)
    lspec = next(c for c in method.components if c.name == "photometric")
    loss = resolve(lspec.factory)(lspec)
    g = torch.Generator(device="cpu").manual_seed(5)
    target = torch.rand(H, W, 3, generator=g).to(device)
    rgb = torch.rand(H // 4, W // 4, 3, generator=g).to(device).requires_grad_()
    small = F.avg_pool2d(target.permute(2, 0, 1)[None], 4)[0].permute(1, 2, 0)
    from splatstudio.losses import create_photometric

    ref = create_photometric(SimpleNamespace(config=dict(lspec.config)))({}, {"rgb": rgb}, small)
    val = loss({}, {"rgb": rgb, "downscale": 4}, target)
    if abs(val.item() - ref.item()) > 1e-5:
        raise AssertionError(f"downscaled photometric {val.item()} != area reference {ref.item()}")
    try:
        loss({}, {"rgb": rgb, "downscale": 1}, target)
        raise AssertionError("size mismatch must be rejected")
    except ValueError:
        pass
    return {"sizes": sizes, "loss": val.item()}


def renderer_gradient_scheduled(method, device):
    """Base renderer-gradient probe for a renderer with a resolution schedule. The base probe
    compares the first (training) render with a camera-resolution target, so it is run on the
    same method with num_downscales = 0; the scheduled 1/4-resolution step is then checked for
    finite nonzero signed and absolute screen-space gradients."""
    from dataclasses import replace

    from splatstudio.probes import renderer_gradient
    from splatstudio.registry import resolve
    from splatstudio.state import sample_scene

    comps = tuple(replace(c, config={**c.config, "num_downscales": 0}) if c.name == "dn-render"
                  else c for c in method.components)
    evidence = renderer_gradient(replace(method, components=comps), device)
    rspec = next(c for c in method.components if c.name == "dn-render")
    r = resolve(rspec.factory)(rspec)
    params, _ = sample_scene(method, device)
    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[80.0, 0, 32], [0, 80.0, 32], [0, 0, 1]], device=device),
           "width": 64, "height": 64}
    out = r.render(params, cam, sh_degree=0)
    if out["downscale"] <= 1:
        raise AssertionError("first training render must be downscaled")
    out["info"]["means2d"].retain_grad()
    target = torch.rand_like(out["rgb"])
    (out["rgb"] - target).square().mean().backward()
    m2 = out["info"]["means2d"]
    for g in (m2.grad, getattr(m2, "absgrad", None)):
        if g is None or not torch.isfinite(g).all() or not torch.count_nonzero(g):
            raise AssertionError("scheduled render has no finite nonzero screen-space gradient")
    evidence["scheduled_rgb_shape"] = list(out["rgb"].shape)
    return evidence
