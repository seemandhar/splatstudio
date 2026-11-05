"""Mechanism probes for the AGS-Mesh port. Each one compares the implementation with an
independently written formula or an analytic case, so removing or altering the mechanism fails.
scale_loss, gaussian_normals, density_recipe and sidecar_init are copied from ss_dnsplatter.probes
(reproductions/dn-splatter); dnc_filter, depth_schedule, normal_anr and surface_normals are new
(probes_ags.py)."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _component(method, name):
    from splatstudio.registry import resolve

    spec = next(c for c in method.components if c.name == name)
    return spec, resolve(spec.factory)(spec)


# ------------------------------------------------------------------------ losses
def scale_loss(method, device):
    spec, fn = _component(method, "ags-scale")
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


from .probes_ags import depth_schedule, dnc_filter, normal_anr, surface_normals  # noqa: E402,F401
