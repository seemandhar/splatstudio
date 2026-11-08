"""Mechanism probes for nerf-init-gs. Each raises if its mechanism is removed or wrong.

`hashgrid`, `proposal_sampler` and the sphere scene of `ingp_fit` are adapted from the SPLATIFY
ConeGS probes (reproductions/conegs/plugin/ss_conegs/probes.py).
"""

import math

import torch
from splatstudio.components import build
from splatstudio.state import check_state, sample_scene

from . import ingp as ingp_mod
from . import sidecar
from .density import NerfInitDensity
from .hashgrid import HashGrid, reference_encode
from .ingp import (
    INGP,
    RayCaster,
    expected_depth,
    importance_sampling,
    pdf_loss,
    s_to_t,
    surface_samples,
    t_to_s,
)
from .losses import depth_weight


def _spec(method, kind, name=None):
    return next(c for c in method.components if c.kind == kind and (name is None or c.name == name))


def _need_cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")


def hashgrid(method, device):
    """CUDA hash encoding equals the independent torch reference (dense and hashed levels)."""
    _need_cuda()
    torch.manual_seed(0)
    grid = HashGrid(16, 19, 16, ingp_mod.per_level(16, 4096, 16)).to(device)
    grid.params.data.uniform_(-1, 1)
    res = grid.resolutions.tolist()
    sizes = (grid.offsets[1:] - grid.offsets[:-1]).tolist()
    dense = [r**3 <= s for r, s in zip(res, sizes)]
    if not (any(dense) and not all(dense)):
        raise AssertionError("Expected both dense and hashed levels")
    if res[0] != 16 or res[-1] != 4096 or sizes[-1] != 1 << 19:
        raise AssertionError(f"tcnn level layout mismatch: res {res[0]}..{res[-1]}")
    x = torch.rand(4096, 3, device=device)
    y = grid(x)
    ref = reference_encode(x, grid.params, grid.offsets, grid.scales, grid.resolutions)
    fwd = (y - ref).abs().max().item()
    g = torch.randn_like(y)
    ga = torch.autograd.grad((y * g).sum(), grid.params)[0]
    gr = torch.autograd.grad((ref * g).sum(), grid.params)[0]
    bwd = (ga - gr).abs().max().item()
    if fwd > 2e-3 or bwd > 2e-3:
        raise AssertionError(f"hash kernel differs from reference: fwd {fwd} bwd {bwd}")
    return {"forward_max_abs": fwd, "backward_max_abs": bwd, "levels": len(res),
            "dense_levels": int(sum(dense))}


def proposal_sampler(method, device):
    """nerfacc importance_sampling docstring cases, lindisp endpoints/inverse, pdf loss."""
    e1 = torch.tensor([[0.0, 1.0]], device=device)
    c1 = torch.tensor([[0.0, 0.5]], device=device)
    edges, samples = importance_sampling(e1, c1, 2, False)
    torch.testing.assert_close(edges, torch.tensor([[0.0, 0.5, 1.0]], device=device))
    torch.testing.assert_close(samples, torch.tensor([[0.25, 0.75]], device=device))
    e2 = torch.tensor([[0.0, 1.0, 2.0]], device=device)
    c2 = torch.tensor([[0.0, 0.5, 1.0]], device=device)
    _, samples = importance_sampling(e2, c2, 2, False)
    torch.testing.assert_close(samples, torch.tensor([[0.5, 1.5]], device=device))
    t = s_to_t(torch.tensor([0.0, 0.3, 1.0], device=device), 0.2, 1e3)
    torch.testing.assert_close(t[[0, 2]], torch.tensor([0.2, 1e3], device=device), rtol=1e-4,
                               atol=1e-4)
    torch.testing.assert_close(t_to_s(t, 0.2, 1e3), torch.tensor([0.0, 0.3, 1.0], device=device),
                               rtol=1e-4, atol=1e-5)
    e3 = torch.linspace(0, 1, 11, device=device)[None]
    c3 = torch.tensor([[0, 0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1.0]], device=device)
    loss0 = pdf_loss(e3, c3, e3, c3).sum().item()
    loss1 = pdf_loss(e3, c3, e3, e3).sum().item()
    if loss0 > 1e-9 or loss1 <= 1e-3:
        raise AssertionError(f"pdf loss wrong: same {loss0}, flat proposal {loss1}")
    return {"pdf_loss_same": loss0, "pdf_loss_flat": loss1}


class _TwoSurfaceINGP(INGP):
    """Analytic field: a semi-transparent surface (alpha a1) at s in [0.3, 0.4] and an opaque
    one at s in [0.6, 0.7] along every ray (lindisp space), constant colour."""

    def __init__(self, a1=0.3):
        super().__init__()
        self.a1 = a1
        self.edges_s = torch.linspace(0, 1, 11)

    def sample(self, origins, dirs, stratified, requires_grad=False, cache=None):
        t = s_to_t(self.edges_s.to(origins.device), self.cfg["near"], self.cfg["far"])
        t = t.expand(len(origins), -1)
        return t[:, :-1], t[:, 1:]

    def render(self, origins, dirs, ts, te, view_dirs=True):
        alphas = torch.zeros_like(ts)
        alphas[:, 3] = self.a1
        alphas[:, 6] = 1.0
        trans = torch.cumprod(torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas[:, :-1]], 1),
                              1)
        return None, {"trans": trans, "alphas": alphas, "mid": (ts + te) / 2}


def surface_sampling(method, device):
    """Eq. 4: one inverse-CDF sample per ray follows the volume-rendering weights.

    Two surfaces with weights 0.3 / 0.7: the sample fractions match, every sample lies inside a
    surface interval (in s), and the returned depth is Eq. 6. A median/argmax placement (ConeGS)
    would put 100% on the second surface; a uniform sampler would put 80% outside both."""
    torch.manual_seed(0)
    model = _TwoSurfaceINGP().to(device)
    model.field = _ConstantField().to(device)
    n = 20000
    o = torch.zeros(n, 3, device=device)
    d = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
    gen = torch.Generator(device=device).manual_seed(0)
    pts, rgb, t, depth = surface_samples(model, o, d, generator=gen)
    s = t_to_s(t, model.cfg["near"], model.cfg["far"])
    first = ((s >= 0.3 - 1e-4) & (s <= 0.4 + 1e-4)).float().mean().item()
    second = ((s >= 0.6 - 1e-4) & (s <= 0.7 + 1e-4)).float().mean().item()
    if abs(first - 0.3) > 0.02 or abs(second - 0.7) > 0.02 or first + second < 0.999:
        raise AssertionError(f"surface sample fractions {first:.3f}/{second:.3f} != 0.3/0.7")
    torch.testing.assert_close(pts, o + d * t[:, None])
    edges_t = s_to_t(model.edges_s.to(device), model.cfg["near"], model.cfg["far"])
    mid = (edges_t[:-1] + edges_t[1:]) / 2
    ref_depth = 0.3 * mid[3] + 0.7 * mid[6]  # Eq. 6 by hand
    torch.testing.assert_close(depth, ref_depth.expand_as(depth), rtol=1e-4, atol=1e-4)
    if not torch.allclose(rgb, torch.full_like(rgb, 0.25)):
        raise AssertionError("sample colour is not the radiance queried at the sample")
    return {"fraction_surface1": first, "fraction_surface2": second,
            "depth": float(depth[0])}


class _ConstantField(torch.nn.Module):
    def forward(self, x, dirs):
        return torch.full_like(x, 0.25), torch.ones_like(x[..., :1])


def _sched(fn):
    return fn.cfg["lambda_init"], fn.cfg["decay"], fn.cfg["decay_steps"]


def depth_loss(method, device):
    """Eq. 7 + Eq. 13: lambda(i) = 0.9 * 0.9^(i/100); L = lambda * mean |D_NeRF - D_GS|."""
    _, _, losses = build(method)
    fn = dict(losses).get("depth-distillation")
    if fn is None:
        raise AssertionError("depth-distillation loss missing")
    for step, ref in ((1, 0.9 * 0.9 ** 0.01), (100, 0.81), (1000, 0.9 * 0.9 ** 10)):
        if abs(depth_weight(step, *_sched(fn)) - ref) > 1e-9:
            raise AssertionError(f"lambda({step}) = {depth_weight(step, *_sched(fn))} != {ref}")
    torch.manual_seed(0)
    d_gs = torch.rand(8, 8, 1, device=device, requires_grad=True)
    d_nerf = torch.rand(8, 8, 1, device=device) * 3
    unit = 0.25 if fn.cfg["units"] == "nerf" else 1.0  # world -> nerfacc scene units
    value = fn({}, {"step": 100, "depth_accum": d_gs, "nerf_depth": d_nerf, "nerf_scale": 0.25},
               None)
    ref = 0.81 * unit * (d_nerf - d_gs).abs().mean()
    torch.testing.assert_close(value, ref)
    g = torch.autograd.grad(value, d_gs)[0]
    torch.testing.assert_close(g, -0.81 * unit * torch.sign(d_nerf - d_gs) / 64)
    try:
        fn({}, {"step": 1, "depth_accum": d_gs}, None)
    except KeyError:
        pass
    else:
        raise AssertionError("depth loss silently ran without NeRF depth")
    return {"loss_step100": value.item(), "lambda_5000": depth_weight(5000, *_sched(fn))}


class _Table:
    def __init__(self, viewmat, depth, sha=None, log_scales=None):
        self.viewmats = viewmat.detach().cpu()[None]
        self.depth = depth
        self._init = (sha, log_scales)

    def index(self, viewmat):
        return 0 if torch.allclose(viewmat.detach().cpu(), self.viewmats[0]) else None

    def depth_view(self, i, device):
        return self.depth.to(device)

    def init(self):
        return self._init


def renderer_depth(method, device):
    """D_GS of Eq. 5 is the un-normalised sum_i w_i z_i (independent gsplat 'D' render), and the
    sidecar NeRF depth is attached to training renders only."""
    _need_cuda()
    from gsplat import rasterization

    renderer, _, _ = build(method)
    params, _ = sample_scene(method, device)
    with torch.no_grad():
        params["opacities"].fill_(-1.0)  # partial coverage: ED != D
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
              "width": 16, "height": 16}
    nerf = torch.full((16, 16, 1), 2.5, device=device)
    renderer.table_override = _Table(camera["viewmat"], nerf)
    out = renderer.render(params, camera, sh_degree=0)
    ref, alpha, _ = rasterization(
        params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        torch.cat((params["sh0"], params["shN"]), 1), camera["viewmat"][None], camera["K"][None],
        16, 16, sh_degree=0, packed=False, render_mode="D")
    torch.testing.assert_close(out["depth_accum"], ref[0], rtol=1e-4, atol=1e-5)
    if torch.allclose(out["depth_accum"], out["depth"]):
        raise AssertionError("test scene is degenerate (alpha == 1 everywhere)")
    if "nerf_depth" not in out or not torch.equal(out["nerf_depth"], nerf):
        raise AssertionError("training render lacks the sidecar NeRF depth")
    with torch.no_grad():
        ev = renderer.render(params, camera, sh_degree=0)
    if "nerf_depth" in ev:
        raise AssertionError("evaluation render touched the training sidecar")
    renderer.table_override = None
    return {"max_accum": float(ref.max()), "coverage_mean": float(alpha.mean())}


def init_loading(method, device):
    """Point cloud loaded like 3DGS SfM init: exact 3-NN scales from init.npz (SHA-checked),
    opacity 0.1, camera-extent scene scale for the means LR and grow/prune thresholds."""
    params, opts = sample_scene(method, device)
    n = len(params["means"])
    log_scales = torch.linspace(-5, -1, n)
    spec = _spec(method, "density")
    dens = NerfInitDensity(spec)
    dens.table_override = _Table(torch.eye(4), None, sidecar.points_sha256(params["means"]),
                                 log_scales)
    lr0 = opts["means"].param_groups[0]["lr"]
    dens.initialize(params, opts, scene_scale=2.0)
    torch.testing.assert_close(params["scales"].detach().cpu(), log_scales[:, None].expand(-1, 3))
    torch.testing.assert_close(params["opacities"].sigmoid().detach().cpu(),
                               torch.full((n,), 0.1), rtol=1e-5, atol=1e-6)
    cams = []
    for x in (-1.0, 1.0, 3.0):
        c2w = torch.eye(4, device=device)
        c2w[0, 3] = x
        cams.append({"viewmat": torch.linalg.inv(c2w)})
    dens.set_training_cameras(cams)
    extent = 1.1 * 2.0  # centroid x = 1, max distance 2
    if abs(dens.state["scene_scale"] - extent) > 1e-5:
        raise AssertionError(f"scene scale {dens.state['scene_scale']} != camera extent {extent}")
    lr = opts["means"].param_groups[0]["lr"]
    if abs(lr - lr0 / 2.0 * extent) > 1e-12:
        raise AssertionError("means LR not rescaled to the camera extent")
    check_state(method, params, opts)
    bad = NerfInitDensity(spec)
    bad.table_override = _Table(torch.eye(4), None, "0" * 64, log_scales)
    try:
        bad.initialize(*sample_scene(method, device), scene_scale=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("SHA mismatch between means and init.npz was not detected")
    return {"scene_scale": dens.state["scene_scale"], "means_lr": lr}


def ingp_fit(method, device):
    """A short NerfAcc-iNGP fit on a textured unit sphere (64 orbiting cameras): the Eq. 6 depth
    and the Eq. 4 samples land on the sphere (whole path: normalisation, rays, hash grids,
    proposal sampling, training loop, weight sampling, depth)."""
    _need_cuda()
    torch.manual_seed(0)
    H = W = 64
    f = 32.0
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                            indexing="ij")
    pix = torch.stack([(xs + 0.5 - W / 2) / f, (ys + 0.5 - H / 2) / f,
                       torch.ones(H, W, device=device)], -1)
    c2ws, imgs = [], []
    for i in range(64):
        ang = i / 64 * 2 * math.pi
        el = 0.5 * math.sin(5 * ang)
        c = 3 * torch.tensor([math.cos(ang) * math.cos(el), math.sin(ang) * math.cos(el),
                              math.sin(el)], device=device)
        c2w = _look_at(c, device)
        dw = pix @ c2w[:3, :3].T
        dw = dw / dw.norm(dim=-1, keepdim=True)
        t, hit = _sphere_hit(c.expand_as(dw), dw)
        img = 0.5 + 0.4 * torch.sin(9 * (c + dw * t[..., None]))
        img[~hit] = 0.9
        c2ws.append(c2w)
        imgs.append(img)
    transform, scale = ingp_mod.gs2nerf_transform(torch.stack(c2ws).cpu().numpy())
    caster = RayCaster(transform, scale, device)
    cams = [caster.camera(K, torch.linalg.inv(c)) for c in c2ws]
    K_inv, R, o = (torch.stack(x) for x in zip(*cams))
    model = INGP({"iterations": 3000, "rays": 2048}).to(device)
    ingp_mod.train_ingp(model, torch.stack(imgs), K_inv, R, o, log=lambda *_: None)
    px = torch.randint(26, 38, (2048,), device=device)
    py = torch.randint(26, 38, (2048,), device=device)
    origins, dirs, norm, _ = RayCaster.rays(K_inv[5], R[5], o[5], px, py)
    cam = c2ws[5][:3, 3]
    target, hit = _sphere_hit(cam.expand(len(px), 3), (caster.to_world(origins + dirs) - cam)
                              / (1 / scale))
    depth = expected_depth(model, origins, dirs) / scale
    pts, _, _, depth2 = surface_samples(model, origins, dirs)
    torch.testing.assert_close(depth2, depth * scale, rtol=1e-3, atol=1e-3)
    dist = (caster.to_world(pts) - cam).norm(dim=-1)
    err_depth = (depth - target)[hit].abs().median().item()
    err_pts = (dist - target)[hit].abs().median().item()
    if hit.float().mean() < 0.99 or err_depth > 0.05 or err_pts > 0.1:
        raise AssertionError(f"iNGP depth/sample error {err_depth:.3f}/{err_pts:.3f}")
    return {"median_abs_depth_error": err_depth, "median_abs_sample_error": err_pts,
            "surface_distance": float(target.median())}


def _look_at(c, device):
    z = -c / c.norm()
    up = torch.tensor([0, 0, -1.0], device=device)
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    c2w = torch.eye(4, device=device)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, torch.linalg.cross(z, x), z, c
    return c2w


def _sphere_hit(o, d):
    b = (o * d).sum(-1)
    disc = b * b - ((o * o).sum(-1) - 1.0)
    return -b - torch.sqrt(disc.clamp_min(0)), disc > 0
