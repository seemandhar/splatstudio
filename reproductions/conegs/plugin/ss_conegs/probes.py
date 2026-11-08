"""Mechanism probes for ConeGS. Each raises if its mechanism is removed or wrong."""

import math

import torch
from splatstudio.components import build
from splatstudio.state import check_state, sample_scene

from . import ingp as ingp_mod
from .density import ConeGSDensity, append_gaussians, rays_to_cast
from .hashgrid import HashGrid, reference_encode
from .ingp import INGP, RayCaster, importance_sampling, pdf_loss, s_to_t
from .losses import penalty


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


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
    # hashing matters: a dense-only index would differ on hashed levels
    return {"forward_max_abs": fwd, "backward_max_abs": bwd, "levels": len(res),
            "dense_levels": int(sum(dense)), "params": int(grid.offsets[-1])}


def proposal_sampler(method, device):
    """nerfacc importance_sampling docstring cases, lindisp endpoints, zero pdf loss."""
    e1 = torch.tensor([[0.0, 1.0]], device=device)
    c1 = torch.tensor([[0.0, 0.5]], device=device)
    edges, samples = importance_sampling(e1, c1, 2, False)
    torch.testing.assert_close(edges, torch.tensor([[0.0, 0.5, 1.0]], device=device))
    torch.testing.assert_close(samples, torch.tensor([[0.25, 0.75]], device=device))
    e2 = torch.tensor([[0.0, 1.0, 2.0]], device=device)
    c2 = torch.tensor([[0.0, 0.5, 1.0]], device=device)
    edges, samples = importance_sampling(e2, c2, 2, False)
    torch.testing.assert_close(edges, torch.tensor([[0.0, 1.0, 2.0]], device=device))
    torch.testing.assert_close(samples, torch.tensor([[0.5, 1.5]], device=device))
    # peaked cdf concentrates samples in the mass
    e3 = torch.linspace(0, 1, 11, device=device)[None]
    c3 = torch.tensor([[0, 0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1.0]], device=device)
    _, s3 = importance_sampling(e3, c3, 64, True)
    if not ((s3 >= 0.4) & (s3 <= 0.6)).all():
        raise AssertionError("importance sampling ignores the cdf")
    t = s_to_t(torch.tensor([0.0, 1.0], device=device), 0.2, 1e6)
    torch.testing.assert_close(t, torch.tensor([0.2, 1e6], device=device), rtol=1e-4, atol=1e-4)
    loss0 = pdf_loss(e3, c3, e3, c3).sum().item()
    flat = torch.linspace(0, 1, 11, device=device)[None]
    loss1 = pdf_loss(e3, c3, e3, flat).sum().item()
    if loss0 > 1e-9 or loss1 <= 1e-3:
        raise AssertionError(f"pdf loss wrong: same {loss0}, flat proposal {loss1}")
    return {"pdf_loss_same": loss0, "pdf_loss_flat": loss1}


class _PlaneINGP(INGP):
    """Real median_points with a synthetic opaque plane z_cam = depth (camera frame = world)."""

    def __init__(self, depth, scale, color=(0.2, 0.6, 0.9), steps=4096, far=20.0):
        super().__init__()
        self.depth, self.scale_, self.color, self.steps, self.far = depth, scale, color, steps, far

    def sample(self, origins, dirs, stratified, requires_grad=False, cache=None):
        t = torch.linspace(0.01, self.far, self.steps + 1, device=origins.device)
        t = t.expand(len(origins), -1)
        return t[:, :-1], t[:, 1:]

    def render(self, origins, dirs, ts, te, view_dirs=True):
        mid = (ts + te) / 2
        z = (origins[:, None, 2] + dirs[:, None, 2] * mid) / self.scale_
        alphas = (z >= self.depth).float()
        trans = torch.cumprod(torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas[:, :-1]], 1), 1)
        color = torch.tensor(self.color, device=origins.device).expand(len(origins), 3)
        return color, {"trans": trans, "alphas": alphas, "mid": mid}


def _plane_density(method, device, depth, scale, size_mult=None):
    spec = _spec(method, "density")
    cfg = dict(spec.config)
    if size_mult is not None:
        cfg["size_mult"] = size_mult
    dens = ConeGSDensity(type(spec)(spec.name, spec.kind, spec.factory, spec.requires,
                                    spec.provides, cfg))
    model = _PlaneINGP(depth, scale).to(device)
    caster = RayCaster(torch.eye(4), scale, device)
    dens.model, dens.caster = model, caster
    return dens


def cone_placement(method, device):
    """Eq. 7/11 + median depth + error weighting on an analytic plane.

    New Gaussians lie on their pixel rays at the plane depth, carry the rendered colour, and have
    std x f / z = size_mult (1 pixel footprint); all samples fall in the high-error quadrant."""
    W, H, f, depth, scale = 64, 48, 50.0, 3.0, 0.5
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=device)
    image = torch.zeros(H, W, 3, device=device)
    rgb = torch.zeros(H, W, 3, device=device)
    rgb[H // 2 :, W // 2 :] = 0.8  # error only in the bottom-right quadrant
    outputs = {"rgb": rgb, "conegs_frame": {"K": K, "viewmat": torch.eye(4, device=device),
                                            "width": W, "height": H, "image": image}}
    results = {}
    for mult in (1.0, 2.0):
        dens = _plane_density(method, device, depth, scale, size_mult=mult)
        n = dens.cast(outputs, 200)
        s = dens.stash[0]
        means = s["means"]
        std = s["scales"][:, 0].exp()
        u = means[:, 0] / means[:, 2] * f + W / 2
        v = means[:, 1] / means[:, 2] * f + H / 2
        footprint = (std * f / means[:, 2]).mean().item()
        results[mult] = footprint
        if n != 200:
            raise AssertionError(f"expected 200 new Gaussians, got {n}")
        if not ((u >= W // 2) & (v >= H // 2)).all():
            raise AssertionError("error-weighted sampling picked zero-error pixels")
        frac = torch.frac(u)
        if ((frac - 0.5).abs() > 1e-3).any() or ((torch.frac(v) - 0.5).abs() > 1e-3).any():
            raise AssertionError("Gaussian centres are not on the pixel-centre rays")
        if ((means[:, 2] - depth).abs() > 2 * 20.0 / 4096 / scale).any():
            raise AssertionError("Gaussian centres are not at the median depth")
        col = s["sh0"][:, 0] * ingp_mod.SH_C0 + 0.5
        torch.testing.assert_close(col, torch.tensor([0.2, 0.6, 0.9], device=device).expand_as(col))
    configured = float(_spec(method, "density").config.get("size_mult", 1.0))
    if abs(results[1.0] - 1.0) > 0.01 or abs(results[2.0] - 2.0) > 0.02:
        raise AssertionError(f"footprint scale wrong: {results}")
    if configured != 1.0:
        raise AssertionError("configured size_mult differs from the official 1.0 pixel footprint")
    return {"footprint_px_mult1": results[1.0], "footprint_px_mult2": results[2.0]}


def median_depth(method, device):
    """First sample whose inclusive transmittance drops below 0.5 (tau_{k-1} > 0.5 >= tau_k)."""

    class Profile(INGP):
        def sample(self, origins, dirs, stratified, requires_grad=False, cache=None):
            t = torch.arange(9, device=origins.device, dtype=torch.float32).expand(len(origins), -1)
            return t[:, :-1] + 1, t[:, 1:] + 1

        def render(self, origins, dirs, ts, te, view_dirs=True):
            a = torch.tensor([0.2, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 1.0], device=origins.device)
            a = a.expand(len(origins), -1)
            trans = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
            return torch.full((len(origins), 3), 0.5, device=origins.device), {
                "trans": trans, "alphas": a, "mid": (ts + te) / 2}

    model = Profile().to(device)
    o = torch.zeros(4, 3, device=device)
    d = torch.tensor([[0, 0, 1.0]], device=device).expand(4, -1)
    _, _, t, _, valid = model.median_points(o, d, torch.ones(4, device=device),
                                            torch.ones(4, device=device))
    # inclusive T: .8, .64, .448 -> index 2 -> midpoint 3.5
    if not torch.allclose(t, torch.full_like(t, 3.5)) or not valid.all():
        raise AssertionError(f"median depth wrong: {t.tolist()}")
    return {"median_t": t[0].item(), "expected": 3.5}


def budget(method, device):
    """Prune + merge keep N <= budget, refill exactly the pruned slots with the oldest stash rows,
    zero Adam moments for new rows, keep fields trainable; official ray-count formula."""
    spec = _spec(method, "density")
    if rays_to_cast(1_000_000, None, 0.02) != 201 or rays_to_cast(1_000_000, 50_000, 0.02) != 601:
        raise AssertionError("ray-count formula differs from get_num_rays_to_cast")
    if rays_to_cast(1000, 5, 0.02) != 10:
        raise AssertionError("minimum of 10 rays missing")
    params, opts = sample_scene(method, device)
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step()
        o.zero_grad(set_to_none=True)
    n0 = len(params["means"])
    cfg = {**spec.config, "budget": n0, "camera_extent_lr": False}
    dens = ConeGSDensity(type(spec)(spec.name, spec.kind, spec.factory, spec.requires,
                                    spec.provides, cfg))
    with torch.no_grad():
        params["opacities"][:3] = -10.0  # three dead Gaussians
    keep_means = params["means"][3:].detach().clone()
    dens.extent_applied = True
    dens.n_rays = 10
    dens.state = {"pruned": 0, "added": 0, "cast": 0, "merges": []}
    stash_means = torch.arange(15, dtype=torch.float32, device=device)[:, None].expand(-1, 3) + 100
    dens.stash = [{"means": stash_means[:8], "scales": torch.zeros(8, 3, device=device),
                   "sh0": torch.zeros(8, 1, 3, device=device)},
                  {"means": stash_means[8:], "scales": torch.zeros(7, 3, device=device),
                   "sh0": torch.zeros(7, 1, 3, device=device)}]
    dens.cast = lambda outputs, n: 0
    dens.after_backward(params, opts, 100, {})
    check_state(method, params, opts)
    if len(params["means"]) != n0:
        raise AssertionError("budget not refilled / exceeded")
    torch.testing.assert_close(params["means"][: n0 - 3], keep_means)
    torch.testing.assert_close(params["means"][n0 - 3 :], stash_means[:3])
    for name, p in params.items():
        if not (p.is_leaf and p.requires_grad):
            raise AssertionError(f"{name} frozen by the merge")
        if torch.count_nonzero(opts[name].state[p]["exp_avg"][n0 - 3 :]):
            raise AssertionError("new rows must start with zero Adam moments")
    torch.testing.assert_close(params["opacities"][n0 - 3 :].sigmoid(),
                               torch.full((3,), 0.1, device=device))
    if dens.n_rays != rays_to_cast(n0, 3, cfg["growth"]):
        raise AssertionError("ray count not updated from the number added")
    # no budget: every stashed Gaussian is added
    cfg2 = {**cfg, "budget": 0}
    dens2 = ConeGSDensity(type(spec)(spec.name, spec.kind, spec.factory, spec.requires,
                                     spec.provides, cfg2))
    dens2.stash = [{"means": stash_means, "scales": torch.zeros(15, 3, device=device),
                    "sh0": torch.zeros(15, 1, 3, device=device)}]
    added = dens2.merge(params, opts)
    check_state(method, params, opts)
    if added != 15:
        raise AssertionError("unbudgeted merge must add every stashed Gaussian")
    new = {"means": torch.zeros(2, 3, device=device)}
    append_gaussians(params, opts, new)
    check_state(method, params, opts)
    return {"budget": n0, "pruned": 3, "refilled": 3, "unbudgeted_added": added}


def opacity_penalty(method, device):
    """Official signed mean of logits: d/d logit = lambda/N everywhere (also for logits > 0)."""
    spec = next(c for c in method.components if c.name == "conegs-opacity")
    weight = float(spec.config["weight"])
    _, _, losses = build(method)
    fn = dict(losses)["conegs-opacity"]
    logits = torch.tensor([-6.0, -1.0, 0.5, 3.0], device=device, requires_grad=True)
    value = fn({"opacities": logits}, {}, None)
    grad = torch.autograd.grad(value, logits)[0]
    torch.testing.assert_close(grad, torch.full_like(grad, weight / 4))
    post = torch.autograd.grad(penalty(logits, weight, "post"), logits)[0]
    if torch.allclose(post, grad):
        raise AssertionError("pre-activation penalty indistinguishable from MCMC-style penalty")
    if abs(weight - 2e-4) > 1e-12:
        raise AssertionError("lambda_o differs from the paper's 0.0002")
    return {"weight": weight, "gradient": grad.tolist(), "post_gradient": post.tolist()}


def renderer_sh(method, device):
    """All SH bands rendered from step 1 regardless of the runtime degree; frame passthrough."""
    _need_cuda()
    from splatstudio.backends.gsplat import Renderer

    renderer, _, _ = build(method)
    params, _ = sample_scene(method, device)
    with torch.no_grad():
        params["shN"].normal_(0, 0.5)
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
              "width": 16, "height": 16, "image": torch.zeros(16, 16, 3, device=device)}
    out = renderer.render(params, camera, sh_degree=0)
    rspec = _spec(method, "renderer")
    inner = Renderer(type(rspec)(rspec.name, "renderer", rspec.factory, rspec.requires, (), {}))
    full = inner.render(params, camera, sh_degree=3)
    ref3, ref0 = full["rgb"], inner.render(params, camera, sh_degree=0)["rgb"]
    if renderer.background == "black":  # official white_background: false
        ref3, ref0 = ref3 - (1 - full["alpha"]), ref0 - (1 - full["alpha"])
        if (full["alpha"] > 0.999).all():
            raise AssertionError("SH test scene has no background pixels")
    torch.testing.assert_close(out["rgb"], ref3)
    if torch.allclose(ref3, ref0):
        raise AssertionError("SH test scene is degenerate")
    if out["conegs_frame"] is not camera:
        raise AssertionError("training frame not passed through")
    return {"diff_vs_degree0": (ref3 - ref0).abs().max().item()}


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


def ingp_fit(method, device):
    """A short iNGP fit on a textured unit sphere seen by 64 orbiting cameras recovers the
    sphere surface as the median depth (whole iNGP path: normalisation, rays, hash grids,
    proposal sampling, training loop, median query)."""
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
        p = c + dw * t[..., None]
        img = 0.5 + 0.4 * torch.sin(9 * p)
        img[~hit] = 0.9
        c2ws.append(c2w)
        imgs.append(img)
    transform, scale = ingp_mod.gs2nerf_transform(torch.stack(c2ws).cpu().numpy())
    caster = RayCaster(transform, scale, device)
    cams = [caster.camera(K, torch.linalg.inv(c)) for c in c2ws]
    K_inv, R, o = (torch.stack(x) for x in zip(*cams))
    model = INGP({"iterations": 3000, "rays": 2048}).to(device)
    ingp_mod.train_ingp(model, torch.stack(imgs), K_inv, R, o, log=lambda *_: None)
    px = torch.randint(24, 40, (512,), device=device)
    py = torch.randint(24, 40, (512,), device=device)
    origins, dirs, norm, radii = RayCaster.rays(K_inv[5], R[5], o[5], px, py)
    pts, _, _, _, valid = model.median_points(origins, dirs, norm, radii)
    world = caster.to_world(pts)
    cam = c2ws[5][:3, 3]
    dist = (world - cam).norm(dim=-1)
    target, hit = _sphere_hit(cam.expand_as(world), (world - cam) / dist[:, None])
    err = (dist - target)[hit & valid].abs().median().item()
    if err > 0.05 or hit.float().mean() < 0.99:
        raise AssertionError(f"iNGP median depth off the sphere by {err}")
    return {"median_abs_depth_error": err, "surface_distance": float(target.median())}
