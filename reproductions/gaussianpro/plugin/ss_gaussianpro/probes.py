"""Mechanism probes for the GaussianPro port. Each raises AssertionError when the mechanism
is removed or wrong and returns measured evidence."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from . import geometry as G
from . import losses as L

# --- synthetic textured plane seen by 5 cameras -------------------------------------------------

W, H, FOCAL = 256, 192, 220.0


def _look(center, target):
    z = F.normalize(target - center, dim=0)
    x = F.normalize(torch.linalg.cross(torch.tensor([0.0, -1.0, 0.0]), z), dim=0)
    y = torch.linalg.cross(z, x)
    vm = torch.eye(4)
    vm[:3, :3] = torch.stack([x, y, z])
    vm[:3, 3] = -vm[:3, :3] @ center
    return vm


class PlaneScene:
    """Plane through P0 with normal n (facing the cameras), smooth random texture."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.K = torch.tensor([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
        self.n = F.normalize(torch.tensor([0.3, 0.2, -1.0]), dim=0)
        self.p0 = torch.tensor([0.0, 0.0, 4.0])
        self.e1 = F.normalize(torch.linalg.cross(self.n, torch.tensor([0.0, 1.0, 0.0])), dim=0)
        self.e2 = torch.linalg.cross(self.n, self.e1)
        self.freqs = torch.randn(40, 2, generator=g) * 6
        self.phases = torch.rand(40, generator=g) * 6.28
        self.amps = torch.rand(40, generator=g)
        offsets = [(0, 0), (0.4, 0), (-0.4, 0), (0, 0.3), (0, -0.3)]
        self.viewmats = [_look(torch.tensor([dx, dy, 0.0]), self.p0) for dx, dy in offsets]

    def texture(self, u, v):
        val = (self.amps[:, None] * torch.sin(self.freqs[:, :1] * u[None]
                                              + self.freqs[:, 1:] * v[None]
                                              + self.phases[:, None])).sum(0)
        return ((val + self.amps.sum()) / (2 * self.amps.sum())).clamp(0, 1)

    def render(self, vm):
        """Analytic image [H,W,3] in [0,1] and camera-z depth [H,W]."""
        c2w = torch.linalg.inv(vm)
        y, x = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
        d_cam = torch.stack([(x - self.K[0, 2]) / FOCAL, (y - self.K[1, 2]) / FOCAL,
                             torch.ones_like(x)], -1)
        d_w = d_cam @ c2w[:3, :3].T
        o = c2w[:3, 3]
        t = ((self.p0 - o) @ self.n) / (d_w @ self.n)
        rel = o + t[..., None] * d_w - self.p0
        img = self.texture((rel @ self.e1).flatten(), (rel @ self.e2).flatten()).view(H, W)
        img = (img * 255).round() / 255
        return img[..., None].expand(H, W, 3).contiguous(), t

    def cameras(self, device):
        return [{"viewmat": vm.to(device), "K": self.K.to(device), "width": W, "height": H}
                for vm in self.viewmats]

    def plane_coords(self, pts):
        rel = pts - self.p0.to(pts)
        return rel @ self.e1.to(pts), rel @ self.e2.to(pts), rel @ self.n.to(pts)


def _need_cuda(device):
    if not str(device).startswith("cuda"):
        from splatstudio.verification import ProbeUnavailable

        raise ProbeUnavailable("GaussianPro CUDA probes need a CUDA device")


def propagation_plane(method, device):
    """C2: the vendored kernel recovers plane depth/normal from a corrupted rendered depth."""
    _need_cuda(device)
    from . import kernels

    scene = PlaneScene()
    renders = [scene.render(vm) for vm in scene.viewmats]
    truth = renders[0][1]
    g = torch.Generator().manual_seed(1)
    init = truth * (0.8 + 0.4 * torch.rand(truth.shape, generator=g))
    init[60:90] = 0.0  # unrendered band -> random initialisation inside the kernel
    imgs = torch.stack([(im * 255).to(torch.uint8) for im, _ in renders])
    interval = torch.tensor([0.1, (20 - 0.1) / 192, 192, 20]).expand(5, 4)
    depth, normal = kernels.propagate(imgs, scene.K.expand(5, 3, 3), torch.stack(scene.viewmats),
                                      init, interval, 20)
    rel = (depth - truth).abs() / truth
    init_rel = (init - truth).abs() / truth
    angle = torch.rad2deg(torch.acos((normal @ scene.n).clamp(-1, 1)))
    band = rel[60:90]
    out = {"initial_rel_err_median": float(init_rel.median()),
           "rel_err_median": float(rel.median()), "frac_rel_err_below_1pct": float((rel < 0.01).float().mean()),
           "band_rel_err_median": float(band.median()), "normal_angle_median_deg": float(angle.median())}
    if not (out["initial_rel_err_median"] > 0.05):
        raise AssertionError(f"probe setup: initial depth not corrupted {out}")
    if not (out["rel_err_median"] < 0.01 and out["frac_rel_err_below_1pct"] > 0.8
            and out["band_rel_err_median"] < 0.02 and out["normal_angle_median_deg"] < 5.0):
        raise AssertionError(f"propagation did not recover the plane: {out}")
    return out


def _plane_gaussians(scene, device, hole, spacing=0.05, extent=3.0):
    """Flat Gaussians tiling the plane (normal = scene.n), optionally with a square hole."""
    from splatstudio.state import initialize

    from . import method as make_method

    ticks = torch.arange(-extent, extent + 1e-6, spacing)
    u, v = torch.meshgrid(ticks, ticks, indexing="ij")
    u, v = u.flatten(), v.flatten()
    if hole is not None:
        keep = ~((u.abs() < hole) & (v.abs() < hole))
        u, v = u[keep], v[keep]
    pts = scene.p0 + u[:, None] * scene.e1 + v[:, None] * scene.e2
    colors = torch.full_like(pts, 0.5)
    params, optimizers = initialize(make_method(), pts.to(device), colors.to(device))
    # rotation whose columns are (e1, e2, n): the shortest (third) axis is the plane normal
    R = torch.stack([scene.e1, scene.e2, scene.n], dim=1)
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = torch.tensor([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                      (R[1, 0] - R[0, 1]) / (4 * w)])
    with torch.no_grad():
        params["quats"].copy_(q.expand(len(pts), 4).to(device))
        params["scales"].copy_(torch.tensor([spacing * 0.7, spacing * 0.7, 1e-3]).log()
                               .expand(len(pts), 3).to(device))
        params["opacities"].fill_(5.0)
    return params, optimizers


def _density(device):
    from splatstudio.specs import ComponentSpec

    from .density import Density
    from .link import LINK
    from .renderer import Renderer

    Renderer(SimpleNamespace(config={}))
    dens = Density(ComponentSpec("gaussianpro-density", "density",
                                 "ss_gaussianpro.density:create_density", (), (), {}))
    return dens, LINK


def _run_selection(device, hole):
    from splatstudio.state import check_state

    from . import method as make_method

    scene = PlaneScene()
    images = [scene.render(vm)[0] for vm in scene.viewmats]
    cams = scene.cameras(device)
    dens, link = _density(device)
    neighbours = [[j for j in range(5) if j != i] for i in range(5)]
    dens.set_training_frames(cams, images, neighbours)
    params, optimizers = _plane_gaussians(scene, device, hole)
    dens.initialize(params, optimizers, 4.0)
    n0 = len(params["means"])
    dens.base.state["grad2d"] = torch.ones(n0, device=device)
    dens.base.state["count"] = torch.ones(n0, device=device)
    step = 6500  # mid window: sigma = 0.9 (pixels with no Gaussians have abs-rel error 1)
    event = dens.propagate(params, optimizers, step, 0)
    check_state(make_method(), params, optimizers)
    new = params["means"][n0:].detach()
    u, v, h = scene.plane_coords(new)
    in_hole = (u.abs() < hole + 0.1) & (v.abs() < hole + 0.1) if hole else torch.zeros_like(u).bool()
    cam = cams[0]
    zc = G.camera_depths(new, cam["viewmat"])
    return {"event": event, "n_new": len(new), "in_hole_frac": float(in_hole.float().mean()) if len(new) else 0.0,
            "max_plane_dist_rel": float((h.abs() / zc).max()) if len(new) else 0.0,
            "stats_zeroed": bool(dens.base.state["grad2d"].abs().sum() == 0) if len(new) else None,
            "normal_valid_frac": float((link.normal_targets[0][..., 0] != G.INVALID_NORMAL).float().mean()),
            "new_opacity": float(params["opacities"][n0:].sigmoid().min()) if len(new) else None}


def propagation_selection(method, device):
    """C2/C4: Gaussians are added only where rendered and propagated depth disagree (a hole)."""
    _need_cuda(device)
    with_hole = _run_selection(device, hole=0.6)
    without = _run_selection(device, hole=None)
    out = {"hole": with_hole, "no_hole": without}
    h = with_hole
    if not (h["n_new"] > 20 and h["in_hole_frac"] > 0.9 and h["max_plane_dist_rel"] < 0.02):
        raise AssertionError(f"propagation did not fill the hole on the plane: {out}")
    if not (h["stats_zeroed"] and h["new_opacity"] == 1.0 and h["normal_valid_frac"] > 0.5):
        raise AssertionError(f"insertion side effects differ from the official code: {out}")
    if without["n_new"] != 0:
        raise AssertionError(f"Gaussians added although rendered depth agrees: {out}")
    return out


def geometry_render(method, device):
    """C1: rendered camera normal = flipped shortest axis in camera frame; depth = Σ z α T."""
    _need_cuda(device)
    from .renderer import Renderer

    cam = {"viewmat": torch.eye(4, device=device),
           "K": torch.tensor([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], device=device),
           "width": 64, "height": 64}
    # two flat Gaussians on the optical axis; the shortest axis of the front one points away
    # from the camera (+z) and must be flipped; the back one is tilted about x by 30 degrees.
    a = math.radians(30)
    quats = torch.tensor([[1.0, 0, 0, 0], [math.cos(a / 2), math.sin(a / 2), 0, 0]], device=device)
    scales = torch.tensor([[0.5, 0.5, 0.001], [0.5, 0.5, 0.001]], device=device).log()
    means = torch.tensor([[0.0, 0, 2.0], [0.0, 0, 3.0]], device=device)
    op = torch.tensor([0.6, 0.9], device=device)
    params = {"means": means, "quats": quats, "scales": scales, "opacities": torch.logit(op),
              "sh0": torch.zeros(2, 1, 3, device=device), "shN": torch.zeros(2, 15, 3, device=device)}
    rend = Renderer(SimpleNamespace(config={"st_alpha_clamp": False}))
    with torch.no_grad():
        geo = rend.geometry(params, cam)
    c = (32, 32)
    # independent: axis 3 of each rotation; flip to face the camera at the origin
    n_front = torch.tensor([0.0, 0, -1.0])           # (0,0,1) flipped
    n_back = torch.tensor([0.0, -math.sin(a), math.cos(a)])
    n_back = -n_back if float(n_back @ torch.tensor([0.0, 0, 3.0])) >= 0 else n_back
    a1, a2 = 0.6, 0.9  # center pixel: G = 1 (gsplat evaluates the pixel center 32.5; tiny offset)
    T = 1 - a1
    blend = a1 * n_front + T * a2 * n_back
    expected_n = blend / blend.norm()
    got_n = geo["normal"][c].cpu()
    expected_d = a1 * 2.0 + T * a2 * 3.0
    got_d = float(geo["depth"][c])
    out = {"normal": got_n.tolist(), "expected_normal": expected_n.tolist(),
           "depth": got_d, "expected_depth": expected_d}
    if float((got_n - expected_n).abs().max()) > 0.02 or abs(got_d - expected_d) > 0.03 * expected_d:
        raise AssertionError(f"geometry pass differs from independent compositing: {out}")
    return out


def planar_losses(method, device):
    """C3: Eq. 8 normal loss and the flatten loss against independent formulas."""
    g = torch.Generator().manual_seed(0)
    n = F.normalize(torch.randn(8, 9, 3, generator=g), dim=-1).requires_grad_(True)
    t = F.normalize(torch.randn(8, 9, 3, generator=g), dim=-1)
    t[:3] = G.INVALID_NORMAL
    c = dict(L.NORMAL_DEFAULTS)
    value = L.normal_loss(n, t, c)
    total_l1, total_cos, count = 0.0, 0.0, 0
    for i in range(8):
        for j in range(9):
            if float(t[i, j, 0]) == G.INVALID_NORMAL:
                continue
            total_l1 += float((n[i, j] - t[i, j]).abs().sum())
            total_cos += 1 - float((n[i, j] * t[i, j]).sum())
            count += 1
    expected = 0.001 * total_l1 / count + 0.001 * total_cos / count
    value.backward()
    grad_invalid = float(n.grad[:3].abs().max())
    zero = float(L.normal_loss(t[3:].clone(), t[3:], c))
    s = torch.tensor([[0.1, 0.02, 0.3], [0.5, 0.4, 0.05]]).log().requires_grad_(True)
    fl = L.flatten_loss(s, dict(L.FLATTEN_DEFAULTS))
    fl.backward()
    expected_fl = 100 * (0.02 + 0.05) / 2
    mask = s.grad != 0
    out = {"normal_loss": float(value), "expected": expected, "grad_invalid": grad_invalid,
           "at_target": zero, "flatten": float(fl), "expected_flatten": expected_fl,
           "flatten_grad_pattern": mask.int().tolist()}
    if abs(float(value) - expected) > 1e-6 or grad_invalid != 0 or abs(zero) > 1e-7:
        raise AssertionError(f"normal loss differs from Eq. 8 as coded: {out}")
    if abs(float(fl) - expected_fl) > 1e-5 or mask.int().tolist() != [[0, 1, 0], [0, 0, 1]]:
        raise AssertionError(f"flatten loss differs from 100 * mean(min scale): {out}")
    return out


def opacity_mask(method, device):
    """C4: before step 1000 only pixels with opacity > 0.999 drive the photometric loss."""
    rgb = torch.rand(32, 32, 3).requires_grad_(True)
    target = torch.rand(32, 32, 3)
    alpha = torch.ones(32, 32, 1)
    alpha[:, :16] = 0.5
    c = {**L.PHOTO_DEFAULTS, "ssim_weight": 0.0}
    L.masked_photometric(rgb, target, alpha, 500, c).backward()
    early = (float(rgb.grad[:, :16].abs().max()), float(rgb.grad[:, 16:].abs().max()))
    rgb.grad = None
    L.masked_photometric(rgb, target, alpha, 1500, c).backward()
    late = float(rgb.grad[:, :16].abs().max())
    full = L.masked_photometric(rgb, target, alpha, 500, dict(L.PHOTO_DEFAULTS))
    out = {"early_grad_masked": early[0], "early_grad_opaque": early[1], "late_grad": late,
           "full_loss": float(full)}
    if early[0] != 0 or early[1] == 0 or late == 0 or not math.isfinite(float(full)):
        raise AssertionError(f"opacity mask schedule wrong: {out}")
    return out


def consistency_check(method, device):
    """C2: multi-view geometric filter accepts the true plane and rejects a 5 % offset."""
    scene = PlaneScene()
    (_, d0), (_, d1) = scene.render(scene.viewmats[0]), scene.render(scene.viewmats[1])
    K = scene.K
    c2w0, c2w1 = (torch.linalg.inv(scene.viewmats[i]) for i in (0, 1))
    good = G.check_geometric_consistency(d0, K, c2w0, d1, K, c2w1)
    bad = G.check_geometric_consistency(d0 * 1.05, K, c2w0, d1, K, c2w1)
    out = {"consistent_frac": float(good.float().mean()), "offset_frac": float(bad.float().mean())}
    if out["consistent_frac"] < 0.7 or out["offset_frac"] > 0.05:
        raise AssertionError(f"geometric consistency check wrong: {out}")
    return out
