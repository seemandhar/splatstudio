"""Mechanism probes for the PGSR port. Each raises on failure and returns measurements."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G

C0 = 0.28209479177387814


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _spec(method, name):
    return next(c for c in method.components if c.name == name)


def _camera(device, w=48, h=40, f=40.0, R=None, t=None):
    vm = torch.eye(4, device=device)
    if R is not None:
        vm[:3, :3] = torch.as_tensor(R, dtype=torch.float32, device=device)
    if t is not None:
        vm[:3, 3] = torch.as_tensor(t, dtype=torch.float32, device=device)
    return {"viewmat": vm, "K": torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]],
                                             device=device), "width": w, "height": h}


def _rot_y(deg):
    a = math.radians(deg)
    return [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]


def _rot_x(deg):
    a = math.radians(deg)
    return [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]


def _params(means, scales, quats, opac, colors, device):
    n = len(means)
    p = {"means": torch.tensor(means, dtype=torch.float32),
         "scales": torch.tensor(scales, dtype=torch.float32).log(),
         "quats": torch.tensor(quats, dtype=torch.float32),
         "opacities": torch.logit(torch.tensor(opac, dtype=torch.float32)),
         "sh0": ((torch.tensor(colors, dtype=torch.float32) - 0.5) / C0)[:, None, :],
         "shN": torch.zeros(n, 15, 3)}
    return {k: torch.nn.Parameter(v.to(device)) for k, v in p.items()}


def _plane_cam(camera, n_w, d_w):
    """World plane n_w . X = d_w in camera frame: n_c . Xc = d_w + n_c . t (n_c = R n_w)."""
    vm = camera["viewmat"].float()
    n_c = vm[:3, :3] @ n_w
    off = d_w + float(n_c @ vm[:3, 3])
    if off > 0:  # orient the normal towards the camera (n . X < 0 on the plane)
        n_c, off = -n_c, -off
    return n_c, -off  # (normal facing camera, positive distance)


def _analytic_depth(camera, n_w, d_w):
    n_c, dist = _plane_cam(camera, n_w, d_w)
    rays = G.camera_rays(camera)
    return -dist / (rays @ n_c)  # z of the ray-plane intersection (ray z = 1)


# ---------------------------------------------------------------------------------------
def plane_depth(method, device):
    """C2: official kernel plane depth = ray-plane intersection of a flattened Gaussian."""
    _cuda(device)
    from .raster import render

    cam = _camera(device, 64, 64, 60.0)
    q = [math.cos(math.radians(17.5)), math.sin(math.radians(17.5)), 0.0, 0.0]  # 35 deg about x
    params = _params([[0.1, -0.05, 3.0]], [[1.2, 0.9, 1e-4]], [q], [0.95], [[0.6, 0.4, 0.2]],
                     device)
    out = render(params, cam, 0, torch.zeros(3, device=device), render_geo=True)
    alpha = out["alpha"][..., 0]
    m = alpha > 0.3
    R = torch.tensor(_rot_x(35.0), device=device)
    n_w = R[:, 2]
    d_w = float(n_w @ params["means"][0].detach())
    ref = _analytic_depth(cam, n_w, d_w)
    depth = out["depth"][..., 0]
    err = (depth - ref).abs()[m].max().item()
    n_c, dist = _plane_cam(cam, n_w, d_w)
    nrm = out["normal"][m] / alpha[m][:, None]
    nerr = (nrm - n_c[None]).abs().max().item()
    derr = (out["distance"][..., 0][m] / alpha[m] - dist).abs().max().item()
    spread = (depth[m].max() - depth[m].min()).item()
    if int(m.sum()) < 200 or err > 2e-3 * 3.0 or nerr > 1e-3 or derr > 1e-3:
        raise AssertionError(f"plane depth/normal mismatch: err {err}, normal {nerr}, "
                             f"distance {derr}, pixels {int(m.sum())}")
    if spread < 0.3:  # an alpha-blended centre z would be constant (3.0) across the splat
        raise AssertionError(f"plane depth does not vary over the oblique plane ({spread})")
    observed = int(out["info"]["observe"][0])
    expect = int((alpha >= 1.0 / 255.0).sum())
    if abs(observed - expect) > 2:
        raise AssertionError(f"out_observe {observed} != contributing pixels {expect}")
    loss = out["depth"].sum() + out["rgb"].sum() + out["normal"].sum()
    loss.backward()
    grads = {k: float(params[k].grad.abs().sum()) for k in ("means", "quats", "scales",
                                                            "opacities")}
    if not all(math.isfinite(v) and v > 0 for v in grads.values()):
        raise AssertionError(f"bad gradients {grads}")
    return {"max_depth_error": err, "max_normal_error": nerr, "max_distance_error": derr,
            "depth_spread": spread, "observe": observed, "grad_abs_sums": grads}


def flatten_loss(method, device):
    """C1: L_s = lambda_1 mean(min(s)) on visible Gaussians; gradient on the min axis only."""
    from .losses import create_flatten

    loss_fn = create_flatten(_spec(method, "pgsr-flatten"))
    s = torch.tensor([[0.3, 0.02, 0.5], [0.04, 0.2, 0.1], [0.001, 0.5, 0.5]], device=device)
    params = {"scales": torch.nn.Parameter(s.log()), "means": torch.nn.Parameter(
        torch.zeros(3, 3, device=device))}
    radii = torch.tensor([3, 5, 0], device=device)
    val = loss_fn(params, {"info": {"radii": radii}}, None)
    ref = 100.0 * (0.02 + 0.04) / 2
    if abs(float(val) - ref) > 1e-5:
        raise AssertionError(f"L_s {float(val)} != {ref}")
    val.backward()
    g = params["scales"].grad
    expect = torch.zeros_like(g)
    expect[0, 1] = 100.0 / 2 * 0.02
    expect[1, 0] = 100.0 / 2 * 0.04
    if (g - expect).abs().max() > 1e-5:
        raise AssertionError(f"L_s gradient wrong {g.tolist()}")
    return {"value": float(val), "reference": ref}


def _grad_weight_reference(img):
    """Independent loop implementation of get_img_grad_weight. img HWC numpy."""
    H, W, _ = img.shape
    g = np.zeros((H - 2, W - 2))
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            gx = np.abs(img[i, j + 1] - img[i, j - 1]).mean()
            gy = np.abs(img[i - 1, j] - img[i + 1, j]).mean()
            g[i - 1, j - 1] = max(gx, gy)
    g = (g - g.min()) / (g.max() - g.min())
    out = np.ones((H, W))
    out[1:-1, 1:-1] = g
    return out


def single_view(method, device):
    """C3: depth->normal on an analytic plane; L_svgeom vs independent formula; schedule."""
    from .losses import create_single_view
    from .raster import normal_from_depth

    cam = _camera(device, 40, 30, 30.0)
    n_w = F.normalize(torch.tensor([0.3, -0.2, -1.0], device=device), dim=0)
    depth = _analytic_depth(cam, n_w, -2.5)
    n_c, _ = _plane_cam(cam, n_w, -2.5)
    dn = normal_from_depth(cam, depth)
    inner = (dn[1:-1, 1:-1] - n_c).abs().max().item()
    if inner > 1e-4 or dn[0].abs().max() > 0:
        raise AssertionError(f"normal_from_depth error {inner}")
    g = torch.Generator().manual_seed(0)
    img = torch.rand(30, 40, 3, generator=g)
    img[:, 20:] += 0.8  # a strong vertical edge
    img = (img / img.max()).to(device)
    normal = torch.nn.Parameter((n_c + torch.tensor([0.2, 0.0, 0.1], device=device)).expand(
        30, 40, 3).clone())
    w = torch.tensor(_grad_weight_reference(img.cpu().numpy()), dtype=torch.float32,
                     device=device)
    ref = (((1 - w).clamp(0, 1) ** 2) * (dn - normal).abs().sum(-1)).mean().item() * 0.015
    fn = create_single_view(_spec(method, "pgsr-single-view"))
    outputs = {"normal": normal, "depth_normal": dn, "geometry": True,
               "step": 7001}
    params = {"means": torch.nn.Parameter(torch.zeros(1, 3, device=device))}
    val = fn(params, outputs, img)
    off = fn(params, {**outputs, "step": 7000}, img)
    if abs(float(val) - ref) > 1e-6 * max(1.0, ref) or float(off) != 0.0:
        raise AssertionError(f"L_svgeom {float(val)} vs {ref}; at 7000 {float(off)}")
    unweighted = 0.015 * (dn - normal).abs().sum(-1).mean().item()
    if abs(unweighted - ref) < 1e-4:
        raise AssertionError("edge weight has no effect")
    val.backward()
    if normal.grad is None or normal.grad.abs().sum() == 0:
        raise AssertionError("no gradient to the rendered normal")
    return {"normal_error": inner, "loss": float(val), "reference": ref,
            "unweighted": unweighted}


def _two_views(device):
    ref = _camera(device, 64, 48, 50.0)
    R = torch.tensor(_rot_y(-6.0))
    c = torch.tensor([0.35, 0.05, 0.0])  # neighbour centre
    nb = _camera(device, 64, 48, 50.0, R=R, t=-(R @ c))
    n_w = F.normalize(torch.tensor([0.15, -0.1, -1.0], device=device), dim=0)
    return ref, nb, n_w, -3.0


def _homography_phi(ref, nb, n_w, d_ref, d_nb):
    """|| p - H_nr H_rn p || with H from the two (possibly different) world planes."""
    n_r, dist_r = _plane_cam(ref, n_w, d_ref)
    n_n, dist_n = _plane_cam(nb, n_w, d_nb)
    H, W = int(ref["height"]), int(ref["width"])
    P = H * W
    H_rn = G.homography(ref, nb, n_r.expand(P, 3), torch.full((P,), dist_r, device=n_r.device))
    H_nr = G.homography(nb, ref, n_n.expand(P, 3), torch.full((P,), dist_n, device=n_r.device))
    iy, ix = torch.meshgrid(torch.arange(H, device=n_r.device), torch.arange(W,
                            device=n_r.device), indexing="ij")
    p = torch.stack([ix, iy, torch.ones_like(ix)], -1).reshape(-1, 3).float()
    q = torch.einsum("bij,bj->bi", H_nr @ H_rn, p)
    return torch.norm(q[:, :2] / q[:, 2:] - p[:, :2], dim=-1)


def mv_geometry(method, device):
    """C4: forward-backward reprojection error and occlusion weights."""
    ref, nb, n_w, d = _two_views(device)
    dr = _analytic_depth(ref, n_w, d)
    dn = _analytic_depth(nb, n_w, d)
    noise, mask = G.forward_backward_error(ref, dr, nb, dn)
    if int(mask.sum()) < 0.5 * mask.numel():
        raise AssertionError("too few pixels project into the neighbour")
    consistent = noise[mask].max().item()
    if consistent > 0.02:
        raise AssertionError(f"consistent planes give phi up to {consistent}")
    out = {"consistent_max_phi": consistent}
    for scale, label in ((1.01, "small"), (1.6, "large")):
        dn2 = _analytic_depth(nb, n_w, d * scale)
        noise2, mask2 = G.forward_backward_error(ref, dr, nb, dn2)
        phi_h = _homography_phi(ref, nb, n_w, d, d * scale)
        sel = mask2 & (phi_h < 50)
        rel = ((noise2[sel] - phi_h[sel]).abs() / phi_h[sel].clamp_min(1e-3)).median().item()
        if rel > 0.05:
            raise AssertionError(f"{label}: reprojection != homography phi (median rel {rel})")
        w, keep = G.occlusion_weights(noise2, mask2, 1.0, True)
        if label == "large" and keep.float().mean() > 0.05:
            raise AssertionError("large inconsistency not rejected by phi < 1")
        if label == "small":
            if keep.sum() == 0 or (w[keep] - torch.exp(-noise2[keep])).abs().max() > 1e-6:
                raise AssertionError("weights != exp(-phi) on kept pixels")
            if w.requires_grad:
                raise AssertionError("occlusion weights must be detached")
        out[f"{label}_median_phi"] = noise2[mask2].median().item()
        out[f"{label}_kept_fraction"] = keep.float().mean().item()
        out[f"{label}_rel_vs_homography"] = rel
    return out


def _texture(X):
    return 0.5 + 0.22 * torch.sin(9.0 * X[:, 0] + 2.0 * X[:, 1]) + 0.2 * torch.cos(
        7.0 * X[:, 1] - 3.0 * X[:, 0])


def mv_ncc(method, device):
    """C5: homography-warped NCC on a textured plane."""
    ref, nb, n_w, d = _two_views(device)
    grays = []
    for cam in (ref, nb):
        X = G.depth_to_world(cam, _analytic_depth(cam, n_w, d))
        grays.append(_texture(X).reshape(int(cam["height"]), int(cam["width"])))
    n_r, dist = _plane_cam(ref, n_w, d)
    g = torch.Generator().manual_seed(1)
    xs = torch.randint(12, 52, (300,), generator=g)
    ys = torch.randint(10, 38, (300,), generator=g)
    pixels = torch.stack([xs, ys], -1).float().to(device)
    w = torch.ones(300, device=device)
    # homography maps reference pixels to the projection of the plane point
    Hm = G.homography(ref, nb, n_r.expand(300, 3), torch.full((300,), dist, device=device))
    p = torch.cat([pixels, torch.ones(300, 1, device=device)], -1)
    q = torch.einsum("bij,bj->bi", Hm, p)
    q = q[:, :2] / q[:, 2:]
    dr = _analytic_depth(ref, n_w, d)
    X = G.depth_to_world(ref, dr).reshape(48, 64, 3)[ys.to(device), xs.to(device)]
    proj = G.project(nb, G.to_cam(nb, X))
    herr = (q - proj).abs().max().item()
    if herr > 1e-2:
        raise AssertionError(f"homography != projection ({herr} px)")
    res = {"homography_error_px": herr}
    for scale, label in ((1.0, "true"), (1.3, "wrong")):
        normal = torch.nn.Parameter(n_r.expand(300, 3).clone())
        distance = torch.nn.Parameter(torch.full((300,), dist * scale, device=device))
        ncc, keep = G.ncc_terms(ref, nb, grays[0], grays[1], normal, distance, pixels, w, 3)
        if ncc.numel():
            ncc.mean().backward()
        res[f"{label}_median_ncc"] = G.lncc(
            G.sample_gray(grays[0], pixels[:, None] + G.patch_offsets(3, device).float()),
            G.sample_gray(grays[1], G.patch_warp(G.homography(ref, nb, normal.detach(),
                          distance.detach()), pixels[:, None] + G.patch_offsets(3, device)
                          .float())))[0].median().item()
        res[f"{label}_kept"] = int(keep.sum())
        if label == "wrong":
            gd = distance.grad
            if gd is None or not torch.isfinite(gd).all() or gd.abs().sum() == 0:
                raise AssertionError("no gradient to the rendered distance")
    if res["true_median_ncc"] > 0.02 or res["wrong_median_ncc"] < 3 * res[
            "true_median_ncc"] + 0.05:
        raise AssertionError(f"NCC does not separate true/wrong planes: {res}")
    return res


def exposure(method, device):
    """C6: exposure-compensated L1 (gated) and coefficient learning (pgsr-tnt config)."""
    from dataclasses import replace

    from .link import LINK
    from .losses import create_rgb, ssim_same

    spec = _spec(method, "pgsr-rgb")
    spec = replace(spec, config={**spec.config, "exposure_compensation": True})
    saved = (LINK.cameras, LINK.gray, LINK.neighbours, LINK._viewmats, LINK.extra_optimizers)
    try:
        LINK.extra_optimizers = []
        cams = [_camera(device, 32, 32, 30.0, t=[0.1 * i, 0, 0]) for i in range(3)]
        LINK.set_views(cams, [torch.zeros(32, 32, device=device)] * 3, [[], [], []])
        fn = create_rgb(spec)
        g = torch.Generator().manual_seed(2)
        target = (torch.rand(32, 32, 3, generator=g) * 0.6 + 0.2).to(device)
        rgb = torch.nn.Parameter(target * 0.8)
        out = {"rgb": rgb, "camera": cams[1], "step": 1001}
        val = fn(None, out, target)
        s = 1 - ssim_same(rgb, target)
        raw = 0.8 * (rgb - target).abs().mean() + 0.2 * s
        if abs(float(val) - float(raw)) > 1e-6:  # a = b = 0 initially -> identical
            raise AssertionError("initial exposure must be the identity")
        before = float(fn(None, {**out, "step": 1000}, target))
        val.backward()
        ab = fn.exposure.ab
        if ab.grad is None or ab.grad[1].abs().sum() == 0 or ab.grad[0].abs().sum() != 0:
            raise AssertionError("exposure gradient must reach only the rendered view's (a, b)")
        LINK.step_extra()
        for _ in range(300):
            fn(None, out, target).backward()
            LINK.step_extra()
        a, b = ab[1].tolist()
        after = float(fn(None, out, target))
        ref = 0.8 * (math.exp(a) * rgb - target + b).abs().mean() + 0.2 * s
        if abs(after - float(ref)) > 1e-5:
            raise AssertionError("loss != (1-l) L1(exp(a) I + b) + l (1-SSIM(I))")
        if not after < before - 0.01:
            raise AssertionError(f"exposure did not reduce the loss ({before} -> {after})")
        at_1000 = float(fn(None, {**out, "step": 1000}, target))
        if abs(at_1000 - before) > 1e-6:
            raise AssertionError("exposure active at step <= 1000")
        noise = torch.rand(32, 32, 3, generator=g).to(device)
        gated = float(fn(None, {**out, "rgb": noise}, target))
        sn = 1 - ssim_same(noise, target)
        if float(sn) < 0.5 or abs(gated - float(0.8 * (noise - target).abs().mean()
                                                  + 0.2 * sn)) > 1e-6:
            raise AssertionError("SSIM gate (1-SSIM >= 0.5 -> raw image) not applied")
        return {"loss_before": before, "loss_after": after, "a": a, "b": b,
                "expected_a": math.log(1 / 0.8)}
    finally:
        (LINK.cameras, LINK.gray, LINK.neighbours, LINK._viewmats,
         LINK.extra_optimizers) = saved


def neighbour_graph(method, device):
    """Official image graph: distance-sorted (ties by angle), filtered, capped."""
    cfg = _spec(method, "pgsr-density").config
    # distance is the primary key: view 2 (0.10 m, 20 deg) precedes view 3 (0.12 m, 0 deg)
    centres = [[0, 0, 0], [0.2, 0, 0], [0.1, 0, 0], [0.12, 0, 0], [2.0, 0, 0], [0.005, 0, 0],
               [0.05, 0, 0], [0, 0.15, 0]]
    rots = [0, 0, 20, 0, 0, 0, 40, 25]
    vms = []
    for c, r in zip(centres, rots):
        R = torch.tensor(_rot_y(r))
        vm = torch.eye(4)
        vm[:3, :3] = R
        vm[:3, 3] = -(R @ torch.tensor(c, dtype=torch.float32))
        vms.append(vm)
    vms = torch.stack(vms).to(device)
    got = G.neighbour_graph(vms, cfg["multi_view_num"], cfg["multi_view_max_angle"],
                            cfg["multi_view_min_dis"], cfg["multi_view_max_dis"])[0]
    cand = []
    for j in range(1, len(centres)):
        dist = float(np.linalg.norm(np.array(centres[j], dtype=np.float64)))
        ang = abs(rots[j])
        if ang < 30 and 0.01 < dist < 1.5:
            cand.append((dist, ang, j))
    expect = [j for _, _, j in sorted(cand)][: cfg["multi_view_num"]]
    if got != expect:
        raise AssertionError(f"neighbours {got} != {expect}")
    capped = G.neighbour_graph(vms, 2)[0]
    if capped != expect[:2]:
        raise AssertionError("max neighbour count not applied")
    return {"neighbours_of_0": got}


def _density(method, device, **overrides):
    from dataclasses import replace

    from .density import Density

    spec = _spec(method, "pgsr-density")
    return Density(replace(spec, config={**spec.config, **overrides}))


def _state(device, means, scales, opac):
    n = len(means)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(means, dtype=torch.float32, device=device)),
        "scales": torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32,
                                                  device=device).log()),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * n, device=device)),
        "opacities": torch.nn.Parameter(torch.logit(torch.tensor(opac, device=device))),
        "sh0": torch.nn.Parameter(torch.arange(n, dtype=torch.float32, device=device)[:, None,
                                  None].expand(n, 1, 3).clone()),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=device)),
    })
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for p in params.values():
        p.grad = torch.ones_like(p)
    for o in opts.values():
        o.step()
    return params, opts


def density_recipe(method, device):
    """C7: official densify / prune / reset / LR rules on hand-built statistics."""
    from splatstudio.state import check_state

    def scenario(**over):
        d = _density(method, device, use_multi_view_trim=False, **over)
        s, b = 0.0005, 0.01  # extent 1 -> percent_dense * extent = 0.001
        means = [[i * 1.0, 0, 3] for i in range(7)]
        scales = [[s] * 3, [b, s, s], [b, s, s], [b, s, s], [s] * 3, [0.2, s, s], [b, s, s]]
        opac = [0.5, 0.5, 0.5, 0.5, 0.003, 0.5, 0.5]
        params, opts = _state(device, means, scales, opac)
        d.initialize(params, opts, 1.0)
        st = d.stats
        st["denom"][:] = 1
        st["denom_abs"][:] = 1
        st["accum"][:] = torch.tensor([3e-4, 3e-4, 1e-4, 1e-4, 0, 0, 0], device=device)
        st["accum_abs"][:] = torch.tensor([0, 0, 5e-4, 5e-4, 0, 0, 0], device=device)
        st["max_radii"][:] = torch.tensor([5, 5, 30, 10, 5, 5, 100.0], device=device)
        return d, params, opts

    d, params, opts = scenario()
    sh_before = params["sh0"][:, 0, 0].tolist()
    d.densify_and_prune(params, opts, world_prune=True)
    last = d.state["last"]
    sh = sorted(params["sh0"][:, 0, 0].tolist())
    # 0 cloned (x2), 1 split (2 children), 2 abs-split (2 children), 3 kept (radius 10),
    # 4 pruned (opacity), 5 pruned (world size), 6 kept (screen-size prune not applied)
    expect = sorted([0, 0, 1, 1, 2, 2, 3, 6])
    if [round(v) for v in sh] != expect or last["clone"] != 1 or last["abs_split"] != 1:
        raise AssertionError(f"densify result {sh} / {last} (input ids {sh_before})")
    check_state(type("M", (), {"fields": ()})(), params, opts)
    ids = params["sh0"][:, 0, 0].round()
    clones = params["means"][ids == 0]
    if (clones[0] - clones[1]).abs().max() == 0:
        raise AssertionError("clone position not resampled")
    child = params["scales"][ids == 1].exp()
    if (child[:, 0] - 0.01 / 1.6).abs().max() > 1e-4:  # Adam warm-up step moved s by ~0.1%
        raise AssertionError("split scale != s / 1.6")
    d, params, opts = scenario()
    d.densify_and_prune(params, opts, world_prune=False)
    if 5 not in [round(v) for v in params["sh0"][:, 0, 0].tolist()]:
        raise AssertionError("world-size prune applied before step 3000")
    d, params, opts = scenario(max_abs_split_points=0)
    d.densify_and_prune(params, opts, world_prune=True)
    if d.state["last"]["abs_split"] != 0:
        raise AssertionError("max_abs_split_points cap ignored")
    # accumulate: max_radii only where observed; abs gradient norm of [:, :2]
    d, params, opts = scenario()
    d._reset_stats(len(params["means"]), device)
    g = torch.zeros(7, 3, device=device)
    g[:, 0], g[:, 1] = 3.0, 4.0
    m2 = torch.zeros(7, 3, device=device, requires_grad=True)
    m2.grad = g.clone()
    ma = torch.zeros(7, 3, device=device, requires_grad=True)
    ma.grad = 2 * g
    info = {"means2d": m2, "means2d_abs": ma,
            "radii": torch.tensor([4, 4, 0, 4, 4, 4, 4], device=device),
            "observe": torch.tensor([1, 0, 1, 1, 1, 1, 1], device=device)}
    d.accumulate(info)
    if d.stats["max_radii"].tolist() != [4, 0, 0, 4, 4, 4, 4] or abs(
            float(d.stats["accum"][0]) - 5) > 1e-6 or abs(float(d.stats["accum_abs"][0])
                                                         - 10) > 1e-6:
        raise AssertionError(f"accumulate wrong {d.stats}")
    # reset at 3000 (< densify_until) and xyz LR schedule
    d, params, opts = scenario()
    d.after_backward(params, opts, 3000, {"info": info})
    if params["opacities"].sigmoid().max() > 0.0100001:
        raise AssertionError("opacity reset not applied")
    if opts["opacities"].state[params["opacities"]]["exp_avg"].abs().sum() != 0:
        raise AssertionError("opacity moments not zeroed")
    lr_end, lr_mid = d.xyz_lr(30000), d.xyz_lr(15000)
    if abs(lr_end - 1.6e-6) > 1e-12 or abs(lr_mid - math.sqrt(1.6e-4 * 1.6e-6)) > 1e-10:
        raise AssertionError("xyz LR schedule wrong")
    return {"densify": last, "lr_mid": lr_mid}


def multi_view_trim(method, device):
    """C7: trim removes Gaussians observed (T > 0.5) in < 2 training views."""
    _cuda(device)
    from .link import LINK
    from .renderer import create_renderer

    create_renderer(_spec(method, "pgsr-plane"))
    d = _density(method, device)
    cams = [_camera(device, 32, 32, 30.0, t=[-x, 0, 0]) for x in (0.0, 0.2, -0.2)]
    # A visible everywhere in front, B only in camera 1, C hidden behind opaque D, D wall
    params, opts = _state(device, [[0, 0, 1.5], [0.95, 0, 1.5], [0.3, 0.3, 4.0], [0, 0, 2.0]],
                          [[0.05] * 3, [0.015] * 3, [0.05] * 3, [3.0, 3.0, 0.01]],
                          [0.9, 0.9, 0.9, 0.99])
    d.initialize(params, opts, 1.0)
    d.set_training_frames(cams, [torch.zeros(32, 32, 3, device=device)] * 3)
    counts = d.observe_counts(params).tolist()
    d.trim(params, opts)
    kept = sorted(round(v) for v in params["sh0"][:, 0, 0].tolist())
    LINK.cameras = LINK.gray = LINK.neighbours = None
    if counts[0] < 2 or counts[1] != 1 or counts[2] != 0 or kept != [0, 3]:
        raise AssertionError(f"observe counts {counts}, kept {kept}")
    return {"observe_counts": counts, "kept": kept}
