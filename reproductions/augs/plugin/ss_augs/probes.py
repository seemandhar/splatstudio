"""Mechanism probes: each compares the implementation to an independently written reference and
fails when the mechanism is removed or wrong."""

import math

import numpy as np
import torch


def _fail(cond, msg):
    if not cond:
        raise AssertionError(msg)


def lobe_probe(method, device):
    from .renderer import augmented_opacities

    dev = torch.device(device)
    n = 5
    means = torch.zeros(n, 3, device=dev)
    campos = torch.tensor([0.0, 0.0, 5.0], device=dev)
    # lobe directions at increasing angles from the view direction (+z)
    angles = torch.tensor([0.0, 0.3, 0.8, 1.4, 0.5], device=dev)
    lobe_dir = torch.stack((torch.sin(angles), torch.zeros(n, device=dev), torch.cos(angles)), 1)
    params = {
        "means": means.clone().requires_grad_(True),
        "opacities": torch.full((n,), 0.4, device=dev).requires_grad_(True),
        "lobe_dir": (lobe_dir * 2.0).requires_grad_(True),  # unnormalised on purpose
        "lobe_t": torch.tensor([0.5, 0.5, 0.2, 1.0, 0.5], device=dev).requires_grad_(True),
        "lobe_beta": torch.tensor([0.0, 0.7, 0.0, -0.5, 0.0], device=dev).requires_grad_(True),
        "aug_flag": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], device=dev),
    }
    out = augmented_opacities(params, campos)
    # independent numpy Eq. 1
    a = 1 / (1 + np.exp(-0.4))
    ref = []
    for i in range(n):
        if i == 4:
            ref.append(a)
            continue
        th = float(np.arccos(np.clip(np.cos(float(angles[i])), -0.9999, 0.9999)))
        T = float(params["lobe_t"][i])
        b = 0.5 * (np.cos(min(max(th / T, 0.0), np.pi)) + 1)
        ref.append(a * (b ** np.exp(float(params["lobe_beta"][i])) if b >= 1e-5 else 0.0))
    ref = np.array(ref)
    err = float(np.abs(out.detach().cpu().numpy() - ref).max())
    _fail(err < 1e-5, f"lobe opacity mismatch {err}")
    _fail(float(out[2]) == 0.0, "lobe must vanish beyond T*pi")
    _fail(float(out[1]) < a * 0.999 and float(out[0]) > a * 0.999, "lobe must peak on its axis")
    out[:4].sum().backward()
    for k in ("lobe_dir", "lobe_t", "lobe_beta"):
        g = params[k].grad
        _fail(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"no grad for {k}")
    return {"max_abs_error": err, "opacities": out.detach().cpu().tolist()}


def median_depth_probe(method, device):
    from .renderer import median_depth

    dev = torch.device(device)
    W = H = 32
    K = torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=dev)
    viewmat = torch.eye(4, device=dev)
    cam = {"K": K, "viewmat": viewmat, "width": W, "height": H}
    results = {}
    for name, (o_front, expect) in {"opaque_front": (0.5, 2.0), "faint_front": (0.2, 4.0)}.items():
        # two large fronto-parallel Gaussians on the optical axis at z = 2 and z = 4
        params = {
            "means": torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], device=dev),
            "scales": torch.log(torch.tensor([[1.0, 1.0, 0.01], [2.0, 2.0, 0.01]], device=dev)),
            "quats": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=dev),
            "opacities": torch.logit(torch.tensor([o_front, 0.9], device=dev)),
        }
        d = median_depth(params, cam, t_threshold=0.7)
        c = float(d[16, 16])
        # independent reference at the centre pixel: alphas ~ o (the centre ray hits both means)
        T, chosen = 1.0, None
        for z, o in ((2.0, o_front), (4.0, 0.9)):
            if chosen is None or T >= 0.7:
                chosen = z
            T *= 1 - o
        _fail(abs(chosen - expect) < 1e-6, "reference compositing broken")
        _fail(abs(c - expect) < 0.02, f"{name}: depth {c} != {expect}")
        results[name] = c
    return results


def fit2d_probe(method, device):
    from .fit2d import fit_view, image_loss, render2d, ssim_map
    from .kernels import fused_ssim_map

    dev = torch.device(device)
    a = torch.rand(3, 40, 56, device=dev, requires_grad=True)
    b = torch.rand(3, 40, 56, device=dev)
    fused = fused_ssim_map(a[None], b[None])[0]
    ref = ssim_map(a, b)
    ga = torch.autograd.grad(fused.mean(), a)[0]
    gr = torch.autograd.grad(ref.mean(), a)[0]
    ssim_err = float((fused - ref).abs().max())
    grad_err = float((ga - gr).abs().max() / gr.abs().max())
    _fail(ssim_err < 1e-4 and grad_err < 1e-3, f"fused SSIM differs from reference {ssim_err} {grad_err}")
    H, W = 320, 400  # init scale 0.005 * min(H, W) = 1.6 px as on real images
    bg = torch.full((3, H, W), 0.5, device=dev)
    gt = bg.clone()
    gt[:, 60:90, 180:220] = torch.tensor([0.9, 0.2, 0.1], device=dev)[:, None, None]
    depth = torch.full((H, W), 3.0, device=dev)
    cfg = {"xy_lr": 0.001, "color_lr": 0.01, "opacity_lr": 0.02, "scale_lr": 0.001,
           "rotation_lr": 0.02, "add_fraction": 0.2, "coarse_steps": 60, "finetune_steps": 150,
           "init_opacity": 0.02, "init_scale": 0.005}
    gen = torch.Generator(device=dev).manual_seed(0)
    before = float(image_loss(bg, gt))
    g, info = fit_view(bg, gt, depth, 100, cfg, gen)
    after = float(image_loss(render2d(g, bg)[0], gt))
    xy = g.p["xy"].detach()
    inside = ((xy[:, 0] >= 55) & (xy[:, 0] <= 95) & (xy[:, 1] >= 175) & (xy[:, 1] <= 225)).float()
    _fail(g.size() > 0, "all 2D kernels pruned")
    _fail(after < 0.9 * before, f"2D kernels did not reduce the loss ({before} -> {after})")
    _fail(float(inside.mean()) > 0.8, "kernels not placed in the error region")
    return {"loss_before": before, "loss_after": after, "kernels": g.size(),
            "fused_ssim_max_error": ssim_err, "fused_ssim_grad_rel_error": grad_err,
            "fraction_in_error_region": float(inside.mean()), **info}


def _reproject_cov(mean, quat, log_scale, K, viewmat):
    from gsplat.utils import normalized_quat_to_rotmat

    R = normalized_quat_to_rotmat(torch.nn.functional.normalize(quat[None], dim=-1))[0].double()
    S = torch.diag(log_scale.double().exp() ** 2)
    cov3 = R @ S @ R.T
    t = viewmat[:3, :3].double() @ mean.double() + viewmat[:3, 3].double()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    J = torch.tensor([[fx / t[2], 0, -fx * t[0] / t[2] ** 2],
                      [0, fy / t[2], -fy * t[1] / t[2] ** 2]], dtype=torch.float64)
    Wr = viewmat[:3, :3].double()
    return J @ Wr @ cov3 @ Wr.T @ J.T


def inverse_probe(method, device):
    from .density import DEFAULTS, lobe_t_init
    from .fit2d import cov2d
    from .inverse import inverse_splat

    dev = torch.device("cpu")
    H, W = 60, 80
    K = torch.tensor([[70.0, 0, 40], [0, 70.0, 30], [0, 0, 1]])
    cams = [{"K": K, "viewmat": torch.eye(4), "width": W, "height": H}]
    # second camera shifted along x, looking the same way (sees the point, small angle)
    v2 = torch.eye(4)
    v2[0, 3] = -0.5
    cams.append({"K": K, "viewmat": v2, "width": W, "height": H})
    depth = torch.full((H, W), 3.0)
    depth[:, 44:] = 10.0  # depth discontinuity right of column 44
    xy = torch.tensor([[30.0, 40.0], [20.0, 20.0]])
    log_scale = torch.log(torch.tensor([[3.0, 1.5], [2.0, 2.0]]))
    rot = torch.tensor([0.4, 0.0])
    opac = torch.logit(torch.tensor([0.6, 0.5]))
    color = torch.tensor([[0.2, 0.5, 0.8], [0.5, 0.5, 0.5]])
    res = inverse_splat(xy, log_scale, rot, opac, color, depth, 0, cams)
    m = res["means"]
    _fail(abs(float(m[0, 2]) - 3.0) < 1e-4, f"kernel across the discontinuity not on the near "
          f"surface: z={float(m[0, 2])}")
    target = cov2d(log_scale.exp(), rot)
    errs = []
    for i in range(2):
        c = _reproject_cov(m[i], res["quats"][i], res["scales"][i], K, cams[0]["viewmat"])
        tgt = torch.tensor([[target[i, 0], target[i, 1]], [target[i, 1], target[i, 2]]],
                           dtype=torch.float64)
        errs.append(float((c - tgt).norm() / tgt.norm()))
    _fail(max(errs) < 0.25, f"reprojected covariance error {errs}")
    from gsplat.utils import normalized_quat_to_rotmat

    R = normalized_quat_to_rotmat(res["quats"])
    normal_z = R[:, 2, 2]  # third axis (smallest variance) . camera z
    _fail(bool((normal_z < -0.99).all()), f"normal must face the camera: {normal_z.tolist()}")
    d = res["lobe_dir"][1]
    expect = -m[1] / m[1].norm()
    _fail(float((d - expect).norm()) < 1e-5, "lobe direction must point to the camera")
    theta = res["theta"]
    p = m[1].double()
    c2 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    ang = math.acos(float(((-p) / p.norm()) @ ((c2 - p) / (c2 - p).norm())))
    _fail(abs(float(theta[1]) - ang) < 1e-4, "theta_bar must be the angle to the other camera")
    T = lobe_t_init(theta, DEFAULTS)
    _fail(abs(float(T[1]) - min(max(7 * ang / math.pi, 0.01), 1.0)) < 1e-6, "T init rule")
    return {"depths": m[:, 2].tolist(), "cov_rel_error": errs, "theta": theta.tolist(),
            "T": T.tolist(), "device": str(dev)}


def joint_probe(method, device):
    from .optim import SparseRowAdam

    torch.manual_seed(0)
    n_base, n = 4, 7
    p = torch.nn.Parameter(torch.randn(n, 3))
    o = torch.nn.Parameter(torch.randn(n))
    p0, o0 = p.detach().clone(), o.detach().clone()
    opt_p = SparseRowAdam(p, 0.1, n_base)
    opt_o = SparseRowAdam(o, 0.1, n_base, include_used_base=True)
    gp = torch.randn(n, 3)
    go = torch.randn(n)
    go[1] = 0.0  # an unused original Gaussian
    p.grad, o.grad = gp.clone(), go.clone()
    opt_p.step()
    opt_o.step()
    # independent first step without bias correction: m = 0.1 g, v = 0.001 g^2
    step_p = -0.1 * (0.1 * gp) / ((0.001 * gp * gp).sqrt() + 1e-15)
    _fail(torch.equal(p.detach()[:n_base], p0[:n_base]), "original rows must stay frozen")
    _fail(torch.allclose(p.detach()[n_base:], p0[n_base:] + step_p[n_base:], atol=1e-6),
          "augmented rows: not the no-bias-correction Adam step")
    step_o = -0.1 * (0.1 * go) / ((0.001 * go * go).sqrt() + 1e-15)
    used = go != 0
    _fail(torch.allclose(o.detach()[used], o0[used] + step_o[used], atol=1e-6),
          "opacity of used Gaussians must update")
    _fail(float(o.detach()[1]) == float(o0[1]), "unused original opacity must stay frozen")
    _fail(float(opt_o.state[o]["exp_avg"][1]) == 0.0, "unused rows keep their moments")
    return {"aug_step_norm": float(step_p[n_base:].norm()), "used_rows": int(used.sum())}
