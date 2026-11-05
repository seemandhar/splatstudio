"""Mechanism probes for the DOF-GS port. Each raises AssertionError when the mechanism is
removed or wrong and returns its measurements."""

import math

import numpy as np
import torch

from . import bank as bank_mod
from .losses import create_mask_losses, create_objective, l_rec, pearson_mask_loss, ssim_same
from .raster import rasterize

C_COC = 1.0 / (2.0 * math.log(4.0))  # a = R^2 / (2 ln 4)
SH_C0 = 0.28209479177387814


class _Spec:
    def __init__(self, config):
        self.config = config


def _component_config(method, name):
    return dict(next(c for c in method.components if c.name == name).config)


def _camera(device, W=64, H=48, f=60.0):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device)
    return {"K": K, "viewmat": torch.eye(4, device=device), "width": W, "height": H}


def _scene(device, n=12, seed=0, opacity_max=0.3):
    g = torch.Generator().manual_seed(seed)
    means = torch.empty(n, 3)
    means[:, 0] = (torch.rand(n, generator=g) - 0.5) * 1.2
    means[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.9
    means[:, 2] = 2.0 + torch.rand(n, generator=g) * 4.0
    scales = 0.03 + torch.rand(n, 3, generator=g) * 0.08
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opac = 0.05 + torch.rand(n, generator=g) * (opacity_max - 0.05)
    colors = 0.1 + torch.rand(n, 3, generator=g) * 0.8
    sh = torch.zeros(n, 16, 3)
    sh[:, 0] = (colors - 0.5) / SH_C0
    return {k: v.to(device) for k, v in dict(means=means, scales=scales, quats=quats,
                                              opac=opac, sh=sh, colors=colors).items()}


def _quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def reference_render(scene, camera, F, K, kernel_size=0.1, background=(0.0, 0.0, 0.0),
                     coc_on=True, detach_z_in_coc=True):
    """Independent dense float64 EWA renderer of the DOF model (paper Sec. 3.2, supp. A):
    Sigma'' = J W Sigma W^T J^T + aI + kI, opacity * sqrt(det(S'+aI)/det(S'+aI+kI)),
    sorted by ray distance, alpha <= 0.99, skip alpha < 1/255, stop when T < 1e-4."""
    dd = torch.float64
    means = scene["means"].to(dd)
    Kc, V = camera["K"].to(dd), camera["viewmat"].to(dd)
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]
    t = means @ V[:3, :3].T + V[:3, 3]
    z = t[:, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tx = torch.clamp(t[:, 0] / z, -1.3 * tanx, 1.3 * tanx) * z
    ty = torch.clamp(t[:, 1] / z, -1.3 * tany, 1.3 * tany) * z
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zero, -fx * tx / z**2], -1),
                     torch.stack([zero, fy / z, -fy * ty / z**2], -1)], -2)
    R = _quat_to_rot(scene["quats"].to(dd))
    S = torch.diag_embed(scene["scales"].to(dd))
    M = R @ S
    cov3 = M @ M.transpose(1, 2)
    T = J @ V[:3, :3]
    cov2 = T @ cov3 @ T.transpose(1, 2)
    zc = z.detach() if detach_z_in_coc else z
    rad = 0.5 * K * torch.abs(1.0 / F - 1.0 / zc) if coc_on else torch.zeros_like(z)
    a = C_COC * rad**2
    eye = torch.eye(2, dtype=dd)
    covb = cov2 + a[:, None, None] * eye
    covk = covb + kernel_size * eye
    det_b = torch.linalg.det(covb).clamp_min(1e-6)
    det_k = torch.linalg.det(covk).clamp_min(1e-6)
    opac = scene["opac"].to(dd) * torch.sqrt(det_b / (det_k + 1e-6) + 1e-6)
    conic = torch.linalg.inv(covk)
    center = torch.stack([fx * t[:, 0] / z + cx - 0.5, fy * t[:, 1] / z + cy - 0.5], -1)
    color = (scene["sh"][:, 0].to(dd) * SH_C0 + 0.5).clamp_min(0)
    dist = torch.linalg.vector_norm(t, dim=-1)
    order = torch.argsort(dist.detach())
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dd), torch.arange(W, dtype=dd), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    Tr = torch.ones(len(pix), dtype=dd)
    done = torch.zeros(len(pix), dtype=torch.bool)
    out = torch.zeros(len(pix), 3, dtype=dd)
    dep = torch.zeros(len(pix), dtype=dd)
    coc = torch.zeros(len(pix), dtype=dd)
    for i in order.tolist():
        if float(z[i]) <= 0.2:
            continue
        d = center[i] - pix
        power = -0.5 * (conic[i, 0, 0] * d[:, 0] ** 2 + conic[i, 1, 1] * d[:, 1] ** 2) \
            - conic[i, 0, 1] * d[:, 0] * d[:, 1]
        alpha = torch.clamp(opac[i] * torch.exp(power), max=0.99)
        use = (power <= 0) & (alpha >= 1 / 255) & ~done
        test_T = Tr * (1 - alpha)
        stop = use & (test_T < 1e-4)
        done = done | stop
        use = use & ~stop
        w = torch.where(use, alpha * Tr, torch.zeros_like(Tr))
        out = out + w[:, None] * color[i]
        dep = dep + w * dist[i]
        coc = coc + w * rad[i]
        Tr = torch.where(use, test_T, Tr)
    out = out + Tr[:, None] * torch.tensor(background, dtype=dd)
    return {"rgb": out.reshape(H, W, 3), "depth": dep.reshape(H, W, 1),
            "coc": coc.reshape(H, W, 1), "radius": rad}


def _kernel(scene, camera, F, K, device, kernel_size=0.1):
    Ft = torch.tensor([[F]], device=device, dtype=torch.float32) if not torch.is_tensor(F) else F
    Kt = torch.tensor([[K]], device=device, dtype=torch.float32) if not torch.is_tensor(K) else K
    bg = torch.zeros(3, device=device)
    return rasterize(scene["means"], scene["scales"], scene["opac"], scene["quats"], scene["sh"],
                     camera, Ft, Kt, 0, 0, bg, kernel_size)


def _cuda(device):
    if not str(device).startswith("cuda"):
        raise AssertionError("DOF kernel probes need CUDA")
    return torch.device(device)


def dof_kernel_reference(method, device="cuda"):
    device = _cuda(device)
    cam = _camera(device)
    scene = _scene(device)
    out = {}
    for F, K in ((3.5, 0.0), (3.5, 40.0), (2.5, 120.0)):
        k = _kernel(scene, cam, F, K, device)
        ref = reference_render({kk: v.cpu() for kk, v in scene.items()},
                               {kk: (v.cpu() if torch.is_tensor(v) else v)
                                for kk, v in cam.items()}, F, K)
        e_rgb = float((k["rgb"].cpu().double() - ref["rgb"]).abs().max())
        e_dep = float((k["depth"].cpu().double() - ref["depth"]).abs().max())
        e_coc = float((k["coc"].cpu().double() - ref["coc"]).abs().max())
        out[f"F{F}_K{K}"] = {"rgb_max_err": e_rgb, "depth_max_err": e_dep, "coc_max_err": e_coc,
                             "max_R_px": float(ref["radius"].max())}
        assert e_rgb < 2e-3, f"DOF kernel RGB differs from reference ({e_rgb})"
        assert e_dep < 2e-2, f"ray-distance depth differs ({e_dep})"
        assert e_coc < 2e-2 * max(1.0, float(ref["radius"].max())), f"CoC map differs ({e_coc})"
        if K > 0:  # the CoC term matters: a renderer without it is far off
            nococ = reference_render({kk: v.cpu() for kk, v in scene.items()},
                                     {kk: (v.cpu() if torch.is_tensor(v) else v)
                                      for kk, v in cam.items()}, F, K, coc_on=False)
            gap = float((k["rgb"].cpu().double() - nococ["rgb"]).abs().max())
            out[f"F{F}_K{K}"]["no_coc_gap"] = gap
            assert gap > 20 * e_rgb, "kernel output indistinguishable from a CoC-free render"
    return out


def pinhole_limit(method, device="cuda"):
    device = _cuda(device)
    cam = _camera(device)
    scene = _scene(device, seed=1)
    k0 = _kernel(scene, cam, 3.0, 0.0, device)
    ksmall = _kernel(scene, cam, 3.0, 1e-3, device)
    cpu_scene = {kk: v.cpu() for kk, v in scene.items()}
    cpu_cam = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in cam.items()}
    ref = reference_render(cpu_scene, cpu_cam, 3.0, 0.0, coc_on=False)
    e0 = float((k0["rgb"].cpu().double() - ref["rgb"]).abs().max())
    e_small = float((ksmall["rgb"] - k0["rgb"]).abs().max())
    assert e0 < 2e-3, f"Q=0 render is not the standard (pinhole) render ({e0})"
    assert float(k0["coc"].abs().max()) == 0.0, "CoC map must vanish for Q=0"
    assert e_small < 1e-4, f"Q->0 is discontinuous ({e_small})"
    # Focal-plane invariance: a Gaussian at z = f is unblurred for any aperture.
    one = {kk: v[:1].clone() for kk, v in scene.items()}
    one["means"][0] = torch.tensor([0.05, -0.02, 4.0], device=device)
    sharp = _kernel(one, cam, 4.0, 0.0, device)["rgb"]
    focused = _kernel(one, cam, 4.0, 500.0, device)["rgb"]
    defocused = _kernel(one, cam, 2.0, 500.0, device)["rgb"]
    e_plane = float((focused - sharp).abs().max())
    e_off = float((defocused - sharp).abs().max())
    assert e_plane < 1e-5 and e_off > 1e-2, "focal-plane Gaussian blurred / off-plane not blurred"
    return {"q0_vs_pinhole_reference": e0, "q1e-3_vs_q0": e_small, "focal_plane_err": e_plane,
            "off_plane_change": e_off}


def analytic_blur(method, device="cuda"):
    device = _cuda(device)
    W = H = 96
    cam = _camera(device, W, H, 80.0)
    z, s, o, F = 4.0, 0.1, 0.05, 2.0
    scene = {"means": torch.tensor([[0.0, 0.0, z]], device=device),
             "scales": torch.full((1, 3), s, device=device),
             "quats": torch.tensor([[1.0, 0, 0, 0]], device=device),
             "opac": torch.tensor([o], device=device),
             "sh": torch.zeros(1, 16, 3, device=device)}
    scene["sh"][0, 0] = (1.0 - 0.5) / SH_C0  # white
    ys, xs = torch.meshgrid(torch.arange(H, device=device).double(),
                            torch.arange(W, device=device).double(), indexing="ij")
    d2 = (xs - (W / 2 - 0.5)) ** 2 + (ys - (H / 2 - 0.5)) ** 2
    sigma2 = (80.0 * s / z) ** 2
    out = {}
    for Q in (0.0, 30.0, 80.0):
        img = _kernel(scene, cam, F, Q, device)["rgb"][..., 0].double()
        R = 0.5 * Q * abs(1.0 / z - 1.0 / F)
        v_true = sigma2 + 0.1 + C_COC * R * R
        peak_true = o * (sigma2 + C_COC * R * R) / (sigma2 + C_COC * R * R + 0.1)
        m = img > 1e-3
        A = torch.stack([torch.ones_like(d2[m]), d2[m]], -1)
        coef = torch.linalg.lstsq(A, torch.log(img[m])[:, None]).solution[:, 0]
        v_fit, peak_fit = float(-0.5 / coef[1]), float(torch.exp(coef[0]))
        out[f"Q{Q}"] = {"R_px": R, "var_fit": v_fit, "var_true": v_true, "peak_fit": peak_fit,
                        "peak_true": peak_true}
        assert abs(v_fit / v_true - 1) < 0.02, f"blur variance {v_fit} != {v_true}"
        assert abs(peak_fit / peak_true - 1) < 0.02, f"blur peak {peak_fit} != {peak_true}"
    assert out["Q80.0"]["var_fit"] > 3 * out["Q0.0"]["var_fit"], "aperture does not blur"
    return out


def fk_gradients(method, device="cuda"):
    device = _cuda(device)
    cam = _camera(device)
    scene = _scene(device, seed=2)
    g = torch.Generator().manual_seed(5)
    wimg = torch.rand(48, 64, 3, generator=g)
    F0, K0 = 3.2, 60.0
    Ft = torch.tensor([[1.0], [F0]], device=device, requires_grad=True)
    Kt = torch.tensor([[0.0], [K0]], device=device, requires_grad=True)
    means = scene["means"].clone().requires_grad_(True)
    opac = scene["opac"].clone().requires_grad_(True)
    bg = torch.zeros(3, device=device)
    k = rasterize(means, scene["scales"], opac, scene["quats"], scene["sh"], cam, Ft, Kt, 1, 0,
                  bg, 0.1)
    (k["rgb"].cpu() * wimg).sum().backward()
    res = {}
    for detach in (True, False):
        Fr = torch.tensor(F0, dtype=torch.float64, requires_grad=True)
        Kr = torch.tensor(K0, dtype=torch.float64, requires_grad=True)
        sc = {kk: v.detach().cpu() for kk, v in scene.items()}
        sc["means"] = sc["means"].double().requires_grad_(True)
        sc["opac"] = sc["opac"].double().requires_grad_(True)
        ref = reference_render(sc, {kk: (v.cpu() if torch.is_tensor(v) else v)
                                    for kk, v in cam.items()}, Fr, Kr, detach_z_in_coc=detach)
        (ref["rgb"] * wimg.double()).sum().backward()
        res[detach] = (Fr.grad.item(), Kr.grad.item(), sc["means"].grad, sc["opac"].grad)
    gF, gK = Ft.grad[1, 0].item(), Kt.grad[1, 0].item()
    rF, rK, rM, rO = res[True]
    eF, eK = abs(gF - rF) / (abs(rF) + 1e-6), abs(gK - rK) / (abs(rK) + 1e-6)
    eM = float((means.grad.cpu().double() - rM).abs().max() / (rM.abs().max() + 1e-12))
    eO = float((opac.grad.cpu().double() - rO).abs().max() / (rO.abs().max() + 1e-12))
    eM_att = float((means.grad.cpu().double() - res[False][2]).abs().max()
                   / (res[False][2].abs().max() + 1e-12))
    assert Ft.grad[0, 0].item() == 0.0 and Kt.grad[0, 0].item() == 0.0, "gradient leaked to vIdx 0"
    assert eF < 0.03 and eK < 0.03, f"dL/dF or dL/dK wrong (rel {eF:.3g}, {eK:.3g})"
    assert abs(rF) > 1e-6 and abs(rK) > 1e-6, "degenerate fixture"
    assert eM < 0.03 and eO < 0.03, f"means/opacity gradients wrong ({eM:.3g}, {eO:.3g})"
    return {"dL_dF": [gF, rF], "dL_dK": [gK, rK], "rel_err_F": eF, "rel_err_K": eK,
            "rel_err_means": eM, "rel_err_opacity": eO,
            "rel_err_means_vs_coc_z_attached": eM_att}


def camera_init(method, device="cpu"):
    rng = np.random.default_rng(0)
    z = rng.uniform(6.0, 12.0, size=101)  # diopter range small enough that Q_init > 101
    inv = np.sort(1.0 / z)
    z_med = 1.0 / inv[50] + 0.7
    z_max, z_min = 1.0 / inv[5], 1.0 / inv[95]
    kv = 15.0 / (abs(1 / z_min - 1 / z_max) + 1e-6)
    fs, ks = bank_mod.initial_camera_params(torch.tensor(z), 0.7, 15.0)
    assert ks > 0.01, "fixture must not hit the aperture clamp"
    assert abs(fs - z_med / 10) < 1e-9 and abs(ks - 10 * (kv - 100) / 1000) < 1e-6
    f_init = float(bank_mod.focal_from_raw(torch.tensor(fs)))
    q_init = float(bank_mod.aperture_from_raw(torch.tensor(ks)))
    assert abs(f_init - z_med) < 1e-6 and abs(q_init - kv) < 1e-3, "reparameterisation broken"
    cams = [{"viewmat": torch.eye(4) + i * 0.01, "width": 16, "height": 12} for i in range(3)]
    cfg = {"coc_max": 15.0, "fdist_lr": 0.05, "kapert_lr": 0.01, "mask_nn_lr": 5e-4,
           "iter_defocus_start": 100, "iter_stop_cam_gd": 150}
    bank = bank_mod.DofBank(cams, torch.tensor(z, dtype=torch.float32), 0.7, cfg, "cpu")
    f0 = bank.Fs.detach().clone()
    for step in range(1, 100):
        (bank.focal().sum() + bank.aperture().sum()).backward()
        bank.after_step(step)
    assert torch.equal(bank.Fs.detach(), f0), "F updated before iter_defocus_start"
    assert abs(float(bank.Fs.grad[0, 0]) - 99 * 10.0) < 1e-3, "grads must accumulate before 100"
    for step in range(100, 151):
        (bank.focal().sum() + bank.aperture().sum()).backward()
        bank.after_step(step)
    f150 = bank.Fs.detach().clone()
    assert not torch.equal(f150, f0) and not bank.Fs.requires_grad, "schedule 100..15000 broken"
    assert bank.view_index(cams[2]["viewmat"]) == 2 and bank.view_index(torch.zeros(4, 4)) is None
    return {"f_init": f_init, "q_init": q_init, "cam_steps": bank.counts["cam_steps"],
            "f_after": float(f150[0, 0] * 10)}


def _fake_outputs(step, H=24, W=32, seed=0, with_aif=True):
    g = torch.Generator().manual_seed(seed)
    cams = [{"viewmat": torch.eye(4) + i * 0.01, "width": W, "height": H} for i in range(2)]
    cfg = {"coc_max": 15.0, "fdist_lr": 0.05, "kapert_lr": 0.01, "mask_nn_lr": 5e-4,
           "iter_defocus_start": 100, "iter_stop_cam_gd": 15000}
    torch.manual_seed(seed)
    bank = bank_mod.DofBank(cams, torch.linspace(1, 9, 50), 0.0, cfg, "cpu")
    rgb = torch.rand(H, W, 3, generator=g).requires_grad_(True)
    aif = torch.rand(H, W, 3, generator=g).requires_grad_(True) if with_aif else None
    outputs = {"rgb": rgb, "rgb_aif": aif, "depth": torch.rand(H, W, 1, generator=g) * 5,
               "coc": torch.rand(H, W, 1, generator=g) * 4, "step": step,
               "dof": {"view": 1, "bank": bank}}
    return outputs, torch.rand(H, W, 3, generator=g)


def iln_mask_losses(method, device="cpu"):
    outputs, gt = _fake_outputs(6000)
    bank = outputs["dof"]["bank"]
    net = bank.mask_net
    shapes = {n: tuple(p.shape) for n, p in net.named_parameters()}
    assert shapes["conv_rgb.0.weight"] == (48, 5, 3, 3)
    assert shapes["conv_rgb.3.weight"] == (16, 48, 3, 3)
    assert shapes["mlp_base_mlp.0.weight"] == (20, 64, 1, 1)
    assert shapes["mlp_mask1.weight"] == (1, 20, 1, 1)
    assert shapes["embedding_camera.weight"] == (2, 32)
    assert tuple(bank.pos_enc.shape) == (1, 24, 32, 16)
    # Pearson / entropy vs numpy
    g = np.random.default_rng(1)
    a, b = g.random(500), g.random(500) * 3 + 1
    pcc = np.mean((a - a.mean()) * (b - b.mean())) / (np.std(a, ddof=1) * np.std(b, ddof=1))
    ours = float(pearson_mask_loss(torch.tensor(a), torch.tensor(b)))
    assert abs(ours - (1 - pcc)) < 1e-9, "Pearson mask loss formula"
    mcfg = _component_config(method, "dofgs-mask")
    lam, reg = float(mcfg["lambda_mask_loss"]), float(mcfg["reg_factor"])
    mask_loss = create_mask_losses(_Spec(mcfg))
    val = float(mask_loss(None, outputs, gt))
    m = outputs["dof"]["mask"].detach().numpy()[0]
    coc = outputs["coc"].numpy()[..., 0]
    lc = np.log1p(coc)
    t = (lc - lc.min()) / (abs(lc.max() - lc.min()) + 1e-5)
    mm, tt = m.ravel(), t.ravel() * (m.max() - m.min()) + m.min()
    p = np.mean((mm - mm.mean()) * (tt - tt.mean())) / (np.std(mm, ddof=1) * np.std(tt, ddof=1))
    expect = lam * (1 - p) - lam * reg * np.mean(m * np.log(m + 1e-6))
    assert abs(val - expect) < 1e-6, f"mask losses {val} != {expect}"
    # Gradient routing: d/d rgb only from L_a, d/d aif only from L_b (official two backwards)
    outputs, gt = _fake_outputs(6000)
    obj = create_objective(_Spec({}))
    obj(None, outputs, gt).backward()
    g_rgb, g_aif = outputs["rgb"].grad.clone(), outputs["rgb_aif"].grad.clone()
    o2, gt2 = _fake_outputs(6000)
    mk = o2["dof"]["bank"].mask_net(1, o2["dof"]["bank"].pos_enc,
                                    torch.cat([o2["rgb"].permute(2, 0, 1),
                                               _mm(o2["depth"].permute(2, 0, 1))]).detach(),
                                    _mm(o2["coc"].permute(2, 0, 1)).detach()).detach()
    img, aifc, gtc = (x.permute(2, 0, 1) for x in (o2["rgb"], o2["rgb_aif"], gt2))
    la = l_rec((1 - mk) * aifc.detach() + mk * img, gtc)
    lb = l_rec((1 - mk) * aifc + mk * img.detach(), gtc)
    ga = torch.autograd.grad(la, o2["rgb"], retain_graph=True)[0]
    gb = torch.autograd.grad(lb, o2["rgb_aif"])[0]
    e_rgb, e_aif = float((ga - g_rgb).abs().max()), float((gb - g_aif).abs().max())
    assert e_rgb < 1e-6 and e_aif < 1e-6, "composite gradient routing differs from official"
    has_mask_grad = all(p.grad is not None for p in outputs["dof"]["bank"].mask_net.parameters()
                        if p.requires_grad and p is not
                        outputs["dof"]["bank"].mask_net.embedding_camera.weight)
    assert has_mask_grad, "ILN receives no gradient from L_detail"
    return {"mask_loss": val, "routing_err_rgb": e_rgb, "routing_err_aif": e_aif}


def _mm(t, eps=1e-5):
    return (t - t.min()) / (t.max() - t.min() + eps)


def objective_schedule(method, device="cpu"):
    obj = create_objective(_Spec({}))
    mask = create_mask_losses(_Spec({}))
    o, gt = _fake_outputs(5000)
    warm = float(obj(None, o, gt))
    ref = float(l_rec(o["rgb"].permute(2, 0, 1), gt.permute(2, 0, 1)))
    assert abs(warm - ref) < 1e-7 and float(mask(None, o, gt)) == 0.0, "warm-up must be L_rec"
    # SSIM same padding == valid SSIM on a zero-padded image
    x, y = o["rgb"].detach().permute(2, 0, 1), gt.permute(2, 0, 1)
    s_same = float(ssim_same(x, y))
    w = torch.tensor([math.exp(-((i - 5) ** 2) / 4.5) for i in range(11)])
    w = w / w.sum()
    w2 = (w[:, None] * w[None]).expand(3, 1, 11, 11)
    xp, yp = (torch.nn.functional.pad(t[None], (5, 5, 5, 5)) for t in (x, y))

    def cv(t):
        return torch.nn.functional.conv2d(t, w2, groups=3)

    mx, my = cv(xp), cv(yp)
    sx, sy, sxy = cv(xp * xp) - mx**2, cv(yp * yp) - my**2, cv(xp * yp) - mx * my
    s_ref = float((((2 * mx * my + 1e-4) * (2 * sxy + 9e-4))
                   / ((mx**2 + my**2 + 1e-4) * (sx + sy + 9e-4))).mean())
    assert abs(s_same - s_ref) < 1e-6, "SSIM must use same padding (Inria)"
    o, gt = _fake_outputs(5001)
    full = float(obj(None, o, gt))
    o_noaif, _ = _fake_outputs(5001, with_aif=False)
    no_aif = float(obj(None, o_noaif, gt))
    assert abs(full - no_aif) > 1e-4, "detail-enhancement branch inactive after step 5000"
    assert abs(no_aif - float(l_rec(o_noaif["rgb"].permute(2, 0, 1), gt.permute(2, 0, 1)))) < 1e-7
    return {"warmup_loss": warm, "detail_loss": full, "without_aif": no_aif, "ssim_same": s_same}


def density_recipe(method, device="cuda"):
    from .density import Density, compactness_candidates, distance_to_gaussian_surface, extra_points

    dev = torch.device(device)
    dens = Density(_Spec({}))
    dens.extent = 2.0
    lr40k = dens.xyz_lr(40000)
    assert abs(lr40k - 5e-4 * 2 * math.sqrt(0.01)) < 1e-12, "xyz LR must decay over 80000 steps"

    def make(n, seed):
        g = torch.Generator().manual_seed(seed)
        p = {"means": torch.rand(n, 3, generator=g) * 2,
             "scales": torch.log(torch.full((n, 3), 0.01)),
             "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
             "opacities": torch.randn(n, generator=g),
             "sh0": torch.randn(n, 1, 3, generator=g), "shN": torch.randn(n, 15, 3, generator=g),
             "filter_3d": torch.zeros(n)}
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(dev)) for k, v in p.items()})
        opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
        return params, opts

    # extra points: 2M uniform samples, IDW attributes of k=4 nearest
    params, opts = make(40, 0)
    old = {k: v.detach().clone() for k, v in params.items()}
    torch.manual_seed(0)
    added = extra_points(params, opts, M=50, k=4, dist_thres=15.0, range_factor=0.25)
    assert added == 100, f"expected 2M = 100 candidates kept, got {added}"
    q = params["means"][40].detach()
    d = torch.norm(old["means"] - q, dim=1)
    dv, iv = torch.topk(d, 4, largest=False)
    wv = 1 / (dv + 1e-5)
    wv = wv / wv.sum()
    e_sh = float((params["sh0"][40].detach() - (wv[:, None, None] * old["sh0"][iv]).sum(0))
                 .abs().max())
    e_sc = float((params["scales"][40].detach() - torch.log(dv.min())).abs().max())
    assert e_sh < 1e-5 and e_sc < 1e-5, "extra point attributes not IDW / min-distance scale"
    # compactness: independent per-pair formula
    params, opts = make(2, 1)
    with torch.no_grad():
        # raw log-scales enter as svec (official): surface distances ~ |log s|^2, so the pair
        # must be far apart for a bridging point
        params["means"].copy_(torch.tensor([[0.0, 0, 0], [40.0, 2.0, 1.0]], device=dev))
    idx = torch.tensor([1, 0], device=dev)
    new = compactness_candidates(params, idx, 1.5)
    p0, p1 = params["means"][0].detach(), params["means"][1].detach()
    sv = (params["scales"] / 1.5).detach()
    rot = torch.eye(3, device=dev)[None]
    s01 = distance_to_gaussian_surface(p0[None], sv[:1], rot, p1[None])[0]
    s10 = distance_to_gaussian_surface(p1[None], sv[1:], rot, p0[None])[0]
    dist = torch.norm(p1 - p0)
    m_expect = p0 + (p1 - p0) / dist * (dist + s01 - s10) / 2
    assert len(new["means"]) == 2, "both Gaussians should spawn a bridging point"
    e_c = float((new["means"][0] - m_expect).abs().max())
    s_expect = torch.log(torch.clamp((dist - s01 - s10) / 6, 1e-6, 10))
    assert e_c < 1e-5 and abs(float(new["scales"][0, 0] - s_expect)) < 1e-5, "compactness math"
    # densify_and_prune: thresholds, clone vs split, prune < 0.005
    params, opts = make(6, 2)
    with torch.no_grad():
        params["scales"].copy_(torch.log(torch.tensor([[0.01] * 3, [0.2] * 3, [0.01] * 3,
                                                       [0.01] * 3, [0.2] * 3, [0.01] * 3],
                                                      device=dev)))
        params["opacities"].copy_(torch.logit(torch.tensor([0.5, 0.5, 0.004, 0.5, 0.5, 0.5],
                                                           device=dev)))
    dens.params = params
    dens._reset_stats(6, dev)
    dens.stats["grad_accum"] += torch.tensor([3e-4, 3e-4, 1e-4, 1e-4, 1e-4, 2.5e-4], device=dev)
    dens.stats["denom"] += 1
    dens.densify_and_prune(params, opts)
    last = dens.state["last"]
    # percent_dense 0.03 * extent 2 = 0.06: clone {0,5}; split {1}; prune {2}
    assert (last["clone"], last["split"], last["prune"]) == (2, 1, 1), last
    assert len(params["means"]) == 6 + 2 + 1 - 1
    # filter-aware opacity reset
    dens.cfg = {**dens.cfg}
    with torch.no_grad():
        params["filter_3d"].fill_(0.005)
    dens.reset_opacity(params, opts)
    from .filters import filtered_scales_opacities
    _, opf = filtered_scales_opacities(params["scales"], params["opacities"],
                                       params["filter_3d"])
    assert float(opf.max()) <= 0.01 + 1e-6, "filtered opacity not reset to <= 0.01"
    return {"xyz_lr_40k": lr40k, "extra_added": added, "extra_err": [e_sh, e_sc],
            "compactness_err": e_c, "densify": last}


def official_config(method, device="cpu"):
    """Pin the variant's configuration to the official config_real_<scene>.yaml values
    (transcribed in variants.py; tests/test_dofgs.py checks the transcription against the
    yaml files)."""
    from .variants import OFFICIAL_REAL, REG_FACTOR, SCENE_VARIANT, SHARED_KEYS, SHARED_VALUES

    scenes = [s for s, v in SCENE_VARIANT.items() if v == method.name]
    assert scenes, f"{method.name} serves no official real scene"
    comp = {"renderer": _component_config(method, "dof-raster"),
            "density": _component_config(method, "dofgs-density"),
            "objective": _component_config(method, "dofgs-objective"),
            "mask": _component_config(method, "dofgs-mask")}
    for s in scenes:
        bl, lam = OFFICIAL_REAL[s]
        assert comp["renderer"]["band_limit"] is bl, f"{s}: renderer band_limit != {bl}"
        assert comp["density"]["band_limit"] is bl, f"{s}: density band_limit != {bl}"
        assert comp["mask"]["lambda_mask_loss"] == lam, f"{s}: lambda_mask_loss != {lam}"
        assert comp["mask"]["reg_factor"] == REG_FACTOR, f"{s}: lambda_reg != 0.1 lambda_mk"
    for key, (c, opt) in SHARED_KEYS.items():
        ours, theirs = comp[c][opt], SHARED_VALUES[key]
        same = (list(ours) == list(theirs) if isinstance(theirs, list)
                else float(ours) == float(theirs))
        assert same, f"{key}: plugin {c}.{opt}={ours} != official {theirs}"
    return {"variant": method.name, "scenes": scenes, "band_limit": comp["density"]["band_limit"],
            "lambda_mask_loss": comp["mask"]["lambda_mask_loss"],
            "lambda_reg": comp["mask"]["lambda_mask_loss"] * comp["mask"]["reg_factor"]}


class _DummyBank:
    step = 0

    def after_step(self, step):
        self.step = step

    def export(self):
        return {}


def _filter_updates(band_limit, device, cam):
    """Run the controller through a post-densification filter step and count recomputes."""
    from .density import Density

    dens = Density(_Spec({"band_limit": band_limit}))
    dens.extent, dens.cameras, dens.bank = 1.0, [cam], _DummyBank()
    sc = _scene(device, n=8, seed=4)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(sc["means"]), "scales": torch.nn.Parameter(sc["scales"].log()),
        "opacities": torch.nn.Parameter(torch.logit(sc["opac"])),
        "filter_3d": torch.nn.Parameter(torch.zeros(8, device=device))})
    opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    dens.after_backward(params, opts, 26100, {})
    return dens.counts["filter_updates"], float(params["filter_3d"].abs().max())


def band_limit_switch(method, device="cuda"):
    """band_limit=false (official caps/cupcake/cups): no 3D filter in rendering, no filter
    recompute, plain reset_opacity; band_limit=true: filtered render and filter-aware reset."""
    from .density import Density
    from .renderer import Renderer, aif_camera

    dev = _cuda(device)
    cam = _camera(dev)
    sc = _scene(dev, n=12, seed=3, opacity_max=0.6)
    params = {"means": sc["means"], "scales": sc["scales"].log(),
              "opacities": torch.logit(sc["opac"]),
              "quats": sc["quats"], "sh0": sc["sh"][:, :1].contiguous(),
              "shN": sc["sh"][:, 1:].contiguous(),
              "filter_3d": torch.full((12,), 0.03, device=dev)}  # deliberately non-zero
    saved, bank_mod.ACTIVE = bank_mod.ACTIVE, None
    try:
        with torch.no_grad():
            out_off = Renderer(_Spec({"band_limit": False})).render(params, cam, 0)["rgb"]
            out_on = Renderer(_Spec({"band_limit": True})).render(params, cam, 0)["rgb"]
            f0, k0 = aif_camera(dev)
            ref = rasterize(params["means"], params["scales"].exp(),
                            params["opacities"].sigmoid(), sc["quats"], sc["sh"],
                            cam, f0, k0, 0, 0, torch.zeros(3, device=dev), 0.1)["rgb"]
            zero = dict(params, filter_3d=torch.zeros(12, device=dev))
            out_on0 = Renderer(_Spec({"band_limit": True})).render(zero, cam, 0)["rgb"]
    finally:
        bank_mod.ACTIVE = saved
    e_ref = float((out_off - ref).abs().max())
    e_zero = float((out_off - out_on0).abs().max())
    gap = float((out_on - out_off).abs().max())
    assert e_ref < 1e-6, f"band_limit=false render != unfiltered kernel call ({e_ref})"
    assert e_zero < 1e-5, f"band_limit=false != band_limit=true with zero filter ({e_zero})"
    assert gap > 1e-3, f"3D filter has no effect with band_limit=true ({gap})"
    # opacity reset: plain vs filter-aware
    resets = {}
    for bl in (False, True):
        dens = Density(_Spec({"band_limit": bl}))
        p = torch.nn.ParameterDict({
            "scales": torch.nn.Parameter(sc["scales"].log().cpu()),
            "opacities": torch.nn.Parameter(torch.linspace(-6.0, 4.0, 12)),
            "filter_3d": torch.nn.Parameter(torch.full((12,), 0.03))})
        o0 = p["opacities"].detach().clone()
        opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in p.items()}
        dens.reset_opacity(p, opts)
        resets[bl] = (o0, p["opacities"].detach().clone())
    o0, plain = resets[False]
    official = torch.sigmoid(o0).clamp(max=0.01)
    official = torch.log(official / (1 - official))  # inverse_sigmoid(min(sigmoid(o), 0.01))
    e_reset = float((plain - official).abs().max())
    assert e_reset < 1e-4, f"plain reset != inverse_sigmoid(min(sigmoid(o), 0.01)) ({e_reset})"
    gap_reset = float((resets[True][1] - plain).abs().max())
    assert gap_reset > 1e-2, "filter-aware reset should differ from the plain reset"
    # filter recompute schedule
    n_off, f_off = _filter_updates(False, dev, cam)
    n_on, f_on = _filter_updates(True, dev, cam)
    assert n_off == 0 and f_off == 0.0, "band_limit=false must not compute the 3D filter"
    assert n_on == 1 and f_on > 0.0, "band_limit=true must recompute the filter at step 26100"
    return {"variant_band_limit": _component_config(method, "dofgs-density")["band_limit"],
            "render_err_vs_unfiltered": e_ref, "render_err_vs_zero_filter": e_zero,
            "filter_render_gap": gap, "plain_reset_err": e_reset, "reset_gap": gap_reset,
            "filter_updates": {"off": n_off, "on": n_on}}
