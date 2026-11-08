"""Mechanism probes for VoD-3DGS. Each compares against an independent formulation."""

import math

import torch
from splatstudio.components import build
from splatstudio.state import initialize

from . import vod


def _scene(method, device, n=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5).to(device)
    pts[:, 2] += 3.0
    cols = (torch.rand(n, 3, generator=g) * 0.6 + 0.2).to(device)
    params, opts = initialize(method, pts, cols)
    with torch.no_grad():
        params["vod_S"].copy_((torch.randn(n, 6, generator=g) * 2.0).to(device))
        params["scales"].fill_(math.log(0.08))
    return params, opts


def _camera(device, eye=(0.0, 0.0, 0.0)):
    viewmat = torch.eye(4, device=device)
    viewmat[:3, 3] = -torch.tensor(eye, device=device)
    K = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)
    return {"viewmat": viewmat, "K": K, "width": 64, "height": 64}


def _independent_opacity(gamma, S6, means, eye):
    out = []
    for k in range(len(means)):
        a, b, c, d, e, f = S6[k].tolist()
        M = torch.tensor([[a, b, c], [b, d, e], [c, e, f]], dtype=torch.float64)
        w = torch.tensor(eye, dtype=torch.float64) - means[k].double().cpu()
        w = w / w.norm()
        out.append(1 / (1 + math.exp(-(float(gamma[k]) + float(w @ M @ w)))))
    return torch.tensor(out)


def vod_opacity_probe(method, device):
    renderer, _, _ = build(method)
    params, _ = _scene(method, device)
    eye = (0.1, -0.05, 0.0)
    cam = _camera(device, eye)
    opac, center = renderer.opacities(params, cam)
    ref = _independent_opacity(params["opacities"].detach().cpu(), params["vod_S"].detach().cpu(),
                               params["means"].detach().cpu(), eye)
    err = float((opac.detach().cpu().double() - ref).abs().max())
    if err > 1e-5:
        raise AssertionError(f"opacity differs from sigma(gamma + w^T S w): {err}")
    # sign invariance of the view vector
    flipped = vod.quad_form(params["vod_S"], -vod.view_dirs(params["means"], center))
    if not torch.allclose(flipped, vod.quad_form(params["vod_S"], vod.view_dirs(params["means"], center))):
        raise AssertionError("quadratic form not even in w")
    out = renderer.render(params, cam, sh_degree=0)
    out["rgb"].mean().backward()
    gS = params["vod_S"].grad
    if gS is None or not torch.isfinite(gS).all() or float(gS.abs().sum()) == 0:
        raise AssertionError("vod_S receives no gradient through the renderer")
    with torch.no_grad():
        saved = params["vod_S"].clone()
        params["vod_S"].zero_()
        rgb0 = renderer.render(params, cam, sh_degree=0)["rgb"]
        params["vod_S"].copy_(saved)
        rgb1 = renderer.render(params, cam, sh_degree=0)["rgb"]
    diff = float((rgb1 - rgb0).abs().max())
    if diff < 1e-3:
        raise AssertionError("S has no effect on the rendered image")
    return {"max_opacity_error": err, "image_change_from_S": diff, "grad_S_l1": float(gS.abs().sum())}


def max_view_prune_probe(method, device):
    _, density, _ = build(method)
    params, opts = _scene(method, device, n=3)
    with torch.no_grad():
        params["opacities"].fill_(math.log(0.001 / 0.999))  # sigma(gamma) = 0.001 < tau
        params["vod_S"].zero_()
        params["means"].copy_(torch.tensor([[0.0, 0, 3], [0.5, 0, 3], [0.0, 0.5, 3]], device=device))
        # Gaussian 0: +8 along x -> opaque only when viewed along +-x; Gaussian 1: S = 0; 2: -8 along x.
        params["vod_S"][0, 0] = 8.0
        params["vod_S"][2, 0] = -8.0
    density.initialize(params, opts, 1.0)
    cams = [_camera(device, (0, 0, 0)), _camera(device, (3.0, 0, 3.0))]  # second camera looks along x at G0
    density.set_training_cameras(cams)
    state = density.state
    n_pruned = density.strategy._prune_gs(params, opts, state, step=600)
    if n_pruned != 2 or len(params["means"]) != 1 or float(params["vod_S"][0, 0]) != 8.0:
        raise AssertionError(f"expected only the view-opaque Gaussian to survive; pruned {n_pruned}")
    a = torch.sigmoid(params["opacities"][0] + 8.0).item()
    return {"pruned": n_pruned, "survivor_max_view_opacity": a, "survivor_sigma_gamma": 0.001}


def eigen_reset_probe(method, device):
    from .density import reset_vod_S

    _, density, _ = build(method)
    variant = density.variant
    params, opts = _scene(method, device, n=16)
    # Allocate Adam moments.
    params["vod_S"].square().sum().backward()
    opts["vod_S"].step()
    S = params["vod_S"].detach().clone()
    p_before = params["vod_S"]
    reset_vod_S(params, opts, variant)
    # Independent: numpy-free power-free reference via eigh on each full matrix in float64.
    for k in range(16):
        M = vod.to_matrix(S[k].double().cpu())
        lam, Q = torch.linalg.eigh(M)
        idx = 0 if variant == "S" else 2
        ref = lam[idx] * torch.outer(Q[:, idx], Q[:, idx])
        got = vod.to_matrix(params["vod_S"][k].detach().double().cpu())
        if not torch.allclose(got, ref, atol=1e-4):
            raise AssertionError("eigen reset differs from rank-1 projection")
        if torch.linalg.matrix_rank(got, atol=1e-5) > 1:
            raise AssertionError("reset matrix is not rank 1")
    if params["vod_S"] is not p_before or not params["vod_S"].requires_grad:
        raise AssertionError("reset rebuilt or froze the vod_S parameter")
    st = opts["vod_S"].state[params["vod_S"]]
    if float(st["exp_avg"].abs().sum()) != 0 or float(st["exp_avg_sq"].abs().sum()) != 0:
        raise AssertionError("Adam moments not zeroed at reset")
    from splatstudio.backends.gsplat import resets_opacity

    steps = [s for s in range(1, 30001) if resets_opacity(density.strategy, s)]
    if steps != [3000, 6000, 9000, 12000]:
        raise AssertionError(f"unexpected reset schedule {steps}")
    return {"variant": variant, "reset_steps": steps}


def vc_loss_probe(method, device):
    _, density, losses = build(method)
    vc = dict(losses)["vod-view-consistency"]
    params, opts = _scene(method, device, n=32)
    eyes = [(0, 0, 0), (0.3, 0, 0), (0, 0, 6.0)]  # third camera behind the Gaussians
    cams = [_camera(device, e) for e in eyes]
    density.initialize(params, opts, 1.0)
    density.set_training_cameras(cams)
    vc.sampling = "uniform"  # synthetic cameras have no SfM file; covisibility tested below
    vc.centers = None
    ci = vod.camera_center(cams[0]["viewmat"])
    cj = vod.camera_center(cams[2]["viewmat"])
    val = vod.view_consistency(params["opacities"], params["vod_S"], params["means"], ci, cj)
    # independent per-Gaussian evaluation of Eq. 12
    a_i = _independent_opacity(params["opacities"].detach().cpu(), params["vod_S"].detach().cpu(),
                               params["means"].detach().cpu(), eyes[0])
    a_j = _independent_opacity(params["opacities"].detach().cpu(), params["vod_S"].detach().cpu(),
                               params["means"].detach().cpu(), eyes[2])
    ref = 0.0
    for k in range(32):
        wi = torch.tensor(eyes[0], dtype=torch.float64) - params["means"][k].detach().double().cpu()
        wj = torch.tensor(eyes[2], dtype=torch.float64) - params["means"][k].detach().double().cpu()
        cos = float(wi @ wj / (wi.norm() * wj.norm()))
        ref += max(cos, 0.0) * float(a_i[k] - a_j[k]) ** 2
    ref /= 32
    if abs(float(val) - ref) > 1e-6:
        raise AssertionError(f"L_vc {float(val)} != reference {ref}")
    if ref != 0.0:
        raise AssertionError("opposite views must give zero L_vc (cos clamp)")
    cj = vod.camera_center(cams[1]["viewmat"])
    val = vod.view_consistency(params["opacities"], params["vod_S"], params["means"], ci, cj)
    val.backward()
    if params["vod_S"].grad is None or float(params["vod_S"].grad.abs().sum()) == 0:
        raise AssertionError("L_vc gives no gradient to vod_S")
    if params["means"].grad is not None and float(params["means"].grad.abs().sum()) != 0:
        raise AssertionError("L_vc must not move the means")
    # covisibility-weighted sampling: probabilities follow the match matrix
    vc.sampling = "covisibility"
    m = torch.tensor([[0, 30, 10], [30, 0, 0], [10, 0, 0]], dtype=torch.float64)
    vc.centers = vod._CONTEXT["train_centers"]
    vc.probs = (m / m.sum(1, keepdim=True).clamp_min(1)).float().to(device)
    torch.manual_seed(0)
    js = [vc.sample(ci)[1] for _ in range(4000)]
    f1 = js.count(1) / len(js)
    if abs(f1 - 0.75) > 0.03 or js.count(0) != 0:
        raise AssertionError(f"sampling frequency {f1} != 0.75 or self-pair drawn")
    return {"vc_value": float(val), "opposite_view_value": ref, "covis_freq_view1": f1}


def opacity_lr_probe(method, device):
    _, density, _ = build(method)
    params, opts = _scene(method, device, n=8)
    density.initialize(params, opts, 1.0)
    lo, ls = opts["opacities"].param_groups[0]["lr"], opts["vod_S"].param_groups[0]["lr"]
    if abs(lo - 0.0125) > 1e-12 or abs(ls - 0.0125) > 1e-12:
        raise AssertionError(f"learning rates {lo}, {ls} != 0.05/4")
    return {"opacity_lr": lo, "vod_S_lr": ls}
