"""Mechanism probes for Denoising-GS. Reference formulas are written independently of core.py;
each probe raises AssertionError when its mechanism is removed or wrong."""

import math
from dataclasses import replace

import numpy as np
import torch

from . import core
from . import density as D


def _spec(method):
    return next(c for c in method.components if c.name == "denoising-density")


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def _density(method, **cfg):
    spec = _spec(method)
    return D.DenoisingDensity(replace(spec, config={**spec.config, "cap_max": 10**6, **cfg}))


def _scene(method, n, device, seed=0, depth=3.0, spread=0.6, scale=0.05):
    from splatstudio.state import initialize

    g = torch.Generator().manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5) * spread
    pts[:, 2] += depth
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_((math.log(scale) + 0.3 * torch.randn(n, 3, generator=g)).to(device))
        params["quats"].copy_(torch.nn.functional.normalize(torch.randn(n, 4, generator=g),
                                                            dim=-1).to(device))
    return params, opts


def _camera(device, w=48, h=40, f=40.0, tx=0.0):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], device=device)
    V = torch.eye(4, device=device)
    V[0, 3] = tx
    return {"K": K, "viewmat": V, "width": w, "height": h}


def _rotmat_np(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ------------------------------------------------------------------------------------------
def momentum_drift(method, device):
    """C1: 3 steps with known gradients through the real hooks, Adam step and after_backward.
    Default mode: -alpha*lr*m^/(sqrt(v^)+eps) with an independent lagged EMA m and Adam v;
    raw mode: -alpha*lr*m. alpha = 0 ablation must differ."""
    dev = "cpu"
    alpha, beta1, lr = 0.05, 0.9, 0.5
    out = {}

    def run(a, mode):
        dens = _density(method, alpha=a, beta1=beta1, momentum_mode=mode, spatial_denoise=False,
                        noise_lr=0.0, refine_start_iter=10**9, fisher_prune=False, scr=False)
        params, opts = _scene(method, 16, dev)
        dens.initialize(params, opts)
        opts["means"].param_groups[0]["lr"] = lr
        b2, eps = opts["means"].param_groups[0]["betas"][1], opts["means"].param_groups[0]["eps"]
        g = torch.Generator().manual_seed(1)
        coefs = [torch.randn(16, 3, generator=g) for _ in range(3)]
        m_ref, v_ref = torch.zeros(16, 3), torch.zeros(16, 3)
        disp, ref = [], []
        for step, cf in enumerate(coefs, start=1):
            dens.before_backward(params, opts, step, {})
            (params["means"] * cf).sum().backward()
            assert torch.allclose(params["means"].grad, cf), "hook/grad mismatch"
            opts["means"].step()
            opts["means"].zero_grad(set_to_none=True)
            opts["means"].param_groups[0]["lr"] = lr
            v_ref = b2 * v_ref + (1 - b2) * cf**2
            before = params["means"].detach().clone()
            dens.after_backward(params, opts, step, {})
            disp.append(params["means"].detach() - before)
            if mode == "raw":
                ref.append(-a * lr * m_ref)  # m_t holds gradients up to t-1 (Eq. m2)
            else:
                bias = 1 - beta1 ** (step - 1) if step > 1 else 1.0
                denom = (v_ref / (1 - b2**step)).sqrt() + eps
                ref.append(-a * lr * (m_ref / bias) / denom)
            m_ref = beta1 * m_ref + (1 - beta1) * cf
        return disp, ref

    for mode in ("adam", "raw"):
        disp, ref = run(alpha, mode)
        err = max(float((d - r).abs().max()) for d, r in zip(disp, ref))
        assert err < 1e-5, f"{mode} momentum drift mismatch {err}"
        assert float(disp[2].abs().max()) > 1e-4, f"{mode} momentum drift inactive"
        abl, _ = run(0.0, mode)
        assert float((abl[2] - ref[2]).abs().max()) > 1e-4, "alpha=0 ablation indistinguishable"
        out[mode] = {"max_error": err, "step3_drift": float(disp[2].abs().max())}
    # The Adam-normalised drift is O(alpha * lr); the raw one scales with |grad|.
    assert out["adam"]["step3_drift"] > 0.5 * alpha * lr * 0.1
    return out


def spatial_denoise(method, device):
    """C2: independent loop + analytic cases + density applies beta2 R Delta-mu."""
    g = torch.Generator().manual_seed(2)
    n = 64
    gm, gs = torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g)
    q = torch.randn(n, 4, generator=g)
    world, aligned = core.denoise_term(gm, gs, q)
    ref = np.zeros((n, 3))
    n_al = 0
    for i in range(n):
        R = _rotmat_np(q[i].double().numpy())
        gl = R.T @ gm[i].double().numpy()
        jm, js = int(np.argmax(np.abs(gl))), int(np.argmax(np.abs(gs[i].double().numpy())))
        if jm == js:
            n_al += 1
            dmu = np.zeros(3)
            dmu[jm] = -abs(float(gs[i, jm])) * np.sign(gl[jm])  # along the descent update
            ref[i] = R @ dmu
    err = float(np.abs(world.double().numpy() - ref).max())
    assert err < 1e-5, f"denoise term mismatch {err}"
    assert 0 < n_al < n and int(aligned.sum()) == n_al
    eye = torch.tensor([[1.0, 0, 0, 0]] * 2)
    w2, _ = core.denoise_term(torch.tensor([[-0.3, 0.1, 0.0], [-0.3, 0.1, 0.0]]),
                              torch.tensor([[0.2, 0.05, 0.0], [0.01, 0.4, 0.0]]), eye)
    assert torch.allclose(w2[0], torch.tensor([0.2, 0.0, 0.0]))  # aligned on x, along -dM
    # Delta-mu must point along the gradient-descent update for every aligned Gaussian.
    assert bool(((world * gm).sum(-1)[aligned] < 0).all()), "Delta-mu points up the gradient"
    lit, _ = core.denoise_term(gm, gs, q, sign_mode="literal")
    assert torch.allclose(lit, -world), "literal option is not the sign-flipped term"
    assert torch.allclose(w2[1], torch.zeros(3))  # misaligned -> no term
    # Through the density: the mean moves by exactly beta2 * world.
    dens = _density(method, momentum=False, noise_lr=0.0, refine_start_iter=10**9,
                    fisher_prune=False, scr=False, beta2=0.5)
    params, opts = _scene(method, n, "cpu")
    dens.initialize(params, opts)
    with torch.no_grad():
        params["quats"].copy_(q)
    dens.before_backward(params, opts, 1, {})
    ((params["means"] * gm).sum() + (params["scales"] * gs).sum()).backward()
    before = params["means"].detach().clone()
    dens.after_backward(params, opts, 1, {})
    moved = params["means"].detach() - before
    err2 = float((moved - 0.5 * world).abs().max())
    assert err2 < 1e-6 and float(moved.abs().max()) > 1e-3, f"density C2 mismatch {err2}"
    return {"max_error": err, "aligned": n_al, "density_error": err2}


def fisher_prune(method, device):
    """C3 against an explicit 6x6 Fisher matrix and SVD."""
    _cuda(device)
    n = 200
    params, opts = _scene(method, n, device, spread=0.8)
    cams = [_camera(device, tx=t) for t in (-0.15, 0.0, 0.15)]
    g = torch.Generator().manual_seed(3)
    imgs = [torch.rand(40, 48, 3, generator=g).to(device) for _ in cams]
    dens = _density(method, momentum=False, spatial_denoise=False, noise_lr=0.0,
                    refine_start_iter=10**9, scr=False, total_steps=10, prune_offset=5)
    dens.initialize(params, opts)  # sets the MCMC initial opacity before u is measured
    with torch.no_grad():
        params["opacities"].copy_(torch.logit(torch.rand(n, generator=g) * 0.8 + 0.1).to(device))
    u = D.fisher_uncertainty(params, cams, imgs)
    F_ = torch.zeros(n, 6, 6, device=device, dtype=torch.float64)
    for cam, img in zip(cams, imgs):
        m = params["means"].detach().clone().requires_grad_(True)
        s = params["scales"].detach().clone().requires_grad_(True)
        from gsplat import rasterization

        rgb, a, _ = rasterization(m, params["quats"].detach(), s.exp(),
                                  params["opacities"].detach().sigmoid(),
                                  torch.cat((params["sh0"], params["shN"]), 1).detach(),
                                  cam["viewmat"][None], cam["K"][None], cam["width"],
                                  cam["height"], sh_degree=3)
        loss = ((rgb[0] + 1 - a[0]) - img).pow(2).sum()
        gm, gsc = torch.autograd.grad(loss, (m, s))
        v = torch.cat([gm, gsc], -1).double()
        F_ += v[:, :, None] * v[:, None, :]
    u_ref = torch.linalg.svdvals(F_).sum(-1)
    rel = float(((u.double() - u_ref).abs() / u_ref.clamp_min(1e-30)).max())
    assert rel < 1e-3 and float(u_ref.max()) > 0, f"Fisher uncertainty mismatch {rel}"
    dens.set_training_cameras(cams)
    dens.set_training_images(imgs)
    keep_ref = torch.sort(torch.topk(u, n - int(0.1 * n)).values).values
    sh0_ref = params["sh0"].detach().clone()
    dens.before_backward(params, opts, 5, {})
    dens.after_backward(params, opts, 5, {})
    assert len(params["means"]) == n - int(0.1 * n)
    for k, p in params.items():
        st = opts[k].state[opts[k].param_groups[0]["params"][0]]
        assert opts[k].param_groups[0]["params"][0] is p and p.requires_grad, k
        assert all(v.shape[0] == len(p) for v in st.values() if torch.is_tensor(v) and v.dim()), k
    kept = (params["sh0"].detach()[:, None] == sh0_ref[None]).all(-1).all(-1).any(0)
    assert int(kept.sum()) == len(params["means"])
    assert torch.equal(torch.sort(u[kept]).values, keep_ref), "pruned set is not the lowest-u 10 %"
    # Ablation guard: a reversed or random ranking would remove some higher-u primitive.
    assert float(u[~kept].max()) <= float(u[kept].min())
    return {"u_rel_error": rel, "removed": int((~kept).sum()),
            "removed_u_max": float(u[~kept].max()), "kept_u_min": float(u[kept].min())}


def scr_densify(method, device):
    """C4: brute-force kNN, sparsest selection, Eqs. p6/p7 with N = 2."""
    _cuda(device)
    n = 4000
    params, opts = _scene(method, n, device, spread=0.5)
    with torch.no_grad():
        params["means"][7] = torch.tensor([5.0, 5.0, 8.0], device=device)  # isolated primitive
        params["means"][11] = torch.tensor([-4.0, 3.0, 9.0], device=device)
        params["opacities"].copy_(torch.logit(torch.rand(n, device=device) * 0.8 + 0.1))
    d = core.knn_sparsity(params["means"], 3)
    d2 = torch.cdist(params["means"].detach().double(), params["means"].detach().double()) ** 2
    d2.fill_diagonal_(float("inf"))
    d_ref = d2.topk(3, largest=False).values.mean(1)
    rel = float(((d.double() - d_ref).abs() / d_ref).max())
    assert rel < 1e-4, f"kNN sparsity mismatch {rel}"
    dens = _density(method, momentum=False, spatial_denoise=False, noise_lr=0.0,
                    refine_start_iter=10**9, fisher_prune=False, total_steps=1000,
                    prune_offset=500, densify_every=100, densify_fraction=0.0005)
    dens.initialize(params, opts)
    with torch.no_grad():
        params["opacities"].copy_(torch.logit(torch.rand(n, device=device) * 0.8 + 0.1))
    o_old = params["opacities"].detach().sigmoid().clone()
    s_old = params["scales"].detach().exp().clone()
    m_old = params["means"].detach().clone()
    dens.before_backward(params, opts, 600, {})
    dens.after_backward(params, opts, 600, {})
    assert len(params["means"]) == n + 2, len(params["means"])
    o_new_ref = 1 - torch.sqrt(1 - o_old[[7, 11]])  # Eq. p6, N = 2
    denom = 2 * o_new_ref - o_new_ref**2 / math.sqrt(2)  # Eq. p7 inner sum, N = 2
    s_new_ref = (o_old[[7, 11]] / denom)[:, None] * s_old[[7, 11]]
    o, s = params["opacities"].detach().sigmoid(), params["scales"].detach().exp()
    copies = params["means"].detach()[n:]
    assert {tuple(c.tolist()) for c in copies} == {tuple(m_old[7].tolist()), tuple(m_old[11].tolist())}
    for j, src in enumerate((7, 11)):
        cp = n + int((copies == m_old[src]).all(-1).nonzero()[0])
        for idx in (src, cp):
            assert abs(float(o[idx]) - float(o_new_ref[j])) < 1e-5, "Eq. p6 opacity"
            assert torch.allclose(s[idx], s_new_ref[j], rtol=1e-4), "Eq. p7 scale"
    assert float((o[[7, 11]] - o_old[[7, 11]]).abs().min()) > 1e-3, "ablation: no split"
    return {"knn_rel_error": rel, "added": 2, "opacity_new": o_new_ref.tolist()}


def schedule_probe(method, device):
    c = _spec(method).config
    prune, scr = core.schedule(c["total_steps"], c["prune_offset"], c["densify_every"],
                               c["densify_times"])
    assert prune == c["total_steps"] - c["prune_offset"] and len(scr) == 5
    assert scr == [prune + c["densify_every"] * k for k in range(1, 6)] and scr[-1] <= c["total_steps"]
    if c["total_steps"] == 35000:
        assert prune == 30000 and scr == [30500, 31000, 31500, 32000, 32500]
    assert (c["refine_start_iter"], c["refine_stop_iter"], c["refine_every"]) == (500, 25000, 100)
    assert core.SCENE_CAPS["bicycle"] == 5_900_000 and core.SCENE_CAPS["counter"] == 1_200_000
    assert c["prune_fraction"] == 0.10 and c["densify_fraction"] == 0.0005 and c["knn_k"] == 3
    return {"prune_step": prune, "scr_steps": scr}


def split_finite(method, device):
    """Regression for the counter 35k failure: SCR split of near-transparent sparse primitives
    (o down to 1e-9), followed by full training steps (render, loss, Adam, after_backward with
    C1/C2), must keep every field finite; the closed form must agree with gsplat's
    compute_relocation where that is well conditioned."""
    _cuda(device)
    from gsplat.relocation import compute_relocation
    from splatstudio.state import check_state

    o = torch.tensor([1e-3, 0.05, 0.5, 0.9, 0.999], device=device)
    s = torch.rand(5, 3, device=device) + 0.1
    binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(51)]
                           for n in range(51)], dtype=torch.float32, device=device)
    o_ref, s_ref = compute_relocation(o, s, torch.full((5,), 2, device=device, dtype=torch.int32),
                                      binoms)
    o_new, s_new = core.split_in_two(o, s)
    assert torch.allclose(o_new, o_ref, rtol=1e-4) and torch.allclose(s_new, s_ref, rtol=1e-4)
    n = 3000
    params, opts = _scene(method, n, device, spread=0.5)
    dens = _density(method, noise_lr=5e5, refine_start_iter=10**9, fisher_prune=False,
                    total_steps=1000, prune_offset=500, densify_every=100, densify_fraction=0.002)
    dens.initialize(params, opts)
    far = torch.tensor([[5.0, 5.0, 8.0], [-4.0, 3.0, 9.0], [3.0, -4.0, 7.0], [-5.0, -5.0, 9.0],
                        [0.0, 6.0, 10.0], [6.0, 0.0, 9.0]], device=device)
    with torch.no_grad():
        params["means"][:6] = far
        params["opacities"][:6] = torch.logit(torch.tensor([1e-9, 1e-8, 1e-7, 1e-6, 0.3, 0.9],
                                                           device=device))
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.losses import create_photometric

    spec = next(c for c in method.components if c.kind == "renderer")
    renderer = Renderer(spec)
    photo = create_photometric(next(c for c in method.components if c.name == "photometric"))
    cam = _camera(device)
    target = torch.rand(40, 48, 3, device=device)
    n0 = len(params["means"])
    for step in (600, 601, 602, 603):
        out = renderer.render(params, cam, sh_degree=3)
        out["step"] = step
        dens.before_backward(params, opts, step, out)
        photo(params, out, target).backward()
        for opt in opts.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        dens.after_backward(params, opts, step, out)
        for k, p in params.items():
            assert torch.isfinite(p).all(), f"{k} nonfinite after step {step}"
    assert len(params["means"]) == n0 + 6, "split did not run"
    check_state(method, params, opts)
    return {"added": len(params["means"]) - n0, "min_opacity_after":
            float(params["opacities"].sigmoid().min()), "max_scale": float(params["scales"].max())}
