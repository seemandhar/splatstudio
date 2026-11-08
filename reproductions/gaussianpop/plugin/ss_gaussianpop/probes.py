"""Executable mechanism checks for GaussianPOP (each fails if its mechanism is removed/wrong)."""

import math
from dataclasses import replace

import torch


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("The GaussianPOP error kernel needs CUDA")


def _scene(n, device, seed, opacity=(0.05, 0.3), sh_rest=0.1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-0.5, 0.5, generator=g)
    means[:, 2] = torch.empty(n).uniform_(-0.4, 0.4, generator=g)
    params = {
        "means": means,
        "scales": torch.empty(n, 3).uniform_(math.log(0.08), math.log(0.25), generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.empty(n).uniform_(*opacity, generator=g)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * sh_rest,
    }
    return {k: v.to(device) for k, v in params.items()}


def _camera(device, angle=0.15, width=72, height=56, cx=35.3, cy=27.1, f=60.0):
    """OpenCV world-to-camera looking at the origin from distance 3."""
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = torch.tensor([0.05, -0.03, 3.0])
    K = torch.tensor([[f, 0.0, cx], [0.0, f * 1.02, cy], [0.0, 0.0, 1.0]])
    return {"K": K.to(device), "viewmat": viewmat.to(device), "width": width, "height": height}


def _render(params, camera, keep=None, bg=1.0):
    """Independent reference: gsplat `rasterization` (degree 3) over a constant background."""
    from gsplat import rasterization

    p = params if keep is None else {k: v[keep] for k, v in params.items()}
    colors = torch.cat((p["sh0"], p["shN"]), dim=1)
    render, alpha, _ = rasterization(
        means=p["means"], quats=p["quats"], scales=p["scales"].exp(),
        opacities=p["opacities"].sigmoid(), colors=colors, viewmats=camera["viewmat"][None],
        Ks=camera["K"][None], width=camera["width"], height=camera["height"], sh_degree=3,
        packed=False,
    )
    return render[0] + bg * (1 - alpha[0])


def _spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return float(torch.corrcoef(torch.stack((ra, rb)))[0, 1])


@torch.no_grad()
def delta_se_oracle(method, device):
    """Kernel Δ_SE (Alg. 1) == brute-force removal error Σ_pix ||C − C'_{−k}||² (Eq. 4)."""
    _require_cuda(device)
    from . import kernels

    n = 24  # opacities <= 0.3: T >= 0.7^24 > 1e-4, so no termination changes on removal
    params = _scene(n, device, seed=3)
    cam = _camera(device)
    full = _render(params, cam)
    se = torch.zeros(n, device=device)
    c_kernel = kernels.view_error(params, cam, se, background=(1.0, 1.0, 1.0), n_max=64)
    render_err = float((c_kernel - full).abs().max())
    if render_err > 2e-4:
        raise AssertionError(f"kernel C_render differs from gsplat render by {render_err}")
    oracle = torch.zeros(n, device=device)
    weight = torch.zeros(n, device=device)
    for k in range(n):
        keep = torch.ones(n, dtype=torch.bool, device=device)
        keep[k] = False
        oracle[k] = ((full - _render(params, cam, keep)) ** 2).sum()
    visible = oracle > 1e-6
    if int(visible.sum()) < n // 2:
        raise AssertionError("fixture too sparse: fewer than half the Gaussians are visible")
    rel = float(((se - oracle).abs() / oracle.clamp_min(1e-6))[visible].max())
    if rel > 2e-3:
        raise AssertionError(f"kernel Δ_SE differs from brute-force removal error (rel {rel})")
    # Discrimination: a wrong background (black instead of the white render background) and
    # a truncated contributor list must change the scores; blending weight is not the criterion.
    se_black = torch.zeros(n, device=device)
    kernels.view_error(params, cam, se_black, background=(0.0, 0.0, 0.0), n_max=64)
    rel_black = float(((se_black - oracle).abs() / oracle.clamp_min(1e-6))[visible].max())
    if rel_black < 1e-2:
        raise AssertionError("background term has no effect: probe cannot discriminate")
    se_trunc = torch.zeros(n, device=device)
    kernels.view_error(params, cam, se_trunc, n_max=2)
    if not float(se_trunc.sum()) < 0.999 * float(se.sum()):
        raise AssertionError("n_max truncation has no effect")
    # Blending weight Σ_pix T α (LightGaussian-style importance) via one-hot colours.
    from gsplat import rasterization

    for k in range(n):
        colors = torch.zeros(n, 3, device=device)
        colors[k] = 1.0
        img, _, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(), colors=colors,
            viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=cam["width"],
            height=cam["height"], sh_degree=None, packed=False,
        )
        weight[k] = img[0, ..., 0].sum()
    rho = _spearman(se[visible], weight[visible])
    if rho > 0.999:
        raise AssertionError("Δ_SE ranking identical to blending weight: probe not discriminative")
    return {
        "render_max_abs_err": render_err,
        "delta_se_max_rel_err": rel,
        "black_bg_rel_err": rel_black,
        "n_max2_total_ratio": float(se_trunc.sum() / se.sum()),
        "spearman_vs_blend_weight": rho,
        "visible": int(visible.sum()),
    }


def prune_cycles(method, device):
    """C cycles of re-quantification + lowest-error removal, fresh Adam, LR schedule."""
    _require_cuda(device)
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.losses import create_photometric
    from splatstudio.specs import ComponentSpec
    from splatstudio.state import check_state, initialize

    from . import kernels, schedule
    from .density import create_density

    spec = next(c for c in method.components if c.kind == "density")
    ratio, cycles = float(spec.config["prune_ratio"]), int(spec.config["cycles"])
    config = dict(spec.config, base_steps=4, prune_step=4)
    density = create_density(replace(spec, config=config))
    loss_fn = create_photometric(next(c for c in method.components if c.name == "photometric"))

    n0 = 400
    scene = _scene(n0, device, seed=5, opacity=(0.1, 0.8))
    scene["means"][:10, 2] = -6.0  # behind every camera -> zero error, must be pruned first
    params, optimizers = initialize(method, scene["means"].clone(), torch.rand(n0, 3, device=device))
    with torch.no_grad():
        for k in ("scales", "quats", "opacities", "sh0", "shN"):
            params[k].copy_(scene[k])
    cameras = [_camera(device, angle=a) for a in (-0.3, -0.1, 0.1, 0.3)]
    density.initialize(params, optimizers, 1.0)
    density.set_training_cameras(cameras)
    lr0 = optimizers["means"].param_groups[0]["lr"]
    renderer = Renderer(ComponentSpec("gsplat", "renderer", "", config={}))
    target = torch.rand(56, 72, 3, device=device)

    # Record each cycle's input and check its removal against an independent recomputation.
    seen = []
    original = density.quantify

    def recording(p):
        before = p["means"].detach().clone()
        ref = kernels.scene_error(p, cameras, background=(1.0, 1.0, 1.0), n_max=64)
        scores = original(p)
        seen.append((before, ref))
        return scores

    density.quantify = recording
    expected_keep = schedule.keep_counts(n0, ratio, cycles, spec.config.get("split", "compound"))
    for step in range(1, 7):
        out = renderer.render(params, cameras[step % 4], sh_degree=3)
        out["step"] = step
        density.before_backward(params, optimizers, step, out)
        loss_fn(params, out, target).backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] = lr0 * 0.01 ** (step / 6)  # runtime's decay
        density.after_backward(params, optimizers, step, out)
        check_state(method, params, optimizers)
        lr = optimizers["means"].param_groups[0]["lr"]
        expected_lr = lr0 * 0.01 ** (min(step, 4) / 4)
        if not math.isclose(lr, expected_lr, rel_tol=1e-9):
            raise AssertionError(f"step {step}: means LR {lr} != {expected_lr}")
        if step == 4:
            if len(seen) != cycles:
                raise AssertionError(f"{len(seen)} quantifications, expected {cycles} cycles")
            if any(len(o.state) for o in optimizers.values()):
                raise AssertionError("Adam state was not reset after pruning")
            for p in params.values():
                if not (p.is_leaf and p.requires_grad):
                    raise AssertionError("a field is frozen after pruning")
    counts = [len(b) for b, _ in seen] + [len(params["means"])]
    if counts[1:] != expected_keep:
        raise AssertionError(f"per-cycle counts {counts[1:]} != schedule {expected_keep}")
    for c, (before, ref) in enumerate(seen):
        nxt = seen[c + 1][0] if c + 1 < len(seen) else None
        if nxt is None:
            break
        survivors = {tuple(r) for r in nxt.cpu().numpy().round(6)}
        kept = torch.tensor([tuple(r) in survivors for r in before.cpu().numpy().round(6)],
                            device=before.device)
        thr = ref[~kept].max() if (~kept).any() else -math.inf
        tol = 1e-4 * max(1.0, float(thr))
        bad = int((ref[kept] < thr - tol).sum())
        if bad:
            raise AssertionError(f"cycle {c + 1}: {bad} kept Gaussians had lower error than removed")
        if c == 0:
            unseen = before[:10].cpu().numpy().round(6)
            if any(tuple(r) in survivors for r in unseen):
                raise AssertionError("zero-error (unseen) Gaussians survived the first cycle")
    if len(seen) >= 2 and torch.equal(seen[0][1][: len(seen[1][1])], seen[1][1]):
        raise AssertionError("scores were not re-quantified between cycles")
    return {"counts": counts, "expected": expected_keep, "cycles": len(seen)}
