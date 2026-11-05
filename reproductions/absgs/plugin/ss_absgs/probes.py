"""Mechanism probes for AbsGS. Each raises AssertionError when the mechanism is absent/wrong."""

import math

import torch
from splatstudio.components import build

# Paper values (independent of DEFAULTS): §4.1 / Table 1.
PAPER = {
    "absgs": {"split": 0.0004, "reset": True},
    "absgs-0008": {"split": 0.0008, "reset": True},
    "absgs-noreset": {"split": 0.0004, "reset": False},
}
TAU_CLONE = 0.0002
TAU_S = 0.001


def _scene(n, device, means, scales, opacity=0.8):
    params = {
        "means": torch.tensor(means, dtype=torch.float32, device=device),
        "scales": torch.log(torch.tensor(scales, dtype=torch.float32, device=device)),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * n, device=device),
        "opacities": torch.full((n,), math.log(opacity / (1 - opacity)), device=device),
        "sh0": torch.full((n, 1, 3), 0.5, device=device),
        "shN": torch.zeros((n, 15, 3), device=device),
    }
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in params.items()})
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    return params, opts


def homodirectional_gradient(method, device):
    """C1: absgrad == sum_j |dL_j/dmu_x|, sum_j |dL_j/dmu_y| (independent per-pixel autograd)."""
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is required for the gsplat rasterizer")
    renderer, density, _ = build(method)
    # One large Gaussian centered in view (gradient collision) + one small off-center one.
    params, opts = _scene(
        2, device, [[0.0, 0.0, 4.0], [0.9, -0.6, 4.0]], [[1.2, 1.2, 1.2], [0.15, 0.15, 0.15]]
    )
    w = h = 16
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
        "width": w,
        "height": h,
    }
    target = torch.full((h, w, 3), 0.1, device=device)
    target[:, :, 0] += 0.3 * (torch.arange(w, device=device) % 3 == 0).float()[None, :]
    out = renderer.render(params, camera, sh_degree=0)
    density.initialize(params, opts, 1.0)
    density.before_backward(params, opts, 1, out)
    per_pixel = (out["rgb"] - target).abs().sum(-1)  # L_j, L1 per pixel
    means2d = out["info"]["means2d"]
    expected = torch.zeros_like(means2d)
    for j in range(h * w):
        (g,) = torch.autograd.grad(per_pixel.view(-1)[j], means2d, retain_graph=True)
        expected += g.abs()
    per_pixel.sum().backward()
    absgrad, signed = means2d.absgrad, means2d.grad
    if absgrad is None:
        raise AssertionError("Renderer did not produce absolute screen gradients")
    err = (absgrad - expected).abs().max().item()
    scale = expected.abs().max().item()
    if not err <= 1e-3 * scale + 1e-7:
        raise AssertionError(f"absgrad differs from independent sum |dL_j/dmu| ({err:.3e}/{scale:.3e})")
    big_abs, big_signed = absgrad[0, 0].norm().item(), signed[0, 0].norm().item()
    if not big_abs > 3.0 * big_signed:
        raise AssertionError(
            f"No gradient collision on the large Gaussian (|abs|={big_abs:.3e}, |signed|={big_signed:.3e})"
        )
    # The density controller must consume absgrad into its own accumulator.
    density.after_backward(params, opts, 1, out)
    state = density.state
    acc_abs, acc = state["grad2d_abs"][0].item(), state["grad2d"][0].item()
    ndc = torch.tensor([w / 2.0, h / 2.0], device=device)
    if not math.isclose(acc_abs, (absgrad[0, 0] * ndc).norm().item(), rel_tol=1e-4):
        raise AssertionError("Density does not accumulate the NDC-scaled homodirectional gradient")
    if not math.isclose(acc, (signed[0, 0] * ndc).norm().item(), rel_tol=1e-4, abs_tol=1e-9):
        raise AssertionError("Density does not accumulate the NDC-scaled signed gradient")
    return {
        "max_abs_error": err,
        "absgrad_scale": scale,
        "large_gaussian_abs_norm": big_abs,
        "large_gaussian_signed_norm": big_signed,
        "collision_ratio": big_abs / max(big_signed, 1e-12),
    }


def split_clone_criteria(method, device):
    """C2: clone <- signed >= 0.0002 & small; split <- abs >= tau_split & large; tau_S=0.001."""
    if method.name not in PAPER:
        raise AssertionError(f"No paper values for method {method.name}")
    paper = PAPER[method.name]
    tau_split = paper["split"]
    _, density, _ = build(method)
    small, large, huge = 0.5 * TAU_S, 5.0 * TAU_S, 0.05  # scene_scale 1; 0.005 < 3DGS tau 0.01
    rows = [  # (max scale, signed grad, abs grad, visible) -> expected (clone, split)
        (small, 1.5 * TAU_CLONE, 1.5 * TAU_CLONE, True, (True, False)),
        (small, 0.5 * TAU_CLONE, 4.5 * TAU_CLONE, True, (False, False)),  # abs high: no clone
        (large, 0.5 * TAU_CLONE, 1.25 * tau_split, True, (False, True)),  # collision: split
        (large, 1.5 * TAU_CLONE, 0.875 * tau_split, True, (False, False)),  # 3DGS would split
        (large, 1.0, 1.0, False, (False, False)),  # never visible
        (small, 0.95 * TAU_CLONE, 0.95 * TAU_CLONE, True, (False, False)),
        (huge, 0.0, 2.0 * tau_split, True, (False, True)),
    ]
    n = len(rows)
    means = [[0.1 * i, 0.0, 0.0] for i in range(n)]
    params, opts = _scene(n, device, means, [[r[0], 0.5 * r[0], 0.5 * r[0]] for r in rows], 0.5)
    density.initialize(params, opts, 1.0)
    strat, state = density.strategy, density.state
    w, h = 64, 48

    def info():
        m2d = torch.zeros((1, n, 2), device=device)
        # raw (pixel-normalized) grads; density multiplies by W/2 (x) and H/2 (y)
        m2d.grad = torch.tensor(
            [[[r[1] * 0.6 / (w / 2), r[1] * 0.8 / (h / 2)] for r in rows]], device=device
        )
        m2d.absgrad = torch.tensor(
            [[[r[2] * 0.8 / (w / 2), r[2] * 0.6 / (h / 2)] for r in rows]], device=device
        )
        radii = torch.tensor([[[3, 3] if r[3] else [0, 0] for r in rows]], device=device)
        return {"means2d": m2d, "width": w, "height": h, "n_cameras": 1, "radii": radii,
                "gaussian_ids": None}

    for _ in range(3):
        strat._update_state(params, state, info())
    if not torch.equal(state["count"].cpu(), torch.tensor([3.0, 3, 3, 3, 0, 3, 3])):
        raise AssertionError("View counts must include only visible Gaussians")
    clone, split_ = strat.decisions(params, state)
    exp_clone = torch.tensor([r[4][0] for r in rows])
    exp_split = torch.tensor([r[4][1] for r in rows])
    if not torch.equal(clone.cpu(), exp_clone):
        raise AssertionError(f"Clone mask {clone.tolist()} != expected {exp_clone.tolist()}")
    if not torch.equal(split_.cpu(), exp_split):
        raise AssertionError(f"Split mask {split_.tolist()} != expected {exp_split.tolist()}")

    # Full refine step at 3000: grow, prune, and (paper) opacity reset to 0.01.
    out = {"info": info()}
    density.after_backward(params, opts, 3000, out)
    n_after = len(params["means"])
    # +1 clone, 2 splits (each 1 -> 2), Gaussian 4 is too big (>0.1 extent after 3000) only if
    # huge > 0.1 -- it is not, so no world-size prune; opacity 0.5 > 0.005 so no opacity prune.
    expected_n = n + int(exp_clone.sum()) + int(exp_split.sum())
    if n_after != expected_n:
        raise AssertionError(f"Gaussian count after refine {n_after} != {expected_n}")
    for key in ("grad2d", "grad2d_abs", "count"):
        if len(state[key]) != n_after:
            raise AssertionError(f"State {key} not aligned after densification")
    max_opa = torch.sigmoid(params["opacities"]).max().item()
    if paper["reset"] and not max_opa <= 0.01 + 1e-6:
        raise AssertionError(f"Opacity reset at step 3000 missing (max opacity {max_opa})")
    if not paper["reset"] and max_opa <= 0.011:
        raise AssertionError("no-reset ablation must not reset opacities")
    return {
        "clone_threshold": strat.clone_grad2d,
        "split_threshold": strat.split_grad2d,
        "grow_scale3d": strat.grow_scale3d,
        "clone_mask": clone.tolist(),
        "split_mask": split_.tolist(),
        "gaussians_after_refine": n_after,
        "max_opacity_after_step_3000": max_opa,
    }
