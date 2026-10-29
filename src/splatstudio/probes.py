"""Small diagnostic cases. These are infrastructure checks, not reproduction benchmarks."""

import io

import torch

from .components import build
from .state import check_state, sample_scene


def state_transitions(method, device):
    from gsplat.strategy.ops import duplicate, remove, split

    params, optimizers = sample_scene(method, device)
    # Allocate real Adam moments before testing topology changes.
    sum(p.square().sum() for p in params.values()).backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    original = {k: v.detach().clone() for k, v in params.items()}
    original_moments = {k: optimizers[k].state[p]["exp_avg"].clone() for k, p in params.items()}
    mask = torch.zeros(12, dtype=torch.bool, device=device)
    mask[0] = True
    state = {"probe_index": torch.arange(12, device=device)}
    duplicate(params, optimizers, state, mask)
    check_state(method, params, optimizers)
    for name, p in params.items():
        torch.testing.assert_close(p[-1], original[name][0])
        torch.testing.assert_close(
            optimizers[name].state[p]["exp_avg"][:12], original_moments[name]
        )
        if torch.count_nonzero(optimizers[name].state[p]["exp_avg"][-1]):
            raise AssertionError("Cloned optimizer moments should start at zero")
    mask = torch.zeros(13, dtype=torch.bool, device=device)
    mask[1] = True
    split(params, optimizers, state, mask)
    check_state(method, params, optimizers)
    for f in method.fields:
        torch.testing.assert_close(params[f.name][-2:], original[f.name][1].expand(2, *f.shape))
    mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=device)
    mask[0] = True
    before_remove = {k: v.detach().clone() for k, v in params.items()}
    remove(params, optimizers, state, mask)
    check_state(method, params, optimizers)
    for k, p in params.items():
        torch.testing.assert_close(p, before_remove[k][~mask])
    payload = {
        "params": dict(params),
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location=device, weights_only=True)
    new_params = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v) for k, v in loaded["params"].items()}
    )
    new_opts = {k: torch.optim.Adam([v]) for k, v in new_params.items()}
    for k, opt in new_opts.items():
        opt.load_state_dict(loaded["optimizers"][k])
    check_state(method, new_params, new_opts)
    for k in params:
        torch.testing.assert_close(new_params[k], params[k])
    return {
        "operations": ["duplicate", "split", "prune", "checkpoint"],
        "initial_gaussians": 12,
        "final_gaussians": len(params["means"]),
        "checked_fields": sorted(params),
        "optimizer_moments_checked": True,
    }


def photometric_gradient(method, device):
    from .losses import check_loss

    _, _, losses = build(method)
    rgb = torch.full((16, 16, 3), 0.3, device=device, requires_grad=True)
    target = torch.full_like(rgb, 0.7)
    name, loss = next((n, f) for n, f in losses if n == "photometric")
    value = loss({}, {"rgb": rgb}, target)
    check_loss(value, name)
    grad = torch.autograd.grad(value, rgb)[0]
    if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
        raise AssertionError("Photometric gradient is zero or nonfinite")
    return {"loss": value.item(), "gradient_norm": grad.norm().item()}


def renderer_gradient(method, device):
    from .verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    renderer, density, _ = build(method)
    params, opts = sample_scene(method, device)
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]], device=device),
        "width": 16,
        "height": 16,
    }
    outputs = renderer.render(params, camera, sh_degree=0)
    density.initialize(params, opts)
    density.before_backward(params, opts, 1, outputs)
    # Spatially varying target exercises gradients on both sides of projected centers.
    target = (torch.arange(16, device=device)[None, :, None] % 2).expand(16, 16, 3).float()
    (outputs["rgb"] - target).square().mean().backward()
    xy = outputs["info"][
        "gradient_2dgs" if "gradient.surfel" in renderer.capabilities else "means2d"
    ]
    if xy.grad is None or not torch.isfinite(xy.grad).all() or not torch.count_nonzero(xy.grad):
        raise AssertionError("Renderer has no finite nonzero screen-space gradient")
    evidence = {
        "rgb_shape": list(outputs["rgb"].shape),
        "screen_gradient_norm": xy.grad.norm().item(),
    }
    if "gradient.screen.absolute" in renderer.capabilities:
        absolute = getattr(xy, "absgrad", None)
        if (
            absolute is None
            or not torch.isfinite(absolute).all()
            or not torch.count_nonzero(absolute)
        ):
            raise AssertionError("Renderer declares absolute gradients but does not produce them")
        if torch.any(absolute + 1e-6 < xy.grad.abs()):
            raise AssertionError("Absolute accumulation is smaller than the absolute signed sum")
        evidence["absolute_gradient_norm"] = absolute.norm().item()
    density.after_backward(params, opts, 1, outputs)
    check_state(method, params, opts)
    return evidence
