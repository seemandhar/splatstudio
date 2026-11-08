"""Mechanism probes for the Opti3DGS port. Each raises AssertionError when its mechanism is
removed or wrong; references are written independently of filtering.py."""

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from splatstudio.losses import ssim

from .losses import create_photometric


def _spec(method, kind):
    return next(c for c in method.components if c.kind == kind)


def _loss_cfg(method):
    from .losses import DEFAULTS

    return {**DEFAULTS, **_spec(method, "loss").config}


def _random_image(h=48, w=64, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(h, w, 3, generator=g).to(device)


def mean_filter(method, device):
    """mean_blur vs scipy uniform_filter (mode='mirror' == reflect-101), checkerboard energy."""
    from .filtering import mean_blur

    cfg = _loss_cfg(method)
    if cfg["padding_mode"] != "reflect":
        raise AssertionError("probe reference assumes reflect padding")
    img = _random_image(device=device)
    out = {}
    for k in sorted(set(cfg["kernel_sizes"])) + [1]:
        got = mean_blur(img, k, "reflect").cpu().numpy()
        ref = uniform_filter(img.cpu().numpy().astype(np.float64), size=(k, k, 1), mode="mirror")
        err = float(np.abs(got - ref).max())
        if got.shape != ref.shape or err > 1e-5:
            raise AssertionError(f"mean blur K={k} differs from reference by {err}")
        out[f"max_err_k{k}"] = err
    # Checkerboard (Nyquist) must be strongly attenuated by the largest kernel.
    yy, xx = torch.meshgrid(torch.arange(48), torch.arange(64), indexing="ij")
    board = ((yy + xx) % 2).float()[..., None].expand(48, 64, 3).to(device)
    k0 = max(cfg["kernel_sizes"])
    blurred = mean_blur(board, k0, "reflect")
    hf_in = float((board - board.mean()).std())
    hf_out = float((blurred - blurred.mean()).std())
    if not hf_out < 0.05 * hf_in:
        raise AssertionError(f"K={k0} does not low-pass the checkerboard ({hf_out} vs {hf_in})")
    out.update(checkerboard_std_in=hf_in, checkerboard_std_out=hf_out)
    return out


def schedule(method, device):
    """Fig. 3 step schedule at its boundaries (independent table)."""
    from .filtering import kernel_at

    cfg = _loss_cfg(method)
    expected = {1: 15, 1000: 15, 1001: 9, 4000: 9, 4001: 5, 8000: 5, 8001: 3,
                12000: 3, 12001: 1, 20000: 1, 30000: 1}
    got = {s: kernel_at(s, cfg["kernel_sizes"], cfg["boundaries"]) for s in expected}
    if got != expected:
        raise AssertionError(f"schedule {got} != paper {expected}")
    return {str(s): k for s, k in got.items()}


def loss_target(method, device):
    """The training loss uses the blurred target while filtering is on, the original after."""
    loss = create_photometric(_spec(method, "loss"))
    gt = _random_image(seed=1, device=device)
    rgb = _random_image(seed=2, device=device).requires_grad_(True)
    w = 0.2

    def ref(target):
        return (1 - w) * (rgb - target).abs().mean() + w * (1 - ssim(rgb, target))

    # Independent K=15 reflect box blur via explicit shifts.
    pad = np.pad(gt.detach().cpu().numpy(), ((7, 7), (7, 7), (0, 0)), mode="reflect")
    blur = sum(pad[i:i + 48, j:j + 64] for i in range(15) for j in range(15)) / 225.0
    blur = torch.tensor(blur, dtype=gt.dtype, device=device)
    early = loss({}, {"rgb": rgb, "step": 1}, gt)
    late = loss({}, {"rgb": rgb, "step": 12001}, gt)
    e_ref, l_ref, unfiltered = ref(blur), ref(gt), ref(gt)
    if abs(early.item() - e_ref.item()) > 1e-5:
        raise AssertionError(f"step-1 loss {early.item()} != blurred-target loss {e_ref.item()}")
    if abs(early.item() - unfiltered.item()) < 1e-3:
        raise AssertionError("step-1 loss equals the unfiltered loss: modulation inactive")
    if abs(late.item() - l_ref.item()) > 1e-6:
        raise AssertionError("loss after 12000 does not use the original images")
    grad = torch.autograd.grad(early, rgb)[0]
    if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
        raise AssertionError("loss gradient is zero or nonfinite")
    return {"loss_step1": early.item(), "loss_step1_unfiltered": unfiltered.item(),
            "loss_step12001": late.item(), "gradient_norm": grad.norm().item()}


def density_config(method, device):
    """Ours* settings reach the strategy/optimizer; a real densify keeps fields trainable."""
    from splatstudio.verification import ProbeUnavailable

    from .density import OptiDensity

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    from splatstudio.components import build
    from splatstudio.state import check_state, sample_scene

    spec = _spec(method, "density")
    renderer, density, _ = build(method)
    if not isinstance(density, OptiDensity):
        raise AssertionError("density is not the Opti3DGS density")
    params, opts = sample_scene(method, device, n=64)
    density.initialize(params, opts, scene_scale=1.0)
    s = density.strategy
    lr = opts["scales"].param_groups[0]["lr"]
    if (abs(lr - spec.config["scales_lr"]) > 1e-12 or s.refine_every != spec.config["refine_every"]
            or abs(s.grow_grad2d - spec.config["grow_grad2d"]) > 1e-12):
        raise AssertionError(f"settings not applied: lr={lr}, every={s.refine_every}, "
                             f"grad={s.grow_grad2d}")
    camera = {"viewmat": torch.eye(4, device=device),
              "K": torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]], device=device),
              "width": 32, "height": 32}
    step = s.refine_every * (s.refine_start_iter // s.refine_every + 1)
    n0 = len(params["means"])
    # Force densification: zero threshold, accumulate one backward, densify at a refine step.
    s.grow_grad2d = 0.0
    out = renderer.render(params, camera, sh_degree=0)
    density.before_backward(params, opts, step, out)
    out["rgb"].sum().backward()
    for o in opts.values():
        o.step()
        o.zero_grad()
    density.after_backward(params, opts, step, out)
    check_state(method, params, opts)
    bad = [k for k, p in params.items() if not (p.requires_grad and p.is_leaf)]
    if bad:
        raise AssertionError(f"fields frozen or non-leaf after densification: {bad}")
    lr_after = opts["scales"].param_groups[0]["lr"]
    if abs(lr_after - spec.config["scales_lr"]) > 1e-12:
        raise AssertionError("scales lr lost after densification")
    return {"scales_lr": lr, "refine_every": s.refine_every, "n_before": n0,
            "n_after": len(params["means"]), "scales_lr_after_densify": lr_after}
