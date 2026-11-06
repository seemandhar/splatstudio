"""AdamW-GS optimizer: Sparse Adam + Decoupled Attribute Regularisation (paper Sec. 4.1, 4.3).

Port of the official fused kernel `submodules/diff-gauss/cuda_rasterizer/adam.cu` (MIT,
https://github.com/EliottDJay/3DGS_AdamWGS) and its driver `diff_gauss.SparseGaussianAdam`.
For every element of a primitive that is visible in this iteration (radii > 0):

    bc1 = 1 / (1 - b1^t_i);  bc2 = 1 / (1 - b2^t_i)            # per-primitive step t_i (Eq. 4)
    m = b1 m + (1 - b1) g;   v = b2 v + (1 - b2) g^2           # moments see only grad(l) (Eq. 7)
    denom = sqrt(v bc2) + eps
    reg = min(reg_scale * dR/dtheta / denom, cap)              # DAR (Eq. 8, App. Eq. 13-16)
    theta = theta - lr (m bc1 / denom + reg)

with dR/dtau = sigmoid(tau)(1 - sigmoid(tau)) for logit opacity and dR/dkappa = exp(kappa) for
log scale, `reg_scale = lambda / img_scale` (rasterize_points.cu:adamUpdate). Invisible primitives
keep parameters and moments unchanged (beta' = beta V + (1 - V)). Implemented with float32 torch
ops instead of the fused kernel (same formula; no speed claim).
"""

import torch

REG_NONE, REG_OPACITY, REG_SCALE = 0, 1, 2


def img_scale(n_pixels):
    """Official `SparseGaussianAdam._img_scale`: first two digits x 10^(len-3) (~N_I / 10)."""
    n = int(n_pixels)
    if n < 100:
        raise ValueError("img_scale needs an image with at least 100 pixels")
    digits = str(n)
    return (int(digits[0]) * 10 + int(digits[1])) * 10 ** (len(digits) - 3)


class AdamWGSContext:
    """Per-iteration inputs that `optimizer.step()` does not receive from the runtime.

    `state` is the density controller's running state; its `asystep` tensor (int32 [N]) is kept
    aligned by gsplat's duplicate/split/remove ops like the official `optimizer.asystep`.
    """

    def __init__(self, config, state):
        self.config = config
        self.state = state
        self.visibility = None
        self.step = None
        self.img_scale = None

    def set_image_pixels(self, n_pixels):
        self.img_scale = img_scale(n_pixels)

    def begin_iteration(self, step, visibility):
        """Official `SparseGaussianAdam.step`: `asystep.add_(visibility)` once per iteration."""
        visibility = visibility.reshape(-1).bool()
        asystep = self.state.get("asystep")
        if asystep is None or asystep.shape[0] != visibility.shape[0]:
            raise ValueError("asystep is not aligned with the rendered population")
        asystep.add_(visibility.to(asystep.dtype))
        self.visibility = visibility
        self.step = int(step)

    def regularizer(self, field):
        """(kind, reg_scale, cap) of the DAR term for `field` at the current step."""
        c = self.config
        if self.img_scale is None:
            raise ValueError("AdamW-GS needs the training image size (set_training_cameras)")
        s = self.step
        if field == "opacities" and c["opacity_reg_begin"] <= s <= c["opacity_reg_end"]:
            return REG_OPACITY, c["lambda_opacity"] / self.img_scale, c["opacity_reg_cap"]
        if field == "scales" and c["scale_reg_begin"] <= s <= c["scale_reg_end"]:
            return REG_SCALE, c["lambda_scale"] / self.img_scale, c["scale_reg_cap"]
        return REG_NONE, 0.0, 0.0


@torch.no_grad()
def sparse_adamwgs_update(param, grad, exp_avg, exp_avg_sq, asystep, visible, lr, b1, b2, eps,
                          reg_kind=REG_NONE, reg_scale=0.0, cap=10.0):
    """In-place adam.cu update of the visible rows of one field (all tensors share dim 0)."""
    idx = torch.nonzero(visible, as_tuple=True)[0]
    if idx.numel() == 0:
        return
    shape = (-1,) + (1,) * (param.dim() - 1)
    t = asystep[idx].to(torch.float32).reshape(shape)
    bc1 = 1.0 / (1.0 - torch.pow(torch.tensor(b1, dtype=torch.float32, device=t.device), t))
    bc2 = 1.0 / (1.0 - torch.pow(torch.tensor(b2, dtype=torch.float32, device=t.device), t))
    p = param[idx]
    g = grad[idx]
    m = b1 * exp_avg[idx] + (1.0 - b1) * g
    v = b2 * exp_avg_sq[idx] + (1.0 - b2) * g * g
    denom = torch.sqrt(v * bc2) + eps
    if reg_kind == REG_OPACITY:
        o = torch.sigmoid(p)
        reg = torch.clamp_max(reg_scale * o * (1.0 - o) / denom, cap)
    elif reg_kind == REG_SCALE:
        reg = torch.clamp_max(reg_scale * torch.exp(p) / denom, cap)
    else:
        reg = 0.0
    param[idx] = p - lr * (m * bc1 / denom + reg)
    exp_avg[idx] = m
    exp_avg_sq[idx] = v


class SparseAdamWGS(torch.optim.Adam):
    """One-field optimizer with the runtime's `step()` signature, driven by an AdamWGSContext."""

    def __init__(self, param, lr, context, field, betas=(0.9, 0.999), eps=1e-15):
        super().__init__([param], lr=lr, betas=betas, eps=eps)
        self.context = context
        self.field = field

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("SparseAdamWGS does not support closures")
        group = self.param_groups[0]
        (param,) = group["params"]
        if param.grad is None:  # official: `if param.grad is None: continue`
            return None
        ctx = self.context
        visible = ctx.visibility
        if visible is None or visible.shape[0] != param.shape[0]:
            raise ValueError(f"{self.field}: visibility is not aligned with the parameter")
        state = self.state[param]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["step"] += 1
        kind, scale, cap = ctx.regularizer(self.field)
        b1, b2 = group["betas"]
        sparse_adamwgs_update(param.data, param.grad, state["exp_avg"], state["exp_avg_sq"],
                              ctx.state["asystep"], visible, group["lr"], b1, b2, group["eps"],
                              kind, scale, cap)
        return None
