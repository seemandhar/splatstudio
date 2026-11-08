"""Pure LP-3DGS functions shared by renderer, density, loss and probes.

Official code: github.com/ASU-ESIC-FAN-Lab/LP-3DGS @ b828063, mask.py and
scene/gaussian_model_rad_splat.py.
"""

import torch

FIELD = "lp_mask"
# Non-learned mask state lives on the training ParameterDict (plain attributes, not fields):
#   phase "off"    - stock sigmoid opacity (also the evaluation renderer: the checkpoint dict
#                    carries no attributes, matching the official post-pruning renderer);
#   phase "train"  - opacity * gs(m * S), gradients reach m (official 19520 <= it < 20000);
#   phase "frozen" - opacity * gs(m * S) with m detached (official it == 20000: masked render,
#                    mask optimiser not stepped).
PHASE_ATTR = "_lp3dgs_phase"
SCORE_ATTR = "_lp3dgs_score"
PHASES = ("off", "train", "frozen")


def logistic_noise(shape, device, generator=None, eps=1e-10):
    """mask.py: g0 - g1 = -log(log(u1 + eps) / log(u2 + eps) + eps), u ~ U(0, 1)."""
    u1 = torch.rand(shape, device=device, generator=generator)
    u2 = torch.rand(shape, device=device, generator=generator)
    return -torch.log(torch.log(u1 + eps) / torch.log(u2 + eps) + eps)


def gumbel_sigmoid(x, tau, noise=None):
    """Eq. 7 as implemented (mask.py _gumbel_sigmoid, hard=False): sigmoid((x + n) / tau).

    The code feeds the logit x = m * S directly (the paper writes log(m)); code followed.
    """
    if noise is None:
        noise = logistic_noise(x.shape, x.device)
    return torch.sigmoid((x + noise) / tau)


def phase(params):
    return getattr(params, PHASE_ATTR, "off")


def effective_opacity(params, tau, noise=None):
    """Eq. 8: o_im = o_i * gs(m_i * S_i) while the mask is live, else sigmoid(o)."""
    opacity = params["opacities"].sigmoid()
    current = phase(params)
    if current == "off":
        return opacity
    if current not in PHASES:
        raise ValueError(f"unknown LP-3DGS mask phase {current!r}")
    mask = params[FIELD]
    if current == "frozen":
        mask = mask.detach()
    score = getattr(params, SCORE_ATTR).detach()  # computed under no_grad in the official code
    return opacity * gumbel_sigmoid(mask * score, tau, noise)


def mask_loss(mask, weight):
    """Official addtional_loss: 0.0005 * mean(sigmoid(m)) (paper Eq. 9 writes mean |m|)."""
    return weight * torch.sigmoid(mask).mean()


def in_mask_window(step, start, iters):
    """train.py: iteration > prune_iterations[0] and iteration < prune_iterations[0] + iters."""
    return start < step < start + iters


def is_score_step(step, start, iters, every):
    """train.py: score refreshed before the render of these steps (19520, 19540, ..., 19980)."""
    return in_mask_window(step, start, iters) and (step - start) % every == 0


def prune_selection(mask, tau, noise=None):
    """Official prune at start + iters: get_prune_mask = gs(m) (no score, fresh noise) < 0.5."""
    return (gumbel_sigmoid(mask.detach(), tau, noise) < 0.5).reshape(-1)
