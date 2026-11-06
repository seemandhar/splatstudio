"""Pure tensor helpers of the 3D-HGS density control (CPU-testable)."""

import torch
import torch.nn.functional as F


def averaged(acc, denom):
    """xyz_gradient_accum / denom with NaN (denom 0) -> 0."""
    out = acc / denom
    out[out.isnan()] = 0.0
    return out


CLONE_LOGIT_LIMIT = 40.0


def clone_opacity_logit(logit):
    """densify_and_clone: inverse_sigmoid(1 - sqrt(1 - sigmoid(o))).

    Evaluated in float64 as a / (1 + sqrt(1 - a)). The official float32 expression returns
    -inf for a half with alpha below ~1e-7 (frequent: one half is often transparent) and +inf
    for alpha == 1; such values are inert there (sigmoid gradient 0) but the runtime rejects
    non-finite state, so the logit is clamped to +-40 (alpha 4e-18 / 1 - 4e-18): same render.
    """
    a = torch.sigmoid(logit.double())
    out = torch.logit(a / (1 + torch.sqrt(1 - a)))
    return out.clamp(-CLONE_LOGIT_LIMIT, CLONE_LOGIT_LIMIT).to(logit.dtype)


def build_rotation(quats):
    """utils.general_utils.build_rotation (wxyz, normalised)."""
    q = F.normalize(quats, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def split_offsets(scales, quats, generator=None):
    """densify_and_split, N=2: R @ N(0, diag(s)) for stds = scales.repeat(2, 1)."""
    stds = scales.repeat(2, 1)
    samples = torch.randn(stds.shape, generator=generator, device=stds.device) * stds
    rots = build_rotation(quats).repeat(2, 1, 1)
    return torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
