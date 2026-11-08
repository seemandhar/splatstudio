"""Population operations that keep params, BOGausS optimizer states and density state aligned.

New Parameters are always built from `p.detach()` with `requires_grad` copied from the old
Parameter (a `clone()` under no_grad silently freezes the field; see TODO-runtime.md).
"""

import torch

from . import core


def _swap(params, optimizers, name, new_data, state_fn):
    old = params[name]
    new = torch.nn.Parameter(new_data.detach().contiguous(), requires_grad=old.requires_grad)
    params[name] = new
    opt = optimizers[name]
    st = opt.state.pop(old, {})
    opt.param_groups[0]["params"] = [new]
    if st:
        opt.state[new] = {k: state_fn(k, v) for k, v in st.items()}


@torch.no_grad()
def remove(params, optimizers, state, mask):
    keep = torch.nonzero(~mask).flatten()
    for name in list(params.keys()):
        _swap(params, optimizers, name, params[name].detach()[keep], lambda k, v: v[keep])
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor):
            state[k] = v[keep]


@torch.no_grad()
def split(params, optimizers, state, sel, offset=0.3, alpha_t=1.0, alpha_g=0.2,
          min_size_factor=1.0):
    """Density-preserving split of Gaussians `sel` (long index): child A replaces the mother
    in place, child B is appended. Means shift by +-offset*sigma_long along the longest axis;
    split-axis size *= ratio(o), opacity -> o_child(o) (offline SplitAdaptation); other fields
    copied. Optimizer states inherited: t*alpha_t, m*alpha_g, v*alpha_g^2."""
    if len(sel) == 0:
        return
    log_s = params["scales"].detach()[sel]
    sigma = log_s.exp()
    axis = sigma.argmax(dim=-1)
    rot = core.quat_to_rotmat(params["quats"].detach()[sel])
    rows = torch.arange(len(sel), device=sel.device)
    direction = rot[rows, :, axis]  # [M,3] longest principal axis
    shift = offset * sigma[rows, axis][:, None] * direction
    o = params["opacities"].detach()[sel].sigmoid()
    ratio, o_child = core.split_adaptation(o, offset)
    child_sigma = sigma.clone()
    child_sigma[rows, axis] = sigma[rows, axis] * ratio
    if "delta" in state:
        child_sigma = torch.maximum(child_sigma, min_size_factor * state["delta"][sel][:, None])
    children = {
        "means_a": params["means"].detach()[sel] + shift,
        "means_b": params["means"].detach()[sel] - shift,
        "scales": child_sigma.log(),
        "opacities": torch.logit(o_child.clamp(1e-6, 1 - 1e-6)),
    }

    def inherit(k, v):
        if k == "t":
            mother = v[sel] * alpha_t
        elif k == "exp_avg":
            mother = v[sel] * alpha_g
        elif k == "exp_avg_sq":
            mother = v[sel] * alpha_g * alpha_g
        else:
            raise KeyError(f"unexpected optimizer state {k}")
        v = v.clone()
        v[sel] = mother
        return torch.cat([v, mother])

    for name in list(params.keys()):
        p = params[name].detach()
        if name == "means":
            a, b = children["means_a"], children["means_b"]
        elif name in children:
            a = b = children[name]
        else:
            a = b = p[sel]
        new = p.clone()
        new[sel] = a
        _swap(params, optimizers, name, torch.cat([new, b]), inherit)
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat([v, v[sel]])
