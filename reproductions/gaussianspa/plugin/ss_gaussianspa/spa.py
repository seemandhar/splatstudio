"""Pure tensor functions of GaussianSpa's "optimizing-sparsifying" solver (CPU-testable).

Numerics follow the official `optimizing_spa.py` and `train_opacity.py` / `train_imp_score.py`
of https://github.com/noodle-lab/GaussianSpa. `a` is the activated opacity sigmoid(logit),
`z` the auxiliary sparse copy, `u` the scaled dual multiplier (lambda in the paper).
"""

import torch


def _rank_index(n, ratio):
    index = int(ratio * n)
    if not 1 <= index <= n:
        raise ValueError(f"prune ratio {ratio} leaves no threshold for {n} Gaussians")
    return index


def project_opacity(v, ratio):
    """prox_h(v) for the opacity criterion (Eq. 16, official `prune_z`): with
    thr = sort(v)[int(ratio N) - 1], keep v where v > thr, zero elsewhere."""
    v = v.flatten()
    thr = torch.sort(v).values[_rank_index(v.numel(), ratio) - 1]
    return v * (v > thr)


def project_importance(v, importance, ratio):
    """Importance-ranked projection (Sec. 5.2, official `prune_z_metrics_imp_score`): with
    thr = sort(imp)[int(ratio N) - 1], z = v and z[imp < thr] = 0."""
    v, importance = v.flatten(), importance.flatten()
    thr = torch.sort(importance).values[_rank_index(importance.numel(), ratio) - 1]
    z = v.clone()
    z[importance < thr] = 0
    return z


def penalty(a, z, u, delta):
    """Optimizing-step term (Eq. 11): delta/2 ||a - z + u||_2^2, a sum over all Gaussians
    (official `append_spa_loss`). z and u carry no gradient."""
    return 0.5 * delta * torch.linalg.vector_norm(a - z.detach() + u.detach()) ** 2


def admm_step(a, u, ratio, importance=None):
    """One sparsifying step + multiplier update (Eq. 16-17, official `update`):
    z = prox(a + u); u <- u + a - z. Returns (z, u_new)."""
    v = a + u
    z = project_opacity(v, ratio) if importance is None else project_importance(v, importance,
                                                                                 ratio)
    return z, u + a - z


def penalty_active(step, start, stop, interval, stop_inclusive):
    """Official loss window: it > start and it % interval == 0 and it <= stop (opacity script)
    or it < stop (importance script)."""
    upper = step <= stop if stop_inclusive else step < stop
    return step > start and step % interval == 0 and upper


def update_active(step, start, stop, interval):
    """Official z/u update window (both scripts): start < it <= stop, it % interval == 0."""
    return start < step <= stop and step % interval == 0


def hard_prune_mask(ratio, opacity=None, importance=None):
    """Removal mask of the final hard prune at the ADMM stop iteration.

    opacity (train_opacity.py): keep topk(a, int((1 - ratio) N)), remove the rest.
    importance (train_imp_score.py): remove imp <= sort(imp)[int(ratio N) - 1].
    """
    if (opacity is None) == (importance is None):
        raise ValueError("Give exactly one of opacity or importance")
    if opacity is not None:
        opacity = opacity.flatten()
        k = int((1 - ratio) * opacity.numel())
        remove = torch.ones(opacity.numel(), dtype=torch.bool, device=opacity.device)
        remove[torch.topk(opacity, k, largest=True).indices] = False
        return remove
    importance = importance.flatten()
    thr = torch.sort(importance).values[_rank_index(importance.numel(), ratio) - 1]
    return importance <= thr


def sampling_count(n, prune_ratio1):
    """GaussianSpa's 15k simplification keeps int(N (1 - prune_ratio1)) of ALL Gaussians
    (Mini-Splatting instead uses int(N * 0.5 * nnz / N))."""
    return int(n * (1 - prune_ratio1))


def sampling_weights(importance):
    """GaussianSpa's 15k sampling probability is proportional to imp + 1, so Gaussians with
    zero importance can survive (official train_*.py: prob = (imp+1)/(imp+1).sum())."""
    return importance + 1
