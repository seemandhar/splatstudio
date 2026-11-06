"""Group Training density controller (Wang et al., ICCV 2025, arXiv 2412.07608).

Port of the official plug-in https://github.com/Chengbo-Wang/3DGS-with-Group-Training
(`__init__.py: gaussians_grouping_and_caching`, `grouping_method.py: get_under_training_mask`)
on the runtime's vanilla-3DGS density (gsplat DefaultStrategy + 3DGS opacity reset).

Between groupings only the Under-training group exists in `params`: the Cached group is
physically removed (official `prune_points(mask_caching)`) and held here as detached tensors,
so it is not rendered, receives no gradient and is neither stepped nor decayed by the
runtime's per-field Adam. At the next grouping it is appended back (official
`densification_postfix(**cache)`) with zero Adam moments and all density accumulators are
zeroed, exactly like the official code.
"""

from dataclasses import replace

import torch
from gsplat.strategy.ops import remove
from splatstudio.backends.gsplat import Density

GROUP_DEFAULTS = {
    "grouping": True,  # GroupingParams.Grouping
    "grouping_method": "opacity",  # 'Opacity-weighted' (OPS); 'random' = RS
    "utr": 0.6,  # Under Training Ratio
    "grouping_from_iter": 1000,
    "grouping_until_iter": 29000,  # = I_O (global optimisation merge)
    "grouping_interval": 500,
    "global_densify_iter": 14500,  # = I_D, hard-coded in the official code
    "freeze_sh_at_merge": True,  # f_rest.requires_grad = False at I_D / I_O
    "cached_moments": "reset",  # official: re-appended rows start with zero Adam moments
    "reset_stats_on_merge": True,  # official densification_postfix zeroes the accumulators
    "total_steps": 30000,  # final merge if a cache is still held at the last step
}
METHODS = ("opacity", "random")
MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


def grouping_iterations(config):
    """Official `range(grouping_from_iter, grouping_until_iter + 1, grouping_interval)`."""
    return set(
        range(
            int(config["grouping_from_iter"]),
            int(config["grouping_until_iter"]) + 1,
            int(config["grouping_interval"]),
        )
    )


def is_merge_only(config, iteration):
    """Official `_UTR = 1.0 if iteration in [14500, grouping_until_iter]`: merge, no resample."""
    return iteration in (int(config["global_densify_iter"]), int(config["grouping_until_iter"]))


@torch.no_grad()
def under_training_mask(opacity_logits, utr, method="opacity", generator=None):
    """Official get_under_training_mask: True = under training (Eq. 11 for OPS)."""
    n = opacity_logits.shape[0]
    device = opacity_logits.device
    if method == "random":
        # official: torch.rand([N]) < UTR (Bernoulli, not an exact count)
        return torch.rand(n, device=device, generator=generator) < utr
    if method != "opacity":
        raise ValueError(f"grouping_method must be one of {METHODS}")
    k = int(n * utr)
    theta = torch.sigmoid(opacity_logits.flatten())
    prob = theta / theta.sum()  # Eq. 11: p_i = alpha_i / sum alpha
    if n <= 2**24:
        idx = torch.multinomial(prob, k, replacement=False, generator=generator)
    else:
        # torch.multinomial's category limit; the exponential race is the same successive
        # sampling-without-replacement distribution.
        keys = torch.empty(n, device=device).exponential_(generator=generator) / prob
        idx = torch.topk(keys, k, largest=False).indices
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


@torch.no_grad()
def append_rows(params, optimizers, state, rows, moments=None):
    """Append cached rows to every field (official cat_tensors_to_optimizer).

    New Adam moment rows are zero unless `moments` supplies them; Adam's `step` is kept.
    DefaultStrategy running-state tensors get zero rows.
    """
    m = len(next(iter(rows.values())))
    for name in list(params.keys()):
        old = params[name]
        new = torch.nn.Parameter(torch.cat([old.detach(), rows[name].to(old)]), requires_grad=True)
        params[name] = new
        opt = optimizers[name]
        st = opt.state.pop(old, {})
        for key in list(st.keys()):
            if key in MOMENT_KEYS or key == "max_exp_avg_sq":
                extra = (
                    moments[name][key].to(st[key])
                    if moments is not None and key in moments.get(name, {})
                    else torch.zeros((m, *st[key].shape[1:]), dtype=st[key].dtype, device=st[key].device)
                )
                st[key] = torch.cat([st[key], extra])
        opt.param_groups[0]["params"] = [new]
        if st:
            opt.state[new] = st
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            state[key] = torch.cat([value, torch.zeros((m, *value.shape[1:]), dtype=value.dtype, device=value.device)])


class GroupTrainingDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        config = dict(spec.config)
        self.config = {k: config.pop(k, v) for k, v in GROUP_DEFAULTS.items()}
        c = self.config
        if c["grouping_method"] not in METHODS:
            raise ValueError(f"grouping_method must be one of {METHODS}")
        if not 0 < float(c["utr"]) < 1:
            raise ValueError("utr must be in (0, 1)")
        if c["cached_moments"] not in ("reset", "keep"):
            raise ValueError("cached_moments must be 'reset' (official) or 'keep'")
        self.base = Density(replace(spec, config=config))
        if self.base.requires != self.requires:
            raise ValueError("Group training wraps the signed-gradient 3DGS density only")
        self.iterations = grouping_iterations(c) if c["grouping"] else set()
        self.cache = None  # {"rows": {field: tensor}, "moments": {field: {key: tensor}} | None}
        self.sh_frozen = False
        self.log = []  # (iteration, event, n_active, n_cached) for probes/diagnostics

    @property
    def state(self):
        return self.base.state

    @property
    def strategy(self):
        return self.base.strategy

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def is_refine_step(self, step):
        s = self.strategy
        return (
            s.refine_start_iter < step < s.refine_stop_iter
            and step % s.refine_every == 0
            and step % s.reset_every >= s.pause_refine_after_reset
        )

    def after_backward(self, params, optimizers, step, outputs):
        self.base.after_backward(params, optimizers, step, outputs)
        self.group_step(params, optimizers, step)

    def group_step(self, params, optimizers, step):
        """Grouping work after the 3DGS density step of `step` (schedule of Fig. 5)."""
        if self.sh_frozen and self.is_refine_step(step):
            # official clone/split/prune rebuild f_rest with requires_grad_(True)
            self.set_sh_trainable(params, True)
        if step + 1 in self.iterations:
            # official grouping runs before the render of iteration step + 1
            self.regroup(params, optimizers, step + 1)
        if step >= int(self.config["total_steps"]) and self.cache is not None:
            self.merge(params, optimizers)
            self.log.append((step, "final-merge", len(params["means"]), 0))

    def set_sh_trainable(self, params, trainable):
        params["shN"].requires_grad_(trainable)
        self.sh_frozen = not trainable

    @torch.no_grad()
    def merge(self, params, optimizers):
        """Official `densification_postfix(**_points_caching)` (Fig. 3, groups merging)."""
        if self.cache is not None:
            append_rows(params, optimizers, self.state, self.cache["rows"], self.cache["moments"])
            self.cache = None
            self.sh_frozen = False  # appended parameters are rebuilt with requires_grad=True
            if self.config["reset_stats_on_merge"]:
                for key in ("grad2d", "count", "radii"):
                    if isinstance(self.state.get(key), torch.Tensor):
                        self.state[key].zero_()

    @torch.no_grad()
    def regroup(self, params, optimizers, iteration, mask=None):
        self.merge(params, optimizers)
        c = self.config
        if is_merge_only(c, iteration):
            if c["freeze_sh_at_merge"]:
                self.set_sh_trainable(params, False)
            self.log.append((iteration, "merge", len(params["means"]), 0))
            return
        if mask is None:
            mask = under_training_mask(params["opacities"], float(c["utr"]), c["grouping_method"])
        cached = ~mask
        rows = {name: p.detach()[cached].clone() for name, p in params.items()}
        moments = None
        if c["cached_moments"] == "keep":
            moments = {
                name: {
                    k: v[cached].clone()
                    for k, v in optimizers[name].state.get(p, {}).items()
                    if k in MOMENT_KEYS
                }
                for name, p in params.items()
            }
        self.cache = {"rows": rows, "moments": moments}
        remove(params, optimizers, self.state, cached)  # official prune_points(mask_caching)
        self.set_sh_trainable(params, True)  # official _prune_optimizer: requires_grad_(True)
        self.log.append((iteration, "regroup", len(params["means"]), int(cached.sum())))


def create_density(spec):
    return GroupTrainingDensity(spec)
