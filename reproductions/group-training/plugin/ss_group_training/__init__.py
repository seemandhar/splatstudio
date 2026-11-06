"""Faster and Better 3D Splatting via Group Training (Wang et al., ICCV 2025, arXiv 2412.07608).

Native SplatStudio port of the official plug-in
https://github.com/Chengbo-Wang/3DGS-with-Group-Training on the vanilla-3DGS host (gsplat
renderer, 3DGS ADC, L1 + 0.2 D-SSIM); paper fidelity unverified.

- `group-training`: official default GroupingParams (OPS, UTR 0.6, groupings every 500 iters
  from 1000 to 29000, merge-only at I_D=14500 and I_O=29000) = paper rows "w/ OPS, 0~30K".
- `group-training-smoke2k`: identical but `total_steps=2000`, so the cached group is merged
  back before the runtime's step-2000 checkpoint (smoke tests only).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import GROUP_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2412.07608",
    "https://github.com/Chengbo-Wang/3DGS-with-Group-Training",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_group_training.probes:{fn}", description, cuda)


PROBES = (
    _probe("gt-ops-sampling", "ops_sampling",
           "OPS draws exactly int(N*UTR) distinct Gaussians with inclusion frequencies matching an "
           "independent successive-sampling reference p_i = o_i / sum o (Eq. 11); random sampling "
           "and with-replacement sampling are rejected"),
    _probe("gt-schedule", "schedule",
           "Groupings exactly at 1000..29000 step 500 (run in after_backward of the previous "
           "step), merge-only at I_D=14500 and I_O=29000, SH rest frozen at those merges and "
           "unfrozen by the next refine/regroup, final merge at total_steps"),
    _probe("gt-subset-state", "subset_state",
           "Regroup keeps only the sampled rows in params/optimizers/density state; cached rows "
           "stay bit-identical through Adam steps (while the runtime Adam would move zero-grad "
           "rows); merge restores every row, returned rows have zero moments (official), "
           "retained rows keep theirs, accumulators are zeroed, optimizer state aligned"),
    _probe("gt-cuda-training", "cuda_training",
           "CUDA: training on a synthetic scene with gsplat renders only the under-training group "
           "(means2d rows == active count), densify statistics stay in subset indexing, and the "
           "final merge restores the cached Gaussians unchanged", True),
)
CONTRIBUTIONS = (
    ContributionSpec("group-training", "Subset rendering and optimisation of the Under-training "
                     "group with a frozen Cached group, merged before each regroup (Sec. 3.1, "
                     "Eq. 1-2, Fig. 3)", ("gt-subset-state", "gt-cuda-training")),
    ContributionSpec("opacity-prioritized-sampling", "Opacity-based Prioritized Sampling "
                     "(Sec. 3.2-3.3, Eq. 11)", ("gt-ops-sampling",)),
    ContributionSpec("group-schedule", "Cyclic resampling every 500 iterations with global "
                     "densification (I_D) / optimisation (I_O) merges and SH freezing (Sec. 4.1, "
                     "Fig. 5, Tab. 6)", ("gt-schedule",)),
)


def _make(name, overrides, description):
    base = make_method(name)
    old = base.components[1]
    density = ComponentSpec(
        "group-training-density", "density", "ss_group_training.density:create_density",
        ("gradient.screen.signed",), (), {**old.config, **GROUP_DEFAULTS, **overrides},
    )
    return replace(
        base,
        description=description,
        components=(base.components[0], density, base.components[2]),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _make("group-training", {},
                 "Group Training (arXiv 2412.07608) native port on vanilla 3DGS, official default "
                 "OPS 0~30K schedule; paper fidelity unverified.")


def smoke2k():
    return _make("group-training-smoke2k", {"total_steps": 2000},
                 "Group Training smoke-test variant (final merge at step 2000); not a paper "
                 "configuration.")
