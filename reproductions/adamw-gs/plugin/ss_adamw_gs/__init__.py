"""A Step to Decouple Optimization in 3DGS (AdamW-GS; Ding et al., ICLR 2026, arXiv 2601.16736).
Native SplatStudio port of the official code https://github.com/EliottDJay/3DGS_AdamWGS (MIT)
on the vanilla-3DGS host; paper fidelity unverified.

Methods (released configs of the official repo):
- `adamw-gs`: Mip-NeRF 360 outdoor (`experiments/mipnerf_adamwgsp`): Sparse Adam + DAR + RSR with
  StSS GS2 (0.1/0.3/0.5 from 1k/7k/10k), 3DGS-MCMC position noise in [1k, 14k], no opacity reset.
- `adamw-gs-gs8`: paper headline outdoor row GS8 (no released config): `adamw-gs` plus 3DGS opacity
  reset every 3000 (to 0.01) and the 3DGS world-space big-primitive prune (max scale > 0.1 extent
  after step 3000), which official train.py:219 enables together with resets. The official 20-px
  screen test is dead code (max_radii2D is re-zeroed by densification_postfix before prune()).
- `adamw-gs-indoor`: Mip-NeRF 360 indoor (`experiments/mipnerf_adamwgs`): same without noise,
  StSS GS1 (0.1/0.3 from 1k/7k).

The runtime has no optimizer factory: the density component replaces the runtime's per-field
Adam optimizers by `SparseAdamWGS` in `initialize` (see density.py).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2601.16736",
    "https://github.com/EliottDJay/3DGS_AdamWGS",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_adamw_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("adamwgs-install", "install",
           "density.initialize swaps every runtime Adam for SparseAdamWGS sharing one context; a "
           "runtime-style step through the runtime optimizers dict leaves invisible rows and "
           "moments unchanged"),
    _probe("adamwgs-sparse-step", "sparse_step",
           "SparseAdamWGS equals a scalar transcription of the official adam.cu on visible rows; "
           "invisible rows and moments stay bit-identical; bias correction uses the per-primitive "
           "asystep (a global-step Adam is distinguishable)"),
    _probe("adamwgs-dar", "dar",
           "Opacity/scale DAR term min(lambda/img_scale * dR / (sqrt(v_hat)+eps), 10) matches the "
           "reference, moments contain only the photometric gradient, cap gives lr*10, active "
           "exactly in [3000,15000] (opacity) and [1000,15000] (scale)"),
    _probe("adamwgs-img-scale", "image_scale",
           "N_I normalisation equals the official _img_scale two-significant-digit rule"),
    _probe("adamwgs-rsr", "rsr",
           "Re-State Regularisation rescales exactly int(ratio*N) uniformly sampled primitives in "
           "every optimizer by (alpha1, alpha2) on the milestone StSS schedule"),
    _probe("adamwgs-topology", "topology",
           "Clone/split/prune through the density keep asystep inherited from the parent, children "
           "moments zero, params/optimizers aligned; clone opacity correction 1-sqrt(1-o) on source "
           "and copy"),
    _probe("adamwgs-noise", "noise",
           "Position noise equals gsplat's 3DGS-MCMC inject_noise_to_position with scaler "
           "noise_lr*lr and is scheduled in [1000,14000] (off for the indoor variant)", True),
)
CONTRIBUTIONS = (
    ContributionSpec("sparse-adam", "Sparse Adam with asynchronous per-primitive steps "
                     "(Sec. 4.1, Eq. 4; diff_gauss.SparseGaussianAdam / adam.cu)",
                     ("adamwgs-install", "adamwgs-sparse-step", "adamwgs-topology")),
    ContributionSpec("re-state-regularization", "Re-State Regularisation with milestone StSS "
                     "(Sec. 4.2, Eq. 5; App. 'StSS in RSR')", ("adamwgs-rsr",)),
    ContributionSpec("decoupled-attribute-regularization", "Decoupled opacity/scale "
                     "regularisation preconditioned by the photometric second moment "
                     "(Sec. 4.3, Eq. 7-8; App. Eq. 13-16)", ("adamwgs-dar", "adamwgs-img-scale")),
    ContributionSpec("adamw-gs-recoupling", "AdamW-GS in vanilla 3DGS: exploration noise, "
                     "opacity correction on clone, no opacity reset (Sec. 4.4)",
                     ("adamwgs-noise", "adamwgs-topology")),
)


def _make(name, density_config, description):
    base = make_method(name)
    density = ComponentSpec(
        "adamw-gs-density", "density", "ss_adamw_gs.density:create_density",
        ("gradient.screen.signed",), (), {**DEFAULTS, **density_config},
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
    return _make("adamw-gs", {},
                 "AdamW-GS (arXiv 2601.16736) native port on vanilla 3DGS, Mip-NeRF 360 outdoor "
                 "config (noise, StSS GS2); paper fidelity unverified.")


def indoor():
    return _make("adamw-gs-indoor",
                 {"noise": False, "rsr_ratios": [0.1, 0.3], "rsr_shifts": [1000, 7000]},
                 "AdamW-GS (arXiv 2601.16736) native port on vanilla 3DGS, Mip-NeRF 360 indoor "
                 "config (no noise, StSS GS1); paper fidelity unverified.")


def gs8():
    return _make("adamw-gs-gs8", {"opacity_reset_every": 3000},
                 "AdamW-GS (arXiv 2601.16736) native port on vanilla 3DGS, paper row GS8 (outdoor "
                 "config + opacity reset every 3000 + 3DGS world-space big prune); paper fidelity "
                 "unverified.")
