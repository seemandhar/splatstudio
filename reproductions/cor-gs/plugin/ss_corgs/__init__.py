"""CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization (ECCV 2024) — native port.

Defaults are the official code values (github.com/jiaw-z/CoR-GS, arguments/__init__.py,
train.py, scripts/run_llff.sh: `-r 8 --n_views 3 --random_background --iterations 10000
--densify_grad_threshold 0.0005 --gaussiansN 2 --coprune --coprune_threshold 5 --coreg
--sample_pseudo_interval 1 --start_sample_pseudo 500`), 10 000 iterations.

`cor-gs-single` is the same pipeline with one field and neither co-pruning nor co-reg — the
authors' "3DGS" row (Table 1/4: 19.22 dB) and the protocol-matched baseline for gains.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import COREG_DEFAULTS, FIELD2_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2405.12110", "https://github.com/jiaw-z/CoR-GS",
             "https://jiaw-z.github.io/CoR-GS", "https://github.com/VITA-Group/FSGS")


def _renderer():
    return ComponentSpec("corgs-renderer", "renderer", "ss_corgs.renderer:create_renderer",
                         ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                         dict(RENDERER_DEFAULTS))


def _density(**overrides):
    return ComponentSpec("corgs-density", "density", "ss_corgs.density:create_density",
                         ("gradient.screen.signed",), (), {**DENSITY_DEFAULTS, **overrides})


SCHEDULE_PROBE = ProbeSpec(
    "corgs-schedule", "ss_corgs.probes:corgs_schedule",
    "SH min(t//500,3), random train / black eval background, opacity resets 501/3501/6501/9501, "
    "co-prune 1000..10000 every 500")


def method():
    base = make_method("cor-gs")
    photometric = base.components[2]  # field 1: 0.8 L1 + 0.2 D-SSIM (Eq. 6)
    field2 = ComponentSpec("corgs-field2", "loss", "ss_corgs.losses:create_field2",
                           ("render.camera",), (), dict(FIELD2_DEFAULTS))
    coreg = ComponentSpec("corgs-coreg", "loss", "ss_corgs.losses:create_coreg",
                          ("render.camera",), (), dict(COREG_DEFAULTS))
    probes = (
        ProbeSpec("co-pruning", "ss_corgs.probes:co_pruning",
                  "Both fields pruned exactly where the nearest counterpart is > tau (brute-force "
                  "Eq. 3 reference), optimizer state aligned, gated to t>500 & t%500==0"),
        ProbeSpec("pseudo-coregularization", "ss_corgs.probes:pseudo_coregularization",
                  "Pseudo-view loss = independent L1/SSIM formula with detached targets, no "
                  "cross-field gradient, active for 500<=t<=10000, zero for identical fields",
                  True),
        ProbeSpec("two-fields", "ss_corgs.probes:two_fields",
                  "Second field = independent copy of field 1, own gradient/Adam/densification, "
                  "absent from runtime params (evaluation sees field 1 only)", True),
        SCHEDULE_PROBE,
    )
    contributions = (
        ContributionSpec("co-pruning", "Prune Gaussians without a counterpart within tau=5 in "
                         "the other field, every 500 its (§4.1, Eq. 2-3)", ("co-pruning",)),
        ContributionSpec("pseudo-view-coregularization",
                         "Rendering-disagreement loss between the two fields on online pseudo "
                         "views, lambda_p = 1 (§4.2, Eq. 4-7)", ("pseudo-coregularization",)),
        ContributionSpec("dual-radiance-fields", "Two 3DGS fields trained simultaneously; field "
                         "1 is kept for inference (§4)", ("two-fields",)),
        ContributionSpec("official-training-protocol", "FSGS-fork densification, random "
                         "background, SH every 500, resets after 500", ("corgs-schedule",)),
    )
    return replace(
        base,
        description="CoR-GS native port (two co-regularised 3DGS fields: co-pruning + "
        "pseudo-view co-regularisation; field 1 evaluated); paper fidelity unverified.",
        components=(_renderer(), _density(), photometric, field2, coreg),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method_single():
    base = make_method("cor-gs-single")
    return replace(
        base,
        description="CoR-GS pipeline with a single field (no co-pruning, no co-reg) = the "
        "authors' sparse-view 3DGS baseline; paper fidelity unverified.",
        components=(_renderer(), _density(n_fields=1, coprune=False), base.components[2]),
        probes=base.probes + (SCHEDULE_PROBE,),
        contributions=base.contributions + (
            ContributionSpec("official-training-protocol", "Same protocol as cor-gs with one "
                             "field", ("corgs-schedule",)),),
        citations=CITATIONS,
    )
