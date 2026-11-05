"""DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting (CVPR 2025,
arXiv 2504.00773) — native SplatStudio port.

Defaults are the paper / official-code values (github.com/DCVL-3D/DropGaussian_release).
`dropgaussian-nodrop` is the identical pipeline with gamma = 0 (paper Table 5, first row):
the protocol-matched baseline for measuring the dropout gain.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2504.00773",
    "https://github.com/DCVL-3D/DropGaussian_release",
)


def _build(name, gamma, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "dropgaussian-renderer",
        "renderer",
        "ss_dropgaussian.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        tuple(sorted(CAPABILITIES)),
        {**RENDERER_DEFAULTS, "gamma": gamma},
    )
    density = ComponentSpec(
        "dropgaussian-density",
        "density",
        "ss_dropgaussian.density:create_density",
        ("gradient.screen.signed",),
        (),
        dict(DENSITY_DEFAULTS),
    )
    photometric = ComponentSpec(
        "photometric",
        "loss",
        "ss_dropgaussian.losses:create_photometric",
        ("render.rgb",),
        (),
        dict(LOSS_DEFAULTS),
    )
    probes = (
        ProbeSpec("drop-compensation", "ss_dropgaussian.probes:drop_compensation",
                  "Eq. 5 mask: kept opacity o/(1-r), dropped 0, keep rate 1-r, E[o~]=o"),
        ProbeSpec("drop-schedule", "ss_dropgaussian.probes:drop_schedule",
                  "Eq. 6 r_t = gamma t / t_total; counter only on training renders; "
                  "density rejects a desynchronised counter"),
        ProbeSpec("dropout-in-render", "ss_dropgaussian.probes:dropout_in_render",
                  "Training render drops Gaussians (zero gradient for dropped ones)", True),
        ProbeSpec("test-time-all-gaussians", "ss_dropgaussian.probes:test_time_all_gaussians",
                  "No-grad render equals an independent all-Gaussian black-background render",
                  True),
        ProbeSpec("official-densification", "ss_dropgaussian.probes:official_densification",
                  "3DGS clone/split/prune with tau 5e-4, no size pruning, resets 3000/6000/9000"),
        ProbeSpec("loss-weights", "ss_dropgaussian.probes:loss_weights",
                  "L1 + 0.2 (1 - SSIM_same) against an independent formula"),
    )
    contributions = (
        ContributionSpec("dropgaussian-compensation",
                         "Random Gaussian dropping with 1/(1-r) opacity compensation in "
                         "training renders, all Gaussians at test time (paper §3.2, Eq. 5)",
                         ("drop-compensation", "dropout-in-render", "test-time-all-gaussians")),
        ContributionSpec("progressive-drop-rate",
                         "Dropping rate grows linearly with the iteration (Eq. 6, gamma 0.2)",
                         ("drop-schedule",)),
        ContributionSpec("official-training-setup",
                         "Official 3DGS settings of the release (tau 5e-4, no size pruning, "
                         "L1 weight 1.0, black background, clamped renders)",
                         ("official-densification", "loss-weights")),
    )
    if gamma == 0.0:  # ablation: keep only the training-setup evidence
        drop_names = {"drop-compensation", "drop-schedule", "dropout-in-render",
                      "test-time-all-gaussians"}
        probes = tuple(p for p in probes if p.name not in drop_names)
        contributions = contributions[2:]
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _build("dropgaussian", RENDERER_DEFAULTS["gamma"],
                  "DropGaussian native port (scheduled random Gaussian dropout with opacity "
                  "compensation); paper fidelity unverified.")


def method_nodrop():
    return _build("dropgaussian-nodrop", 0.0,
                  "DropGaussian pipeline with gamma = 0 (no dropout): protocol-matched 3DGS "
                  "baseline of the official release (paper Table 5 first row).")
