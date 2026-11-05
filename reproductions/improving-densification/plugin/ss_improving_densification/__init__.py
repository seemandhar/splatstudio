"""Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering (ImprovedGS;
Deng, Diao, Li, Yu, Xu; arXiv 2508.12313). Native SplatStudio port of the official code
https://github.com/XiaoBin2001/Improved-GS (20b5db3) with its rasterizer vendored unmodified;
paper fidelity unverified.

Per-scene Gaussian budget N_max (paper Sec. 4.2) is looked up from the scene directory name
(`ss-train --data .../garden`); set density config `budget` for other scenes.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, REQUIRES
from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2508.12313",
    "https://github.com/XiaoBin2001/Improved-GS",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_improving_densification.probes:{fn}", description, cuda)


PROBES = (
    _probe("raster-vendored", "raster_vendored",
           "Vendored official rasterizer matches gsplat's classic RGB on a random scene; "
           "absolute-gradient channels 2:4 dominate |signed| channels 0:2", True),
    _probe("eas-score", "eas_score",
           "Edge-weighted blend-weight sums (kernel) equal an independent gsplat colour-gradient "
           "computation; edge maps equal an explicit Laplacian loop; edge Gaussians score higher",
           True),
    _probe("eas-sampling", "eas_sampling",
           "Split parents are sampled only among abs-gradient qualifiers with frequency "
           "proportional to the score; view pool yields 10,10,5,... and all views at 3400/6400"),
    _probe("las-geometry", "las_geometry",
           "Long-axis split: children at +-0.45 L0 along the rotated longest axis, scales "
           "(1-d, sqrt(1-d^2)), opacity x0.6, parent removed, zero Adam moments"),
    _probe("growth-control", "growth_control",
           "Budget equals N_max*sqrt((it-500)/14000); a densification step lands exactly on it "
           "and never exceeds it"),
    _probe("rap-schedule", "rap_schedule",
           "Early prune at 300; opacity reset to 0.05 at 3000..12000; 20% lowest-opacity prune "
           "exactly at 3300 and 6300"),
    _probe("mu-gated-adam", "mu_gated_adam",
           "Optimizer steps skipped per the MU schedule (5 from 20000, 20 from 24000, none at "
           "30000); an update equals Adam on the summed gradients"),
    _probe("lr-schedule", "lr_schedule",
           "Position lr 4e-5 -> 2e-6 (x extent) exponential overrides the runtime schedule; "
           "opacity lr 0.025"),
)
CONTRIBUTIONS = (
    ContributionSpec("edge-aware-score", "EAS (Sec. 3.2, Eq. 5-7): Laplacian edge-weighted "
                     "blend-weight score selects split parents among abs-gradient candidates",
                     ("eas-score", "eas-sampling", "raster-vendored")),
    ContributionSpec("long-axis-split", "LAS (Sec. 3.3, Eq. 8): symmetric split along the "
                     "longest axis, d=0.45, opacity x0.6; only densification op",
                     ("las-geometry",)),
    ContributionSpec("recovery-aware-pruning", "RAP (Sec. 3.4): prune lowest 20% opacity 300 "
                     "iterations after the 3k/6k resets (reset to 0.05)", ("rap-schedule",)),
    ContributionSpec("multi-view-update", "MU (Sec. 3.5): summed-gradient updates every 5 / 20 "
                     "iterations after densification (code schedule 20000/24000)",
                     ("mu-gated-adam",)),
    ContributionSpec("growth-control", "GC (Sec. 3.6, Eq. 9): sqrt budget curve to the "
                     "per-scene N_max", ("growth-control",)),
    ContributionSpec("improvedgs-hyperparameters", "App. A.2: position lr 4e-5 -> 2e-6, "
                     "tau=0.0003 (abs), no size pruning; official opacity lr 0.025",
                     ("lr-schedule",)),
)


def _make(name, density_config, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "improvedgs-raster", "renderer", "ss_improving_densification.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        {"background": [0.0, 0.0, 0.0]},
    )
    density = ComponentSpec(
        "improvedgs-density", "density", "ss_improving_densification.density:create_density",
        tuple(sorted(REQUIRES)), (), {**DEFAULTS, **density_config},
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, base.components[2]),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _make("improving-densification", {},
                 "ImprovedGS (arXiv 2508.12313) native port with the official rasterizer "
                 "(EAS + LAS + RAP + MU + GC, official code defaults); paper fidelity "
                 "unverified.")
