"""Improving Adaptive Density Control for 3D Gaussian Splatting (Grubert, Barthel, Hilsmann,
Eisert; GRAPP 2025, arXiv 2503.14274). Native SplatStudio port of the official code
https://github.com/fraunhoferhhi/Improving-ADC-3DGS (65901d8); paper fidelity unverified.

Three contributions (corrected scene extent, exponentially ascending gradient threshold,
significance-aware pruning) on top of the paper's base (PixelGS pixel-aware depth-scaled gradient
and Bulo et al.'s revised opacity), all inside one density component; renderer and loss are the
stock 3DGS/gsplat ones.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2503.14274",
    "https://github.com/fraunhoferhhi/Improving-ADC-3DGS",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_improving_adc.probes:{fn}", description, cuda)


PROBES = (
    _probe("adc-stats-kernel", "stats_kernel",
           "CUDA per-Gaussian pixel count and blend-weight sum sigma (Eq. 15) equal a brute-force "
           "per-pixel loop and the official fake-colour autograd route", True),
    _probe("adc-scene-extent", "scene_extent",
           "Corrected extent equals Eq. 13 on a clustered-camera fixture (e' >> 3DGS extent), "
           "drives clone/split and the means LR; the camera ablation does not"),
    _probe("adc-threshold", "threshold_schedule",
           "Per-call thresholds equal Eq. 14 at k/145; a Gaussian with tau'=2.5e-4 densifies at "
           "call 1 but not at call 144; the constant ablation differs"),
    _probe("adc-significance-prune", "significance_prune",
           "Background preservation: a low-opacity, high-sigma background Gaussian survives while "
           "occluded low-sigma ones are pruned (mask equals an independent re-implementation); "
           "opacity-only pruning removes it; no size pruning", True),
    _probe("adc-pixel-gradient", "pixel_gradient",
           "Accumulated tau' equals an independent sum f*count*|g| / sum count with the squared "
           "PixelGS depth factor, and differs from the 3DGS per-view mean", True),
    _probe("adc-revised-opacity", "revised_opacity",
           "Clone parent+copy and split children get 1-sqrt(1-o); new Adam moments zero, kept "
           "parents' moments unchanged"),
    _probe("adc-schedule", "schedule",
           "Densification at 600..14900 (144 calls), opacity reset at 3000..12000, no statistics "
           "or density work at/after 15000"),
)
CONTRIBUTIONS = (
    ContributionSpec("corrected-scene-extent", "Sec. 4.1, Eq. 13: extent = mean distance of SfM "
                     "points to the mean camera centre (also the means LR scale, official "
                     "spatial_lr_scale)", ("adc-scene-extent",)),
    ContributionSpec("exponential-gradient-threshold", "Sec. 4.2, Eq. 14: threshold rises "
                     "exponentially 1e-4 -> 4e-4 over the densification calls",
                     ("adc-threshold", "adc-schedule")),
    ContributionSpec("significance-aware-pruning", "Sec. 4.3, Eq. 15-16, Alg. 2: prune "
                     "low-opacity Gaussians only if their blend-weight sum is in the bottom "
                     "N_prune; no size pruning (custom CUDA statistics)",
                     ("adc-stats-kernel", "adc-significance-prune")),
    ContributionSpec("pixelgs-gradient-base", "Paper base (Sec. 2, Eq. 9-10): PixelGS "
                     "pixel-count-weighted, depth-scaled 2D gradient", ("adc-stats-kernel",
                                                                          "adc-pixel-gradient")),
    ContributionSpec("revised-opacity-base", "Paper base (Sec. 2, Eq. 8): Bulo et al. opacity "
                     "1-sqrt(1-o) for cloned and split Gaussians", ("adc-revised-opacity",)),
)


def method():
    base = make_method("improving-adc")
    density = ComponentSpec(
        "improving-adc-density", "density", "ss_improving_adc.density:create_density",
        ("gradient.screen.signed",), (), dict(DEFAULTS),
    )
    return replace(
        base,
        description="Improving ADC for 3DGS (arXiv 2503.14274) native port; paper fidelity "
                    "unverified.",
        components=(base.components[0], density, base.components[2]),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
