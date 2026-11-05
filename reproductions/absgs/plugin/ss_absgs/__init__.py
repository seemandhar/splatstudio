"""AbsGS: Recovering Fine Details for 3D Gaussian Splatting (arXiv 2404.10484).

Native SplatStudio port. Reuses the built-in gsplat renderer with ``absgrad=True`` (identical
per-pixel |dL/dmu| accumulation to the official diff-gaussian-rasterization-abs) and the
built-in photometric loss; replaces the density controller (``ss_absgs.density``).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATION = "https://arxiv.org/abs/2404.10484"
CODE = "https://github.com/TY424/AbsGS"


def _make(name, label, **overrides):
    base = make_method(name, absgrad=True)
    renderer, _, photometric = base.components
    density = ComponentSpec(
        "absgs-density",
        "density",
        "ss_absgs.density:create_density",
        ("gradient.screen.signed", "gradient.screen.absolute"),
        (),
        {**DEFAULTS, **overrides},
    )
    probes = base.probes + (
        ProbeSpec(
            "homodirectional-gradient",
            "ss_absgs.probes:homodirectional_gradient",
            "Renderer absgrad equals an independent per-pixel autograd sum of |dL_j/dmu| (x, y "
            "separately) and exceeds the colliding signed sum on a large Gaussian",
            True,
        ),
        ProbeSpec(
            "split-clone-criteria",
            "ss_absgs.probes:split_clone_criteria",
            "Fixture: clone uses signed grad >= clone_grad2d, split uses abs grad >= "
            "split_grad2d, tau_S scale boundary, NDC accumulation, opacity reset at 3000",
        ),
    )
    contributions = base.contributions + (
        ContributionSpec(
            "homodirectional-gradient",
            "C1 (§3.3): densification statistic sums per-pixel absolute gradient components",
            ("homodirectional-gradient",),
        ),
        ContributionSpec(
            "decoupled-split-clone",
            "C2 (§4.1): split guided by homodirectional gradient with larger threshold; clone "
            "keeps the 3DGS signed gradient; tau_S = 0.001",
            ("split-clone-criteria",),
        ),
    )
    return replace(
        base,
        components=(renderer, density, photometric),
        probes=probes,
        contributions=contributions,
        description=f"AbsGS ({label}) native port; paper fidelity unverified.",
        citations=(CITATION, CODE),
    )


def method():
    """AbsGS-0004 (tau_split = 0.0004, tau_S = 0.001): main/best paper row, code default."""
    return _make("absgs", "AbsGS-0004")


def absgs_0008():
    """AbsGS-0008 (tau_split = 0.0008, tau_S = 0.001): lighter paper variant."""
    return _make("absgs-0008", "AbsGS-0008", split_grad2d=0.0008)


def absgs_noreset():
    """Ablation: AbsGS-0004 without opacity reset, matching gsplat 1.5.3 DefaultStrategy."""
    return _make("absgs-noreset", "AbsGS-0004, no opacity reset ablation", opacity_reset=False)
