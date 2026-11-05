"""PUP 3D-GS (arXiv 2406.10219) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2406.10219",
    "https://github.com/j-alex-hanson/gaussian-splatting-pup",
)


def _make(name, ratios, label):
    base = make_method(name)
    renderer, baseline_density, photometric = base.components
    config = {**DEFAULTS, "prune_ratios": list(ratios)}
    config["base"] = dict(baseline_density.config)
    density = ComponentSpec(
        "pup-density",
        "density",
        "ss_pup3dgs.density:create_density",
        baseline_density.requires,
        (),
        config,
    )
    probes = (
        ProbeSpec(
            "fisher-oracle",
            "ss_pup3dgs.probes:fisher_oracle",
            "Official CUDA Fisher (mean+log-scale, 1/4-res patches) vs per-pixel autograd oracle",
            True,
        ),
        ProbeSpec(
            "prune-schedule",
            "ss_pup3dgs.probes:prune_schedule",
            "Two prune rounds: lowest log|F| removed per official percentile rule, fresh Adam, "
            "official pretrain/fine-tune learning rates",
            True,
        ),
    )
    contributions = (
        ContributionSpec(
            "fisher-sensitivity",
            "C1+C2: per-Gaussian log det of the 6x6 Fisher block over (mean, scale), "
            "accumulated over 4x4 patches of all training views",
            ("fisher-oracle",),
        ),
        ContributionSpec(
            "multi-round-prune-refine",
            f"C3: prune {label} after 30k-step 3DGS, 5k-step refinement per round",
            ("prune-schedule",),
        ),
    )
    return replace(
        base,
        description=(
            f"PUP 3D-GS native port ({label}, single 40k-step run: 3DGS 30k + 2x(prune, 5k "
            "refine)); paper fidelity unverified"
        ),
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    """CVPR / arXiv v3 pipeline: 80% then 50% (90% total)."""
    return _make("pup-3dgs", (0.8, 0.5), "80% then 50% = 90%")


def v1():
    """arXiv v1 pipeline: 66% then 66% (88.44% total)."""
    return _make("pup-3dgs-v1", (0.66, 0.66), "66% then 66% = 88.44%")
