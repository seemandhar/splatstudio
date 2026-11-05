"""SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding Catastrophic Scene Destruction
(arXiv 2405.17793, WACV 2026). Native SplatStudio port of the official code
https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS; paper fidelity unverified.

Vanilla 3DGS for 30k steps with one pixel-level importance-rank pruning event at step 20000:
top-1 by Score_SG_36 = CS1(c_gt, c_i) * alpha_i * T_i, union over all training views.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, TARGET_CAPABILITY
from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2405.17793",
    "https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS",
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_safeguardgs.probes:{fn}", description, cuda)


def _make(name, overrides, label):
    base = make_method(name)
    renderer_base, baseline_density, photometric = base.components
    renderer = ComponentSpec(
        "sg-renderer",
        "renderer",
        "ss_safeguardgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        CAPABILITIES,
        dict(renderer_base.config),
    )
    config = {**DEFAULTS, **overrides, "base": dict(baseline_density.config)}
    density = ComponentSpec(
        "sg-density",
        "density",
        "ss_safeguardgs.density:create_density",
        tuple(baseline_density.requires) + (TARGET_CAPABILITY,),
        (),
        config,
    )
    probes = (
        _probe("sg-topk-kernel", "topk_kernel",
               "CUDA slot keys (official last-contributor-per-batch rule) and per-tile / "
               "per-pixel top-k keep masks equal a sequential torch blend over gsplat tile "
               "lists, for SG_36 and SG_4, k = 1, 2"),
        _probe("sg-color-score", "color_score",
               "Red GT, blue front Gaussian with larger alpha*T, red back one: SG_4 keeps the "
               "front, SG_36 the back; pixel-level pruning leaves no covered ray empty"),
        _probe("sg-prune-schedule", "prune_schedule",
               "One prune at step 20000 (top-1, 0x24) after every view's GT is cached; survivors "
               "= union of per-view masks; params / Adam / strategy state aligned"),
    )
    contributions = (
        ContributionSpec(
            "pixel-level-rank-pruning",
            "Keep the top-k Gaussians per ray (official kernel: per 16x16 tile of per-pixel "
            "batch samples), union over training views (Sec. 3.2, 3.4)",
            ("sg-topk-kernel", "sg-color-score"),
        ),
        ContributionSpec(
            "color-weighted-score",
            "Score_SG_36 = CS1(c_gt, c_i) * alpha_i * T_i (Sec. 3.3, Eq. 2, Tab. 1)",
            ("sg-color-score", "sg-topk-kernel"),
        ),
        ContributionSpec(
            "prune-schedule",
            "Vanilla 3DGS, pruning once at 20k of 30k iterations with top-1 (Sec. 4.1)",
            ("sg-prune-schedule",),
        ),
    )
    return replace(
        base,
        description=f"SafeguardGS (arXiv 2405.17793) native port, {label}; "
        "paper fidelity unverified.",
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    """Paper setting: top-1 by Score_SG_36 at step 20000, official per-tile kernel semantics."""
    return _make("safeguardgs", {}, "top-1 Score_SG_36 at 20k")
