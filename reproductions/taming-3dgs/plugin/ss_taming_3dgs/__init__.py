"""Taming 3DGS: High-Quality Radiance Fields with Limited Resources (Mallick, Goel et al.,
SIGGRAPH Asia 2024, arXiv 2406.15643). Native SplatStudio port of the official code
https://github.com/humansensinglab/taming-3dgs (fd0f7d9); paper fidelity unverified.

Methods
- `taming-3dgs`: "Ours (#3DGS)" (full_eval.py --mode big --sh_lower): final count = the 3DGS
  count of the scene, densification every 100 steps.
- `taming-3dgs-budget`: "Ours" (--mode budget --sh_lower): final count = multiplier x SfM count
  (15 outdoor Mip-NeRF360, 2 indoor / T&T, 5 Deep Blending), densification every 500 steps.
The per-scene budget is looked up from the initial SfM point count (see core.SCENE_SFM_COUNTS);
set density config `budget` for other scenes.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from . import core
from .density import DEFAULTS
from .renderer import TARGET_CAPABILITY

CITATIONS = (
    "https://arxiv.org/abs/2406.15643",
    "https://github.com/humansensinglab/taming-3dgs",
)
RENDERER_CAPS = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                 TARGET_CAPABILITY)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_taming_3dgs.probes:{fn}", description, cuda)


PROBES = (
    _probe("taming-stats-kernel", "stats_kernel",
           "CUDA per-Gaussian pixel count / centre distance / saliency / blend-weight sums "
           "equal a brute-force per-pixel loop over gsplat's projection"),
    _probe("taming-growth-curve", "growth_curve",
           "Count array equals an independent parabola through (0,S) peaking at (N,B); a "
           "densification call lands exactly on its target", False),
    _probe("taming-score", "score",
           "Score of a tiny scene equals an independent re-implementation of "
           "compute_gaussian_score; low-opacity Gaussians score lower"),
    _probe("taming-sampled-densify", "sampled_densify",
           "Only gradient-qualified Gaussians are cloned/split, budgets split proportionally, "
           "sampling follows the score, new Gaussians start with zero Adam moments"),
    _probe("taming-guided-prune", "guided_prune",
           "Pruned set is a subset of the prune mask sampled by 1/score; no pruning after "
           "densification call 27"),
    _probe("taming-high-opacity", "high_opacity",
           "Opacities train before step 15000 and stay frozen afterwards (official "
           "modify_functions)", False),
    _probe("taming-sh-interval", "sh_interval",
           "SH-rest Adam steps every 16 iterations on summed gradients (differs from per-step "
           "Adam), drops the sum when the parameter is replaced; lr overrides 0.025 / 2.5e-4",
           False),
    _probe("taming-schedule", "schedule",
           "Densification every 100 (big; 600..14900) / 500 (budget; 1000..14500) steps, "
           "opacity reset at 3000..12000, nothing at/after 15000", False),
)
CONTRIBUTIONS = (
    ContributionSpec("growth-curve-budget", "Predictable parabolic growth to a user budget "
                     "(Sec. 3.2, Eq. 2; code get_count_array)", ("taming-growth-curve",)),
    ContributionSpec("score-based-densification", "Steerable densification: saliency (Eq. 3), "
                     "per-Gaussian score (Eq. 4, Alg. 1, custom CUDA statistics), multinomial "
                     "resampling of clone/split candidates and score-guided pruning (Sec. 3.3)",
                     ("taming-stats-kernel", "taming-score", "taming-sampled-densify",
                      "taming-guided-prune", "taming-schedule")),
    ContributionSpec("high-opacity", "High-opacity switch at 15k (Sec. 3.4; official code "
                     "freezes the opacities)", ("taming-high-opacity",)),
    ContributionSpec("batched-sh-updates", "SH bands >= 1 updated every 16 iterations with a "
                     "separate Adam; SH/opacity lr changes (Sec. 4.2, Sec. 5)",
                     ("taming-sh-interval",)),
)


def _make(name, density_config, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "taming-renderer", "renderer", "ss_taming_3dgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    density = ComponentSpec(
        "taming-density", "density", "ss_taming_3dgs.density:create_density",
        ("gradient.screen.signed", TARGET_CAPABILITY), (), {**DEFAULTS, **density_config},
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
    """'Ours (#3DGS)': final count = 3DGS count, densification interval 100."""
    return _make("taming-3dgs", {},
                 "Taming 3DGS (arXiv 2406.15643) native port, 'Ours (#3DGS)' big mode; "
                 "paper fidelity unverified.")


def budget():
    """'Ours': final count = multiplier x SfM count, densification interval 500."""
    return _make("taming-3dgs-budget",
                 {"mode": "multiplier", "densification_interval": 500,
                  "budget_table": core.table_by_count(core.BUDGET_MULTIPLIERS)},
                 "Taming 3DGS (arXiv 2406.15643) native port, budget mode ('Ours'); "
                 "paper fidelity unverified.")
