"""Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians (ECCV 2024,
arXiv 2403.14166). Native SplatStudio port of the official code
https://github.com/fatPeter/mini-splatting; paper fidelity unverified.

Methods: `mini-splatting` (importance metric for outdoor scenes: Mip-NeRF360 outdoor, T&T),
`mini-splatting-indoor` (Mip-NeRF360 indoor, Deep Blending), `mini-splatting-d`
(Mini-Splatting-D, densification only).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2403.14166", "https://github.com/fatPeter/mini-splatting")
RENDERER_CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    "training.target_image",
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_mini_splatting.probes:{fn}", description, cuda)


def _make(name, density_config, description):
    base = make_method(name)
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "ms-renderer",
        "renderer",
        "ss_mini_splatting.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    density = ComponentSpec(
        "ms-density",
        "density",
        "ss_mini_splatting.density:create_density",
        ("gradient.screen.signed", "training.target_image"),
        (),
        {**DEFAULTS, **density_config},
    )
    simplify = bool(density.config["simplify"])
    probes = [
        _probe("ms-stats-kernel", "stats_kernel",
               "CUDA blend statistics (sum w, #pixels, S_i, argmax, T) equal a brute-force "
               "per-pixel loop; T matches gsplat's alpha"),
        _probe("ms-depth-midpoint", "depth_midpoint",
               "Depth kernel point = argmax-weight Gaussian's density maximum along the ray "
               "(independent Sigma^-1 formula), App. 'Calculation of Gaussian Depth'"),
        _probe("ms-blur-split", "blur_split",
               "Zero-gradient Gaussian covering the view is split by Eq. 1; not with "
               "theta_blur disabled"),
        _probe("ms-depth-reinit", "depth_reinit",
               "Reinit replaces the population with num_depth points on the max-weight "
               "surface, GT colours, opacity 0.1, fresh Adam"),
        _probe("ms-schedule", "schedule",
               "Means LR restart lr(it-10000) after the restart step; SH degree 0 until it, "
               "+1 per 1000 steps; inactive shN coefficients and moments stay zero", False),
    ]
    contributions = [
        ContributionSpec("blend-statistics", "Per-Gaussian rendering statistics from all views "
                         "(custom CUDA on gsplat tile lists)", ("ms-stats-kernel",)),
        ContributionSpec("blur-split", "Blur split of Gaussians with large max-contribution "
                         "area (Sec. 4.1, Eq. 1)", ("ms-blur-split",)),
        ContributionSpec("depth-reinitialization", "Mid-point depth of the max-weight Gaussian, "
                         "re-sampled points every 5000 steps (Sec. 4.1, App.)",
                         ("ms-depth-midpoint", "ms-depth-reinit")),
        ContributionSpec("pipeline-schedule", "SH 0 during densification, centre reinit and LR "
                         "restart at 15k (Algorithm 1, official code)", ("ms-schedule",)),
    ]
    if simplify:
        probes.append(_probe(
            "ms-simplification", "simplification",
            "Never-argmax Gaussian gets zero importance and is dropped (Eq. 2); kept count "
            "int(0.5 nnz) by importance sampling; 1% CDF prune at 20k"))
        contributions.append(ContributionSpec(
            "simplification", "Intersection preserving + importance-weighted sampling "
            "(Sec. 4.2, App. importance metric)", ("ms-simplification",)))
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base.probes + tuple(probes),
        contributions=base.contributions + tuple(contributions),
        citations=CITATIONS,
    )


def method():
    """Mini-Splatting with the outdoor importance metric I^2 (Mip-NeRF360 outdoor, T&T)."""
    return _make("mini-splatting", {"imp_metric": "outdoor"},
                 "Mini-Splatting (ECCV 2024) native port, outdoor importance metric; "
                 "paper fidelity unverified.")


def indoor():
    """Mini-Splatting with the indoor importance metric I^1 (Mip-NeRF360 indoor, DB)."""
    return _make("mini-splatting-indoor", {"imp_metric": "indoor"},
                 "Mini-Splatting (ECCV 2024) native port, indoor importance metric; "
                 "paper fidelity unverified.")


def densified():
    """Mini-Splatting-D: densification only (official ms_d/train.py)."""
    return _make("mini-splatting-d",
                 {"simplify": False, "num_max": None, "depth_with_replacement": True},
                 "Mini-Splatting-D (ECCV 2024) native port, densification only; "
                 "paper fidelity unverified.")
