"""Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification
(Fang & Wang, arXiv 2411.12788v1). Native SplatStudio port of the official code
https://github.com/fatPeter/mini-splatting2 (config/fast, 18000 iterations); paper fidelity
unverified.

Methods: `mini-splatting2` (importance metric for outdoor scenes: Mip-NeRF360 outdoor, T&T),
`mini-splatting2-indoor` (Mip-NeRF360 indoor, Deep Blending), and the dense variants
`mini-splatting2-d` / `mini-splatting2-d-indoor` (MSv2-D: importance pruning instead of
simplification). Train with `--steps 18000`.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CULLING_CAPABILITY, TARGET_CAPABILITY

CITATIONS = ("https://arxiv.org/abs/2411.12788v1", "https://github.com/fatPeter/mini-splatting2")
RENDERER_CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    TARGET_CAPABILITY,
    CULLING_CAPABILITY,
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_mini_splatting2.probes:{fn}", description, cuda)


def _make(name, density_config, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "msv2-renderer",
        "renderer",
        "ss_mini_splatting2.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    density = ComponentSpec(
        "msv2-density",
        "density",
        "ss_mini_splatting2.density:create_density",
        ("gradient.screen.signed", TARGET_CAPABILITY, CULLING_CAPABILITY),
        (),
        {**DEFAULTS, **density_config},
    )
    photometric = ComponentSpec(
        "photometric",
        "loss",
        "ss_mini_splatting2.losses:create_photometric",
        ("render.rgb",),
        (),
        {"ssim_weight": 0.2},
    )
    probes = (
        _probe("msv2-stats-kernel", "stats_kernel",
               "Copied CUDA blend statistics (sum w, #pixels, max-weight area, argmax, T) equal "
               "a brute-force per-pixel loop"),
        _probe("msv2-critical-clone", "critical_clone",
               "culling_with_clone: critical set = ever-argmax & 0.99 CDF of sum w (independent "
               "reference); clone opacity/scale = 3DGS-MCMC binomial formula with N=2; copies "
               "start with zero moments"),
        _probe("msv2-visibility-culling", "visibility_culling",
               "Per-view masks = 0.99 CDF of that view's weights; culled Gaussians are absent "
               "from the training render; factor = count_vis/(count_rad+0.1)"),
        _probe("msv2-sparse-adam", "sparse_adam",
               "Sparse Adam leaves invisible rows and moments untouched and equals torch Adam on "
               "always-visible rows"),
        _probe("msv2-depth-reinit", "depth_reinit",
               "Depth reinit at 2000: num_depth = N * factor points on the max-weight surface, "
               "GT colours from half-res targets, opacity 0.1, fresh Adam, culling cleared"),
        _probe("msv2-half-res", "half_res",
               "Training views render at int(W/2) x int(H/2) with a matching loss target while "
               "step <= 3000; full resolution afterwards and for non-training views"),
        _probe("msv2-schedule", "schedule",
               "Event steps of config/fast (clone 500..2750/250 w/o 2000, densify, reinit 2000, "
               "simplify 3000/8000, culling off 13000), lr(30000 horizon), SH, opacity lr", False),
    )
    contributions = (
        ContributionSpec("blend-statistics", "Per-Gaussian rendering statistics from all "
                         "training views (custom CUDA on gsplat tile lists)",
                         ("msv2-stats-kernel",)),
        ContributionSpec("critical-gaussian-identification", "Max-blending-weight Gaussians "
                         "filtered by the 0.99 CDF (Sec. 4, Eq. 3)", ("msv2-critical-clone",)),
        ContributionSpec("aggressive-gaussian-clone", "Opacity/covariance-preserving clone of "
                         "all critical Gaussians every 250 steps (Sec. 4)",
                         ("msv2-critical-clone",)),
        ContributionSpec("aggressive-densification-pipeline", "Depth reinit at 2k with N "
                         "points, densification until 3k, simplification 3k/8k, 18k steps "
                         "(Sec. 4 'Overall pipeline', config/fast)",
                         ("msv2-depth-reinit", "msv2-schedule")),
        ContributionSpec("visibility-culling", "Precomputed per-view visibility excludes "
                         "Gaussians from rasterization, 500-13k (Sec. 5, Eq. 4)",
                         ("msv2-visibility-culling",)),
        ContributionSpec("accelerated-training", "3DGS-accel sparse Adam, 0.5x resolution and "
                         "SH off during densification (Sec. 6.1)",
                         ("msv2-sparse-adam", "msv2-half-res", "msv2-schedule")),
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    """MSv2 with the outdoor importance metric (Mip-NeRF360 outdoor, T&T)."""
    return _make("mini-splatting2", {"imp_metric": "outdoor"},
                 "Mini-Splatting2 (arXiv 2411.12788v1) native port, outdoor importance metric; "
                 "train 18000 steps; paper fidelity unverified.")


def indoor():
    """MSv2 with the indoor importance metric (Mip-NeRF360 indoor, Deep Blending)."""
    return _make("mini-splatting2-indoor", {"imp_metric": "indoor"},
                 "Mini-Splatting2 (arXiv 2411.12788v1) native port, indoor importance metric; "
                 "train 18000 steps; paper fidelity unverified.")


def dense():
    """MSv2-D (official msv2_d/train.py), outdoor metric for the 2k intersection preserving."""
    return _make("mini-splatting2-d", {"imp_metric": "outdoor", "simplify": "importance"},
                 "Mini-Splatting2-D (arXiv 2411.12788v1) native port, importance pruning; "
                 "train 18000 steps; paper fidelity unverified.")


def dense_indoor():
    """MSv2-D with the indoor metric."""
    return _make("mini-splatting2-d-indoor", {"imp_metric": "indoor", "simplify": "importance"},
                 "Mini-Splatting2-D (arXiv 2411.12788v1) native port, importance pruning, "
                 "indoor metric; train 18000 steps; paper fidelity unverified.")
