"""FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting (ECCV 2024) — native port.

Defaults are the official code values (github.com/VITA-Group/FSGS, arguments/__init__.py,
train.py, README LLFF command `--n_views 3 --sample_pseudo_interval 1`), 10 000 iterations.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import DEPTH_DEFAULTS, PSEUDO_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS


def method():
    base = make_method("fsgs")
    renderer = ComponentSpec(
        "fsgs-renderer",
        "renderer",
        "ss_fsgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        tuple(sorted(CAPABILITIES)),
        dict(RENDERER_DEFAULTS),
    )
    density = ComponentSpec(
        "fsgs-density",
        "density",
        "ss_fsgs.density:create_density",
        ("gradient.screen.signed",),
        (),
        dict(DENSITY_DEFAULTS),
    )
    photometric = base.components[2]  # 0.8 L1 + 0.2 D-SSIM (Eq. 8)
    depth = ComponentSpec(
        "fsgs-depth",
        "loss",
        "ss_fsgs.losses:create_depth",
        ("render.depth.accumulated_z", "render.camera"),
        (),
        dict(DEPTH_DEFAULTS),
    )
    pseudo = ComponentSpec(
        "fsgs-pseudo",
        "loss",
        "ss_fsgs.losses:create_pseudo",
        ("render.depth.accumulated_z", "render.camera"),
        (),
        dict(PSEUDO_DEFAULTS),
    )
    probes = (
        ProbeSpec("proximity-unpooling", "ss_fsgs.probes:proximity_unpooling",
                  "Unpooled Gaussians at 3-NN edge midpoints with destination scale/opacity, "
                  "identity rotation, zero SH; brute-force reference; gated to step < 2000"),
        ProbeSpec("fsgs-densify", "ss_fsgs.probes:fsgs_densify",
                  "FSGS distance split rule, prune 0.005, opacity resets at 2001/5001/8001"),
        ProbeSpec("pearson-depth", "ss_fsgs.probes:pearson_depth",
                  "Train-view Pearson depth loss vs numpy corrcoef; weight 0.05 -> 0.001"),
        ProbeSpec("pseudo-views", "ss_fsgs.probes:pseudo_views",
                  "RegNeRF random LLFF poses from official bounds; 2000<t<9500 ramp; gradient "
                  "through DPT into pseudo-render colours", True),
        ProbeSpec("sh-schedule", "ss_fsgs.probes:sh_schedule",
                  "Training renders use SH degree min(t // 500, 3); evaluation degree 3"),
    )
    contributions = (
        ContributionSpec("proximity-guided-unpooling",
                         "Proximity graph (K=3) Gaussian unpooling + FSGS distance split (§3.2)",
                         ("proximity-unpooling", "fsgs-densify")),
        ContributionSpec("pseudo-view-augmentation",
                         "Online pseudo views with DPT-depth Pearson supervision (§3.3, Eq. 5)",
                         ("pseudo-views",)),
        ContributionSpec("monocular-depth-pearson",
                         "Pearson correlation between rendered and DPT depth on training views "
                         "(Eq. 6-8)", ("pearson-depth",)),
        ContributionSpec("official-training-protocol",
                         "Official SH schedule (every 500 steps)", ("sh-schedule",)),
    )
    return replace(
        base,
        description="FSGS native port (proximity unpooling, pseudo views, DPT Pearson depth); "
        "paper fidelity unverified.",
        components=(renderer, density, photometric, depth, pseudo),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=("https://arxiv.org/abs/2312.00451", "https://github.com/VITA-Group/FSGS",
                   "https://github.com/isl-org/MiDaS"),
    )
