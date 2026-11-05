"""Mip-Splatting (Yu et al., CVPR 2024, arXiv 2311.16493) as a native SplatStudio method."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES

# Values from the official code (commit 1ecf7d5) and scripts/run_mipnerf360.py.
RENDERER_CONFIG = {"kernel_size_2d": 0.1}
DENSITY_CONFIG = {
    # gsplat DefaultStrategy, identical to the shared baseline (3DGS defaults).
    "absgrad": False,
    "grow_grad2d": 0.0002,
    # Mip-Splatting 3D filter.
    "filter_coef": 0.2,
    "near": 0.2,
    "frustum_margin": 0.15,
    "filter_interval": 100,
    "densify_until": 15000,
    "total_steps": 30000,
    # Filter-aware opacity reset (official reset_opacity).
    "opacity_reset_every": 3000,
    "opacity_reset_value": 0.01,
}


def _build(name, reset_every, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "mip-gsplat",
        "renderer",
        "ss_mip_splatting.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        tuple(sorted(CAPABILITIES)),
        dict(RENDERER_CONFIG),
    )
    density = ComponentSpec(
        "mip-density",
        "density",
        "ss_mip_splatting.density:create_density",
        ("gradient.screen.signed",),
        (),
        {**DENSITY_CONFIG, "opacity_reset_every": reset_every},
    )
    photometric = base.components[2]
    probes = (
        ProbeSpec(
            "filter3d-frequency-bound",
            "ss_mip_splatting.probes:frequency_bound",
            "3D filter std equals sqrt(0.2) * min visible depth / max focal (Eq. 8, official "
            "compute_3D_filter), checked against an independent loop and an analytic case",
        ),
        ProbeSpec(
            "filter3d-convolution",
            "ss_mip_splatting.probes:convolution",
            "Rendered image equals gsplat with explicit covariance Sigma + f^2 I and opacity "
            "scaled by sqrt(det Sigma / det(Sigma + f^2 I)) (Eq. 9)",
            True,
        ),
        ProbeSpec(
            "mip-filter-2d",
            "ss_mip_splatting.probes:mip_filter_2d",
            "Peak alpha of an on-axis Gaussian equals opacity * sqrt(det S2 / det(S2 + 0.1 I)) "
            "with analytic S2 (Eq. 10, kernel_size 0.1)",
            True,
        ),
        ProbeSpec(
            "filter-aware-reset",
            "ss_mip_splatting.probes:filter_aware_reset",
            "Opacity reset clamps the filtered opacity to 0.01 and zeroes Adam moments",
        ),
        ProbeSpec(
            "single-reset",
            "ss_mip_splatting.probes:single_reset",
            "Driving the density controller through a reset step applies exactly one "
            "filter-aware reset (a missing or doubled/unfiltered reset fails); none off-schedule",
            True,
        ),
        ProbeSpec(
            "camera-hook",
            "ss_mip_splatting.probes:camera_hook",
            "Filter is computed from set_training_cameras; training without the hook raises",
        ),
        ProbeSpec(
            "filter-schedule",
            "ss_mip_splatting.probes:schedule",
            "3D filter recompute and opacity reset steps match official train.py boundaries",
        ),
    )
    contributions = (
        ContributionSpec(
            "3d-smoothing-filter",
            "Per-Gaussian low-pass filter bounded by the training views' maximal sampling rate, "
            "with opacity compensation, recomputed every 100 steps and filter-aware reset",
            ("filter3d-frequency-bound", "filter3d-convolution", "filter-aware-reset",
             "single-reset", "camera-hook", "filter-schedule"),
        ),
        ContributionSpec(
            "2d-mip-filter",
            "Screen-space box-filter approximation Sigma2D + 0.1 I with opacity compensation "
            "(replaces 3DGS dilation)",
            ("mip-filter-2d",),
        ),
    )
    # The built-in renderer-gradient probe drives density.initialize/before_backward without
    # the runtime's set_training_cameras hook, which MipDensity requires; use an equivalent
    # probe that sets the probe camera as the training camera first.
    base_probes = tuple(
        replace(p, function="ss_mip_splatting.probes:renderer_gradient")
        if p.name == "renderer-gradient" else p
        for p in base.probes
    )
    base_contributions = tuple(base.contributions)
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base_probes + probes,
        contributions=base_contributions + contributions,
        # filter_3d is a constant (official: plain tensor, recomputed from cameras). It is used
        # detached, so it never receives a gradient; the runtime requires lr > 0 for a field.
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=(
            "https://arxiv.org/abs/2311.16493",
            "https://github.com/autonomousvision/mip-splatting",
        ),
    )


def method():
    return _build(
        "mip-splatting",
        DENSITY_CONFIG["opacity_reset_every"],
        "Mip-Splatting native port (3D smoothing + 2D Mip filter, filter-aware opacity reset); "
        "paper fidelity unverified.",
    )
