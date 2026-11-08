"""LPM: Improving Gaussian Splatting with Localized Points Management (CVPR 2025, arXiv
2406.04251). Native SplatStudio port of https://github.com/Surrey-UP-Lab/GS-LPM (@7c06026)
on vanilla 3DGS; paper fidelity unverified.

Methods:
  lpm         official defaults / README outdoor setting: reset_interval 100, angle 45 deg
  lpm-indoor  README note for indoor scenes: reset_interval 200, angle 90 deg
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, REQUIRES

CITATIONS = (
    "https://arxiv.org/abs/2406.04251",
    "https://github.com/Surrey-UP-Lab/GS-LPM",
    "https://github.com/cvg/LightGlue",
)
RENDERER_CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    "training.target_image",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_lpm.probes:{fn}", description, cuda)


def _make(name, reset_every, angle, description):
    base = make_method(name)
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "lpm-renderer", "renderer", "ss_lpm.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    density = ComponentSpec(
        "lpm-density", "density", "ss_lpm.density:create_density", REQUIRES, (),
        {**DEFAULTS, "reset_every": reset_every, "angle": angle},
    )
    probes = (
        _probe("lpm-error-map", "error_map_probe",
               "Error map (normalised L1, 0.4 quantile, 16px patches >85%) equals an independent "
               "numpy reference and marks exactly a synthetic error block"),
        _probe("lpm-zone-projection", "zone_projection_probe",
               "Ray closest points equal least squares; the zone from two views of a known "
               "rectangle encloses it tightly"),
        _probe("lpm-localized-densify", "localized_densify_probe",
               "Inside a zone tau/2 <= g < tau Gaussians are cloned/split, outside not; the "
               "n_extra lowest-opacity Gaussians are pruned; ratio 1 == plain 3DGS"),
        _probe("lpm-localized-reset", "localized_reset_probe",
               "Only the high-opacity Gaussian in front of the zone (camera cone) is reset to "
               "0.01; behind/outside/below-threshold Gaussians untouched"),
        _probe("lpm-matching", "matching_probe",
               "SuperPoint+LightGlue recover a known image shift; region pairing maps an error "
               "box by that shift", True),
        _probe("lpm-density-cycle", "density_cycle_probe",
               "Full after_backward on a rendered plane with an error block and true matches: a "
               "zone is found, extra densification beyond 3DGS happens, params/optimizers aligned", True),
    )
    contributions = (
        ContributionSpec("error-map", "Rendering error map of the current view (Sec. 3.2)",
                         ("lpm-error-map",)),
        ContributionSpec("cross-view-region-mapping", "Error regions matched to a neighbouring "
                         "view with SuperPoint+LightGlue (Sec. 3.2 (i))",
                         ("lpm-matching", "lpm-density-cycle")),
        ContributionSpec("zone-projection", "2D->3D error-source zone from the paired regions' "
                         "rays (Sec. 3.2 (ii))", ("lpm-zone-projection", "lpm-density-cycle")),
        ContributionSpec("localized-point-addition", "Lower (x0.5) clone/split threshold inside "
                         "zones + adaptive low-opacity pruning (Sec. 3.2 Points manipulation)",
                         ("lpm-localized-densify", "lpm-density-cycle")),
        ContributionSpec("localized-reset", "Opacity reset of high-opacity Gaussians in front of "
                         "zones (Sec. 3.2 Points manipulation)",
                         ("lpm-localized-reset",)),
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
    return _make("lpm", 100, 45.0,
                 "LPM (CVPR 2025) native port on 3DGS, outdoor/default setting "
                 "(reset_interval 100, angle 45); paper fidelity unverified.")


def indoor():
    return _make("lpm-indoor", 200, 90.0,
                 "LPM (CVPR 2025) native port on 3DGS, README indoor setting "
                 "(reset_interval 200, angle 90); paper fidelity unverified.")
