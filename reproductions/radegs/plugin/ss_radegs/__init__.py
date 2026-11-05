"""RaDe-GS: Rasterizing Depth in Gaussian Splatting (Zhang et al., arXiv 2406.01467) — native
SplatStudio port of the official code at the paper-v2 formulation (github.com/HKUST-SAIL/RaDe-GS
commit 2d4bc08, 2024-06-21; later commits change the depth formulation, drop the distortion loss
and add PGSR-style multi-view regularisation) with its CUDA rasterizer vendored unmodified.

Mesh extraction (TSDF / marching tetrahedra) and DTU / TnT geometry evaluation are not ported.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import PHOTO_DEFAULTS, REG_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2406.01467",
    "https://github.com/HKUST-SAIL/RaDe-GS/tree/2d4bc087f1b4bd62c96054fbe89d273490526b81",
    "https://arxiv.org/abs/2404.10772",
    "https://arxiv.org/abs/2311.16493",
)
CODE_COMMIT = "2d4bc087f1b4bd62c96054fbe89d273490526b81"


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_radegs.probes:{fn}", desc, cuda)


def method():
    base = make_method("radegs")
    comps = (
        ComponentSpec("radegs-raster", "renderer", "ss_radegs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("radegs-density", "density", "ss_radegs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("photometric", "loss", "ss_radegs.losses:create_photometric",
                      ("render.rgb",), (), dict(PHOTO_DEFAULTS)),
        ComponentSpec("radegs-reg", "loss", "ss_radegs.losses:create_regularizer",
                      ("render.alpha", "render.distortion.radegs", "render.normal.camera",
                       "render.normal_from_depth.camera"), (), dict(REG_DEFAULTS)),
    )
    probes = (
        _probe("rasterized-depth", "rasterized_depth",
               "Kernel per-pixel depth is affine in the pixel offset with the slope of the exact "
               "float64 ray/Gaussian maximum (Eq. 3, 6, 14); constant centre depth fails", True),
        _probe("rasterized-normal", "rasterized_normal",
               "Kernel normal = normal of the intersection plane from the exact depth's "
               "derivatives; flat Gaussian -> short axis facing the camera (Eq. 18-22)", True),
        _probe("kernel-gradients", "kernel_gradients",
               "Backward of rgb/depth/median/alpha/normal/distortion vs float64 central finite "
               "differences (means, scales, quats, opacity, SH)", True),
        _probe("distortion-formula", "distortion_formula",
               "Distortion = sum_{j<i} w_i w_j (m_i - m_j)^2 on NDC depth (Eq. 23); weights "
               "detached (zero opacity gradient), depth gradients present", True),
        _probe("regularizer-schedule", "regularizer_schedule",
               "100 L_d + 0.05 (0.4 L_exp + 0.6 L_med) from step 15000 vs independent formula "
               "(edge-weighted, alpha^2-normalised); analytic plane normal from depth"),
        _probe("densify-metric", "densify_metric",
               "Kernel abs-gradient channel >= per-pixel L1 of the signed gradient; cancelling "
               "gradients densify only with the GOF metric", True),
        _probe("density-recipe", "density_recipe",
               "Clone (resampled) / split (s/1.6) / prune (<0.05, >0.1 extent) / filter-aware "
               "reset (0.01) / 3D filter vs independent formula / schedules / xyz LR"),
    )
    contribs = (
        ContributionSpec("rasterized-depth", "Spatially varying rasterized depth of 3D Gaussian "
                         "splats from the ray-space intersection plane (Sec. 3.2)",
                         ("rasterized-depth", "kernel-gradients")),
        ContributionSpec("rasterized-normal", "Rasterized normal of 3D Gaussian splats "
                         "(Sec. 3.3)", ("rasterized-normal", "kernel-gradients")),
        ContributionSpec("geometry-regularization", "Depth distortion + depth-normal "
                         "consistency on the rasterized maps from 15k (Sec. 3.4, 4.1)",
                         ("distortion-formula", "regularizer-schedule")),
        ContributionSpec("gof-mip-recipe", "GOF densification and Mip-Splatting 3D filter as "
                         "in the official code (Sec. 4.1)", ("densify-metric", "density-recipe")),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base,
        description="RaDe-GS native port (official rasterized depth/normal CUDA @2d4bc08, "
                    "distortion + depth-normal losses, GOF densification, Mip 3D filter); "
                    "paper fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        # filter_3d: constant recomputed from cameras (lr > 0 required by the runtime; it never
        # receives a gradient because it is used detached).
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=CITATIONS,
    )
