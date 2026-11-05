"""Gaussian Opacity Fields (Yu, Sattler, Geiger; SIGGRAPH Asia 2024, arXiv 2404.10772) —
native SplatStudio port of the official code (github.com/autonomousvision/
gaussian-opacity-fields, commit 5245b20) with its ray-tracing CUDA rasterizer vendored.

Opacity field queries: ``ss_gof.raster.opacity_field(params, points, cameras)``.
Tetrahedral mesh extraction (marching tetrahedra + binary search) is not ported.
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
    "https://arxiv.org/abs/2404.10772",
    "https://github.com/autonomousvision/gaussian-opacity-fields",
    "https://arxiv.org/abs/2311.16493",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_gof.probes:{fn}", desc, cuda)


def method():
    base = make_method("gof")
    comps = (
        ComponentSpec("gof-raytrace", "renderer", "ss_gof.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("gof-density", "density", "ss_gof.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("photometric", "loss", "ss_gof.losses:create_photometric",
                      ("render.rgb",), (), dict(PHOTO_DEFAULTS)),
        ComponentSpec("gof-reg", "loss", "ss_gof.losses:create_regularizer",
                      ("render.distortion.gof", "render.normal.world",
                       "render.normal_from_depth.world"), (), dict(REG_DEFAULTS)),
    )
    probes = (
        _probe("ray-intersection", "ray_intersection",
               "Official GOF kernel RGB/normal/median depth/alpha/distortion and gradients "
               "(means, scales, quats, opacity, SH) equal an independent float64 ray-Gaussian "
               "volume renderer (Eq. 2-7); projected splatting (gsplat) differs", True),
        _probe("opacity-field", "opacity_field",
               "integrate kernel: O(t) = o G(min(t, t*)) along a ray, constant after t*, "
               "minimum over views", True),
        _probe("distortion-formula", "distortion_formula",
               "Distortion map = NDC 2DGS form; weights detached (zero opacity gradient) "
               "while depth gradients match the reference", True),
        _probe("normal-consistency", "normal_consistency",
               "depth_to_normal recovers an analytic plane normal; L_d x100 and L_n x0.05 "
               "switch on at step 15000"),
        _probe("densify-metric", "densify_metric",
               "Kernel abs channel >= per-pixel L1 of the signed gradient; a Gaussian with "
               "cancelling gradients is densified only with the GOF metric", True),
        _probe("density-recipe", "density_recipe",
               "Clone (resampled) / split (s/1.6) / prune (<0.05, >0.1 extent) / filter-aware "
               "reset (0.01) / 3D filter vs independent formula / schedules / xyz LR"),
    )
    contribs = (
        ContributionSpec("ray-gaussian-rendering", "Ray-Gaussian intersection volume rendering "
                         "with the official CUDA rasterizer (Sec. 3.1)", ("ray-intersection",)),
        ContributionSpec("opacity-field", "Gaussian opacity field of arbitrary 3D points "
                         "(Sec. 3.2)", ("opacity-field",)),
        ContributionSpec("distortion-detached", "Depth distortion on intersection depths with "
                         "detached weights (Sec. 3.3)", ("distortion-formula",)),
        ContributionSpec("plane-normal-consistency", "Intersection-plane normals with "
                         "depth-normal consistency (Sec. 3.3)",
                         ("ray-intersection", "normal-consistency")),
        ContributionSpec("abs-gradient-densification", "Densification by accumulated "
                         "per-pixel gradient norms (Sec. 3.3, Eq. M)", ("densify-metric",)),
        ContributionSpec("mip-3dgs-recipe", "3DGS density control with Mip-Splatting 3D filter "
                         "as in the official code", ("density-recipe",)),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base,
        description="Gaussian Opacity Fields native port (official ray-tracing CUDA, GOF "
                    "densification, distortion + depth-normal losses, Mip 3D filter); "
                    "paper fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        # filter_3d: constant recomputed from cameras (lr > 0 required by the runtime; it never
        # receives a gradient because it is used detached).
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=CITATIONS,
    )
