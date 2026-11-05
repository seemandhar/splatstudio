"""High-quality Surface Reconstruction using Gaussian Surfels (Dai, Xu, Xie, Liu, Wang, Xu;
SIGGRAPH 2024, arXiv 2404.17774) — native SplatStudio port of the official code
(github.com/turandai/gaussian_surfels, commit 01d1249) with its CUDA rasterizer vendored
unmodified.

``gaussian-surfels``        full method (L_p + L_m + L_c + L_n + curvature + L_o).
``gaussian-surfels-nogeo``  same pipeline with only L_p + L_m (no geometry terms): in-house
                            control, not the paper's 3DGS row.
Meshing (volumetric cutting + screened Poisson) and DTU Chamfer: ``gsurfels-dtu-eval``.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import (
    CONS_DEFAULTS,
    MASK_DEFAULTS,
    OPAC_DEFAULTS,
    PHOTO_DEFAULTS,
    PRIOR_DEFAULTS,
)
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2404.17774",
    "https://github.com/turandai/gaussian_surfels",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_gsurfels.probes:{fn}", desc, cuda)


def _loss(name, fn, requires, defaults):
    return ComponentSpec(name, "loss", f"ss_gsurfels.losses:{fn}", requires, (), dict(defaults))


def _build(name, geometry, description):
    base = make_method("gsplat")
    comps = [
        ComponentSpec("gsurfels-raster", "renderer", "ss_gsurfels.renderer:create_renderer",
                      ("state.surfel2d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("gsurfels-density", "density", "ss_gsurfels.density:create_density",
                      ("gradient.gsurfels",), (), dict(DENSITY_DEFAULTS)),
        _loss("gsurfels-photometric", "create_photometric", ("render.rgb",), PHOTO_DEFAULTS),
        _loss("gsurfels-mask", "create_mask", ("render.alpha",), MASK_DEFAULTS),
    ]
    if geometry:
        comps += [
            _loss("gsurfels-consistency", "create_consistency",
                  ("render.depth.expected_z", "render.normal.camera"), CONS_DEFAULTS),
            _loss("gsurfels-normal-prior", "create_normal_prior", ("render.normal.camera",),
                  PRIOR_DEFAULTS),
            _loss("gsurfels-opacity", "create_opacity", ("render.alpha",), OPAC_DEFAULTS),
        ]
    probes = [
        _probe("surfel-raster", "surfel_raster",
               "Official CUDA: slanted analytic plane -> per-pixel ray/plane depth, plane normal, "
               "third scale ignored, rendered-normal gradient scaled x10", True),
        _probe("density-recipe", "density_recipe",
               "Prune (opacity/big/tiny/invisible over a full view window), clone pre-mask, "
               "in-plane split s/1.6, x0.12 opacity reset at 1001, 15k xyz LR", True),
        _probe("cutting-mesh", "cutting_mesh",
               "Occupancy grid of one surfel vs analytic G*o at voxel centres; points in "
               "empty voxels are cut", True),
        _probe("schedule-recipe", "schedule_recipe",
               "Coarse-to-fine factors 4/2/1 at training renders 2016/5040, random training "
               "background, black evaluation background"),
    ]
    contribs = [
        ContributionSpec("surfel-representation",
                         "Zero-thickness Gaussian surfels with per-pixel tangent-plane depth and "
                         "x10 normal gradient (official CUDA)", ("surfel-raster",)),
        ContributionSpec("optimisation-recipe",
                         "Random visual-hull init, coarse-to-fine, mask loss, invisible-point "
                         "pruning, official LRs/density", ("density-recipe", "schedule-recipe")),
        ContributionSpec("volumetric-cutting-meshing",
                         "Volumetric cutting (512^3 opacity grid) + screened Poisson",
                         ("cutting-mesh",)),
    ]
    if geometry:
        probes += [
            _probe("consistency-formula", "consistency_formula",
                   "depth2normal of an analytic plane = plane normal; L_c 0 when consistent, "
                   "> 0 when tilted; weight 0.01 + 0.1 min(2 it/T, 1)"),
            _probe("normal-prior-formula", "normal_prior_formula",
                   "cos-prior and curvature terms vs independent loops; weight 0.04 - 0.02 it/T"),
            _probe("opacity-formula", "opacity_formula",
                   "Per-Gaussian exp(-(o-0.5)^2/0.05) with the official 0.01/1 bands"),
        ]
        contribs += [
            ContributionSpec("depth-normal-consistency", "L_c (Eq. 10)", ("consistency-formula",)),
            ContributionSpec("normal-prior", "L_n with Omnidata prior + curvature (Eq. 11)",
                             ("normal-prior-formula",)),
            ContributionSpec("opacity-loss", "L_o (Eq. 12)", ("opacity-formula",)),
        ]
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(base, name=name, description=description, components=tuple(comps),
                   probes=base_probes + tuple(probes),
                   contributions=base_contribs + tuple(contribs), citations=CITATIONS,
                   representation="surfel2d")


def method():
    return _build("gaussian-surfels", True,
                  "Gaussian Surfels native port (official CUDA rasterizer); paper fidelity "
                  "unverified.")


def method_nogeo():
    return _build("gaussian-surfels-nogeo", False,
                  "Gaussian Surfels pipeline without L_c/L_n/curvature/L_o (in-house control); "
                  "paper fidelity unverified.")
