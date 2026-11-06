"""GaussianPro: 3D Gaussian Splatting with Progressive Propagation (Cheng et al., ICML 2024,
arXiv 2402.14650) — native SplatStudio port of the official code
(github.com/kcheng1021/GaussianPro @ 86bb048). The official propagation CUDA
(``submodules/Propagation``) and Inria simple-knn are vendored unmodified in ``csrc/``.

``gaussianpro``: Mip-NeRF 360 settings of the paper (§5.1: 3DGS schedule/hyper-parameters,
propagation every 50 steps, σ 0.8 (annealed from 1.0 as in demo.sh), β = 0.001, γ = 100),
dataset mode '360' (neighbours = 4 views with most shared SfM points, depth range [0.1, 20]).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import FLATTEN_DEFAULTS, NORMAL_DEFAULTS, PHOTO_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2402.14650",
    "https://proceedings.mlr.press/v235/cheng24f.html",
    "https://github.com/kcheng1021/GaussianPro",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_gaussianpro.probes:{fn}", desc, cuda)


PROBES = (
    _probe("propagation-plane", "propagation_plane",
           "Vendored PatchMatch kernel recovers a textured plane's depth (<1 % median) and "
           "normal (<5 deg) from a rendered depth corrupted by +-20 % and an unrendered band", True),
    _probe("propagation-selection", "propagation_selection",
           "Density propagation step on a plane tiled by flat Gaussians with a hole adds Gaussians "
           "only inside the hole, on the plane; none without a hole; statistics zeroed; opacity 1",
           True),
    _probe("geometry-render", "geometry_render",
           "Geometry pass: camera normal = camera-facing shortest axis, depth = sum z alpha T "
           "(independent compositing of two Gaussians)", True),
    _probe("planar-losses", "planar_losses",
           "Eq. 8 normal loss (channel-sum L1 + 1-cos, means over valid pixels) and the "
           "flatten loss (100 * mean min scale) vs independent formulas"),
    _probe("opacity-mask", "opacity_mask",
           "Photometric loss ignores pixels with opacity <= 0.999 before step 1000 only"),
    _probe("consistency-check", "consistency_check",
           "Multi-view geometric filter accepts true plane depths and rejects a 5 % offset"),
)

CONTRIBUTIONS = (
    ContributionSpec("hybrid-geometry", "Depth and shortest-axis normal maps rendered by alpha "
                     "blending (§4.1)", ("geometry-render",)),
    ContributionSpec("progressive-propagation", "PatchMatch propagation of plane hypotheses, "
                     "geometric filtering, and Gaussian insertion where rendered depth deviates "
                     "(§4.2)", ("propagation-plane", "propagation-selection", "consistency-check")),
    ContributionSpec("planar-constraint", "Normal consistency with the propagated normal plus "
                     "scale flattening (§4.3, Eq. 8-9)", ("planar-losses",)),
    ContributionSpec("training-strategy", "Propagation schedule and the official opacity-masked "
                     "photometric loss (§4.4, Eq. 10)", ("opacity-mask", "propagation-selection")),
)


def components(density=None):
    return (
        ComponentSpec("gaussianpro-render", "renderer", "ss_gaussianpro.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("gaussianpro-density", "density", "ss_gaussianpro.density:create_density",
                      ("gradient.screen.signed",), (), {**DENSITY_DEFAULTS, **(density or {})}),
        ComponentSpec("gaussianpro-photometric", "loss", "ss_gaussianpro.losses:create_photometric",
                      ("render.rgb", "render.alpha"), (), dict(PHOTO_DEFAULTS)),
        ComponentSpec("gaussianpro-normal", "loss", "ss_gaussianpro.losses:create_normal",
                      ("render.normal.camera",), (), dict(NORMAL_DEFAULTS)),
        ComponentSpec("gaussianpro-flatten", "loss", "ss_gaussianpro.losses:create_flatten",
                      ("state.gaussian3d",), (), dict(FLATTEN_DEFAULTS)),
    )


def method():
    base = make_method("gaussianpro")
    keep = tuple(p for p in base.probes if p.name == "state-transitions")
    return replace(
        base,
        description="GaussianPro (ICML 2024) native port of the official code with the "
                    "official propagation CUDA; paper fidelity unverified.",
        components=components(),
        probes=keep + PROBES,
        contributions=tuple(c for c in base.contributions if c.name == "state-lifecycle")
        + CONTRIBUTIONS,
        citations=CITATIONS,
    )
