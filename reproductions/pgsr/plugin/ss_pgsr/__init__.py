"""PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction
(Chen et al., TVCG 2024, arXiv 2406.06521) — native SplatStudio port of the official code
(github.com/zju3dv/PGSR @de24f1a) with its plane rasterizer vendored unmodified (csrc/).

``pgsr``      official run_mip360.py settings (exposure off, abs-grad threshold 0.0002)
``pgsr-tnt``  official run_tnt.py settings (exposure on, abs 0.00015, opacity cull 0.05;
              ncc_scale 0.5 of the official script is not supported -> 1.0)
Mesh extraction (TSDF) and DTU / TnT geometry evaluation are not part of this port.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import FLATTEN_DEFAULTS, GEO_DEFAULTS, NCC_DEFAULTS, RGB_DEFAULTS, SINGLE_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2406.06521",
    "https://github.com/zju3dv/PGSR",
    "https://arxiv.org/abs/2404.10484",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_pgsr.probes:{fn}", desc, cuda)


def _components(rgb=None, density=None):
    return (
        ComponentSpec("pgsr-plane", "renderer", "ss_pgsr.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("pgsr-density", "density", "ss_pgsr.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, **(density or {})}),
        ComponentSpec("pgsr-rgb", "loss", "ss_pgsr.losses:create_rgb",
                      ("render.rgb", "render.camera"), (), {**RGB_DEFAULTS, **(rgb or {})}),
        ComponentSpec("pgsr-flatten", "loss", "ss_pgsr.losses:create_flatten",
                      ("gradient.screen.signed",), (), dict(FLATTEN_DEFAULTS)),
        ComponentSpec("pgsr-single-view", "loss", "ss_pgsr.losses:create_single_view",
                      ("render.normal.camera", "render.normal_from_depth.camera"), (),
                      dict(SINGLE_DEFAULTS)),
        ComponentSpec("pgsr-mv-geo", "loss", "ss_pgsr.losses:create_mv_geo",
                      ("render.depth.plane", "render.camera"), (), dict(GEO_DEFAULTS)),
        ComponentSpec("pgsr-mv-ncc", "loss", "ss_pgsr.losses:create_mv_ncc",
                      ("render.normal.camera", "render.distance.plane", "render.camera"), (),
                      dict(NCC_DEFAULTS)),
    )


PROBES = (
    _probe("plane-depth", "plane_depth",
           "Official kernel: depth of an oblique flattened Gaussian equals the analytic "
           "ray-plane intersection (not the alpha-blended centre z); rendered normal and "
           "distance equal the plane's; out_observe counts T>0.5 pixels; finite gradients", True),
    _probe("flatten-loss", "flatten_loss",
           "L_s = 100 * mean(min scale) over radii>0 only; gradient only on the min axis"),
    _probe("single-view", "single_view",
           "normal_from_depth recovers an analytic plane normal; L_svgeom equals an independent "
           "formula with the (1 - normalised image gradient)^2 edge weight; 0.015 from > 7000"),
    _probe("mv-geometry", "mv_geometry",
           "Forward-backward reprojection error ~0 for consistent analytic plane depths, equals "
           "the homography error H_nr H_rn p for planes, perturbed depths rejected (phi >= 1) "
           "or down-weighted exp(-phi)"),
    _probe("mv-ncc", "mv_ncc",
           "Plane-induced homography warps a textured plane between views (NCC ~ 0 at the true "
           "plane, larger for a wrong plane distance); gradient reaches normal and distance"),
    _probe("exposure", "exposure",
           "L1 uses exp(a) I + b only when step > 1000 and 1-SSIM < 0.5; SSIM on raw render; "
           "gradient reaches (a, b); coefficients fit a brightness change"),
    _probe("neighbour-graph", "neighbour_graph",
           "Neighbours sorted by distance (ties by angle), angle < 30 deg, 0.01 < d < 1.5, <= 8"),
    _probe("density-recipe", "density_recipe",
           "Resampled clone / split s/1.6 / AbsGS split radius > 20 & cap / opacity cull / "
           "world prune / screen-size prune not applied / reset 0.01 / xyz LR"),
    _probe("multi-view-trim", "multi_view_trim",
           "Trim prunes Gaussians observed (T > 0.5) in fewer than 2 training views", True),
)

CONTRIBUTIONS = (
    ContributionSpec("planar-flattening", "Min-scale flattening loss L_s (Sec. 3.1)",
                     ("flatten-loss",)),
    ContributionSpec("unbiased-plane-depth", "Plane normal / distance rendering and depth = "
                     "distance / (N K^-1 p) with the official CUDA kernel (Sec. 3.1)",
                     ("plane-depth",)),
    ContributionSpec("edge-aware-single-view", "Local-plane depth normal vs rendered normal, "
                     "image-edge weighted (Sec. 3.2.1)", ("single-view",)),
    ContributionSpec("multi-view-geometric", "Forward-backward plane-depth reprojection with "
                     "occlusion weight, one random neighbour per step (Sec. 3.2.2)",
                     ("mv-geometry", "neighbour-graph")),
    ContributionSpec("multi-view-photometric", "Homography-warped 7x7 grey patch NCC "
                     "(Sec. 3.2.2)", ("mv-ncc", "neighbour-graph")),
    ContributionSpec("exposure-compensation", "Per-image affine exposure in the L1 term "
                     "(Sec. 3.3; on in pgsr-tnt only, as in the official scripts)",
                     ("exposure",)),
    ContributionSpec("absgs-densification-trim", "AbsGS + official densification and "
                     "multi-view observation trim (Sec. 4 impl. details, code)",
                     ("density-recipe", "multi-view-trim")),
)


def _build(name, description, rgb=None, density=None):
    base = make_method(name)
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, description=description, components=_components(rgb, density),
        probes=base_probes + PROBES, contributions=base_contribs + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _build("pgsr", "PGSR native port (official plane CUDA rasterizer, flattening, "
                  "single/multi-view regularisation, official densification + trim; Mip-NeRF "
                  "360 settings, exposure off); paper fidelity unverified.")


def method_tnt():
    return _build("pgsr-tnt", "PGSR native port with the official Tanks&Temples settings "
                  "(exposure compensation on, abs-grad 0.00015, opacity cull 0.05; ncc_scale 1 "
                  "instead of 0.5); paper fidelity unverified.",
                  rgb={"exposure_compensation": True},
                  density={"densify_abs_grad_threshold": 0.00015,
                           "opacity_cull_threshold": 0.05})
