"""2D Gaussian Splatting for Geometrically Accurate Radiance Fields (Huang et al., SIGGRAPH 2024,
arXiv 2403.17888) — native SplatStudio port of the official code
(github.com/hbb1/2d-gaussian-splatting @f3e3b9f) with its surfel rasteriser
(github.com/hbb1/diff-surfel-rasterization @e0ed020) vendored unmodified in ``csrc/``.

``2dgs``            released-code settings (scripts/m360_eval.py: lambda_dist 0, lambda_normal 0.05,
                    depth_ratio 0, black background)
``2dgs-paperdist``  same with the paper's unbounded-scene distortion weight lambda_dist = 100
Mesh extraction (TSDF) and DTU / TnT geometry evaluation are not part of this port.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DIST_DEFAULTS, NORMAL_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2403.17888",
    "https://github.com/hbb1/2d-gaussian-splatting",
    "https://github.com/hbb1/diff-surfel-rasterization",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_twodgs.probes:{fn}", desc, cuda)


def _components(dist=None):
    return (
        ComponentSpec("2dgs-surfel", "renderer", "ss_twodgs.renderer:create_renderer",
                      ("state.surfel2d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("2dgs-density", "density", "ss_twodgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("photometric", "loss", "splatstudio.losses:create_photometric",
                      ("render.rgb",), (), {"ssim_weight": 0.2}),
        ComponentSpec("2dgs-distortion", "loss", "ss_twodgs.losses:create_distortion",
                      ("render.distortion.ndc_l2",), (), {**DIST_DEFAULTS, **(dist or {})}),
        ComponentSpec("2dgs-normal", "loss", "ss_twodgs.losses:create_normal",
                      ("render.normal.world.unnormalized",
                       "render.normal_from_depth.world.alpha"), (), dict(NORMAL_DEFAULTS)),
    )


PROBES = (
    _probe("ray-splat", "ray_splat",
           "Official kernel vs an independent torch ray-plane evaluation: alpha = min(0.99, "
           "o exp(-rho/2)), rho = min(|uv|^2, 2|x-c|^2); expected depth = per-pixel intersection "
           "depth (not centre z); normal map = camera-facing splat normal; removing the low-pass "
           "or using centre depth is detected", True),
    _probe("distortion-kernel", "distortion_kernel",
           "Two fronto-parallel splats: distortion = w1 w2 (m1 - m2)^2 with NDC m (n 0.2, f 100); "
           "0 for one splat; gradient separates/pulls the splats", True),
    _probe("projected-center-grad", "projected_center_grad",
           "means2D.grad = dL/dX * Z * W/(2 fx) (projected 3D-centre gradient, backward.cu:634), "
           "zero z component, zero for culled splats", True),
    _probe("normal-loss", "normal_loss",
           "depth_to_normal recovers an analytic plane normal facing the camera; "
           "L_n = 0.05 mean(1 - <sum w n, N alpha>) only for step > 7000"),
    _probe("distortion-loss", "distortion_loss",
           "L_d = lambda mean(dist) only for step > 3000; lambda 0 (code) / 100 (paperdist)"),
    _probe("density-recipe", "density_recipe",
           "Exact clone / in-plane split (zero normal std, s / 1.6, third scale untouched) / "
           "opacity cull 0.05 / world prune only after 3000 / reset 0.01 / uniform quat init / "
           "trainable leaves and aligned Adam state"),
)

CONTRIBUTIONS = (
    ContributionSpec("surfel-ray-splat-rasteriser", "2D surfels rendered by perspective-correct "
                     "ray-splat intersection with object-space low-pass filter (Sec. 4, official "
                     "CUDA)", ("ray-splat",)),
    ContributionSpec("depth-distortion", "Depth distortion loss on intersection depths (Sec. 5, "
                     "App. A)", ("distortion-kernel", "distortion-loss")),
    ContributionSpec("normal-consistency", "Splat normal vs depth-gradient normal (Sec. 5)",
                     ("normal-loss", "ray-splat")),
    ContributionSpec("projected-center-densification", "3DGS adaptive control on the projected "
                     "3D-centre gradient, opacity cull 0.05 (Sec. 6)",
                     ("projected-center-grad", "density-recipe")),
)


def _build(name, description, dist=None):
    base = make_method(name)
    keep = {"state-transitions", "renderer-gradient"}
    base_probes = tuple(p for p in base.probes if p.name in keep)
    base_contribs = tuple(c for c in base.contributions
                          if c.name in {"state-lifecycle", "rasterizer-statistics"})
    return replace(
        base, description=description, components=_components(dist),
        probes=base_probes + PROBES, contributions=base_contribs + CONTRIBUTIONS,
        citations=CITATIONS, representation="surfel2d",
    )


def method():
    return _build("2dgs", "2DGS native port (official surfel CUDA rasteriser, distortion + normal "
                  "consistency, official densification; released-code Mip-NeRF 360 settings, "
                  "lambda_dist 0); paper fidelity unverified.")


def method_paperdist():
    return _build("2dgs-paperdist", "2DGS native port with the paper's unbounded-scene distortion "
                  "weight lambda_dist = 100 (Sec. 5) instead of the released code's 0; paper "
                  "fidelity unverified.", dist={"lambda_dist": 100.0})
