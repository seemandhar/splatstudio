"""3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting (CVPR 2025,
arXiv 2412.12507) — native SplatStudio port of the unsorted "Ours" variant.

Renderer: gsplat 1.5.3's 3DGUT kernels (UT projection + 3D max-response evaluation, contributed
by the paper's authors) driven with the paper's UT parameters (alpha=1, beta=2, kappa=0), which
`gsplat.rasterization` does not expose. Density: the official 3dgrut `GSStrategy`
(github.com/nv-tlabs/3dgrut @ bb203b1, threedgrut/strategy/gs.py) with 3D positional gradients.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2412.12507",
    "https://research.nvidia.com/labs/toronto-ai/3DGUT/",
    "https://github.com/nv-tlabs/3dgrut",
    "https://github.com/nerfstudio-project/gsplat",
)


def _probe(name, fn, desc, cuda=True):
    return ProbeSpec(name, f"ss_3dgut.probes:{fn}", desc, cuda)


def method():
    base = make_method("3dgut")
    photometric = base.components[2]  # 0.8 L1 + 0.2 D-SSIM: official base_gs.yaml loss
    components = (
        ComponentSpec("3dgut-renderer", "renderer", "ss_3dgut.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDER_DEFAULTS)),
        ComponentSpec("3dgut-density", "density", "ss_3dgut.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        photometric,
    )
    keep = {"state-transitions", "photometric-gradient"}  # screen-gradient probe does not apply
    probes = tuple(p for p in base.probes if p.name in keep) + (
        _probe("3dgut-ut-moments", "ut_moments",
               "Renderer UT 2D mean/cov == independent Eq. 5-9 (alpha=1, beta=2, kappa=0) for "
               "pinhole, OpenCV radial+tangential and fisheye cameras; gsplat's default alpha=0.1 "
               "differs"),
        _probe("3dgut-ut-vs-ewa-mc", "ut_vs_ewa_mc",
               "Strong fisheye: UT projection closer (KL) to Monte-Carlo than EWA J Sigma J^T"),
        _probe("3dgut-eval3d-response", "eval3d_response",
               "Rendered alpha == ray-space max response sigma*exp(-0.5|d_g x o_g|^2) (Eq. 10), "
               "not the 2D EWA conic"),
        _probe("3dgut-eval3d-gradient", "eval3d_gradient",
               "Kernel gradients (means, scales, quats, opacity, sh0) == autograd of an "
               "independent float64 3D-evaluation compositor"),
        _probe("3dgut-fisheye-render", "fisheye_render",
               "Fisheye render of a small Gaussian peaks at the analytic equidistant projection"),
        _probe("3dgut-densify-statistic", "densify_statistic",
               "||dL/dmu * dist|| / 2 statistic, clone/split/prune routing, 300/100/3000 cadence",
               cuda=False),
        _probe("3dgut-official-init", "official_init",
               "Official COLMAP init: scale 0.01 x nearest-camera distance, random unit quats",
               cuda=False),
    )
    contributions = tuple(c for c in base.contributions if set(c.probes) <= keep) + (
        ContributionSpec("unscented-transform-projection",
                         "Sigma-point projection replaces EWA linearisation (§4.1, Eq. 5-9)",
                         ("3dgut-ut-moments", "3dgut-ut-vs-ewa-mc")),
        ContributionSpec("3d-particle-evaluation",
                         "Particle response evaluated in 3D at the ray's max-response point, "
                         "gradients bypass the projection (§4.2, Eq. 10, Supp. B)",
                         ("3dgut-eval3d-response", "3dgut-eval3d-gradient")),
        ContributionSpec("distorted-camera-support",
                         "Arbitrary camera models through the same UT/eval3d path (§4.1); "
                         "renderer-level only, core loader is pinhole-only",
                         ("3dgut-ut-moments", "3dgut-fisheye-render")),
        ContributionSpec("3d-gradient-densification",
                         "3DGRT-style densification on 3D positional gradients x dist/2, every "
                         "300 its, official init (§4.4, code gs.py)",
                         ("3dgut-densify-statistic", "3dgut-official-init")),
    )
    return replace(
        base,
        description="3DGUT (CVPR 2025) native port, unsorted variant, on gsplat's UT/eval3d "
                    "kernels with the paper's UT parameters; paper fidelity unverified.",
        components=components, probes=probes, contributions=contributions, citations=CITATIONS,
    )
