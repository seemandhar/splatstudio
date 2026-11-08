"""AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering (ICCV 2025,
arXiv 2504.12811) - native SplatStudio port of github.com/DerThomy/AAA-Gaussians @ dc5b3f2 with
its CUDA rasterizer (DerThomy/AAA-Gaussians-Rasterization @ d03ce2d) vendored unmodified and
used for training and evaluation with configs/aaa.json; density is 3DGS-MCMC with the official
per-scene caps; the training-view 3D filter is computed at the final step as in the official
compute_3D_filter-at-save.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEFAULTS as REG_DEFAULTS
from .raster import AAA
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2504.12811", "https://github.com/DerThomy/AAA-Gaussians",
             "https://github.com/DerThomy/AAA-Gaussians-Rasterization",
             "https://arxiv.org/abs/2402.00525", "https://arxiv.org/abs/2404.09591",
             "https://arxiv.org/abs/2311.16493")


def _probe(name, fn, desc, cuda=True):
    return ProbeSpec(name, f"ss_aaa_gaussians.probes:{fn}", desc, cuda)


def method():
    base = make_method("aaa-gaussians")
    photometric = base.components[2]
    comps = (
        ComponentSpec("aaa-renderer", "renderer", "ss_aaa_gaussians.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {"background": [0.0, 0.0, 0.0], "splatting": AAA}),
        ComponentSpec("aaa-mcmc-density", "density", "ss_aaa_gaussians.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        photometric,
        ComponentSpec("aaa-mcmc-reg", "loss", "ss_aaa_gaussians.losses:create_reg", (), (),
                      dict(REG_DEFAULTS)),
    )
    probes = (
        _probe("aaa-forward-reference", "forward_reference",
               "CUDA image equals an independent float64 dense 3D renderer (ray max "
               "contribution, Eq. 11 scaling, Eq. 12 filter, per-pixel depth order); volume "
               "scaling, no scaling and 2D EWA differ"),
        _probe("aaa-backward-reference", "backward_reference",
               "Kernel gradients (means, scales, quats, opacity, sh0) equal autograd of the "
               "dense reference; the volume-scaled reference does not"),
        _probe("aaa-perp-scaling", "perp_scaling",
               "Face-on thin disk: peak alpha = o * Eq. 11 factor (~1), not Eq. 8 (~0.07)"),
        _probe("aaa-adaptive-filter", "adaptive_filter",
               "Tiny Gaussian: pixel variance 0.3 at z=2 and z=8 (per-view k z^2/f^2), floored "
               "to 1.2 by a training-view filter from 2x farther (v' = min(v_train, v))"),
        _probe("aaa-frustum-bounds", "frustum_bounds",
               "Gaussians with means behind/outside the view that reach into it render "
               "(== reference) while projective bounding drops them; outside ones are culled; "
               "3D tile culling keeps the image and cuts sort entries"),
        _probe("aaa-filter-train", "filter_train",
               "Training-view filter equals an independent loop of mip_filter.py (k=0.3, "
               "unseen -> 0)", cuda=False),
        _probe("aaa-mcmc-schedule", "mcmc_schedule",
               "MCMC init (o=0.5, s x0.1), relocate/add cadence 500<s<25000 every 100, +5% to "
               "cap, noise, filter only at the final step, reg loss"),
    )
    contribs = (
        ContributionSpec("adaptive-3d-filter", "Adaptive 3D smoothing filter with "
                         "perpendicular amplitude scaling, v' = min(v_train, f/z) (§3.2, "
                         "Eq. 9-12)", ("aaa-perp-scaling", "aaa-adaptive-filter",
                                       "aaa-filter-train")),
        ContributionSpec("view-space-bounding", "Perspective-correct view-space angle bounds "
                         "(§3.3, Eq. 13-14)", ("aaa-frustum-bounds",)),
        ContributionSpec("frustum-culling", "Exact 3D frustum and per-tile frustum culling "
                         "(§3.4, Eq. 15)", ("aaa-frustum-bounds",)),
        ContributionSpec("3d-eval-hierarchical-sort", "3D evaluation via screen-space planes "
                         "inside the StopThePop hierarchical sorter, used for training "
                         "(§3.1, §4)", ("aaa-forward-reference", "aaa-backward-reference")),
        ContributionSpec("mcmc-densification", "3DGS-MCMC densification with identical "
                         "settings and the official per-scene caps (§4)",
                         ("aaa-mcmc-schedule",)),
    )
    # The built-in renderer-gradient probe initializes the density without --data (no per-scene
    # cap) and checks 2D screen-space gradients, which the 3D-evaluation kernel does not produce
    # and MCMC never consumes; aaa-backward-reference checks the kernel gradients instead.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
    return replace(
        base,
        description="AAA-Gaussians (ICCV 2025) native port with the official 3D-evaluation "
                    "hierarchical CUDA rasterizer and 3DGS-MCMC; paper fidelity unverified.",
        components=comps, probes=base_probes + probes,
        contributions=base_contribs + contribs,
        fields=base.fields + (FieldSpec("filter_3d", (), learning_rate=1e-8,
                                        initial_value=0.0),),
        citations=CITATIONS,
    )
