"""ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction With Fewer
Primitives (Baranowski, Esposito, Gschossmann, Chen, Geiger; 3DV 2026, arXiv 2511.06810).
Native SplatStudio port of https://github.com/baranowskibrt/conegs; paper fidelity unverified.

Two stages: `ss-conegs-init` (per-scene iNGP + 1M iNGP-initialized points in a dataset copy,
training views only), then `ss-train conegs --data <copy> --recipe splatstudio
--num-gaussians 10000000`.

- `conegs`: fixed budget of 1M primitives (paper Tables 1/2, appendix 1M table).
- `conegs-nobudget`: adaptive growth, beta = 0.02 (appendix "no budget" table).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2511.06810", "https://github.com/baranowskibrt/conegs")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_conegs.probes:{fn}", description, cuda)


PROBES = (
    _probe("conegs-hashgrid", "hashgrid",
           "Hash-grid CUDA kernel (tcnn layout: scale 2^(l log2 b) N_min - 1, coherent-prime "
           "hash, dense coarse levels) equals an independent torch reference, fwd and grad", True),
    _probe("conegs-proposal-sampler", "proposal_sampler",
           "nerfacc importance_sampling docstring cases, lindisp near/far, pdf loss zero for the "
           "same cdf and positive for a flat proposal"),
    _probe("conegs-ingp-fit", "ingp_fit",
           "Short iNGP fit on a synthetic textured plane recovers the plane as median depth", True),
    _probe("conegs-median-depth", "median_depth",
           "Median depth = first sample with inclusive transmittance < 0.5 (tau_{k-1}>0.5>=tau_k)"),
    _probe("conegs-cone-placement", "cone_placement",
           "Error-weighted pixels (only the high-error quadrant), centres on pixel rays at the "
           "median depth, iNGP colour, std x f / z = 1 pixel (Eq. 7/11); size_mult 2 gives 2"),
    _probe("conegs-budget", "budget",
           "Prune opacity<=0.005 then refill the budget with the oldest stashed Gaussians; zero "
           "Adam rows; trainable leaves; get_num_rays_to_cast formula; unbudgeted adds all"),
    _probe("conegs-opacity-penalty", "opacity_penalty",
           "Official pre-activation penalty 2e-4 * mean(logit): gradient lambda/N on every logit, "
           "unlike the MCMC sigmoid penalty"),
    _probe("conegs-renderer-sh", "renderer_sh",
           "All SH bands rendered from step 1 (runtime degree ignored); frame passthrough", True),
)

CONTRIBUTIONS = (
    ContributionSpec("ingp-initialization",
                     "Per-scene iNGP proxy and 1M median-depth initialization (Sec. 4.1; "
                     "preprocessing ss-conegs-init)",
                     ("conegs-hashgrid", "conegs-proposal-sampler", "conegs-ingp-fit",
                      "conegs-median-depth")),
    ContributionSpec("error-guided-cone-densification",
                     "L1-error-weighted pixel sampling, Gaussians at the iNGP median depth on the "
                     "pixel cone, accumulated and merged every 100 steps (Sec. 4.2)",
                     ("conegs-cone-placement", "conegs-median-depth", "conegs-budget")),
    ContributionSpec("pixel-footprint-scaling",
                     "Isotropic scale = pixel cone footprint at the median depth (Eq. 7, 11)",
                     ("conegs-cone-placement",)),
    ContributionSpec("primitive-budgeting",
                     "Fixed budget with ray count max(gN, 1.2 N_last)/100 (Eq. 12) or adaptive "
                     "growth (Eq. 13)", ("conegs-budget",)),
    ContributionSpec("preactivation-opacity-penalty",
                     "Pre-activation opacity penalty, lambda_o = 2e-4 (Sec. 4.2)",
                     ("conegs-opacity-penalty",)),
    ContributionSpec("full-sh-from-start",
                     "All SH bands optimised from the first step (Sec. 4.4)",
                     ("conegs-renderer-sh",)),
)


def _make(name, density_overrides, description, background="white"):
    base = make_method(name)
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "conegs-gsplat", "renderer", "ss_conegs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        {"sh_degree": 3, **({"background": background} if background != "white" else {})},
    )
    density = ComponentSpec(
        "conegs-density", "density", "ss_conegs.density:create_density",
        ("conegs.frame",), (), {**DEFAULTS, **density_overrides},
    )
    opacity = ComponentSpec(
        "conegs-opacity", "loss", "ss_conegs.losses:create_opacity_penalty", (), (),
        {"weight": 0.0002, "mode": "signed"},
    )
    # The built-in renderer-gradient probe initializes the density on a synthetic scene (no
    # conegs_init.npz) and checks screen-gradient statistics ConeGS density never consumes.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric, opacity),
        probes=base_probes + PROBES,
        contributions=base_contribs + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _make("conegs", {},
                 "ConeGS (arXiv 2511.06810) native port, 1M primitive budget: iNGP init "
                 "(ss-conegs-init) + error-guided pixel-cone densification + pre-activation "
                 "opacity penalty; paper fidelity unverified.")


def nobudget():
    return _make("conegs-nobudget", {"budget": 0, "growth": 0.02},
                 "ConeGS (arXiv 2511.06810) native port without a budget (beta = 0.02); paper "
                 "fidelity unverified.")


def blackbg():
    # Diagnostic variant (gap analysis 1): official black background (white_background: false).
    return _make("conegs-blackbg", {},
                 "ConeGS (arXiv 2511.06810) native port, 1M budget, black background as in the "
                 "official code; diagnostic variant; paper fidelity unverified.", background="black")
