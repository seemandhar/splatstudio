"""Evaluating Alternatives to SFM Point Cloud Initialization for Gaussian Splatting
(Foroutan, Rebain, Yi, Tagliasacchi; arXiv 2404.12547). Native SplatStudio port; no official
training code was released (theialab/nerf-3dgs is the project page only); paper fidelity
unverified.

Two stages:
1. `ss-nerfinit-prep --source <native scene> --output <copy>`: NerfAcc proposal iNGP trained for
   5k iterations on the training views, 500k points sampled from the volume-rendering weights
   (Eq. 4), NeRF depth of every training pixel (Eq. 6), hard-linked dataset copy.
2. `ss-train nerf-init-gs --data <copy> --recipe splatstudio --num-gaussians 10000000`:
   unchanged 3DGS training + depth distillation (Eq. 7, lambda = 0.9 * 0.9^(i/100)).

Methods: `nerf-init-gs` = "NerfAcc @ 5k + Depth Loss" (Table 4, the paper's best row);
`nerf-init-gs-nodepth` = "NerfAcc @ 5k" (Table 2/4; same copy, no depth loss).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .losses import DEFAULTS as DEPTH_DEFAULTS
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2404.12547", "https://theialab.github.io/nerf-3dgs")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_nerf_init_gs.probes:{fn}", description, cuda)


INIT_PROBES = (
    _probe("nerfinit-hashgrid", "hashgrid",
           "iNGP hash-grid CUDA kernel (tcnn layout) equals an independent torch reference", True),
    _probe("nerfinit-proposal-sampler", "proposal_sampler",
           "nerfacc importance_sampling docstring cases, lindisp s<->t, pdf loss"),
    _probe("nerfinit-surface-sampling", "surface_sampling",
           "Eq. 4: one inverse-CDF sample per ray follows the volume-rendering weights "
           "(0.3/0.7 two-surface case), colour queried at the sample, Eq. 6 depth"),
    _probe("nerfinit-ingp-fit", "ingp_fit",
           "Short NerfAcc-iNGP fit on a textured sphere: Eq. 6 depth and Eq. 4 samples on the "
           "surface", True),
    _probe("nerfinit-init-loading", "init_loading",
           "3DGS-style loading: SHA-checked exact 3-NN scales, opacity 0.1, camera-extent "
           "scene scale and means LR"),
)
DEPTH_PROBES = (
    _probe("nerfinit-depth-loss", "depth_loss",
           "Eq. 7 L1 depth loss with Eq. 13 schedule 0.9*0.9^(i/100); raises without NeRF depth"),
    _probe("nerfinit-renderer-depth", "renderer_depth",
           "D_GS = sum w z (un-normalised, gsplat 'D' reference); sidecar NeRF depth attached "
           "to training renders only", True),
)

INIT_CONTRIBUTIONS = (
    ContributionSpec("volumetric-initialization",
                     "500k points sampled from the volume-rendering weights of a NerfAcc iNGP "
                     "trained for 5k iterations, coloured by the NeRF radiance (Sec. 3.2, 4.3; "
                     "preprocessing ss-nerfinit-prep)",
                     ("nerfinit-hashgrid", "nerfinit-proposal-sampler",
                      "nerfinit-surface-sampling", "nerfinit-ingp-fit", "nerfinit-init-loading")),
)
DEPTH_CONTRIBUTIONS = (
    ContributionSpec("volumetric-structure-distillation",
                     "L1 distillation of precomputed NeRF depth into the rendered Gaussian depth, "
                     "exponentially decayed weight (Sec. 3.3, 3.4, Supp. Eq. 13)",
                     ("nerfinit-depth-loss", "nerfinit-renderer-depth", "nerfinit-ingp-fit")),
)


def _make(name, depth, description, depth_cfg=None):
    base = make_method(name)
    _, base_density, photometric = base.components
    renderer = ComponentSpec(
        "nerfinit-gsplat", "renderer", "ss_nerf_init_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)), {},
    )
    density = ComponentSpec(
        "nerfinit-density", "density", "ss_nerf_init_gs.density:create_density",
        base_density.requires, (), dict(base_density.config),
    )
    components = [renderer, density, photometric]
    probes, contributions = INIT_PROBES, INIT_CONTRIBUTIONS
    if depth:
        components.append(ComponentSpec(
            "depth-distillation", "loss", "ss_nerf_init_gs.losses:create_depth_loss",
            ("render.depth.accumulated_z", "nerfinit.nerf_depth"), (),
            {**DEPTH_DEFAULTS, **(depth_cfg or {})},
        ))
        probes, contributions = probes + DEPTH_PROBES, contributions + DEPTH_CONTRIBUTIONS
    return replace(
        base,
        description=description,
        components=tuple(components),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("nerf-init-gs", True,
                 "Foroutan et al. (arXiv 2404.12547) 'NerfAcc @ 5k + Depth Loss' native port: "
                 "iNGP volume-weight point init (ss-nerfinit-prep) + NeRF depth distillation; "
                 "paper fidelity unverified.")


def nodepth():
    return _make("nerf-init-gs-nodepth", False,
                 "Foroutan et al. (arXiv 2404.12547) 'NerfAcc @ 5k' native port: iNGP "
                 "volume-weight point init only (ss-nerfinit-prep); paper fidelity unverified.")


def nerfunits():
    return _make("nerf-init-gs-nerfunits", True,
                 "Foroutan et al. (arXiv 2404.12547) post-hoc iteration 1: as nerf-init-gs but the "
                 "depth L1 is measured in nerfacc's normalised scene units (x similarity scale); "
                 "paper fidelity unverified.", {"units": "nerf"})
