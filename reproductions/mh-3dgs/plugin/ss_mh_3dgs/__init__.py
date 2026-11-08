"""Metropolis-Hastings Sampling for 3D Gaussian Reconstruction (MH-3DGS; Kim, Jung, Park,
NeurIPS 2025, arXiv 2506.12945). Native SplatStudio port of the official code
https://github.com/SNU-VGILab/MH-3DGS (commit 71ef8f9, Inria Gaussian-Splatting research
license); paper fidelity unverified.

`mh-3dgs-mcmc-control` is the paper's same-budget 3DGS-MCMC comparison: gsplat MCMCStrategy
with cap = MH-3DGS's reported per-scene count (looked up from the --data basename) and the
same opacity/scale regularisers.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .core import DEFAULTS
from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2506.12945",
    "https://github.com/SNU-VGILab/MH-3DGS",
)

# Paper Appendix Tab. full_360 / full_rest "# of Gaussians" (MH-3DGS = 3DGS-MCMC budget).
PAPER_COUNTS = {
    "bonsai": 777782, "counter": 698871, "kitchen": 623992, "room": 764539,
    "bicycle": 741748, "flowers": 695495, "garden": 811695, "stump": 697792,
    "treehill": 690517, "train": 788942, "truck": 756104, "drjohnson": 766469,
    "playroom": 735409,
}


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_mh_3dgs.probes:{fn}", description, cuda)


PROBES = (
    _probe("importance-formula", "importance_formula",
           "Per-Gaussian importance equals an independent numpy evaluation of "
           "sigmoid(0.8 z(O) + 0.5 z(SSIM) + 0.5 z(L1)) with the official world-xy pixel lookup"),
    _probe("voxel-acceptance", "voxel_acceptance",
           "Empirical acceptance in empty vs crowded voxels matches sigmoid(I-0.5)/(1+5c); "
           "voxel counts equal a Python dict loop"),
    _probe("birth-proposals", "birth_proposals",
           "One MH step: children copy parents except means, offset std = s*scale/(1+ln(1+denom)), "
           "zero Adam moments, trainable leaves, parents unchanged"),
    _probe("relocation", "relocation",
           "Dead Gaussians (opacity<=0.005) are relocated onto alive ones with the MCMC formula; "
           "population size unchanged", True),
    _probe("schedules", "schedules",
           "View subset, round robin, proposal scales, voxel size and batch anneals match the "
           "official formulas; no event outside (500,15000) or off the 100-step grid"),
    _probe("multiview-maps", "multiview_maps",
           "Aggregated SSIM/L1 maps equal an independent per-view render average", True),
    _probe("mh-step", "mh_step_end_to_end",
           "MHDensity.after_backward at step 600 relocates all dead Gaussians, appends the "
           "accepted births, resets statistics and keeps optimizer state aligned", True),
    _probe("regularizers", "regularizers",
           "Regulariser gradients equal 0.01 sigmoid'(o)/N and 0.01 exp(s)/(3N)"),
)
CONTRIBUTIONS = (
    ContributionSpec("multiview-importance", "Sec. 4.2.1-4.2.2: multi-view SSIM/L1 error maps "
                     "over an annealed round-robin view subset fused with opacity into a "
                     "per-Gaussian importance (official world-xy lookup)",
                     ("importance-formula", "multiview-maps", "schedules")),
    ContributionSpec("mh-birth-sampler", "Sec. 4.2.3 / 4.3 / Alg. 1: coarse+fine proposals "
                     "picked by importance, accepted with sigmoid(I-0.5) D(v'), "
                     "D = 1/(1+lambda_v c(v'))", ("birth-proposals", "voxel-acceptance", "mh-step")),
    ContributionSpec("relocation", "Sec. 4.2.4: low-opacity Gaussians relocated (3DGS-MCMC)",
                     ("relocation", "mh-step")),
    ContributionSpec("coarse-to-fine-schedule", "Sec. 5.1: proposal scales 10->5 / 2->1, voxel "
                     "0.02->0.005, batches 4500/16000 decaying, view subset |C|->1",
                     ("schedules",)),
    ContributionSpec("mcmc-regularizers", "Eq. likelihood: 0.01 mean opacity + 0.01 mean scale",
                     ("regularizers",)),
)


def _renderer():
    return ComponentSpec(
        "mh-raster", "renderer", "ss_mh_3dgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        {"background": [0.0, 0.0, 0.0]},
    )


def _regularizers():
    return ComponentSpec(
        "mcmc-regularizers", "loss", "ss_mh_3dgs.losses:create_regularizers",
        ("state.gaussian3d",), (), {"opacity_weight": 0.01, "scale_weight": 0.01},
    )


def method():
    base = make_method("mh-3dgs")
    density = ComponentSpec(
        "mh-density", "density", "ss_mh_3dgs.density:create_density", (), (), dict(DEFAULTS),
    )
    return replace(
        base,
        description="MH-3DGS (arXiv 2506.12945) native port of the official code: MH birth "
                    "sampler + MCMC relocation + regularisers; paper fidelity unverified.",
        components=(_renderer(), density, base.components[2], _regularizers()),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def mcmc_control():
    base = make_method("mh-3dgs-mcmc-control")
    density = ComponentSpec(
        "mcmc-budget-density", "density", "ss_mh_3dgs.control:create_density", (), (),
        {"cap_max": None},
    )
    return replace(
        base,
        description="Same-budget 3DGS-MCMC control for MH-3DGS (gsplat MCMCStrategy, cap = "
                    "MH-3DGS per-scene count); not a reproduction of 3DGS-MCMC.",
        components=(_renderer(), density, base.components[2], _regularizers()),
        citations=("https://arxiv.org/abs/2404.09591",) + CITATIONS,
    )
