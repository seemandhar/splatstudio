"""3D Gaussian Splatting as Markov Chain Monte Carlo (Kheradmand et al., NeurIPS 2024,
arXiv 2404.09591). Faithful native config of the official recipe
(https://github.com/ubc-vision/3dgs-mcmc, commit 7b4fc9f) on gsplat's MCMCStrategy kernels.

Differences from the built-in `gsplat-mcmc`: per-scene cap from the official configs (built-in:
1M everywhere), opacity/scale L1 regularisers 0.01 (0.001 opacity on Deep Blending), initial
opacity 0.5 and initial scale 0.1 x 3-NN (built-in: runtime 0.1 / 1 x), black background
(built-in: white). Paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2404.09591", "https://github.com/ubc-vision/3dgs-mcmc")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_mcmc_3dgs.probes:{fn}", description, cuda)


PROBES = (
    _probe("cap-lookup", "cap_lookup",
           "Per-scene cap_max equals the official configs/<scene>.json; unknown scenes fail"),
    _probe("official-init", "official_init",
           "Initial opacity 0.5 and log-scale log(0.1 sqrt(mean 3-NN d^2)) (brute-force numpy)"),
    _probe("regularizers", "regularizers",
           "Regulariser gradients equal 0.01 sigmoid'(o)/N and 0.01 exp(s)/(3N); DB opacity 0.001"),
    _probe("black-background", "black_background", "Empty render is black", True),
    _probe("mcmc-schedule", "mcmc_schedule",
           "Relocation/births only every 100 steps in (500,25000), +5 % capped; SGLD noise "
           "moves only near-transparent Gaussians", True),
)
CONTRIBUTIONS = (
    ContributionSpec("sgld-noise", "Eq. 9-10: opacity-gated, covariance-shaped position noise "
                     "lambda_noise=5e5 x lr", ("mcmc-schedule",)),
    ContributionSpec("relocation-births", "Sec. 3.3-3.4: dead (o<=0.005) Gaussians relocated, "
                     "+5 % births up to the per-scene cap", ("mcmc-schedule", "cap-lookup")),
    ContributionSpec("regularizers", "Eq. 11: L1 opacity + scale regularisation 0.01",
                     ("regularizers",)),
    ContributionSpec("official-recipe", "Official init (opacity 0.5, scale x0.1) and black "
                     "background", ("official-init", "black-background")),
)


def method():
    base = make_method("mcmc-3dgs")
    renderer = ComponentSpec(
        "mcmc-raster", "renderer", "ss_mcmc_3dgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        {"background": [0.0, 0.0, 0.0]},
    )
    density = ComponentSpec(
        "mcmc-3dgs-density", "density", "ss_mcmc_3dgs.density:create_density", (), (),
        dict(DEFAULTS),
    )
    regs = ComponentSpec(
        "mcmc-regularizers", "loss", "ss_mcmc_3dgs.losses:create_regularizers",
        ("state.gaussian3d",), (), {"opacity_weight": None, "scale_weight": 0.01},
    )
    return replace(
        base,
        description="3DGS-MCMC (arXiv 2404.09591) with the official recipe on gsplat "
                    "MCMCStrategy; paper fidelity unverified.",
        components=(renderer, density, base.components[2], regs),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
