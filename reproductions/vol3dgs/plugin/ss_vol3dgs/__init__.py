"""Volumetrically Consistent 3D Gaussian Rasterization (Vol3DGS; Talegaonkar et al., CVPR 2025,
arXiv 2412.03378). Native SplatStudio port; paper fidelity unverified.

Ported from the official code (MIT): github.com/chinmay0301ucsd/Vol3DGS and its slang
rasterizer github.com/chinmay0301ucsd/slang-gaussian-rasterization @5eca118 (Apache-2.0).
The CUDA kernel adapts gsplat 1.5.3's RasterizeToPixels3DGS{Fwd,Bwd}.cu (Apache-2.0) with the
official `evaluate_splat_volr` alpha; projection/tiling/SH are gsplat's.

Method `vol3dgs`: volumetric alpha renderer + density reparameterisation, Vol3DGS densification,
softplus colours, the official Mip-NeRF 360 recipe (scripts/all_datasets.sh). Dataset-specific
values (T&T: extent_mult 5, min_theta 0.01, clone 5e-4, split 9e-6, interval 100, AbsGS;
DB: extent_mult 1.5, min_theta 0.03) are density config overrides.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import OPTIONS as DENSITY_OPTIONS
from .renderer import CAPABILITIES
from .renderer import OPTIONS as RENDERER_OPTIONS

CITATIONS = (
    "https://arxiv.org/abs/2412.03378",
    "https://github.com/chinmay0301ucsd/Vol3DGS",
    "https://github.com/chinmay0301ucsd/slang-gaussian-rasterization",
    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3",
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_vol3dgs.probes:{fn}", description, cuda)


PROBES = (
    _probe("vol-forward-vs-dense", "forward_vs_dense",
           "CUDA volumetric render (RGB, alpha, depth) equals a dense float64 torch reference of "
           "the analytic ray-integral alpha and differs from gsplat's splatted render"),
    _probe("vol-backward-vs-dense", "backward_vs_dense",
           "CUDA gradients (camera means, inverse covariance, density, colour) and end-to-end "
           "parameter gradients equal autograd of the dense reference"),
    _probe("analytic-transmittance", "analytic_transmittance",
           "Centre-ray alpha of an isotropic Gaussian is 1 - exp(-sqrt(2 pi) kappa s) "
           "(= 0.99 theta with the reparameterisation) and stretching the Gaussian along the "
           "ray raises alpha (a splatted 2D Gaussian is invariant)"),
    _probe("density-reparam", "density_reparam",
           "kappa = -log(1 - 0.99 theta) mean(1/s) / 2.506, its inverse and the halving rule",
           False),
    _probe("densify-rules", "densify_rules",
           "Clone/split/prune rules, density halving, high-density split, world-size prune "
           "after 3000 steps, paper LRs and the exponential means LR", False),
    _probe("world-gradient", "world_gradient",
           "The densification statistic is ||dL/dmu|| (world space) from an independent "
           "autograd computation, accumulated only for visible Gaussians"),
    _probe("recipe-colour", "recipe_colour",
           "Colours are softplus(SH + 0.5, beta 10) and the SH degree rises every 2000 steps"),
)


def _make(name, background, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "vol-renderer", "renderer", "ss_vol3dgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {**RENDERER_OPTIONS, "background": background})
    density = ComponentSpec(
        "vol-density", "density", "ss_vol3dgs.density:create_density",
        ("gradient.world.means",), (), {k: v for k, v in DENSITY_OPTIONS.items()})
    photometric = base.components[2]
    contributions = (
        ContributionSpec(
            "volumetric-alpha",
            "alpha = 1 - exp(-kappa G(gamma d) sqrt(2 pi) beta) from the analytic line integral of "
            "the 3D Gaussian along each pixel ray, inside a tile rasterizer; custom CUDA "
            "forward/backward (Sec. 4.1-4.2, Eq. 13-19)",
            ("vol-forward-vs-dense", "vol-backward-vs-dense", "analytic-transmittance")),
        ContributionSpec(
            "density-reparameterisation",
            "kappa = -log(1 - 0.99 theta) mean(1/s) / sqrt(2 pi), theta = sigmoid "
            "(Sec. 5, Eq. 20; official alpha2density)", ("density-reparam",)),
        ContributionSpec(
            "densification-heuristics",
            "World-space gradient statistic, clone 5e-5 / split 1e-7 thresholds, halved density "
            "for new Gaussians, split theta >= 0.99, prune theta < 0.05 and scale > 0.01 ext, "
            "extent x2, no opacity reset (Sec. 5, Supp. D)",
            ("densify-rules", "world-gradient")),
        ContributionSpec(
            "training-recipe",
            "Softplus colour, SH every 2000 steps, LRs theta 0.03 / SH 0.003 / means "
            "1.6e-4 -> 1e-5 (Supp. D, scripts/all_datasets.sh)", ("recipe-colour", "densify-rules")),
    )
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base_probes + PROBES,
        contributions=base_contribs + contributions,
        citations=CITATIONS,
    )


def vol3dgs():
    return _make("vol3dgs", "white",
                 "Vol3DGS (arXiv 2412.03378) native port: volumetric ray-integral alpha (custom "
                 "CUDA), density reparameterisation and the official densification recipe; white "
                 "background (runtime convention); paper fidelity unverified.")


def vol3dgs_black():
    # official inria_3dgs_volr.py:53 asserts a black background
    return _make("vol3dgs-black", "black",
                 "Vol3DGS (arXiv 2412.03378) native port with the official black background: "
                 "volumetric ray-integral alpha (custom CUDA), density reparameterisation and the "
                 "official densification recipe; paper fidelity unverified.")


method = vol3dgs_black
