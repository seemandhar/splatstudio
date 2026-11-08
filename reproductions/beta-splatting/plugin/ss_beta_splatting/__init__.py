"""Deformable Beta Splatting (Liu et al., SIGGRAPH 2025, arXiv 2501.18630).
Native SplatStudio port; paper fidelity unverified.

Ported from the official code (Apache-2.0) https://github.com/RongLiu-Leo/beta-splatting @ 431d575:
the Beta-kernel rasterizer and Spherical Beta CUDA (submodules/gsplat fork) as a JIT extension
(csrc/beta_raster.cu) on gsplat 1.5.3 projection/tiling; kernel-agnostic MCMC from
`train.py` / `scene/beta_model.py`.

Methods (cap_max from the official benchmark.py; sb = 2, SH degree 0, 30K iterations):
  beta-splatting (1M, train.py default), -garden (5M), -room (1.5M), -bicycle (6M).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2501.18630",
    "https://github.com/RongLiu-Leo/beta-splatting",
)
SB_NUMBER = 2
CAPS = {"beta-splatting": 1_000_000, "beta-splatting-garden": 5_000_000,
        "beta-splatting-room": 1_500_000, "beta-splatting-bicycle": 6_000_000}


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_beta_splatting.probes:{fn}", description, cuda)


PROBES = (
    _probe("beta-forward-vs-dense", "beta_forward_vs_dense",
           "CUDA Beta-kernel render equals a dense float64 reference of min(.999, o(1-r^2)^beta) "
           "compositing (reference validated against gsplat's Gaussian rasterizer) and differs "
           "from the Gaussian render"),
    _probe("beta-backward-vs-dense", "beta_backward_vs_dense",
           "CUDA gradients (means2d, conics, colours, opacities, betas) equal autograd of the "
           "dense reference"),
    _probe("beta-support", "beta_support",
           "Single splat: alpha = o(1-r^2)^beta inside r<1, exactly 0 outside; 1-sigma box; "
           "beta controls the footprint; b=0 (beta=4) is Gaussian-like"),
    _probe("spherical-beta-vs-reference", "spherical_beta_vs_reference",
           "CUDA Spherical Beta colour and gradients equal the torch formula; no view-direction "
           "gradient (official); renderer colours are view-dependent and reduce to SH0 + 0.5"),
    _probe("mcmc-opacity-only", "mcmc_opacity_only",
           "Relocation/growth use o' = 1-(1-o)^(1/(k+1)) without scale change, zero Adam moments, "
           "5% growth to cap; noise = Sigma eps (1-o)^100 noise_lr lr (not the logit noise); "
           "schedule 500<step<25000 every 100", False),
    _probe("regularizer-schedule", "regularizer_schedule",
           "Regulariser = 0.01 mean(o) + 0.01 mean(exp s) only for 500 < step < 25000", False),
)


def _make(name, cap_max):
    base = make_method(name)
    renderer = ComponentSpec(
        "beta-renderer", "renderer", "ss_beta_splatting.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"sb_number": SB_NUMBER, "background": 0.0, "tile_size": 16, "eps2d": 0.3,
         "near_plane": 0.01})
    density = ComponentSpec(
        "kernel-agnostic-mcmc", "density", "ss_beta_splatting.density:create_density", (), (),
        {"cap_max": cap_max, "densify_from": 500, "densify_until": 25_000, "interval": 100,
         "dead_opacity": 0.005, "growth": 1.05, "noise_lr": 5e4, "noise_power": 100.0,
         "init_opacity": 0.5, "init_lobe_angles": True})
    regularizer = ComponentSpec(
        "dbs-regularizer", "loss", "ss_beta_splatting.losses:create_regularizer",
        ("render.rgb",), (), {"opacity_reg": 0.01, "scale_reg": 0.01, "start": 500,
                              "stop": 25_000})
    fields = (
        FieldSpec("betas", (), learning_rate=0.001, initial_value=0.0),
        FieldSpec("sb_params", (SB_NUMBER, 6), learning_rate=0.0025, initial_value=0.0),
    )
    contributions = (
        ContributionSpec("beta-kernel",
                         "Bounded Beta kernel (1-r^2)^(4e^b) with per-primitive learnable b "
                         "replaces the Gaussian; custom CUDA forward/backward (Sec. 4.1-4.2)",
                         ("beta-forward-vs-dense", "beta-backward-vs-dense", "beta-support")),
        ContributionSpec("spherical-beta",
                         "Spherical Beta colour: base colour + 2 lobes dot^(4e^{b_m}) c_m "
                         "replaces SH (Sec. 4.3)", ("spherical-beta-vs-reference",)),
        ContributionSpec("kernel-agnostic-mcmc",
                         "Opacity-only MCMC relocation/growth, Beta noise (1-o)^100, opacity and "
                         "scale regularisers (Sec. 4.4)",
                         ("mcmc-opacity-only", "regularizer-schedule")),
    )
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
    return replace(
        base,
        description=("Deformable Beta Splatting (arXiv 2501.18630) native port of the official "
                     f"code, sb=2, cap_max {cap_max}; paper fidelity unverified."),
        components=(renderer, density, base.components[2], regularizer),
        fields=fields, probes=base_probes + PROBES,
        contributions=base_contribs + contributions, citations=CITATIONS)


def beta_splatting():
    return _make("beta-splatting", CAPS["beta-splatting"])


def beta_splatting_garden():
    return _make("beta-splatting-garden", CAPS["beta-splatting-garden"])


def beta_splatting_room():
    return _make("beta-splatting-room", CAPS["beta-splatting-room"])


def beta_splatting_bicycle():
    return _make("beta-splatting-bicycle", CAPS["beta-splatting-bicycle"])


method = beta_splatting
