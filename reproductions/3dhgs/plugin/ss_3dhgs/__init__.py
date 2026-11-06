"""3D-HGS: 3D Half-Gaussian Splatting (CVPR 2025, arXiv 2406.02720) — native SplatStudio port of
github.com/lihaolin88/3DHGS (commit dde1bd5) with its half-Gaussian CUDA rasterizer vendored.

State: base `opacities` = alpha1 (logit), extra fields `opacity2` = alpha2 (logit) and
`hgs_normal` (raw splitting-plane normal, official `_normal`, init 0.6, lr 0.003).
"""

import math
from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2406.02720", "https://github.com/lihaolin88/3DHGS")
LOGIT_01 = math.log(0.1 / 0.9)  # create_from_pcd: inverse_sigmoid(0.1) for both halves
NORMAL_LR = 0.003  # training_setup hard-codes 0.003 (paper §4.1: 0.3 on Deep Blending)
OPACITY_LR = 0.05  # OptimizationParams.opacity_lr (same group holds both halves)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_3dhgs.probes:{fn}", desc, cuda)


def method(name="3dhgs", normal_lr=NORMAL_LR):
    base = make_method(name)
    photometric = base.components[2]
    comps = (
        ComponentSpec("hgs-renderer", "renderer", "ss_3dhgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("hgs-density", "density", "ss_3dhgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        photometric,
    )
    fields = (
        FieldSpec("opacity2", (), OPACITY_LR, LOGIT_01),
        FieldSpec("hgs_normal", (3,), normal_lr, 0.6),
    )
    probes = (
        _probe("hgs-equal-opacity", "equal_opacity",
               "alpha1 = alpha2: vendored kernel equals an independent float64 3DGS compositor "
               "(no erf) and is invariant to the normal; unequal halves change the image", True),
        _probe("hgs-half-plane", "half_plane",
               "Single Gaussian alpha1=0.9, alpha2=0: kernel alpha = alpha1 * G * P(half) with "
               "P from numerical z-quadrature of the ray-space 3D Gaussian over the half-space; "
               "swapping halves gives the complement", True),
        _probe("hgs-gradients", "gradients",
               "Vendored backward = float64 autograd of the independent compositor for all "
               "fields (opacity <= 0.45); high-opacity forward-only -4.8 cutoff mismatch "
               "measured", True),
        _probe("hgs-test-cull", "test_cull",
               "Half-aware test-time tile rectangles are active only without autograd and "
               "change the render slightly (official efficient rasterizer)", True),
        _probe("hgs-density", "density",
               "Clone 1-sqrt(1-a) per half on copies only, split s/1.6 with copied normal and "
               "opacities, prune on max half < 0.01, reset 0.02 both halves, opacity LR /1.4 "
               "per 5000, densify until 20000"),
    )
    contribs = (
        ContributionSpec("half-gaussian-kernel", "Two opacities + splitting plane per "
                         "primitive; projected half-Gaussian = G * erfc factor (Eq. 7, 8, 10)",
                         ("hgs-equal-opacity", "hgs-half-plane")),
        ContributionSpec("half-gaussian-rasterizer", "Official forward/backward CUDA rasterizer "
                         "and test-time half-aware bounding rectangles (§3.3-3.4, supp. B.1)",
                         ("hgs-gradients", "hgs-test-cull")),
        ContributionSpec("hgs-training-settings", "Normal LR 0.003, opacity LR /1.4 per 5k, "
                         "prune 0.01, reset 0.02, densify until 20k, clone opacity rescale "
                         "(supp. B.2, official train.py)", ("hgs-density",)),
    )
    return replace(
        base,
        description="3D-HGS (CVPR 2025) native port with the official half-Gaussian CUDA "
                    "rasterizer; paper fidelity unverified.",
        components=comps,
        fields=fields,
        probes=base.probes + probes,
        contributions=base.contributions + contribs,
        citations=CITATIONS,
    )


def deep_blending():
    """Paper §4.1 normal LR 0.3 for Deep Blending (the official code ignores it: 0.003)."""
    return method("3dhgs-db-paper", normal_lr=0.3)
