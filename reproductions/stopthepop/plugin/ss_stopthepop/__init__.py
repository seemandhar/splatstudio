"""StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering (TOG 2024,
arXiv 2402.00525) — native SplatStudio port of github.com/r4dl/StopThePop @ 859b11b with its
CUDA rasterizer (r4dl/StopThePop-Rasterization @ d8cad09) vendored unmodified and used for
training and evaluation, `configs/hierarchical.json`; density is the unchanged 3DGS ADC.
"""

import copy
from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .raster import HIERARCHICAL, VANILLA
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2402.00525", "https://github.com/r4dl/StopThePop",
             "https://github.com/r4dl/StopThePop-Rasterization")


def _probe(name, fn, desc, cuda=True):
    return ProbeSpec(name, f"ss_stopthepop.probes:{fn}", desc, cuda)


def _method(name, splatting, density_overrides=None):
    base = make_method(name)
    photometric = base.components[2]
    comps = (
        ComponentSpec("stopthepop-renderer", "renderer", "ss_stopthepop.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {"background": [0.0, 0.0, 0.0], "splatting": copy.deepcopy(splatting)}),
        ComponentSpec("stopthepop-density", "density", "ss_stopthepop.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, **(density_overrides or {})}),
        photometric,
    )
    return base, comps


def method():
    base, comps = _method("stopthepop", HIERARCHICAL)
    probes = (
        _probe("stopthepop-topt", "topt",
               "Tilted vs fronto-parallel splats: PPX_FULL and HIER blend in Eq. (1) t_opt order "
               "(float64 reference), not by d.(mu-o) or the global view z"),
        _probe("stopthepop-perpixel-order", "perpixel_order",
               "Crossing splats: HIER and PPX_FULL images equal an independent per-pixel "
               "t_opt-sorted compositor; GLOBAL equals the global-z reference instead"),
        _probe("stopthepop-backward", "backward",
               "Sorted backward: opacity/sh0/screen-mean gradients equal autograd of the "
               "per-pixel sorted float64 reference, not the global-order one"),
        _probe("stopthepop-popping", "popping",
               "Yaw sweep over two splats whose mean-z order flips: rotation-warped frame "
               "consistency error jumps for GLOBAL (pop), stays smooth for HIER"),
        _probe("stopthepop-culling", "culling",
               "Rect/tight bounding, tile and 4x4 culling leave images unchanged and reduce "
               "sort entries; HIER vs GLOBAL forward timings"),
        _probe("stopthepop-schedule", "schedule",
               "3DGS density cadence: densify 500<s<15000 every 100, reset at 3000, world "
               "prune after 3000, no screen prune; opacity-decay option", cuda=False),
    )
    contribs = (
        ContributionSpec("per-ray-depth", "Blend position t_opt = argmax_t G(o + t d) per ray "
                         "(Eq. 1, S^-1 bounded at 1e3)", ("stopthepop-topt",)),
        ContributionSpec("hierarchical-resorting", "Per-tile-depth global sort + 4x4/2x2/pixel "
                         "queues give per-pixel sorted blending (§3.3-3.4)",
                         ("stopthepop-perpixel-order", "stopthepop-popping")),
        ContributionSpec("tile-culling", "Exact tile/hierarchical culling and tighter bounds "
                         "(§3.4; speed only)", ("stopthepop-culling",)),
        ContributionSpec("sorted-backward", "Front-to-back backward with the same per-pixel "
                         "order, so training uses the sorted renderer (§3.5)",
                         ("stopthepop-backward",)),
        ContributionSpec("3dgs-optimisation", "Unchanged 3DGS ADC / reset schedule of the "
                         "official train.py", ("stopthepop-schedule",)),
    )
    return replace(
        base,
        description="StopThePop (TOG 2024) native port with the official sorted CUDA "
                    "rasterizer for training and rendering; paper fidelity unverified.",
        components=comps, probes=base.probes + probes,
        contributions=base.contributions + contribs, citations=CITATIONS,
    )


def global_control():
    """Control: the same vendored kernels and code base with configs/vanilla.json (3DGS global
    view-space-z sort; the paper's retrained '3DGS' row). Not StopThePop."""
    base, comps = _method("stopthepop-global", VANILLA)
    return replace(base, description="StopThePop code base with configs/vanilla.json (3DGS "
                   "global sort control, not StopThePop); paper fidelity unverified.",
                   components=comps, citations=CITATIONS)
