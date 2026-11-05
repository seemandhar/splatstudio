"""Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives (CVPR 2025,
arXiv 2412.00578) — native SplatStudio port of github.com/j-alex-hanson/speedy-splat (commit
34c45c6) with its CUDA rasterizer (speedy-splat-rasterizer b38abb6) vendored unmodified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2412.00578", "https://github.com/j-alex-hanson/speedy-splat",
             "https://speedysplat.github.io/")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_speedysplat.probes:{fn}", desc, cuda)


def _build(name, rasterizer, prune, description):
    base = make_method(name)
    photometric = base.components[2]
    comps = (
        ComponentSpec("speedy-renderer", "renderer", "ss_speedysplat.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "background": list(RENDERER_DEFAULTS["background"]),
                       "rasterizer": rasterizer}),
        ComponentSpec("speedy-density", "density", "ss_speedysplat.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS,
                       "score_background": list(DENSITY_DEFAULTS["score_background"]),
                       "prune": prune}),
        photometric,
    )
    probes, contribs = [], []
    probes.append(_probe(
        "speedy-adc-schedule", "adc_schedule",
        "3D-GS ADC cadence: densify only at s>500, s%100==0, s<15000; opacity reset at "
        "s%3000; world-size prune after 3000; no screen-size prune"))
    contribs.append(ContributionSpec(
        "3dgs-density-schedule", "Official 3D-GS ADC / opacity reset kept unchanged",
        ("speedy-adc-schedule",)))
    if rasterizer == "speedy":
        probes.append(_probe(
            "speedy-culling-exact", "culling_exact",
            "Vendored SnugBox/AccuTile kernel: image equals an independent float64 dense "
            "compositor (every Gaussian at every pixel, no tiles); tile-Gaussian pairs equal an "
            "independent exact ellipse-tile intersection count and are fewer than the stock "
            "3-sigma rasterizer's", True))
        contribs.append(ContributionSpec(
            "snugbox-accutile", "Precise opacity-aware tile intersection (SnugBox §4.1.1 + "
            "AccuTile Alg. 1) in the official CUDA rasterizer", ("speedy-culling-exact",)))
        probes.append(_probe(
            "speedy-score-kernel", "score_kernel",
            "Kernel pruning score equals sum_views sum_pixels (o * d(sum C)/d alpha)^2 from "
            "float64 autograd of the dense compositor (Eq. 23, train.py score_func)", True))
        contribs.append(ContributionSpec(
            "efficient-pruning-score", "Per-Gaussian score (grad_g I)^2 from the render "
            "backward (§4.2.1)", ("speedy-score-kernel",)))
    if prune:
        probes.append(_probe(
            "speedy-prune-schedule", "prune_schedule",
            "Soft prune 80% at 6000/9000/12000 (after densify, before opacity reset), hard "
            "prune 30% at 15000..27000, nowhere else; percentile rule score <= "
            "sorted[int(r(N-1))]; lowest scores removed, optimizer state aligned"))
        contribs.append(ContributionSpec(
            "soft-pruning", "Soft Pruning during densification (§4.2.2)",
            ("speedy-prune-schedule",)))
        contribs.append(ContributionSpec(
            "hard-pruning", "Hard Pruning after densification (§4.2.3)",
            ("speedy-prune-schedule",)))
    return replace(base, description=description, components=comps,
                   probes=base.probes + tuple(probes),
                   contributions=base.contributions + tuple(contribs), citations=CITATIONS)


def method():
    return _build("speedy-splat", "speedy", True,
                  "Speedy-Splat (CVPR 2025) native port: official SnugBox/AccuTile rasterizer + "
                  "soft/hard sensitivity pruning; paper fidelity unverified.")


def accutile():
    return _build("speedy-splat-accutile", "speedy", False,
                  "Speedy-Splat '+AccuTile' row: official culling rasterizer, 3D-GS training, no "
                  "pruning (lossless speed-up); paper fidelity unverified.")


def stock():
    return _build("speedy-splat-stock", "stock", False,
                  "Control: stock Inria 3D-GS rasterizer (59f5f77, 3-sigma tiles) with the same "
                  "3D-GS density; Speedy-Splat paper 'Baseline' row. Not Speedy-Splat.")
