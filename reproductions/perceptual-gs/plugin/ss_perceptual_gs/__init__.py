"""Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian Splatting (ICML 2025,
arXiv 2506.12400). Native SplatStudio port of the official code
https://github.com/eezkni/Perceptual-GS; paper fidelity unverified.

Requires the per-scene sensitivity sidecar built by `ss-pgs-prep` (C1, offline as in the
official `preprocess.py`).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from . import core
from .density import DEFAULTS as DENSITY_DEFAULTS
from .loss import DEFAULTS as LOSS_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2506.12400", "https://github.com/eezkni/Perceptual-GS")
RENDERER_CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    "training.target_image",
    "training.camera",
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_perceptual_gs.probes:{fn}", description, cuda)


def method():
    base = make_method("perceptual-gs")
    renderer = ComponentSpec(
        "pgs-renderer", "renderer", "ss_perceptual_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    density = ComponentSpec(
        "pgs-density", "density", "ss_perceptual_gs.density:create_density",
        ("gradient.screen.signed", "training.target_image"), (), dict(DENSITY_DEFAULTS),
    )
    photometric = ComponentSpec(
        "pgs-photometric", "loss", "ss_perceptual_gs.loss:create_photometric_weighted",
        ("render.rgb",), (), dict(LOSS_DEFAULTS),
    )
    bce = ComponentSpec(
        "pgs-bce", "loss", "ss_perceptual_gs.loss:create_bce",
        ("training.camera",), (), dict(LOSS_DEFAULTS),
    )
    field = FieldSpec("sensitivity", (), learning_rate=0.05, initial_value=core.logit(0.1))
    probes = (
        _probe("pgs-extraction", "extraction",
               "Sensitivity map equals an explicit-loop Sobel/threshold/pool reference; faint "
               "texture is kept only thanks to the enhancement threshold (Eqs. 5-7)", False),
        _probe("pgs-dual-branch", "dual_branch",
               "Sensitivity branch = (1-A) + R[sigma(eps)]; BCE value and gradient vs finite "
               "difference; 0.9/0.1 weighting until 15k, plain photometric after (Eqs. 8-10)"),
        _probe("pgs-guided-densify", "guided_densify",
               "Zero-gradient Gaussians: sigma>=0.9 & weight>25 densified at 1000k steps, "
               "0.3<sigma<0.9 & weight>10 at 1500k, low sigma never; clone vs split_only; "
               "sensitivity stays trainable (Eqs. 11-14)"),
        _probe("pgs-opacity-decline", "opacity_decline",
               "Clone opacities 1-sqrt(1-alpha^1.2), unchanged with OD disabled (Eqs. 19-20)",
               False),
        _probe("pgs-scene-decision", "scene_decision",
               "gamma vs a sorting reference, gamma>0.55 -> depth reinit, beta>0.85 -> "
               "split_only (Eqs. 15-18)", False),
        _probe("pgs-stats-kernel", "stats_kernel",
               "CUDA per-Gaussian weight sums / argmax / T equal brute-force blending"),
        _probe("pgs-depth-reinit", "depth_reinit",
               "Depth reinit replaces the population with max-weight surface points, GT colours, "
               "opacity/sensitivity 0.1, fresh Adam, trainable fields (Sec. 3.6)"),
    )
    contributions = (
        ContributionSpec("perceptual-sensitivity-extraction",
                         "Binary perceptual sensitivity maps (Sec. 3.3)", ("pgs-extraction",)),
        ContributionSpec("dual-branch-rendering",
                         "Per-Gaussian sensitivity rendered and supervised by BCE (Sec. 3.4)",
                         ("pgs-dual-branch",)),
        ContributionSpec("sensitivity-guided-densification",
                         "High/medium-sensitivity densification with weight constraints and "
                         "scene-sensitivity split mode (Sec. 3.5)",
                         ("pgs-guided-densify", "pgs-stats-kernel")),
        ContributionSpec("scene-adaptive-depth-reinit",
                         "gamma-triggered depth reinitialization (Sec. 3.6)",
                         ("pgs-scene-decision", "pgs-depth-reinit")),
        ContributionSpec("opacity-decline", "Opacity decline for clones (Sec. 3.7)",
                         ("pgs-opacity-decline",)),
    )
    return replace(
        base,
        description="Perceptual-GS (ICML 2025) native port of the official code; paper "
                    "fidelity unverified.",
        components=(renderer, density, photometric, bce),
        fields=(field,),
        # The base photometric probe calls the loss without a training step; the weighted
        # photometric term is covered by pgs-dual-branch instead.
        probes=tuple(p for p in base.probes if p.name != "photometric-gradient") + probes,
        contributions=tuple(c for c in base.contributions if c.name != "photometric-objective")
        + contributions,
        citations=CITATIONS,
    )
