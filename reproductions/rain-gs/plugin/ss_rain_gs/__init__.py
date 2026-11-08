"""RAIN-GS: Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting
(arXiv 2403.09413) — native SplatStudio port of the official `--ours_new --train_from random`
path (github.com/cvlab-kaist/RAIN-GS, MIT License, Copyright (c) 2024 KU-CVLAB).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2403.09413",
    "https://github.com/cvlab-kaist/RAIN-GS",
)


def method():
    base = make_method("rain-gs")
    renderer = ComponentSpec(
        "rain-gs-renderer",
        "renderer",
        "ss_rain_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        tuple(sorted(CAPABILITIES)),
        dict(RENDERER_DEFAULTS),
    )
    density = ComponentSpec(
        "rain-gs-density",
        "density",
        "ss_rain_gs.density:create_density",
        ("gradient.screen.signed",),
        (),
        dict(DENSITY_DEFAULTS),
    )
    probes = (
        ProbeSpec("slv-init", "ss_rain_gs.probes:slv_init",
                  "N=10 random Gaussians in the official 3x tvec cube, 3-NN scales, opacity 0.1"),
        ProbeSpec("lowpass-schedule", "ss_rain_gs.probes:lowpass_schedule",
                  "eps2d = clip(HW/(9 pi N), 0.3, 300) at step 1 and every 1000 below 25000"),
        ProbeSpec("lowpass-in-render", "ss_rain_gs.probes:lowpass_in_render",
                  "Scheduled eps2d enlarges a tiny Gaussian's footprint; test renders use 0.3",
                  True),
        ProbeSpec("abe-split", "ss_rain_gs.probes:abe_split",
                  "Split during warm-up adds a third Gaussian at 0.3*extent*mu with the parent "
                  "scale; children scale/1.4; no ABE after 10000"),
        ProbeSpec("warmup-schedule", "ss_rain_gs.probes:warmup_schedule",
                  "Means LR fixed for 10k then official expon(step-10k); SH from 5000; "
                  "densify until 25000"),
        ProbeSpec("rain-renderer-gradient", "ss_rain_gs.probes:renderer_gradient",
                  "Training render gives screen gradients; one density step keeps state aligned",
                  True),
    )
    contributions = (
        ContributionSpec("slv-init", "Sparse-large-variance random initialization (§4.1, Alg. 1)",
                         ("slv-init",)),
        ContributionSpec("progressive-lowpass",
                         "Progressive Gaussian low-pass filter control (§4.2, Alg. 2 l.8)",
                         ("lowpass-schedule", "lowpass-in-render")),
        ContributionSpec("abe-split", "Adaptive bound-expanding split (§4.3, Alg. 2 l.21-22)",
                         ("abe-split",)),
        ContributionSpec("warmup-schedule",
                         "10k warm-up: fixed means LR, densify until 25k, SH from 5k, ratio 1.4",
                         ("warmup-schedule",)),
    )
    keep = {"state-transitions", "photometric-gradient"}
    return replace(
        base,
        description="RAIN-GS native port (SLV random init, progressive low-pass, ABE-Split); "
                    "paper fidelity unverified.",
        components=(renderer, density, base.components[2]),
        probes=tuple(p for p in base.probes if p.name in keep) + probes,
        contributions=tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
        + (ContributionSpec("rasterizer-statistics",
                            "Renderer supplies density-controller statistics",
                            ("rain-renderer-gradient",)),)
        + contributions,
        citations=CITATIONS,
    )
