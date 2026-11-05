"""Trim 3D Gaussian Splatting for Accurate Geometry Representation (Trim3DGS, arXiv 2406.07499).

Native SplatStudio port of the official code https://github.com/YuxueYang1204/TrimGS
(`Trim3DGS/`, scripts/run_Mipnerf360.py); paper fidelity unverified.

One run of 37000 steps = 30000 steps flat 3DGS pre-training + 7000 steps Trim fine-tune.
Methods: `trimgs` (normal weight 0.1: Mip-NeRF 360 outdoor scenes) and `trimgs-indoor`
(0.01: bonsai, counter, kitchen, room), as in the official per-scene `normal_weights`.
Running `trimgs` for 30000 steps gives the paper's "3DGS*" baseline (flat 3DGS).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPS

CITATIONS = ("https://arxiv.org/abs/2406.07499", "https://github.com/YuxueYang1204/TrimGS")


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_trimgs.probes:{fn}", description, cuda)


def _make(name, normal_weight, description):
    base = make_method(name)
    _, base_density, photometric = base.components
    renderer = ComponentSpec(
        "trim-flat", "renderer", "ss_trimgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPS, {"flat_eps": 1e-8},
    )
    density = ComponentSpec(
        "trim-density", "density", "ss_trimgs.density:create_density",
        ("gradient.screen.signed",), (), {**base_density.config, **DENSITY_DEFAULTS},
    )
    normal = ComponentSpec(
        "trim-normal", "loss", "ss_trimgs.losses:create_normal_loss",
        ("render.flat_normal", "training.camera"), (),
        {**LOSS_DEFAULTS, "weight": normal_weight},
    )
    probes = (
        _probe("contribution-kernel", "contribution_kernel",
               "CUDA Eq. 5 sums / covered-pixel counts equal an independent per-pixel blend "
               "loop on official 3-sigma tiles (gamma 0.5 and 0.25)"),
        _probe("trimming", "trimming",
               "Occluded Gaussian is trimmed, visible ones kept; multi-view list equals the "
               "official record insertion and differs from opacity-only pruning"),
        _probe("scale-split", "scale_split",
               "Large in-box Gaussian split into 3 with s/2.4 and flat axis; small or "
               "out-of-box ones untouched; Adam moments aligned"),
        _probe("normal-loss", "normal_loss",
               "Depth normals of a rendered plane equal the analytic normal; loss ~0 for aligned "
               "flat Gaussians, larger when tilted; finite gradient to quats"),
        _probe("flat-schedule", "flat_schedule",
               "Flat axis pinned and ignored by the renderer; means LR 3DGS-30k then constant; "
               "Adam reset at 30000; normal loss off before 30000", False),
    )
    contributions = (
        ContributionSpec("flat-two-stage", "Flat Gaussians (FlattenGaussianModel) and 30k "
                         "pre-train + 7k fine-tune protocol (official code)", ("flat-schedule",)),
        ContributionSpec("contribution-trimming", "Normalized multi-view contribution and 10% "
                         "trimming every 1000 its (Sec. 4.1, Eq. 5-6)",
                         ("contribution-kernel", "trimming")),
        ContributionSpec("scale-driven-densification", "Split Gaussians larger than tau_s into "
                         "K=3 (Sec. 4.2)", ("scale-split",)),
        ContributionSpec("normal-regularization", "Depth-normal consistency + kNN smoothness "
                         "(Sec. 4.3, Eq. 7-8)", ("normal-loss",)),
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric, normal),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("trimgs", 0.1, "Trim3DGS (arXiv 2406.07499) native port, outdoor normal weight "
                 "0.1; paper fidelity unverified.")


def indoor():
    return _make("trimgs-indoor", 0.01, "Trim3DGS (arXiv 2406.07499) native port, indoor normal "
                 "weight 0.01; paper fidelity unverified.")
