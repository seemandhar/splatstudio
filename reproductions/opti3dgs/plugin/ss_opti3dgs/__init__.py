"""Opti3DGS: Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation
(Farooq et al., arXiv 2503.14475). Native SplatStudio port written from the paper (no official
code released); paper fidelity unverified.

- `opti3dgs`: Table 1 "3DGS + Ours*" (frequency modulation + scaling_lr 0.01,
  densification interval 500, densify grad threshold 1e-4).
- `opti3dgs-filter`: Table 2 "Ours-Filtering" (frequency modulation on default 3DGS).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import OURS_STAR, PLAIN_3DGS
from .losses import DEFAULTS as LOSS_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2503.14475",)

PROBES = (
    ProbeSpec("opti-mean-filter", "ss_opti3dgs.probes:mean_filter",
              "Eq. 2 mean filter equals scipy uniform_filter (reflect-101), keeps size, "
              "K=1 identity, removes Nyquist checkerboard"),
    ProbeSpec("opti-schedule", "ss_opti3dgs.probes:schedule",
              "Fig. 3 step decay 15/9/5/3 with boundaries 1000/4000/8000/12000, off after 12000"),
    ProbeSpec("opti-loss-target", "ss_opti3dgs.probes:loss_target",
              "Training loss = L1 + 0.2 D-SSIM against the blurred image while filtering, "
              "against the original after 12000; finite nonzero gradient"),
    ProbeSpec("opti-density-config", "ss_opti3dgs.probes:density_config",
              "Density settings (scales lr, refine interval, grad threshold) applied and kept "
              "after a real densification; all fields stay trainable leaves", True),
)


def _method(name, density_config, description, settings_contribution):
    base = make_method(name)
    renderer = next(c for c in base.components if c.kind == "renderer")
    density = ComponentSpec(
        "opti-density", "density", "ss_opti3dgs.density:create_density",
        ("gradient.screen.signed",), (), dict(density_config),
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_opti3dgs.losses:create_photometric",
        ("render.rgb",), (), {k: (list(v) if isinstance(v, list) else v)
                              for k, v in LOSS_DEFAULTS.items()},
    )
    # The built-in photometric probe calls the loss without a step; the Opti3DGS loss needs the
    # step for its schedule, so opti-loss-target (which also checks the gradient) replaces it.
    probes = tuple(p for p in base.probes if p.name != "photometric-gradient") + PROBES
    contributions = tuple(
        replace(c, probes=tuple("opti-loss-target" if n == "photometric-gradient" else n
                                for n in c.probes))
        for c in base.contributions
    ) + (
        ContributionSpec("frequency-modulation", "Mean low-pass filtering of training images "
                         "(Sec. 3.3, Eq. 2)", ("opti-mean-filter", "opti-loss-target")),
        ContributionSpec("step-schedule", "Coarse-to-fine step decay of the kernel size, "
                         "original images after 12k (Sec. 3.4, Fig. 3)",
                         ("opti-schedule", "opti-loss-target")),
        ContributionSpec("progressive-growing", "G_n = 3DGS(G_{n-1}, I_n) continued across "
                         "levels (Sec. 3.5, Eq. 3-5)", ("opti-schedule", "opti-loss-target")),
        settings_contribution,
    )
    return replace(
        base, description=description, components=(renderer, density, loss),
        probes=probes, contributions=contributions, citations=CITATIONS,
    )


def method():
    return _method(
        "opti3dgs", OURS_STAR,
        "Opti3DGS (arXiv 2503.14475) '3DGS + Ours*' native port: coarse-to-fine mean-blur "
        "frequency modulation of training images with the paper's modified 3DGS settings; "
        "paper fidelity unverified.",
        ContributionSpec("ours-star-settings", "3DGS+Ours* settings: scaling_lr 0.01, "
                         "densification interval 500, grad threshold 1e-4 (Sec. 4)",
                         ("opti-density-config",)),
    )


def method_filter():
    return _method(
        "opti3dgs-filter", PLAIN_3DGS,
        "Opti3DGS (arXiv 2503.14475) 'Ours-Filtering' (Table 2) native port: frequency "
        "modulation on default 3DGS settings; paper fidelity unverified.",
        ContributionSpec("default-3dgs-settings", "Default 3DGS density settings (Table 2)",
                         ("opti-density-config",)),
    )
