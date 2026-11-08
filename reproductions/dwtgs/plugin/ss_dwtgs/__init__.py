"""DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
(arXiv 2507.15690) — native SplatStudio port, built on the in-repo DropGaussian port as the paper
is built on the DropGaussian release. No official DWTGS code exists; losses follow Eq. 5-6 and
§V-A. `dwtgs-lf` is the paper's Table III "+LF (2-level)" ablation (no HF term).
"""

from dataclasses import replace

import ss_dropgaussian
from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec

from .losses import HF_DEFAULTS, LF_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2507.15690",
    "https://arxiv.org/abs/2504.00773",
    "https://github.com/DCVL-3D/DropGaussian_release",
    "https://github.com/VITA-Group/FSGS",
)


def _build(name, with_hf, description):
    base = ss_dropgaussian.method()
    components = []
    for c in base.components:
        if c.name == "dropgaussian-density":
            c = replace(c, factory="ss_dwtgs.density:create_density")
        components.append(c)
    components.append(ComponentSpec("dwtgs-lf", "loss", "ss_dwtgs.losses:create_lf",
                                    ("render.rgb",), (), dict(LF_DEFAULTS)))
    probes = [
        ProbeSpec("haar-dwt", "ss_dwtgs.probes:haar_dwt_probe",
                  "Eq. 2 averaging Haar (LL = block mean, §IV-B) vs an independent loop; reconstruction"),
        ProbeSpec("lf-loss", "ss_dwtgs.probes:lf_loss_probe",
                  "Eq. 5 multi-level LL L1 (N 2, lambda 0.5); blind to level-1 HH content"),
    ]
    contributions = [
        ContributionSpec("wavelet-lf-supervision",
                         "Multi-level (N = 2) Haar LL-subband (block-mean) L1 supervision, lambda_LL 0.5 "
                         "(§IV-B, Eq. 5)", ("haar-dwt", "lf-loss")),
    ]
    if with_hf:
        components.append(ComponentSpec("dwtgs-hf", "loss", "ss_dwtgs.losses:create_hf",
                                        ("render.rgb",), (), dict(HF_DEFAULTS)))
        probes += [
            ProbeSpec("hf-schedule", "ss_dwtgs.probes:hf_schedule_probe",
                      "HF term every 10 iterations, stopped after 5000; zero otherwise"),
            ProbeSpec("pseudo-poses", "ss_dwtgs.probes:pseudo_poses_probe",
                      "Novel views interpolated between training views (Fig. 2 note 3): slerp/lerp path"),
            ProbeSpec("hf-novel-view", "ss_dwtgs.probes:hf_novel_view_probe",
                      "HF term = mean|HH_1| of an independent novel-view render, with gradient",
                      True),
        ]
        contributions.append(
            ContributionSpec("wavelet-hf-sparsity",
                             "Self-supervised L1 sparsity of the level-1 HH subband of novel-view "
                             "views interpolated between training views (§IV-B, Eq. 6, Fig. 2; every 10 its, "
                             "until 5000)",
                             ("haar-dwt", "hf-schedule", "pseudo-poses", "hf-novel-view")))
    return replace(
        base,
        name=name,
        description=description,
        components=tuple(components),
        probes=base.probes + tuple(probes),
        contributions=base.contributions + tuple(contributions),
        citations=CITATIONS,
    )


def method():
    return _build("dwtgs", True,
                  "DWTGS native port on DropGaussian (wavelet LL supervision + novel-view HH "
                  "sparsity); no official code, paper fidelity unverified.")


def method_lf():
    return _build("dwtgs-lf", False,
                  "DWTGS ablation: DropGaussian + 2-level wavelet LL supervision only (paper "
                  "Table III); paper fidelity unverified.")
