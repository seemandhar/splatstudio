"""VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting (Nowak, Jarosz, Chin; arXiv 2501.17978)
as a native SplatStudio method. No official code was released; implemented from the paper."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .loss import DEFAULTS as LOSS_DEFAULTS

CAPS = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")


def _method(name, variant):
    base = make_method(name)
    renderer = ComponentSpec("vod-renderer", "renderer", "ss_vod3dgs.renderer:create_renderer",
                             ("state.gaussian3d", "camera.pinhole"), CAPS, {})
    density = ComponentSpec("vod-density", "density", "ss_vod3dgs.density:create_density",
                            ("gradient.screen.signed",), (), {**DENSITY_DEFAULTS, "variant": variant})
    vc = ComponentSpec("vod-view-consistency", "loss", "ss_vod3dgs.loss:create_vc_loss",
                       ("render.rgb",), (), {k: v for k, v in LOSS_DEFAULTS.items() if k != "covis_dir"})
    probes = (
        ProbeSpec("vod-opacity", "ss_vod3dgs.probes:vod_opacity_probe",
                  "Renderer opacity equals an independent sigma(gamma + w^T S w) with a full 3x3 "
                  "matrix; S changes the image and receives gradients", True),
        ProbeSpec("max-view-prune", "ss_vod3dgs.probes:max_view_prune_probe",
                  "Prune keeps a low-gamma Gaussian that is opaque from one training view (Eq. 10)"),
        ProbeSpec("eigen-reset", "ss_vod3dgs.probes:eigen_reset_probe",
                  "S reset keeps only the min (S) / max (L) eigen-component at reset steps (Eq. 11)"),
        ProbeSpec("vc-loss", "ss_vod3dgs.probes:vc_loss_probe",
                  "L_vc equals an independent Eq. 12 evaluation; second view sampled by covisibility"),
        ProbeSpec("opacity-lr", "ss_vod3dgs.probes:opacity_lr_probe",
                  "gamma and S learning rates equal lr_gamma/4 after initialization (section 5.1)"),
    )
    contributions = (
        ContributionSpec("view-dependent-opacity", "Opacity sigma(gamma + w^T S w), Eq. 9", ("vod-opacity",)),
        ContributionSpec("max-view-opacity-prune", "Prune on max training-view opacity, Eq. 10",
                         ("max-view-prune",)),
        ContributionSpec("eigen-opacity-reset", f"VoD-3DGS[{variant}] eigen reset of S, Eq. 11",
                         ("eigen-reset",)),
        ContributionSpec("view-consistency-loss", "L_vc with covisibility-weighted view pairs, Eq. 12",
                         ("vc-loss",)),
        ContributionSpec("opacity-learning-rates", "gamma and S at lr_gamma/4", ("opacity-lr",)),
    )
    return replace(
        base,
        components=(renderer, density, base.components[2], vc),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        fields=(FieldSpec("vod_S", (6,), learning_rate=0.05 / 4, initial_value=0.0),),
        description=(f"VoD-3DGS[{variant}] native port (no official code; SfM covisibility replaces "
                     "LightGlue matches); paper fidelity unverified"),
        citations=("https://arxiv.org/abs/2501.17978",),
    )


def method_s():
    return _method("vod3dgs-s", "S")


def method_l():
    return _method("vod3dgs-l", "L")
