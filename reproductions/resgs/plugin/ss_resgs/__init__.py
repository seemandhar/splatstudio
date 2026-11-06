"""ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery (Lyu et al.,
ICCV 2025, arXiv 2412.07494). Native SplatStudio port of the released 'Ours (AbsGS)' variant of
https://github.com/yanzhelyu/ResGS (train.py, scene/gaussian_model.py, arguments/__init__.py);
paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES, PYRAMID_CAPABILITY

CITATIONS = (
    "https://arxiv.org/abs/2412.07494",
    "https://github.com/yanzhelyu/ResGS",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_resgs.probes:{fn}", description, cuda)


PROBES = (
    _probe("resgs-residual-split", "residual_split_probe",
           "Eq. 4-5 against an independent formula: parent kept with opacity beta*o, child "
           "scale S/lambda_s, child opacity/rotation/SH = parent, whitened child offsets ~N(0,I), "
           "parent opacity moments zeroed, other moments kept, child moments zero, level + 1"),
    _probe("resgs-selection", "selection_probe",
           "Eq. 8 thresholds tau/alpha^(k-l) for l<k (alpha = 2^(1/3)) on a level x substage grid; "
           "selection at/below threshold; constant-threshold ablation differs"),
    _probe("resgs-schedule", "schedule_probe",
           "Stage/substage/warm-up/densify/opacity-reduction schedule equals an independent "
           "transcription of official train.py over 30k iterations, plus hand-checked anchors"),
    _probe("resgs-pyramid", "pyramid_probe",
           "Torch pyrDown equals cv2.pyrDown (odd sizes, two levels); loss supervises against the "
           "pyramid image at the rendered level and rejects size mismatches"),
    _probe("resgs-render-scale", "render_scale_probe",
           "Training renders at pyrDown size with K scaled by w'/W, h'/H equal to gsplat directly; "
           "no_grad evaluation render at full resolution", True),
    _probe("resgs-opacity-reduce", "opacity_reduce_probe",
           "AbsGS reduce_opacity: sigmoid(o) x 0.8 with zeroed moments; at 12000 prune < 0.005 "
           "precedes the reduction"),
    _probe("resgs-cycle", "cycle_probe",
           "Native runtime with a compressed schedule: densification sees pyramid levels 2->1->0, "
           "residual splits create level >= 2 Gaussians, finite full-resolution evaluation", True),
)
CONTRIBUTIONS = (
    ContributionSpec("residual-split", "Residual split densification (Sec. 3.2.1, Eq. 4-5)",
                     ("resgs-residual-split", "resgs-cycle")),
    ContributionSpec("image-pyramid", "Image pyramid as supervision, L=3 stages at 1/4, 1/2, 1 "
                     "resolution (Sec. 3.2.2, Eq. 6-7; code change_iter 2500/6000, warm-up 500)",
                     ("resgs-pyramid", "resgs-render-scale", "resgs-schedule", "resgs-cycle")),
    ContributionSpec("varying-threshold", "Coarse-Gaussian selection by varying gradient threshold "
                     "over K=3 substages per stage (Sec. 3.2.3, Eq. 8)",
                     ("resgs-selection", "resgs-schedule")),
    ContributionSpec("absgs-opacity-reduction", "AbsGS host: absolute gradients and extended "
                     "periodic opacity reduction (Sec. 4.1, supp.)",
                     ("resgs-opacity-reduce", "resgs-schedule")),
)


def method():
    base = make_method("resgs", absgrad=True)
    renderer = ComponentSpec(
        "resgs-renderer", "renderer", "ss_resgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, {},
    )
    density = ComponentSpec(
        "resgs-density", "density", "ss_resgs.density:create_density",
        ("gradient.screen.absolute",), (), dict(DEFAULTS),
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_resgs.losses:create_photometric",
        ("render.rgb", PYRAMID_CAPABILITY), (), {"ssim_weight": 0.2, "ssim_padding": "same"},
    )
    return replace(
        base,
        description="ResGS (arXiv 2412.07494) native port of the released AbsGS variant "
        "(residual split, image pyramid, varying threshold); paper fidelity unverified.",
        components=(renderer, density, loss),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
