"""SplatFields (Mihajlovic et al., ECCV 2024) native SplatStudio port; paper fidelity unverified.

Two methods:
* `splatfields`      — SplatFields3D (static): tri-plane CNN + MLP neural splat fields predict
                       every splat attribute from the anchors; mask + splat-norm losses; baked
                       into plain Gaussians at the last step.
* `splatfields-3dgs` — the official `--is_static` 3DGS run (stage 1): the paper's 3DGS row
                       and the point cloud that initialises SplatFields.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, STAGE1_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2409.11211",
    "https://github.com/markomih/SplatFields",
)

CAPS = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed", "render.alpha")


def _probe(name, description, cuda=False):
    return ProbeSpec(name, f"ss_splatfields.probes:{name.replace('-', '_')}", description, cuda)


def _mask_loss():
    return ComponentSpec("mask", "loss", "ss_splatfields.losses:create_mask", ("render.alpha",),
                         (), {"weight": 0.1})


def method():
    base = make_method("splatfields")
    renderer_spec, _, photometric = base.components
    renderer = replace(renderer_spec, name="splatfields-renderer",
                       factory="ss_splatfields.renderer:create_renderer",
                       provides=CAPS + ("render.splat_centres",), config={})
    density = ComponentSpec("splatfields-density", "density",
                            "ss_splatfields.density:create_density",
                            ("gradient.screen.signed",), (), dict(DEFAULTS))
    norm = ComponentSpec("splat-norm", "loss", "ss_splatfields.losses:create_norm",
                         ("render.splat_centres",), (), {"weight": 0.01})
    probes = (
        _probe("triplane-generator", "Tri-plane CNN: 3 x [16,160,160] planes from fixed 8x20x20 "
               "noise, official zero-init residuals, attention == independent softmax, feature "
               "sampling == independent bilinear (align_corners=False, zero padding)"),
        _probe("splat-fields", "GeneralMLP == independent re-implementation; p = x + D(x,f), "
               "s = S + base, sigmoid opacity/RGB, unit quats; gradients reach the networks "
               "and not the anchors"),
        _probe("renderer-uses-fields", "Training render depends on the network weights and "
               "not on the base sh0/opacities/quats", True),
        _probe("bake", "Network render == network-free render of the baked base fields", True),
        _probe("norm-loss", "lambda_3 * mean ||p_hat|| on the predicted centres; gradient on "
               "the deform MLP"),
        _probe("mask-loss", "lambda_2 * L1(clamp(alpha), GT alpha) with the GT alpha looked up "
               "by viewmat; raises for an unknown camera"),
        _probe("schedule", "Network LR 8e-4 -> 1.6e-6 (40k), densify >500 every 100 until "
               "39900 (no refine at total_steps), no opacity reset, bake at total_steps"),
    )
    contributions = (
        ContributionSpec(
            "neural-splat-fields",
            "C1 (§4 'Neural splat fields', supp. B.2): MLP fields f_Theta (deform residual), "
            "f_c, f_s, f_alpha, f_O predict every splat attribute from PE(anchor) + feature; "
            "networks discarded (baked) after training",
            ("splat-fields", "renderer-uses-fields", "bake", "schedule"),
        ),
        ContributionSpec(
            "triplane-cnn-prior",
            "C2 (§4 'Deep structural prior', Eq. 6, supp. B.2): three 2D CNN decoders map fixed "
            "noise to 160x160x16 tri-planes sampled at the anchors (deep image prior)",
            ("triplane-generator",),
        ),
        ContributionSpec(
            "splat-norm-regularisation",
            "C3 (§4, supp. Eq. A.2): lambda_3 = 0.01 * mean ||p_hat||_2",
            ("norm-loss",),
        ),
        ContributionSpec(
            "mask-loss",
            "Protocol (supp. Eq. A.2): lambda_2 = 0.1 L1 between rendered opacity and GT mask "
            "on object scenes",
            ("mask-loss",),
        ),
    )
    return replace(
        base,
        description=("SplatFields3D native port (tri-plane CNN + MLP neural splat fields, "
                     "splat-norm and mask losses, baked to plain 3DGS); static branch only; "
                     "paper fidelity unverified"),
        components=(renderer, density, photometric, _mask_loss(), norm),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method_3dgs():
    base = make_method("splatfields-3dgs")
    renderer_spec, _, photometric = base.components
    renderer = replace(renderer_spec, name="sf3dgs-renderer",
                       factory="ss_splatfields.renderer:create_plain_renderer",
                       provides=CAPS, config={})
    density = ComponentSpec("sf3dgs-density", "density",
                            "ss_splatfields.density:create_stage1_density",
                            ("gradient.screen.signed",), (), dict(STAGE1_DEFAULTS))
    probes = (
        _probe("mask-loss", "lambda_2 * L1(clamp(alpha), GT alpha) with the GT alpha looked up "
               "by viewmat; raises for an unknown camera"),
        _probe("stage1-schedule", "xyz LR 8e-4 -> 8e-6 over 30k then held; densify >500 every "
               "100, last at 39900 (none at total_steps); no opacity reset"),
    )
    contributions = (
        ContributionSpec(
            "official-3dgs-stage",
            "SplatFields' 3DGS baseline / stage 1 (train.py --is_static): 3DGS + mask loss, "
            "densification for the whole run, no opacity reset, official LRs",
            ("mask-loss", "stage1-schedule"),
        ),
    )
    return replace(
        base,
        description=("SplatFields stage-1 / baseline 3DGS protocol (mask loss, densify to the "
                     "end, no opacity reset); paper fidelity unverified"),
        components=(renderer, density, photometric, _mask_loss()),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )
