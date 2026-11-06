"""Spec-Gaussian (Yang et al., NeurIPS 2024, arXiv 2402.15870) -- native SplatStudio port of the
official real-scene code path (github.com/ingra14m/Specular-Gaussians @0b40ce3, train.py with
``--is_real --asg_degree 12`` [``--is_indoor``]), which produced the paper's "Ours" rows.

Per-Gaussian ASG features live in the ``spec_feat`` field; the global decoder (Theta + Psi)
lives in ``ss_spec_gaussian.bank.ACTIVE`` during training and is packed into ``spec_blob`` for
ss-eval. Variants: ``spec-gaussian`` (outdoor decoder), ``spec-gaussian-indoor`` (indoor
decoder), and ``-c2f`` versions with the paper-text resolution coarse-to-fine schedule.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .asg import DEFAULTS as ASG_DEFAULTS
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import BLOB, BLOB_WIDTH, CAPABILITIES, FEAT
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2402.15870",
    "https://github.com/ingra14m/Specular-Gaussians",
)
# name -> (indoor decoder, resolution coarse-to-fine)
VARIANTS = {
    "spec-gaussian": (False, False),
    "spec-gaussian-indoor": (True, False),
    "spec-gaussian-c2f": (False, True),
    "spec-gaussian-indoor-c2f": (True, True),
}
# run_wo_anchor.sh: --is_indoor for these Mip-NeRF 360 scenes
INDOOR_SCENES = ("room", "counter", "kitchen", "bonsai")
OUTDOOR_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill")
FEATURE_LR = 0.0025  # gaussian_model.py: f_asg lr = feature_lr


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_spec_gaussian.probes:{fn}", desc, cuda)


def method(name="spec-gaussian"):
    indoor, c2f = VARIANTS[name]
    base = make_method(name, absgrad=True)
    asg = {**ASG_DEFAULTS, "indoor": indoor}
    renderer = ComponentSpec(
        "spec-asg-raster", "renderer", "ss_spec_gaussian.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        {**RENDERER_DEFAULTS, **asg, "c2f": c2f})
    density = ComponentSpec(
        "spec-density", "density", "ss_spec_gaussian.density:create_density",
        tuple(sorted(DENSITY_REQUIRES)), (), {**DENSITY_DEFAULTS, **asg})
    if c2f:
        loss = ComponentSpec("spec-photometric", "loss",
                             "ss_spec_gaussian.losses:create_photometric", ("render.rgb",), (),
                             {"ssim_weight": 0.2})
    else:
        loss = base.components[2]  # SplatStudio photometric, 0.8 L1 + 0.2 D-SSIM (unchanged)
    probes = (
        _probe("asg-basis", "asg_basis",
               "Fixed ASG axes equal an independent numpy port of init_predefined_omega(2,4) and "
               "are orthonormal; ASG encoding equals a float64 numpy formula; lambda != mu makes "
               "the response anisotropic (isotropic/SG basis fails)"),
        _probe("decoder-architecture", "decoder_architecture",
               "Theta Linear(12->32) and Psi (31->32->3 outdoor, 31->32->32->3 indoor, zero last "
               "bias) shapes; full decoder forward equals an independent numpy forward"),
        _probe("colour-composition", "colour_composition",
               "Rendered RGB equals gsplat rasterization of independently computed "
               "clamp_min(SH+0.5,0)+c_s; c_s is off (and spec_feat/decoder get no gradient) at "
               "step <= 3000 and on afterwards", True),
        _probe("decoder-blob", "decoder_blob",
               "Decoder pack/unpack is exact and the fresh-process (ss-eval) render from the "
               "checkpointed spec_blob equals the live-bank render; corrupt blob is rejected",
               True),
        _probe("absgrad-densify", "absgrad_densify",
               "Density requires absolute screen gradients with grow_grad2d 0.0005; renderer "
               "absgrad >= |signed grad| and differs where pixel gradients cancel", True),
        _probe("official-schedules", "official_schedules",
               "Decoder Adam lr linear 0.0025->0.000125 over 30000 steps; means lr rescaled to "
               "spatial_lr_scale 5; spec_feat lr 0.0025 init 0; decoder stepped once per step",
               True),
        _probe("c2f-resolution", "c2f_resolution",
               "Training render size follows s(i)=min(0.25+0.75 i/5000,1) in -c2f variants and "
               "is constant otherwise; spec-photometric resizes the target bilinearly", True),
    )
    contribs = (
        ContributionSpec(
            "asg-appearance-field",
            "ASG appearance field: per-Gaussian feature f, Theta(f)->{xi,lambda,mu}_8 on fixed "
            "anisotropic axes, decoder Psi(kappa, d, PE(d)) -> c_s, c = c_d(SH) + c_s "
            "(Sec. 3.2, Eq. 6-8; official real-scene configuration)",
            ("asg-basis", "decoder-architecture", "colour-composition", "decoder-blob")),
        ContributionSpec(
            "coarse-to-fine-absgrad",
            "Coarse-to-fine part 1: L1 (absolute) screen-gradient accumulation for densification "
            "with tau_g = 0.0005 (Sec. 3.3)", ("absgrad-densify",)),
        ContributionSpec(
            "coarse-to-fine-resolution",
            "Coarse-to-fine part 2: progressive training resolution 1/4 -> 1 over 5k iterations "
            f"(Sec. 3.3; {'enabled' if c2f else 'disabled as in official train.py'} in this "
            "variant)", ("c2f-resolution",)),
        ContributionSpec(
            "official-schedules",
            "Official optimisation schedule: decoder Adam linear decay, spatial_lr_scale 5, "
            "specular branch from iteration 3001", ("official-schedules",)),
    )
    # the base photometric probe targets the base loss; -c2f variants replace it (c2f probe)
    keep = ("state-transitions",) if c2f else ("state-transitions", "photometric-gradient")
    keep_c = ("state-lifecycle",) if c2f else ("state-lifecycle", "photometric-objective")
    base_probes = tuple(p for p in base.probes if p.name in keep)
    base_contribs = tuple(c for c in base.contributions if c.name in keep_c)
    scenes = INDOOR_SCENES if indoor else OUTDOOR_SCENES
    return replace(
        base, name=name,
        description=(f"Spec-Gaussian native port [{name}: {'indoor' if indoor else 'outdoor'} "
                     f"decoder (Mip-NeRF 360 {', '.join(scenes)}), resolution c2f "
                     f"{'on' if c2f else 'off'}]: ASG appearance field + abs-grad densification "
                     "per the official real-scene code; paper fidelity unverified."),
        components=(renderer, density, loss),
        fields=(FieldSpec(FEAT, (ASG_DEFAULTS["feat_dim"],), learning_rate=FEATURE_LR,
                          initial_value=0.0),
                # decoder weights (Theta + Psi) packed every step; no gradient, never optimised
                FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0)),
        probes=base_probes + probes, contributions=base_contribs + contribs,
        citations=CITATIONS,
    )


def method_indoor():
    return method("spec-gaussian-indoor")


def method_c2f():
    return method("spec-gaussian-c2f")


def method_indoor_c2f():
    return method("spec-gaussian-indoor-c2f")
