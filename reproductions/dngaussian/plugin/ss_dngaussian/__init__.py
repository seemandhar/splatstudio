"""DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth
Normalization (Li et al., CVPR 2024, arXiv 2403.06912) — native SplatStudio port.

Defaults are the official LLFF settings (github.com/Fictionarry/DNGaussian, scripts/run_llff.sh,
random initialization). ``dngaussian-3dgs`` is the same pipeline without the hard / soft depth
passes: the paper's "3DGS†" row (same hyper-parameters and neural colour renderer).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import PHOTO_DEFAULTS, REG_DEFAULTS
from .neural import PACK_WIDTH
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2403.06912",
    "https://github.com/Fictionarry/DNGaussian",
    "https://github.com/ashawkey/torch-ngp",
)

FIELDS = (
    # Inference cache of the Neural Color Renderer (written by the bake; never trained).
    FieldSpec("dng_geo", (64,), learning_rate=1e-3, initial_value=0.0),
    FieldSpec("dng_mlp", (PACK_WIDTH,), learning_rate=1e-3, initial_value=0.0),
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_dngaussian.probes:{fn}", desc, cuda)


def _build(name, depth, description):
    base = make_method(name)
    renderer = ComponentSpec("dngaussian-renderer", "renderer",
                             "ss_dngaussian.renderer:create_renderer",
                             ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                             dict(RENDERER_DEFAULTS))
    density = ComponentSpec("dngaussian-density", "density", "ss_dngaussian.density:create_density",
                            ("gradient.screen.signed",), (),
                            {**DENSITY_DEFAULTS, "hard_depth": depth, "soft_depth": depth})
    photometric = ComponentSpec("photometric", "loss", "ss_dngaussian.losses:create_photometric",
                                ("render.rgb",), (), dict(PHOTO_DEFAULTS))
    regularization = ComponentSpec("dngaussian-regularization", "loss",
                                   "ss_dngaussian.losses:create_regularization", ("render.rgb",),
                                   (), dict(REG_DEFAULTS))
    probes = (
        _probe("glnorm-formula", "glnorm_formula",
               "Eq. 7-9 local/global patch normalisation + tolerance vs an independent loop"),
        _probe("hard-depth", "hard_depth_probe",
               "Eq. 5: depth = tau G z on an analytic splat; gradients reach only the centres",
               True),
        _probe("soft-depth", "soft_depth_probe",
               "Eq. 6: depth = sigmoid(o) G z; gradients reach only the opacities", True),
        _probe("depth-pass-step", "depth_pass_step",
               "Density runs the hard then soft pass with their own optimizer steps; "
               "schedule (soft > 1000, smoothness > 3000)", True),
        _probe("after-backward-schedule", "after_backward_schedule",
               "Real before/after_backward over boundary steps: hard, soft > 1000, smoothness "
               "> 3000, near (s-1) % 25 after 2000, clean at 1/6000, densify, bake", True),
        _probe("neural-color", "neural_color",
               "CUDA hash grid and SH4 encoders vs independent torch formulas; colour gradient "
               "reaches the centres; baked inference colours equal live colours", True),
        _probe("official-density", "official_density",
               "Opacity-gated clone/split (> 0.1), prune < 0.01, near-camera and invisible "
               "pruning, xyz LR schedule", True),
        _probe("photo-pass-losses", "photo_pass_losses",
               "L1 + 0.2 D-SSIM(same) and shape/scale/opacity penalties vs independent formulas"),
        _probe("dng-renderer-gradient", "renderer_gradient",
               "Photometric render executes with the neural colours and supplies screen "
               "gradients", True),
    )
    contributions = (
        ContributionSpec("hard-depth-regularization",
                         "Hard depth (tau = 0.95, frozen s/q), separate centre update (§3.2)",
                         ("hard-depth", "depth-pass-step", "after-backward-schedule")),
        ContributionSpec("soft-depth-regularization",
                         "Soft depth (frozen mu/s/q), separate opacity update after 1000 (§3.2)",
                         ("soft-depth", "depth-pass-step", "after-backward-schedule")),
        ContributionSpec("global-local-depth-normalization",
                         "Patch-wise local + global normalised L2 with error tolerance (§3.3)",
                         ("glnorm-formula",)),
        ContributionSpec("neural-color-renderer",
                         "Hash grid + MLP colour with cached geometry feature at inference (§3.4)",
                         ("neural-color", "dng-renderer-gradient")),
        ContributionSpec("official-training-recipe",
                         "Random init, opacity-gated densification, near/invisible pruning, "
                         "penalties, LR schedule, black background (official code)",
                         ("official-density", "photo-pass-losses", "after-backward-schedule")),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    if not depth:
        keep = {"after-backward-schedule", "neural-color", "official-density", "photo-pass-losses", "dng-renderer-gradient"}
        probes = tuple(p for p in probes if p.name in keep)
        contributions = contributions[3:]
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric, regularization),
        probes=base_probes + probes,
        contributions=base_contribs + contributions,
        fields=FIELDS,
        citations=CITATIONS,
    )


def method():
    return _build("dngaussian", True,
                  "DNGaussian native port (hard/soft depth regularization with global-local depth "
                  "normalization, neural colour renderer); paper fidelity unverified.")


def method_3dgs():
    return _build("dngaussian-3dgs", False,
                  "DNGaussian pipeline without depth regularization (paper's 3DGS-dagger row: same "
                  "hyper-parameters and neural colour renderer); paper fidelity unverified.")
