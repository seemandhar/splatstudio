"""Efficient Confidence-Aware Densification for 3D Gaussian Splatting (arXiv 2610.13541).

Native SplatStudio port written from the paper (no official code released) and CoMe's
confidence code (github.com/r4dl/CoMe); paper fidelity unverified.

Methods:
  confidence-densify  the paper: learned confidence (CoMe loss), Eq. pdf + SSIM-term source
                      sampling, max-blend-weight relocation, sampled positions, GT colour init,
                      within the 3DGS-MCMC fixed budget (default 1M) — custom CUDA statistics.
  cad-mcmc-control    3DGS-MCMC at the same budget (gsplat MCMCStrategy + lambda_reg 0.01 +
                      initial opacity 0.5), the paper's baseline row.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .losses import CONF_DEFAULTS, REG_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2610.13541", "https://arxiv.org/abs/2603.24725",
             "https://arxiv.org/abs/2404.09591")
CAPS = ("render.rgb", "render.depth.expected_z", "render.confidence",
        "gradient.screen.signed", "gradient.screen.absolute")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_confidence_densify.probes:{fn}", description, cuda)


PROBES = (
    _probe("confidence-loss", "confidence_loss",
           "Eq. confidence equals an independent formula, is minimal at C = beta / L_rgb, lowers "
           "gamma where the error exceeds beta; the plain-L_rgb ablation differs"),
    _probe("confidence-render", "confidence_render",
           "Rendered confidence = sum_i b_i exp(gamma_i) (= alpha at gamma=0, 2 alpha at "
           "gamma=log 2); RGB equals gsplat's SH rasterization", True),
    _probe("blend-stats-oracle", "blend_stats_oracle",
           "CUDA max_p alpha T and sum_p err alpha T equal a brute-force per-pixel compositing "
           "loop; sum-instead-of-max ablation differs", True),
    _probe("sampling-pdf", "sampling_pdf",
           "Eq. pdf vs an independent rank implementation, eta ramp, 1.2..0.2 confidence factor, "
           "80/20 split; opacity-sampling ablation picks other sources"),
    _probe("relocation-schedule", "relocation_schedule",
           "Share 10 % / 61 % / 14 % / ~1 % of 10 % at 10k/15k/20k/25k, w=0-only before 5k, "
           "lowest-w selection, exemption of the last relocation's primitives"),
    _probe("densify-step", "densify_step",
           "Growth +5 % to the cap, relocation every 400 it, MCMC opacity rule on sources and "
           "copies, copies sampled around (not on) their source, no position noise, optimizer "
           "alignment", True),
    _probe("color-init", "color_init",
           "A new primitive's SH colour in the first view with radius < 16 px equals the "
           "bilinear GT colour; moments zeroed; timeout after 400 it", True),
    _probe("renderer-gradient-cad", "renderer_gradient",
           "Renderer supplies finite signed and absolute screen gradients through the "
           "confidence loss and a full density step", True),
)

CONTRIBUTIONS = (
    ContributionSpec("learned-confidence", "Sec. 3 / CoMe: per-primitive confidence exp(gamma), "
                     "alpha-blended, confidence-weighted photometric loss (beta 0.075)",
                     ("confidence-loss", "confidence-render")),
    ContributionSpec("error-sampling", "Sec. 3.1: 80 % Eq. pdf (ranked |grad|_abs x confidence "
                     "factor, eta ramp) + 20 % SSIM term; one draw for growth and relocation",
                     ("sampling-pdf", "blend-stats-oracle", "densify-step")),
    ContributionSpec("max-weight-relocation", "Sec. 3.2: relocate lowest max blending weight "
                     "every 400 it with the share schedule (custom CUDA)",
                     ("relocation-schedule", "blend-stats-oracle", "densify-step")),
    ContributionSpec("position-sampling", "Sec. 3.3: MCMC opacity/scale rule, mean drawn from "
                     "the source Gaussian, no position noise", ("densify-step",)),
    ContributionSpec("color-init", "Sec. 3.3: GT colour lookup for new primitives",
                     ("color-init",)),
)


def method():
    base = make_method("confidence-densify")
    renderer = ComponentSpec("cad-renderer", "renderer",
                             "ss_confidence_densify.renderer:create_renderer",
                             ("state.gaussian3d", "camera.pinhole"), CAPS, {})
    density = ComponentSpec("cad-density", "density",
                            "ss_confidence_densify.density:create_density",
                            ("gradient.screen.absolute", "render.confidence"), (), dict(DEFAULTS))
    conf = ComponentSpec("confidence-photometric", "loss",
                         "ss_confidence_densify.losses:create_confidence_loss",
                         ("render.rgb", "render.confidence"), (), dict(CONF_DEFAULTS))
    reg = ComponentSpec("mcmc-reg", "loss", "ss_confidence_densify.losses:create_reg",
                        ("render.rgb",), (), dict(REG_DEFAULTS))
    keep = tuple(p for p in base.probes if p.name == "state-transitions")
    return replace(
        base,
        description="Efficient Confidence-Aware Densification for 3DGS (arXiv 2610.13541) "
                    "native port from the paper (no official code); paper fidelity unverified.",
        components=(renderer, density, conf, reg),
        fields=(FieldSpec("confidence", (1,), learning_rate=2.5e-4, initial_value=0.0),),
        probes=keep + PROBES,
        contributions=(base.contributions[0],) + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def mcmc_control():
    base = make_method("cad-mcmc-control")
    density = ComponentSpec("mcmc-density", "density",
                            "ss_confidence_densify.control:create_control_density", (), (),
                            {"cap_max": 1_000_000, "init_opacity": 0.5})
    reg = ComponentSpec("mcmc-reg", "loss", "ss_confidence_densify.losses:create_reg",
                        ("render.rgb",), (), {**REG_DEFAULTS, "stop": 10**9})
    return replace(
        base,
        description="3DGS-MCMC control for confidence-densify: built-in gsplat MCMCStrategy at "
                    "the same budget + lambda_reg 0.01 + initial opacity 0.5.",
        components=(base.components[0], density, base.components[2], reg),
        citations=("https://arxiv.org/abs/2404.09591",),
    )


def method_sep():
    """Iteration 1 (post-hoc, after the 30k benchmark): confidence trained by Eq. confidence on a
    detached L_rgb(p); geometry/colour optimised with the plain 3DGS-MCMC L_rgb. Everything else
    identical to `confidence-densify`."""
    base = method()
    comps = tuple(
        replace(c, config={**c.config, "weight_photometric": False})
        if c.name == "confidence-photometric" else c
        for c in base.components
    )
    return replace(base, name="confidence-densify-sep", components=comps,
                   description=base.description + " Iteration 1 (post-hoc): confidence does not "
                   "weight the photometric loss.")
