"""WE-GS (Wang, Wang, Qi; arXiv 2406.02407) -- native SplatStudio port from the paper text, with
values the paper omits taken from the authors' released WE-GS inference model
(github.com/yuzewang1998/takinglangsplatw, ``in_the_wild_renderer``; training code unreleased).

The encoder and G_phi live in ``ss_wegs.bank.ACTIVE`` (owned by ``wegs-density``) and are
exported to ``density_state["bank"]``. Evaluate with ``python -m ss_wegs.evaluate --run RUN``;
plain ``ss-eval`` has no bank and refuses to render.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import MASK_REG_DEFAULTS, MASKED_DEFAULTS, SH_REG_DEFAULTS, TS_REG_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2406.02407",
             "https://yuzewang1998.github.io/we-gs.github.io/",
             "https://github.com/yuzewang1998/takinglangsplatw")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_wegs.probes:{fn}", desc, cuda)


def _loss(name, fn, cfg):
    return ComponentSpec(name, "loss", f"ss_wegs.losses:{fn}", ("render.rgb",), (), dict(cfg))


def method(name="we-gs"):
    base = make_method(name)
    comps = (
        ComponentSpec("wegs-renderer", "renderer", "ss_wegs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("wegs-density", "density", "ss_wegs.density:create_density",
                      DENSITY_REQUIRES, (), dict(DENSITY_DEFAULTS)),
        _loss("wegs-masked-photometric", "create_masked_photometric", MASKED_DEFAULTS),
        _loss("wegs-mask-reg", "create_mask_reg", MASK_REG_DEFAULTS),
        _loss("wegs-sh-reg", "create_sh_reg", SH_REG_DEFAULTS),
        _loss("wegs-transient-opacity-reg", "create_transient_reg", TS_REG_DEFAULTS),
    )
    probes = (
        _probe("sh-transfer", "sh_transfer",
               "Rendered image equals gsplat on sh + G_phi(sh, gamma(mu), l) computed "
               "independently; l changes the render; grads reach G_phi and sh (Eq. 9-10)", True),
        _probe("attention-encoder", "attention_encoder",
               "Eq. 15-18 recomputed independently; M in [0,1] at 1/4 of the 256 input; M "
               "changes l_k; output dim 16"),
        _probe("masked-loss", "masked_loss",
               "Eq. 19: no gradient on rendered pixels where M = 0, M receives gradient; "
               "Eq. 20 equals lambda * sum (1-M)^2 at W'xH'; D-SSIM term gives M no gradient"),
        _probe("sh-reg", "sh_reg", "Eq. 13 equals lambda * mean_j ||Delta sh_j|| and has a "
               "gradient"),
        _probe("transient-opacity-reg", "transient_opacity_reg",
               "Eq. 21: opacity gradient only for visible Gaussians projecting where M < eps",
               True),
        _probe("bank-training", "bank_training",
               "One step moves encoder, U-Net and G_phi with lrs 1e-4/1e-4/1e-5; 3DGS density "
               "and position-lr schedule", True),
        _probe("feedforward-eval", "feedforward_eval",
               "Evaluation encodes each test image itself: two test images with different "
               "appearance give different renders of the same pose", True),
    )
    contribs = (
        ContributionSpec("residual-sh-transfer",
                         "Per-image residual SH from G_phi(sh, mu, l_k) + L_regSH (§4.1)",
                         ("sh-transfer", "sh-reg", "bank-training")),
        ContributionSpec("attention-mask-encoder",
                         "Spatial-attention encoder: CAM + U-Net mask shared with the "
                         "appearance encoder (§4.2)", ("attention-encoder", "bank-training")),
        ContributionSpec("masked-photometric", "Static-mask photometric loss + L_regM (§4.3)",
                         ("masked-loss",)),
        ContributionSpec("transient-opacity-reg", "L_regTS on Gaussians in transient regions "
                         "(§4.3)", ("transient-opacity-reg",)),
        ContributionSpec("feedforward-test-appearance", "Test appearance encoded from the test "
                         "image (§5.3 protocol)", ("feedforward-eval",)),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description="WE-GS native port (paper text + released WE-GS inference model for "
                    "omitted values; training code unreleased, loss reductions/eps assumed); "
                    "evaluate with ss_wegs.evaluate; paper fidelity unverified.",
        components=comps, probes=base_probes + probes,
        contributions=base_contribs + contribs, citations=CITATIONS)
