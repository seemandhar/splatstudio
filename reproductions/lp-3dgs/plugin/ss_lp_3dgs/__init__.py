"""LP-3DGS: Learning to Prune 3D Gaussian Splatting (NeurIPS 2024, arXiv 2405.18784) —
native SplatStudio port of github.com/ASU-ESIC-FAN-Lab/LP-3DGS @ b828063 (RadSplat-score
variant, `--prune_method rad_splat --use_importance_mask`).

Standard 3DGS to 19,500 steps; a per-Gaussian mask m (Gumbel-sigmoid of m * S, S = RadSplat max
contribution refreshed every 20 steps) multiplies opacity for 500 steps with a 5e-4 sparsity
loss; at 20,000 Gaussians with gs(m) < 0.5 are removed once and plain 3DGS fine-tunes to 30k.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .core import FIELD
from .density import LP_DEFAULTS, REQUIRES
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2405.18784", "https://github.com/ASU-ESIC-FAN-Lab/LP-3DGS")
MASK_LR = 0.05  # mask.py set_mask: AdamW lr 0.05, eps 1e-15


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_lp_3dgs.probes:{fn}", desc, cuda)


def method():
    base = make_method("lp-3dgs")
    stock_density = base.components[1]
    photometric = base.components[2]
    comps = (
        ComponentSpec("lp-renderer", "renderer", "ss_lp_3dgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("lp-density", "density", "ss_lp_3dgs.density:create_density",
                      tuple(sorted(REQUIRES)), (), {**stock_density.config, **LP_DEFAULTS}),
        photometric,
        ComponentSpec("lp-mask-loss", "loss", "ss_lp_3dgs.losses:create_mask_loss",
                      ("render.rgb",), (), dict(LOSS_DEFAULTS)),
    )
    probes = (
        _probe("lp-gumbel", "gumbel",
               "Gumbel-sigmoid = sigmoid((x + n)/tau) with the official logistic noise; "
               "P(gs(x) > 0.5) = sigmoid(x) over 2e5 draws; analytic derivative"),
        _probe("lp-mask-loss", "mask_loss_probe",
               "lambda * mean(sigmoid(m)) strictly inside (start, start + iters), exactly zero "
               "and no mask gradient outside; analytic gradient"),
        _probe("lp-score-kernel", "score_kernel",
               "CUDA max alpha*T over pixels equals an independent float64 per-pixel compositor "
               "(and differs from the sum)", True),
        _probe("lp-mask-gradient", "mask_gradient",
               "Renderer: phase train gives dL/dm equal to autograd of an independently built "
               "o * sigmoid((m S + n)/tau) graph; phases off/frozen give no mask gradient", True),
        _probe("lp-schedule", "schedule",
               "Density on a toy scene with a compressed window: mask reset, decay, score "
               "refresh steps, phases, one-time prune of exactly gs(m) < 0.5, optimiser "
               "alignment, trainable leaves", True),
    )
    contribs = (
        ContributionSpec("gumbel-sigmoid-mask", "Gumbel-sigmoid mask on the importance score "
                         "multiplied into opacity, gradients reach the mask (Eq. 7-8)",
                         ("lp-gumbel", "lp-mask-gradient")),
        ContributionSpec("mask-sparsity", "Mask sparsity loss 5e-4 * mean(sigmoid(m)) on the "
                         "mask window (Eq. 9-10, official code)", ("lp-mask-loss",)),
        ContributionSpec("radsplat-score", "RadSplat importance S_i = max over training rays "
                         "of alpha_i T_i, refreshed every 20 steps (Eq. 3, §4.1)",
                         ("lp-score-kernel",)),
        ContributionSpec("learn-then-prune", "500-step mask window at 19.5k, one-time pruning "
                         "at 20k, fine-tune to 30k (learned pruning ratio, §3.3/§4.1)",
                         ("lp-schedule",)),
    )
    return replace(
        base,
        description="LP-3DGS (RadSplat score) native port of the official code; paper "
                    "fidelity unverified.",
        components=comps,
        fields=(FieldSpec(FIELD, (), MASK_LR, 1.0),),
        probes=base.probes + probes,
        contributions=base.contributions + contribs,
        citations=CITATIONS,
    )
