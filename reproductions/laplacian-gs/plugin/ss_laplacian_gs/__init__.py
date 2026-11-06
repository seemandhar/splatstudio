"""Laplacian-GS: Laplacian Frequency Hierarchies for Efficient 3D Gaussian Splatting Training
(Yang, Zhang et al., arXiv 2609.03334). Native SplatStudio port of the official code
https://github.com/SorenZhang574/Laplacian-GS (MIT; train_lap.py, render_pn.py,
laplacian/laplacian_torch.py) with its `taming` (3DGS-equivalent) backbone mapped onto gsplat.
Paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES, LEVEL_CAPABILITY, LEVEL_FIELD

CITATIONS = (
    "https://arxiv.org/abs/2609.03334",
    "https://github.com/SorenZhang574/Laplacian-GS",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_laplacian_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("lap-renderer-gradient", "lap_renderer_gradient",
           "Built-in renderer-gradient check at the active pyramid level: finite nonzero screen "
           "gradients reach the per-stage density statistics", True),
    _probe("lap-pyramid", "lap_pyramid",
           "Down/Up equal independent numpy reflect-pad 5x5 binomial conv (+bilinear, blur_up), "
           "pyramid size rule, Laplacian reconstruction identity"),
    _probe("lap-signed-residual", "lap_signed_residual",
           "Residual fields render signed colours without the +0.5 shift (R = B - 0.5 A on a "
           "single Gaussian with negative colour), clamped to [-1, 1]", True),
    _probe("lap-composition", "lap_composition",
           "Evaluation = Up(base@level1) + residual@level0 from raw gsplat calls; residual-stage "
           "training gives zero gradient to archived Gaussians and nonzero to active ones", True),
    _probe("lap-staging", "lap_staging",
           "Stage boundary: xyz-only inheritance (rho N), 3-NN scales, SH 0, opacity 0.1, fresh "
           "Adam (zero moments, step 0), position LR restart; archived rows never move and ADC "
           "never touches them"),
    _probe("lap-loss", "lap_loss",
           "Reconstructed-image supervision: L1 + 0.2 D-SSIM(same) against Down^l(GT)"),
)
CONTRIBUTIONS = (
    ContributionSpec("laplacian-hierarchy",
                     "Per-level Gaussian fields, signed residual renderer and image-domain "
                     "Laplacian composition (Sec. 3.2, Eq. 4-7)",
                     ("lap-pyramid", "lap-signed-residual", "lap-composition")),
    ContributionSpec("staged-archiving",
                     "Coarse-to-fine frequency-staged training with archiving and "
                     "reconstructed-image supervision (Sec. 3.3, Alg. 1)",
                     ("lap-composition", "lap-staging", "lap-loss")),
    ContributionSpec("xyz-inheritance",
                     "Next field inherits only xyz of the previous field, ratio rho (Sec. 3.3, "
                     "supp. B)", ("lap-staging",)),
)


def _build(name, iter_plan, densify_until=None, levels=2):
    base = make_method(name)
    renderer = ComponentSpec(
        "lap-renderer", "renderer", "ss_laplacian_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"levels": levels, "iter_plan": list(iter_plan), "blur_up": True,
         "base_background": "black"},
    )
    dcfg = {**DEFAULTS, "levels": levels, "iter_plan": list(iter_plan),
            "densify_until": densify_until, "lr": dict(DEFAULTS["lr"])}
    density = ComponentSpec(
        "lap-density", "density", "ss_laplacian_gs.density:create_density",
        ("gradient.screen.signed",), (), dcfg,
    )
    loss = ComponentSpec(
        "lap-photometric", "loss", "ss_laplacian_gs.losses:create_loss",
        ("render.rgb", LEVEL_CAPABILITY), (), {"ssim_weight": 0.2, "ssim_padding": "same"},
    )
    # The built-in photometric / renderer-gradient probes compare a full-resolution target to
    # the render; the base stage renders at the coarse pyramid level, so level-aware
    # equivalents replace them.
    swap = {"photometric-gradient": "lap-loss", "renderer-gradient": "lap-renderer-gradient"}
    base_probes = tuple(p for p in base.probes if p.name not in swap)
    base_contribs = tuple(
        replace(c, probes=tuple(swap.get(p, p) for p in c.probes)) for c in base.contributions
    )
    return replace(
        base,
        description="Laplacian-GS (arXiv 2609.03334) native port, Taming(3DGS)-equivalent "
        "backbone on gsplat; paper fidelity unverified.",
        components=(renderer, density, loss),
        fields=(FieldSpec(LEVEL_FIELD, (), 1e-12, float(levels - 1)),),
        probes=base_probes + PROBES,
        contributions=base_contribs + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    """Paper default: 2 levels, iteration split 10K/20K (supp. A, run_mipnerf360_1k_taming.sh)."""
    return _build("laplacian-gs", [10000, 20000])


def smoke_method():
    """Smoke-test-only schedule (1000/1000, densify until 900 per stage); never benchmarked."""
    return _build("laplacian-gs-smoke", [1000, 1000], densify_until=[900, 900])
