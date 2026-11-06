"""RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS (Fu et al., ICCV
2025, arXiv 2506.02751) — native SplatStudio port of the official 3DGS-based code
(github.com/fcyycf/RobustSplat @ a130281).

* ``robustsplat``       full model: delayed Gaussian growth (C1) + scale-cascaded mask
                        bootstrapping (C2) + DINOv2 mask MLP with residual / feature-cosine
                        supervision and early regularisation.
* ``robustsplat-nodg``  ablation "3DGS + Mask + MB" (Table 3): 3DGS densification schedule
                        (500 < t < 15000, opacity reset every 3000 from 3000).
* ``robustsplat-nomb``  ablation "3DGS + Mask + DG": high-resolution supervision throughout.

Needs a scene prepared by ``reproductions/robustsplat/tools/prepare_robustsplat.py`` (coarse
views) and the DINOv2 ViT-S/14-reg weights at ``~/models/dinov2``.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES
from .losses import MLP_DEFAULTS, PHOTO_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2506.02751",
    "https://github.com/fcyycf/RobustSplat",
)
P = "ss_robustsplat.probes:"


def _build(name, description, density_overrides=None, mlp_overrides=None):
    base = make_method(name)
    renderer = ComponentSpec(
        "rs-renderer", "renderer", "ss_robustsplat.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        dict(RENDERER_DEFAULTS))
    density = ComponentSpec(
        "rs-density", "density", "ss_robustsplat.density:create_density",
        tuple(sorted(REQUIRES)), (), {**DENSITY_DEFAULTS, **(density_overrides or {})})
    photo = ComponentSpec(
        "rs-masked-photometric", "loss", "ss_robustsplat.losses:create_photometric",
        ("render.rgb",), (), dict(PHOTO_DEFAULTS))
    mlp_cfg = {**MLP_DEFAULTS, **(mlp_overrides or {})}
    if density_overrides and "reset_start" in density_overrides:
        mlp_cfg["reset_iter"] = density_overrides["reset_start"]   # official shares reset_iter
    mlp = ComponentSpec(
        "rs-mask-mlp", "loss", "ss_robustsplat.losses:create_mask_mlp", ("render.rgb",), (),
        mlp_cfg)
    keep = {"state-transitions", "renderer-gradient"}
    probes = [p for p in base.probes if p.name in keep] + [
        ProbeSpec("delayed-growth", P + "delayed_growth_probe",
                  "rs-density refine/reset step sets and no size pruning on a synthetic scene "
                  "with forced large gradients (C1)"),
        ProbeSpec("bootstrap-scale", P + "bootstrap_probe",
                  "Supervision grid / residual resolution switch at bootstrap_iter, coarse "
                  "residual from the coarse render (C2)", True),
        ProbeSpec("mask-supervision", P + "mask_supervision_probe",
                  "Residual masks + histogram, Eq. 4 cosine map, hinge, e^{-i/beta} vs NumPy"),
        ProbeSpec("masked-photometric", P + "masked_photometric_probe",
                  "Threshold + 7x7 erosion vs NumPy; transient blob receives no gradient; "
                  "MLP equivalence to upsample-then-MLP"),
        ProbeSpec("mlp-schedule", P + "mlp_schedule_probe",
                  "MLP not trained before 500 / in the 300-step post-reset windows"),
        ProbeSpec("dino-parity", P + "dino_parity_probe",
                  "DINOv2 ViT-S/14-reg forward_features vs timm with the same weights", True),
        ProbeSpec("pipeline-step", P + "pipeline_probe",
                  "Renderer -> masked photometric -> mask MLP -> density on a synthetic scene",
                  True),
    ]
    contributions = [c for c in base.contributions
                     if c.name in {"state-lifecycle", "rasterizer-statistics"}] + [
        ContributionSpec("delayed-gaussian-growth",
                         "Densification 10000 < t < 20000, opacity reset only at 15000/18000, "
                         "no size pruning (§4.3, supp.)" if name != "robustsplat-nodg" else
                         "Ablation: 3DGS densification schedule (no delayed growth)",
                         ("delayed-growth", "pipeline-step")),
        ContributionSpec("scale-cascaded-mask-bootstrapping",
                         "Low-resolution (images_{4f} residual, 224² DINOv2) mask supervision "
                         "before bootstrap_iter, high resolution (504²) after (§4.4)"
                         if name != "robustsplat-nomb" else
                         "Ablation: high-resolution supervision throughout (no bootstrapping)",
                         ("bootstrap-scale", "pipeline-step")),
        ContributionSpec("mask-mlp-supervision",
                         "DINOv2 mask MLP gating L1 + D-SSIM; residual hinge (tau .6/.8) + "
                         "feature-cosine supervision (Eq. 3-6, supp. S1)",
                         ("mask-supervision", "masked-photometric", "dino-parity",
                          "pipeline-step")),
        ContributionSpec("early-mask-regularisation",
                         "lambda_reg e^{-i/2000} mean(1 - M), MLP training paused after resets "
                         "(Eq. 7-8)", ("mask-supervision", "mlp-schedule")),
    ]
    components = (renderer, density, photo, mlp)
    return replace(base, description=description, components=components,
                   probes=tuple(probes), contributions=tuple(contributions),
                   citations=CITATIONS)


def method():
    return _build("robustsplat",
                  "RobustSplat native port (official code numerics: delayed growth + "
                  "scale-cascaded DINOv2 mask bootstrapping); paper fidelity unverified.")


def method_nodg():
    return _build("robustsplat-nodg",
                  "RobustSplat ablation 3DGS+Mask+MB (3DGS densification schedule); paper "
                  "fidelity unverified.",
                  density_overrides={"refine_start": 500, "refine_stop": 15000,
                                     "reset_start": 3000})


def method_nomb():
    return _build("robustsplat-nomb",
                  "RobustSplat ablation 3DGS+Mask+DG (no mask bootstrapping); paper fidelity "
                  "unverified.", mlp_overrides={"bootstrap": False})
