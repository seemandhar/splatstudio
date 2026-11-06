"""MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks (CVPR 2025,
arXiv 2412.20522) — native SplatStudio port of github.com/kaikai23/MaskGaussian @ 2f581d0 with
its masked CUDA rasterizer (kaikai23/mask-diff-gaussian-rasterization @ 540dc11) vendored.

Registered methods
  maskgaussian           Ours-alpha: lambda_m = 0.1 on steps 19000..20000 (per-scene tables)
  maskgaussian-beta      Ours-beta:  lambda_m = 0.0005 on 0..30000 (official code default)
  maskgaussian-gamma     Ours-gamma: lambda_m = 0.001 on 0..30000
  maskgaussian-opacitymask  paper ablation "Mask x opacity" (beta loss; mask multiplies opacity)
  maskgaussian-exactgrad  Ours-alpha with the exact supplementary mask gradient (kernel built
                          with -DMASKGS_EXACT_ACCUM); not the released code
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPABILITIES, FIELD
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2412.20522", "https://github.com/kaikai23/MaskGaussian",
             "https://github.com/kaikai23/mask-diff-gaussian-rasterization",
             "https://maskgaussian.github.io/")
MASK_LR = 0.01  # arguments/__init__.py mask_lr
SETTINGS = {  # paper §4.1 / run_all.py
    "alpha": {"lambda_mask": 0.1, "mask_from": 19000, "mask_until": 20000},
    "beta": {"lambda_mask": 0.0005, "mask_from": 0, "mask_until": 30000},
    "gamma": {"lambda_mask": 0.001, "mask_from": 0, "mask_until": 30000},
}


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_maskgaussian.probes:{fn}", desc, cuda)


def _build(name, setting, mask_mode, description, accum="official"):
    base = make_method(name)
    photometric = base.components[2]
    renderer_cfg = {**RENDERER_DEFAULTS, "background": list(RENDERER_DEFAULTS["background"]),
                    "score_offset": list(RENDERER_DEFAULTS["score_offset"]),
                    "mask_mode": mask_mode, "accum": accum}
    density_cfg = {**DENSITY_DEFAULTS, "score_offset": list(renderer_cfg["score_offset"]),
                   "tau": renderer_cfg["tau"]}
    comps = (
        ComponentSpec("maskgs-renderer", "renderer", "ss_maskgaussian.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      renderer_cfg),
        ComponentSpec("maskgs-density", "density", "ss_maskgaussian.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), density_cfg),
        photometric,
        ComponentSpec("maskgs-mask-loss", "loss", "ss_maskgaussian.losses:create_mask_loss",
                      ("render.rgb",), (), {**LOSS_DEFAULTS, **SETTINGS[setting]}),
    )
    probes = (
        _probe("maskgs-raster-exact", "raster_exact",
               "Vendored masked kernel vs float64 dense compositor of Eq. 3-4: image, dL/dM, "
               "dL/dopacity, dL/dcolour (exact build = autograd; official build = independent "
               "replica of backward.cu's accumulation, differs only where adjacent masks "
               "differ); M=1 equals unmasked blending", True),
        _probe("maskgs-unsampled-gradient", "unsampled_gradient",
               "Through renderer + Gumbel ST: unsampled (M=0) Gaussians leave the image "
               "unchanged yet get nonzero mask-score gradients; the same check on the naive "
               "mask-x-opacity renderer finds zero gradients and fails", True),
        _probe("maskgs-gumbel", "gumbel",
               "Hard Gumbel-Softmax: P(M=1) matches softmax(scores+offset)[0] over 20000 draws; "
               "straight-through gradient equals the soft-sample Jacobian; init p = sigmoid(9)"),
        _probe("maskgs-mask-loss", "mask_loss",
               "L = lambda (mean M)^2 inside [from, until] (inclusive), 0 outside; "
               "dL/dM = 2 lambda mean(M) / N"),
        _probe("maskgs-prune-schedule", "prune_schedule",
               "Never-sampled Gaussians removed in densify_and_prune and at s>=15000, s%1000==0 "
               "only; final step keeps one sample; mask scores copied by clone/split; optimizer "
               "state aligned"),
    )
    contribs = (
        ContributionSpec("masked-rasterization", "Mask in alpha blending with gradients for "
                         "masked Gaussians (official CUDA, §3.3-3.4)",
                         ("maskgs-raster-exact", "maskgs-unsampled-gradient")),
        ContributionSpec("probabilistic-masks", "Per-Gaussian 2-way mask scores sampled with "
                         "hard Gumbel-Softmax each step (§3.3)", ("maskgs-gumbel",)),
        ContributionSpec("mask-loss", "Squared mean-mask loss with step window (Eq. Lm, §4.1)",
                         ("maskgs-mask-loss",)),
        ContributionSpec("mask-pruning", "Remove Gaussians never sampled in 10 draws at "
                         "densification steps and every 1000 steps after (§3.5)",
                         ("maskgs-prune-schedule",)),
    )
    if mask_mode != "raster":
        contribs = tuple(c for c in contribs if c.name != "masked-rasterization")
        probes = tuple(p for p in probes if p.name not in ("maskgs-raster-exact",
                                                           "maskgs-unsampled-gradient"))
    return replace(base, description=description, components=comps,
                   fields=(FieldSpec(FIELD, (2,), MASK_LR, 0.0),),
                   probes=base.probes + probes, contributions=base.contributions + contribs,
                   citations=CITATIONS)


def method():
    return _build("maskgaussian", "alpha", "raster",
                  "MaskGaussian-alpha (CVPR 2025) native port: official masked rasterizer, "
                  "Gumbel masks, lambda_m=0.1 on 19k-20k, mask pruning; paper fidelity "
                  "unverified.")


def beta():
    return _build("maskgaussian-beta", "beta", "raster",
                  "MaskGaussian-beta (lambda_m=0.0005 on 0-30k) native port; paper fidelity "
                  "unverified.")


def gamma():
    return _build("maskgaussian-gamma", "gamma", "raster",
                  "MaskGaussian-gamma (lambda_m=0.001 on 0-30k) native port; paper fidelity "
                  "unverified.")


def opacitymask():
    return _build("maskgaussian-opacitymask", "beta", "opacity",
                  "Ablation control (paper 'Mask x opacity'): mask multiplies opacity, no "
                  "gradient to unsampled Gaussians; beta loss. Not MaskGaussian.")


def exactgrad():
    return _build("maskgaussian-exactgrad", "alpha", "raster",
                  "MaskGaussian-alpha with the exact supplementary dL/dM accumulation (the released "
                  "kernel uses the current Gaussian's mask in accum_rec); fidelity variant, not the "
                  "released code.", accum="exact")
