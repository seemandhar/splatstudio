"""FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Gaussian Splatting with
Depth-Feature Consistency (Huang, Wu et al., AAAI 2025, arXiv 2501.04628) — native SplatStudio
port of the official code (github.com/yulunwu0108/FatesGS, commit a721d318).

``fatesgs``       full method: 2DGS backbone + L_r (rank) + L_s (smooth) + L_f (feature).
``fatesgs-2dgs``  same pipeline without the three proposed losses (paper ablation row "none" /
                  the paper's ablation row 'none'; not its separately-run 2DGS row).
Meshes / Chamfer distance: ``fatesgs-dtu-eval`` (ss_fatesgs.dtu) after ss-train.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import FEAT_DEFAULTS, PHOTO_DEFAULTS, RANK_DEFAULTS, REG_DEFAULTS, SMOOTH_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2501.04628",
    "https://github.com/yulunwu0108/FatesGS",
    "https://github.com/hbb1/2d-gaussian-splatting",
)
DEPTH = ("render.depth.expected_z",)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_fatesgs.probes:{fn}", desc, cuda)


def _build(name, priors, description):
    base = make_method(name)
    comps = [
        ComponentSpec("fatesgs-surfel", "renderer", "ss_fatesgs.renderer:create_renderer",
                      ("state.surfel2d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("fatesgs-density", "density", "ss_fatesgs.density:create_density",
                      ("gradient.surfel",), (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("photometric", "loss", "ss_fatesgs.losses:create_photometric",
                      ("render.rgb",), (), dict(PHOTO_DEFAULTS)),
        ComponentSpec("surfel-reg", "loss", "ss_fatesgs.losses:create_surfel_reg",
                      ("render.normal.world", "render.normal_from_depth.world",
                       "render.distortion.2dgs"), (), dict(REG_DEFAULTS)),
    ]
    if priors:
        comps += [
            ComponentSpec("depth-rank", "loss", "ss_fatesgs.losses:create_rank", DEPTH, (),
                          dict(RANK_DEFAULTS)),
            ComponentSpec("depth-smooth", "loss", "ss_fatesgs.losses:create_smooth", DEPTH, (),
                          dict(SMOOTH_DEFAULTS)),
            ComponentSpec("feature", "loss", "ss_fatesgs.losses:create_feature", DEPTH, (),
                          dict(FEAT_DEFAULTS)),
        ]
    probes = [
        _probe("surfel-renderer", "surfel_renderer",
               "Official 2DGS CUDA: analytic plane depth/normal, alpha-normalised depth, "
               "gradients reach means2d/opacities", True),
        _probe("density-recipe", "density_recipe",
               "Clone/split-in-plane/prune(<0.1, >20px, >0.1 extent)/reset(min 0.1, every 1000)"
               " boundaries and the 30000-step exponential xyz LR", True),
        _probe("surfel-reg-formula", "surfel_reg_formula",
               "L_c with same-padding SSIM and L_d/L_n schedules (>3000, >7000) vs formulas"),
    ]
    contribs = [
        ContributionSpec("2dgs-backbone", "2DGS rasterizer, losses and density control of the "
                         "official code (black background, 15k steps)",
                         ("surfel-renderer", "density-recipe", "surfel-reg-formula")),
    ]
    if priors:
        probes += [
            _probe("rank-formula", "rank_formula",
                   "Eq. 8 vs an independent per-pair loop; order-consistent depth -> m-hinge "
                   "only, reversed depth -> large loss"),
            _probe("smooth-formula", "smooth_formula",
                   "Eq. 9 vs an independent loop; edges in the mono depth switch the term off"),
            _probe("feature-consistency", "feature_consistency",
                   "Eq. 14: true depth gives lower L_f than shifted depth on a synthetic "
                   "two-view scene; official visibility bookkeeping vs paper z-buffer"),
        ]
        contribs += [
            ContributionSpec("depth-ranking", "Patch-based monocular depth ranking loss (Eq. 8)",
                             ("rank-formula",)),
            ContributionSpec("depth-smoothness", "Edge-aware depth smoothness loss (Eq. 9)",
                             ("smooth-formula",)),
            ContributionSpec("feature-alignment",
                             "Multi-level multi-view Vis-MVSNet feature loss with visibility "
                             "(Eq. 10-15)", ("feature-consistency",)),
        ]
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, description=description, components=tuple(comps),
        probes=base_probes + tuple(probes), contributions=base_contribs + tuple(contribs),
        citations=CITATIONS, representation="surfel2d",
    )


def method():
    return _build("fatesgs", True,
                  "FatesGS native port (2DGS + depth ranking/smoothness + multi-view feature "
                  "loss); paper fidelity unverified.")


def method_2dgs():
    return _build("fatesgs-2dgs", False,
                  "FatesGS pipeline without L_r/L_s/L_f (2DGS control, same init and TSDF); "
                  "paper fidelity unverified.")
