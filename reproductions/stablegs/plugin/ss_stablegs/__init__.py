"""StableGS: A Floater-Free Framework for 3D Gaussian Splatting (arXiv 2503.18458) — native
SplatStudio port of the official code (github.com/MooreThreads/StableGS @ 397a9c00, run.sh block
1 = Table 1 "StableGS w/o L_prior"). The release lacks `gaussian_renderer/` and `arguments/`;
the renderer is reconstructed from its vendored RaDe-GS@ec9199f rasterizer and the Taming-3DGS
pipeline it forks (see README.md). The DUSt3R depth prior (C3, full "StableGS" row) is not built.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .renderer import CAPABILITIES, PAIR_CAPABILITY
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2503.18458",
    "https://github.com/MooreThreads/StableGS/tree/397a9c00cd322b769801e24b00968cf4f2e24001",
    "https://github.com/HKUST-SAIL/RaDe-GS/tree/ec9199f",
    "https://arxiv.org/abs/2406.15643",
)
CODE_COMMIT = "397a9c00cd322b769801e24b00968cf4f2e24001"
ALPHA_AUX_INIT = 4.59511985013459  # inverse_sigmoid(0.99), gaussian_model.py:195
OPACITY_LR = 0.025  # opacity_lr (Taming default); _opacity1 uses the same lr


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_stablegs.probes:{fn}", desc, cuda)


def _density_config():
    cfg = {k: v for k, v in DENSITY_DEFAULTS.items() if k not in ("pairs", "data", "downscale")}
    return cfg


def build(name="stablegs", consistency_form="official"):
    base = make_method(name)
    pair = (PAIR_CAPABILITY,)
    comps = (
        ComponentSpec("stablegs-raster", "renderer", "ss_stablegs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("stablegs-density", "density", "ss_stablegs.density:create_density",
                      (PAIR_CAPABILITY, "gradient.screen.gof"), (), _density_config()),
        ComponentSpec("pair-color", "loss", "ss_stablegs.losses:create_pair_color", pair, (),
                      {"lambda_dssim": 0.2, "ssim_views": "official"}),
        ComponentSpec("geometric-color", "loss", "ss_stablegs.losses:create_geometric_color",
                      pair, (), {"lambda_s": 1.0}),
        ComponentSpec("depth-normal", "loss", "ss_stablegs.losses:create_depth_normal",
                      pair + ("render.depth.median", "render.normal.camera"), (),
                      {"lambda_depth_normal": 0.05, "depth_ratio": 0.6}),
        ComponentSpec("depth-consistency", "loss", "ss_stablegs.losses:create_depth_consistency",
                      pair + ("render.depth",), (),
                      {"lambda_consis": 0.05, "form": consistency_form}),
    )
    probes = (
        _probe("dual-opacity-paths", "dual_opacity_paths",
               "Appearance path uses α·α', geometric path α only: α'=1 reproduces the geometric "
               "render, α'=0.5 halves a lone splat's alpha; geometric-path losses give α' no "
               "gradient, appearance-path losses do", True),
        _probe("depth-consistency-formula", "depth_consistency_formula",
               "Official one-sided loss vs an independent per-pixel loop; zero for a consistent "
               "fronto-parallel plane pair, positive with a floater; paper form two-sided"),
        _probe("pair-selection", "pair_selection",
               "Covisibility > 30, rotation in [16, 60] deg and homography-inlier < 0.8 each "
               "reject the constructed failing pair and keep the good one"),
        _probe("pair-stack", "pair_stack",
               "Stack = P + |P| inverse-frequency random pairs; P membership flag; "
               "dual/geometric switch after step 15000", True),
        _probe("loss-schedule", "loss_schedule",
               "Colour loss equals the official formula (duplicated view-0 SSIM); geometric "
               "losses are exact zeros up to 15000 and active after", True),
        _probe("gof-qualifier", "gof_qualifier",
               "GOF abs-gradient quantile qualifies Gaussians with cancelling signed gradients; "
               "gof=false removes them"),
    )
    contribs = (
        ContributionSpec("dual-opacity", "Dual Opacity GS: geometric path (α) for depth "
                         "regularization, appearance path (α·α') for rendering (Sec. 3.3)",
                         ("dual-opacity-paths", "loss-schedule")),
        ContributionSpec("depth-consistency", "Self-supervised multi-view depth consistency on "
                         "selected pairs (Sec. 3.4, official one-sided form)",
                         ("depth-consistency-formula", "pair-selection", "pair-stack")),
        ContributionSpec("training-pipeline", "Taming-3DGS budgeted score densification with the "
                         "GOF qualifier, RaDe-GS depth-normal loss on the geometric path "
                         "(official run.sh)", ("gof-qualifier", "loss-schedule")),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base,
        description="StableGS native port (official StableGS@397a9c00 w/o DUSt3R prior: dual "
                    "opacity, depth consistency, RaDe-GS rasterizer, Taming pipeline); paper "
                    "fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        fields=(FieldSpec("opacity_aux", (), learning_rate=OPACITY_LR,
                          initial_value=ALPHA_AUX_INIT),),
        citations=CITATIONS,
    )


def method():
    return build("stablegs", "official")


def method_paper_consistency():
    """Variant with the paper's two-sided Eq. 4 consistency loss (ablation, not the default)."""
    return build("stablegs-paper-consis", "paper")
