"""HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering (Song, Lin, Lei, Liu,
Daniilidis; arXiv 2412.01823) — native SplatStudio port of the official code
(github.com/TimSong412/HDGS, main@b13a359 + texture@01c6fd4) with both CUDA rasterizers
vendored (csrc/).

One run = stage 1 (2DGS + per-ray k-buffer sort + frustum sampling, 40000 its) -> Fisher pruning
(10 %) -> stage 2 (per-surfel textures, geometry frozen, 20000 its): ``--steps 60000``.

``hdgs``         Mip-NeRF 360 outdoor configuration (tex_res 200, k 16).
``hdgs-indoor``  Mip-NeRF 360 indoor configuration (tex_res 500 per official run_360.py; the
                 paper text says 400).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import GEOMETRY_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .texture import FLAG

CITATIONS = ("https://arxiv.org/abs/2412.01823", "https://github.com/TimSong412/HDGS")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_hdgs.probes:{fn}", desc, cuda)


def _build(name, tex_res, description):
    base = make_method("gsplat")
    comps = (
        ComponentSpec("hdgs-raster", "renderer", "ss_hdgs.renderer:create_renderer",
                      ("state.surfel2d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "tex_res": tex_res}),
        ComponentSpec("hdgs-density", "density", "ss_hdgs.density:create_density",
                      ("gradient.hdgs",), (), {**DENSITY_DEFAULTS, "tex_res": tex_res}),
        base.components[2],  # photometric: 0.8 L1 + 0.2 D-SSIM (L_c)
        ComponentSpec("hdgs-geometry", "loss", "ss_hdgs.losses:create_geometry",
                      ("render.normal.world", "render.normal_from_depth.world",
                       "render.distortion.hdgs"), (), dict(GEOMETRY_DEFAULTS)),
    )
    probes = (
        _probe("raster-vs-reference", "raster_vs_reference",
               "Stage-1 official kernel (k-buffer per-ray sort, tile-centre depth keys, 5-sample "
               "frustum G) forward + grads vs an independent dense torch emulation; k=16 and an "
               "overflowing k=4; scene sensitive to sort and frustum", True),
        _probe("texture-vs-reference", "texture_vs_reference",
               "Textured kernel forward + texture/SH-rest grads vs dense torch reference "
               "(bilinear texel at (u/4.5+1)/2 (W-1), SH at the intersection direction)", True),
        _probe("fisher-estimator", "fisher_estimator",
               "Hutchinson diag-F (means, scales) vs exact per-pixel Jacobian sum; 10 % lowest "
               "log-score pruned", True),
        _probe("stage-schedule", "stage_schedule",
               "xyz expon LR (30k), stage-2 freeze (xyz/scale/rot/opacity/DC lr 0, SH rest after "
               "1000), texture StepLR(8000,0.1), regularisers off in stage 2, texture sizes"),
        _probe("density-recipe", "density_recipe",
               "Official 2DGS clone/split(/1.6, in-plane)/cull/world-size prune, stats reset, "
               "optimizer alignment"),
    )
    contribs = (
        ContributionSpec("per-ray-sort", "Per-ray k-buffer sort (k=16) + per-tile centre-depth "
                         "sort (§3.3)", ("raster-vs-reference",)),
        ContributionSpec("frustum-sampling", "5-sample frustum average of G, w=0.2 (§3.2)",
                         ("raster-vs-reference",)),
        ContributionSpec("fisher-pruning", "Prune 10 % lowest Fisher sensitivity after stage 1 "
                         "(§3.3; Hutchinson estimate, code unreleased)", ("fisher-estimator",)),
        ContributionSpec("per-surfel-texture", "Bilinear per-surfel texture DC + SH at the exact "
                         "intersection direction (§3.4)", ("texture-vs-reference",)),
        ContributionSpec("two-stage-optimisation", "Stage 1 2DGS (40k) then texture fine-tune "
                         "with frozen geometry (20k) (§3.5)",
                         ("stage-schedule", "density-recipe")),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name, description=description, components=comps,
        probes=base_probes + probes, contributions=base_contribs + contribs,
        fields=(FieldSpec(FLAG, (1,), learning_rate=1e-12, initial_value=0.0),),
        citations=CITATIONS, representation="surfel2d")


def method():
    return _build("hdgs", 200, "HDGS native port (official CUDA rasterizers), Mip-NeRF 360 "
                  "outdoor configuration (tex_res 200); paper fidelity unverified.")


def method_indoor():
    return _build("hdgs-indoor", 500, "HDGS native port (official CUDA rasterizers), Mip-NeRF "
                  "360 indoor configuration (tex_res 500, official run_360.py); paper fidelity "
                  "unverified.")
