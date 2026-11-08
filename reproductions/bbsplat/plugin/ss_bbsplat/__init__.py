"""BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis
(Svitov, Morerio, Agapito, Del Bue; arXiv 2411.08508) — native SplatStudio port of the official
code (github.com/david-svitov/BBSplat @ cff8d6a) with its CUDA rasterizer
(diff-bbsplat-rasterization @ 2dbc336) vendored in csrc/ and JIT-built.

``bbsplat``       Mip-NeRF 360, paper Tab. 1 budget: cap 60K billboards (50K FPS SfM + 10K sky).
``bbsplat-tnt``   Tanks&Temples, paper Tab. 1 budget: cap 30K (20K FPS SfM + 10K sky).
``bbsplat-300k``  budget of the published scripts/train_all.sh for bicycle/truck/... (cap 300K,
                  290K SfM + 10K sky; corresponds to paper Tab. 2, no per-scene numbers).
All: 32000 iterations (30000 + 2000 SH-rest fine-tuning), black background, no depth-normal
regularisation (scripts), sigma = 100 (code default; paper text 500).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from . import texture as tex
from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import GEOMETRY_DEFAULTS, OPACITY_DEFAULTS, TEXTURE_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2411.08508", "https://github.com/david-svitov/BBSplat",
             "https://github.com/david-svitov/diff-bbsplat-rasterization")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_bbsplat.probes:{fn}", desc, cuda)


def _build(name, cap_max, max_read_points, description, **density_overrides):
    base = make_method("gsplat")
    comps = (
        ComponentSpec("bbsplat-raster", "renderer", "ss_bbsplat.renderer:create_renderer",
                      ("state.surfel2d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("bbsplat-mcmc", "density", "ss_bbsplat.density:create_density", (), (),
                      {**DENSITY_DEFAULTS, "cap_max": cap_max,
                       "max_read_points": max_read_points, **density_overrides}),
        base.components[2],  # photometric: 0.8 L1 + 0.2 D-SSIM (Eq. 10)
        ComponentSpec("bbsplat-texture-reg", "loss", "ss_bbsplat.losses:create_texture_reg",
                      ("render.impact.bbsplat",), (), dict(TEXTURE_DEFAULTS)),
        ComponentSpec("bbsplat-opacity-reg", "loss", "ss_bbsplat.losses:create_opacity_reg",
                      (), (), dict(OPACITY_DEFAULTS)),
        ComponentSpec("bbsplat-geometry", "loss", "ss_bbsplat.losses:create_geometry",
                      ("render.normal.world", "render.normal_from_depth.world",
                       "render.distortion.bbsplat"), (), dict(GEOMETRY_DEFAULTS)),
    )
    probes = (
        _probe("raster-vs-reference", "raster_vs_reference",
               "Official BBSplat kernel forward (rgb, alpha, depth, impact) and gradients (means, "
               "quats, scales, SH, alpha/colour texture logits) vs a dense float64 torch "
               "reference (ray-plane intersection, grid_sample bilinear textures)", True),
        _probe("texture-composition", "texture_composition",
               "Eq. 6: pixel = alpha * (T^RGB + SH colour); removing the texture changes it", True),
        _probe("texture-regularizer", "texture_regularizer",
               "Impact-weighted texture reg vs an independent loop formula; no gradient for "
               "invisible (impact 0) or saturated (impact >= sigma) billboards"),
        _probe("mcmc-relocation", "mcmc_relocation",
               "Dead test mean(alpha) <= 0.005, multinomial(mean alpha) relocation, per-texel "
               "1-(1-a)^(1/(n+1)), unchanged scales, Adam reset at sources, 5% growth to cap, "
               "zero official MCMC noise"),
        _probe("init-sky-fps", "init_sky_fps",
               "FPS == brute-force farthest sampling; Fibonacci sky sphere radius max|xyz|, white, "
               "SH dc - 0.1; textures 0.5 G / 0.1"),
        _probe("schedule", "schedule",
               "Texture lr window [500, 30000), freeze all but SH rest after 30000, no step at "
               "32000, xyz expon lr x extent"),
    )
    contribs = (
        ContributionSpec("textured-billboard-raster", "Planar billboards with learnable 16x16 "
                         "alpha maps, official CUDA rasterizer (§3.2)",
                         ("raster-vs-reference",)),
        ContributionSpec("sh-plus-texture-colour", "c = T^RGB[u] + RGB(SH, d) (Eq. 6)",
                         ("texture-composition", "raster-vs-reference")),
        ContributionSpec("impact-texture-reg", "Impact-weighted texture sparsity regularisation "
                         "(§3.3, code form, sigma 100)", ("texture-regularizer",)),
        ContributionSpec("bbsplat-mcmc", "MCMC densification with alpha-only rescaling and "
                         "mean-alpha dead test (§3.4)", ("mcmc-relocation",)),
        ContributionSpec("sky-fps-init", "FPS SfM subsampling + Fibonacci sky sphere (Supp.)",
                         ("init-sky-fps",)),
        ContributionSpec("sh-finetune-stage", "Texture warm-up (500) and SH-only fine-tuning "
                         "30000-32000 (§4.1)", ("schedule",)),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    fields = (
        FieldSpec(tex.ALPHA, (tex.SIZE, tex.SIZE), learning_rate=0.001, initial_value=0.0),
        FieldSpec(tex.COLOR, (3, tex.SIZE, tex.SIZE), learning_rate=0.0025, initial_value=0.0),
    )
    return replace(
        base, name=name, description=description, components=comps,
        probes=base_probes + probes, contributions=base_contribs + contribs, fields=fields,
        citations=CITATIONS, representation="surfel2d")


def method():
    return _build("bbsplat", 60000, 50000, "BBSplat native port (official CUDA rasterizer), "
                  "Mip-NeRF 360 Tab. 1 budget (cap 60K); paper fidelity unverified.")


def method_tnt():
    return _build("bbsplat-tnt", 30000, 20000, "BBSplat native port (official CUDA rasterizer), "
                  "Tanks&Temples Tab. 1 budget (cap 30K); paper fidelity unverified.")


def method_300k():
    return _build("bbsplat-300k", 300000, 290000, "BBSplat native port (official CUDA "
                  "rasterizer), scripts/train_all.sh budget (cap 300K); paper fidelity "
                  "unverified.")


def method_tnt_sky2k():
    """Gap-analysis iteration 1 (post-hoc, chosen after the truck benchmark): Supp.
    §Initialization "for scenarios when we use 20K points and less, we reduce the number of
    sphere points to 2K" read as applying to Tab. 1 T&T (30K), with the cap = SfM + sky split:
    28K FPS SfM + 2K sky. Also the official xyz lr after 30000 (review-1)."""
    return _build("bbsplat-tnt-sky2k", 30000, 28000, "BBSplat native port, Tanks&Temples Tab. 1 "
                  "budget (cap 30K) with a 2K sky sphere (28K SfM) and official post-30K xyz lr; "
                  "post-hoc variant after the truck benchmark; paper fidelity unverified.",
                  sky_points=2000, xyz_lr_after_freeze="official")
