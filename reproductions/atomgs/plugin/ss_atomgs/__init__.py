"""AtomGS: Atomizing Gaussian Splatting for High-Fidelity Radiance Field (Liu et al., BMVC 2024,
arXiv 2405.12369) — native SplatStudio port of the official code (github.com/RongLiu-Leo/AtomicGS,
commit 5b9ce904) with its feature rasterizer (mean/median depth, alpha) vendored.

Train with ``--recipe 3dgs`` (3-NN initial scales feed the atom-scale initialisation).
Poisson mesh extraction / DTU chamfer evaluation are not ported.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import NORMAL_DEFAULTS, PHOTO_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2405.12369",
    "https://github.com/RongLiu-Leo/AtomicGS",
    "https://rongliu-leo.github.io/AtomGS/",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_atomgs.probes:{fn}", desc, cuda)


def method():
    base = make_method("atomgs")
    comps = (
        ComponentSpec("atomgs-raster", "renderer", "ss_atomgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("atomgs-density", "density", "ss_atomgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("atomgs-photometric", "loss", "ss_atomgs.losses:create_photometric",
                      ("render.rgb",), (), dict(PHOTO_DEFAULTS)),
        ComponentSpec("edge-aware-normal", "loss", "ss_atomgs.losses:create_normal_loss",
                      ("render.normal_from_depth.world",), (), dict(NORMAL_DEFAULTS)),
    )
    probes = (
        _probe("feature-raster", "feature_raster",
               "Official kernel RGB / unnormalised mean depth / alpha / median depth and "
               "d(depth), d(rgb)/d(means) equal an independent float64 EWA compositor", True),
        _probe("normal-from-depth", "normal_from_depth",
               "Official depth_to_normal equals an independent ray unprojection (same (W-1) "
               "grid) and recovers a fronto-parallel plane normal"),
        _probe("edge-aware-normal", "edge_aware_normal",
               "Edge-aware normal loss equals a padded-slice Sobel reference, is suppressed at "
               "image edges ((|dI|-1)^500), weight 0.1, off from step 7000"),
        _probe("ms-ssim", "ms_ssim_probe",
               "Photometric = 0.9 L1 + 0.1 (1 - mean of 3 same-padded SSIM scales) vs a "
               "separable-convolution reference"),
        _probe("atomize", "atomize",
               "Rows with min(s) <= S become isotropic S, others unchanged, scale moments "
               "zeroed, S *= 0.99, runs at 6900 but not 7000"),
        _probe("atom-proliferation", "atom_proliferation",
               "Densify equals a literal transcription of official densify_and_prune "
               "(split-first, warm-up threshold, max(s) > S, index-shifted clone, prunes)"),
        _probe("init-schedules", "init_schedules",
               "Atom scale = P1 of 3-NN distances with init clamp; xyz LR constant after 7000; "
               "opacity reset to 0.1 every 300 before 15000; --recipe 3dgs guard"),
    )
    contribs = (
        ContributionSpec("atomized-proliferation", "Atom scale, atomization, atom-constrained "
                         "split with warm-up, unconstrained clone, 300-step opacity reset "
                         "(Sec. 3.1, supp. Alg. 1)", ("atomize", "atom-proliferation",
                                                      "init-schedules")),
        ContributionSpec("edge-aware-normal-loss", "Edge-aware normal loss on mean-depth "
                         "normals (Sec. 3.2)", ("normal-from-depth", "edge-aware-normal")),
        ContributionSpec("feature-rasterizer", "Official rasterizer with mean/median depth "
                         "and alpha outputs + backward (supp.)", ("feature-raster",)),
        ContributionSpec("ms-ssim-photometric", "L1 + MS-SSIM photometric loss, lambda 0.1 "
                         "(supp. hyper-parameters)", ("ms-ssim",)),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base,
        description="AtomGS native port (official feature-rasterizer CUDA, atomized "
                    "proliferation, edge-aware normal loss, MS-SSIM); paper fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        citations=CITATIONS,
    )
