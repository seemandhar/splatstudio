"""Faster-GS: Analyzing and Improving Gaussian Splatting Optimization (CVPR 2026, arXiv
2602.09999) — native SplatStudio port wrapping the official CUDA backend
(github.com/nerficg-project/faster-gaussian-splatting main @ 090e8ad, Apache-2.0) with the
paper's main configuration (3DGS optimization, Morton ordering, own CUDA Adam).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2602.09999",
             "https://github.com/nerficg-project/faster-gaussian-splatting",
             "https://fhahlbohm.github.io/faster-gaussian-splatting/")


def _probe(name, fn, desc, cuda=True):
    return ProbeSpec(name, f"ss_faster_gs.probes:{fn}", desc, cuda)


PROBES = (
    _probe("fastergs-render-reference", "render_reference",
           "Vendored rasterizer image and all six raw-parameter gradients (fused activations, "
           "separate SH buffers, per-Gaussian backward) equal an independent float64 dense "
           "compositor (every Gaussian at every pixel), SH degree 0 and 3, off-centre camera"),
    _probe("fastergs-tile-culling", "tile_culling",
           "Gaussian/tile instances lie between the tiles holding visible fragments and the "
           "opacity-aware rectangle, and are fewer than the AABB and the 3DGS 3-sigma square for "
           "elongated rotated splats; faint splats touch fewer tiles"),
    _probe("fastergs-adam", "fused_adam",
           "FusedAdam CUDA kernel equals torch.optim.Adam(eps=1e-15) with changing lr; not SGD"),
    _probe("fastergs-morton", "morton",
           "Morton step sorts means by an independent bit-interleave z-order code and permutes "
           "Adam moments with them; schedule {0, 5000, 10000, 15000}"),
    _probe("fastergs-adc", "adc",
           "Official 3DGS ADC (duplicate / split / prune) equals an independent rule on a "
           "fixture; survivor moments kept, new moments zero; 144 ADC steps 600..14900, resets "
           "3000..12000; densification info re-zeroed then disabled"),
    _probe("fastergs-densification-info", "densification_info",
           "Backward accumulates the visibility count and NDC-scaled ||dL/d mu2D|| equal to the "
           "dense reference's mean2d gradients, across iterations"),
    _probe("fastergs-profile", "profile_smoke",
           "Separate forward / backward / update timing of Faster-GS vs gsplat runs"),
)

CONTRIBUTIONS = (
    ContributionSpec("basis-rasterizer", "Refactored 3DGS rasterizer: intrinsics-based "
                     "projection, front-to-back gradients, degenerate handling (§3.1, supp.)",
                     ("fastergs-render-reference",)),
    ContributionSpec("consolidated-bounds-culling", "Consolidated prior work (§3.2): tight "
                     "opacity-aware rectangles + load-balanced exact tile culling (StopThePop), "
                     "separate sorting (Splatshop)", ("fastergs-tile-culling",
                                                       "fastergs-render-reference")),
    ContributionSpec("per-gaussian-backward-fusion", "Per-Gaussian backward (Taming-3DGS) with "
                     "the own shared-memory improvement, fused activations, separate SH buffers, "
                     "fused densification statistics (§3.2, supp.)",
                     ("fastergs-render-reference", "fastergs-densification-info")),
    ContributionSpec("fused-adam", "Own CUDA Adam matching PyTorch (§3.3 Parameter Updates)",
                     ("fastergs-adam",)),
    ContributionSpec("morton-ordering", "Locality-preserving densification: Morton order every "
                     "5000 iterations while densifying (§3.3)", ("fastergs-morton",)),
    ContributionSpec("3dgs-optimization-unchanged", "3DGS ADC / reset / schedule kept as in the "
                     "official code (supp. §A)", ("fastergs-adc",)),
    ContributionSpec("speed-profile", "Forward / backward / update profiled against gsplat "
                     "(timing evidence only, no speed claim)", ("fastergs-profile",)),
)


def method():
    base = make_method("faster-gs")
    photometric = base.components[2]
    comps = (
        ComponentSpec("fastergs-renderer", "renderer", "ss_faster_gs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "background": list(RENDERER_DEFAULTS["background"])}),
        ComponentSpec("fastergs-density", "density", "ss_faster_gs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        photometric,
    )
    keep = {"state-transitions", "photometric-gradient"}
    base_probes = tuple(p for p in base.probes if p.name in keep)
    base_contribs = tuple(c for c in base.contributions if set(c.probes) <= keep)
    return replace(
        base,
        description="Faster-GS (CVPR 2026) native port: official CUDA rasterizer + CUDA Adam + "
                    "Morton ordering, 3DGS optimization (paper main configuration; kernel "
                    "includes borrowed warp-level culling); paper fidelity unverified.",
        components=comps, probes=base_probes + PROBES,
        contributions=base_contribs + CONTRIBUTIONS, citations=CITATIONS)
