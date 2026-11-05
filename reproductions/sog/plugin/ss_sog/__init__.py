"""Self-Organizing Gaussians (Morgenstern et al., ECCV 2024) native SplatStudio port;
paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import DEFAULTS as NEIGHBOR_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2312.13299",
    "https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/",
    "https://github.com/fraunhoferhhi/Self-Organizing-Gaussians",
    "https://github.com/fraunhoferhhi/PLAS",
)


def _probe(name, description, cuda):
    return ProbeSpec(name, f"ss_sog.probes:{name.replace('-', '_')}", description, cuda)


def method(name="sog", total_steps=30000):
    base = make_method(name)
    renderer, _, photometric = base.components
    renderer = replace(renderer, name="sog-renderer", factory="ss_sog.renderer:create_renderer",
                       config={})
    density = ComponentSpec(
        "sog-density", "density", "ss_sog.density:create_density",
        ("gradient.screen.signed",), (), {**DENSITY_DEFAULTS, "total_steps": total_steps},
    )
    neighbor = ComponentSpec("sog-neighbor", "loss", "ss_sog.losses:create_neighbor_loss",
                             ("render.rgb",), (), dict(NEIGHBOR_DEFAULTS))
    probes = (
        _probe("log-xyz", "Log position activation: round trip, renderer(log means) == gsplat("
               "world means), split children sampled in world space, init conversion", True),
        _probe("neighbor-loss", "Huber(detached separable blur 5/sigma 3, circular x / reflect y,"
               " grid) value and gradient vs an independent conv; weights 1/11, 10/11", False),
        _probe("grid-sort", "Prune to the S^2 most opaque in order; PLAS permutation applied to "
               "params, Adam moments and strategy state; neighbour distance drops", True),
        _probe("codec-roundtrip", "imagecodecs 2023.9.18 JXL: decoded lossless grids equal the "
               "independent quantize/normalize formula, DC error bounded, size = sum of files, "
               "sorted grid smaller than shuffled", False),
        _probe("schedule", "Sort / densify / encode steps and densification config equal the "
               "official train.py + yaml; no opacity reset or size pruning", False),
        _probe("decoded-eval", "After the final step the parameters equal the decoded grids and a "
               "fresh renderer renders the decoded model; stored files sum to the size", True),
        _probe("sog-renderer-gradient", "Renderer screen-space gradient for the density, in the "
               "runtime order (initialize before render)", True),
    )
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(
        replace(c, probes=("sog-renderer-gradient",)) if c.name == "rasterizer-statistics" else c
        for c in base.contributions)
    contributions = (
        ContributionSpec(
            "grid-plas-sorting",
            "C1 (§3.1): prune to the largest square grid (lowest opacities removed) and sort all "
            "attributes jointly with PLAS (keys: standardized position, DC colour, scale; "
            "shuffle; improvement break 1e-4) at init and after every densification interval",
            ("grid-sort", "schedule"),
        ),
        ContributionSpec(
            "smoothness-regularization",
            "C2 (§3.2): lambda=1 Huber loss between opacity / rotation grids and their detached "
            "5x5 sigma-3 blur, weights 0.09 / 0.91",
            ("neighbor-loss",),
        ),
        ContributionSpec(
            "log-xyz-activation",
            "C3 (§3.3): positions optimised and stored as sign(x) log(1+|x|)",
            ("log-xyz",),
        ),
        ContributionSpec(
            "jxl-quantized-codec",
            "C4 (§3.4): 14/5/6/6/6-bit quantization, fixed-range normalisation, JPEG XL "
            "(lossy DC, lossless others) with imagecodecs 2023.9.18; evaluation on the decoded "
            "model; size = compressed directory bytes / 1e6",
            ("codec-roundtrip", "decoded-eval"),
        ),
        ContributionSpec(
            "sog-densification",
            "C5 (supp.): densify every 1000 its (500-15000), grad 7e-5, prune opacity < 0.1, "
            "percent dense 0.1, no opacity reset",
            ("schedule",),
        ),
    )
    return replace(
        base,
        description=("Self-Organizing Gaussians native port (PLAS grid sorting + smoothness loss "
                     "+ log positions + JPEG XL grid codec, decoded-model evaluation); paper "
                     "fidelity unverified"),
        components=(renderer, density, photometric, neighbor),
        probes=base_probes + probes,
        contributions=base_contribs + contributions,
        citations=CITATIONS,
    )


def smoke():
    """Diagnostic: identical to `sog` but encodes at step 2000 (not a benchmark configuration)."""
    m = method("sog-smoke", total_steps=2000)
    return replace(m, description="SOG with total_steps=2000 so a smoke run reaches the encode "
                   "step (diagnostic only)")
