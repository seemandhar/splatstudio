"""FCGS (Chen et al., ICLR 2025) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2410.08017",
    "https://github.com/YihangChen-ee/FCGS",
)


def _probe(name, description, cuda):
    return ProbeSpec(name, f"ss_fcgs.probes:{name.replace('-', '_')}", description, cuda)


def _method(name, lmd, label):
    base = make_method(name)
    renderer, baseline_density, photometric = base.components
    density = ComponentSpec(
        "fcgs-density",
        "density",
        "ss_fcgs.density:create_density",
        baseline_density.requires,
        (),
        {**DEFAULTS, "lmd": lmd, "base": dict(baseline_density.config)},
    )
    probes = (
        _probe("official-weights", "Pretrained FCGS weights are the official release (SHA256)",
               False),
        _probe("grid-context", "Official grid-creator (Eq. 4, float and deterministic) and grid "
               "interpolation (Eq. 5) CUDA kernels equal an independent torch implementation",
               True),
        _probe("gmm-cdf", "Arithmetic-coder CDF table equals the softmax-weighted Gaussian "
               "mixture of Eq. 7; lossless decode; coded bits near -log2 p", True),
        _probe("mem-and-quantisation", "Real encode -> bitstream -> decode: m=1 colour = "
               "g_s(round(g_a(x))), m=0 colour and geometry = Q(x) with learned steps, "
               "positions on the 16-bit grid, decode == encoder reconstruction", True),
        _probe("context-models", "Inter/intra-context GMM codes geometry in fewer bits than the "
               "hyperprior Gaussian alone (pretrained model)", True),
        _probe("xyz-lossless", "Position coder (substitute octree when tmc3 is absent) is "
               "lossless incl. duplicates", False),
        _probe("schedule", "Codec runs once, right after the final (30K) step", False),
    )
    contributions = (
        ContributionSpec(
            "mem",
            "C1 (§3.2, Eq. 3): pretrained MLP_m mask (>0.01) routes colour through the "
            "autoencoder (256-d latent, unit-step rounding) or direct quantisation with learned "
            "step; geometry always directly quantised with learned steps",
            ("official-weights", "mem-and-quantisation"),
        ),
        ContributionSpec(
            "inter-gaussian-context",
            "C2 (§3.3, Eqs. 4-5): 4 seeded random batches (1/6,1/6,1/3,1/3); decoded batches "
            "build 3-D (70,80,90) and 2-D (300,400,500) grids by the official CUDA grid creator, "
            "interpolated at later batches + freq. embedding -> MLP_s",
            ("grid-context", "context-models"),
        ),
        ContributionSpec(
            "intra-gaussian-context",
            "C3 (§3.3, Eq. 6): 4x64-channel latent chunks (m=1) and RGB chunks (m=0) "
            "predicted from earlier chunks by MLP_c",
            ("context-models", "mem-and-quantisation"),
        ),
        ContributionSpec(
            "gmm-entropy-coding",
            "C4 (§3.3-3.4, Eq. 7): hyperprior/spatial/channel GMM, factorised z prior, CUDA "
            "arithmetic coding to real bitstreams, Bernoulli mask, 16-bit positions (GPCC or "
            "lossless substitute); the final 30K model is replaced by its decoded version and "
            "size = sum of bitstream bits / 8 MiB as in the official code",
            ("gmm-cdf", "mem-and-quantisation", "xyz-lossless", "schedule"),
        ),
    )
    return replace(
        base,
        description=(f"FCGS native port ({label}, lambda={lmd}): vanilla 3DGS 30K + official "
                     "pretrained feed-forward codec, decoded-model evaluation; paper fidelity "
                     "unverified"),
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    """FCGS 'Ours-highrate' (lambda = 1e-4)."""
    return _method("fcgs", 1e-4, "high rate")


def method_lowrate():
    """FCGS 'Ours-lowrate' (lambda = 16e-4)."""
    return _method("fcgs-lowrate", 16e-4, "low rate")


def smoke():
    """Diagnostic: encode after 2000 steps (smoke tests only, not a benchmark configuration)."""
    base = method()
    comps = tuple(replace(c, config={**c.config, "total_steps": 2000}) if c.kind == "density"
                  else c for c in base.components)
    return replace(base, name="fcgs-smoke", components=comps,
                   description="FCGS 2000-step smoke schedule (diagnostic only)")
