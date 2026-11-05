"""Compressed 3DGS (Niedermayr et al., CVPR 2024) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2401.02436",
    "https://github.com/KeKsBoTer/c3dgs",
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_c3dgs.probes:{name.replace('-', '_')}", description, cuda)


def _make(name, overrides=None, description=None):
    base = make_method(name)
    renderer, baseline_density, photometric = base.components
    renderer = replace(renderer, name="c3dgs-renderer",
                       factory="ss_c3dgs.renderer:create_renderer")
    config = {**DEFAULTS, "base": dict(baseline_density.config), **(overrides or {})}
    density = ComponentSpec("c3dgs-density", "density", "ss_c3dgs.density:create_density",
                            baseline_density.requires, (), config)
    probes = (
        _probe("sensitivity-oracle", "Unclamped |dE/dSH| per view vs W_g Y_k(d) from a one-hot "
               "blend-weight render; shape sensitivity = eta^2 |dE/dSigma|; pixel normalisation; "
               "invisible Gaussians get 0"),
        _probe("vq-official", "Official nearest kernel = brute force; EMA weighted centroid "
               "update; unit-trace covariance codebooks; exact kept vectors; eigh split"),
        _probe("qa-codebook", "8-bit min-max fake quantisers, float16 positions, codebook "
               "gradient = sum of per-Gaussian gradients, official fine-tune LRs"),
        _probe("codec-roundtrip", "Official save_npz layout; decode = fake-quantised model; "
               "size = on-disk savez_compressed size; Morton order applied and shrinks the file"),
        _probe("pipeline", "Tiny end-to-end run: zero-sensitivity pruning, clustering, QA "
               "fine-tune moves codebooks not means, last step skipped, encode -> decode"),
        _probe("schedule", "Compression at 30000, bank steps, encode at 35000, run-length guard, "
               "means LR", False),
    )
    contributions = (
        ContributionSpec(
            "sensitivity-pruning",
            "C1 (§4.1 Eq. 1, supp.): S(p) = mean over training pixels of |dE_i/dp| with "
            "unclamped colours (one backward per view); Gaussians with zero colour sensitivity "
            "are pruned",
            ("sensitivity-oracle", "pipeline"),
        ),
        ContributionSpec(
            "sensitivity-aware-vq",
            "C2 (§4.1): sensitivity-weighted mini-batch k-means (EMA 0.8) of the 48-dim SH "
            "vector and of the trace-normalised covariance into 4096-entry codebooks; vectors "
            "above beta_c = 6e-7 / beta_g = 3e-6 are kept and appended; eigh back to "
            "rotation + scale, per-Gaussian scale factor eta",
            ("vq-official", "pipeline"),
        ),
        ContributionSpec(
            "qa-finetune",
            "C3 (§4.2): 5000 fine-tune iterations with 8-bit min-max fake quantisation "
            "(opacity after sigmoid, scale before normalisation, log eta before exp, rotation "
            "before normalisation), float16 positions, accumulated codebook gradients",
            ("qa-codebook", "pipeline", "schedule"),
        ),
        ContributionSpec(
            "morton-entropy-encoding",
            "C4 (§4.3): Morton-ordered official npz layout compressed with DEFLATE; reported "
            "size = npz bytes / 2**20; evaluation renders the decoded npz",
            ("codec-roundtrip", "pipeline"),
        ),
    )
    return replace(
        base,
        description=description or (
            "Compressed 3DGS native port (single 35k run: 3DGS 30k + sensitivity, VQ, 5k QA "
            "fine-tune, Morton/DEFLATE npz; decoded-model evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("c3dgs")


def smoke():
    """Diagnostic 2000-step schedule (1500 base + 500 fine-tune); not a benchmark config."""
    return _make("c3dgs-smoke", {"base_steps": 1500, "total_steps": 2000},
                 "c3dgs compressed 2000-step smoke schedule (diagnostic only)")
