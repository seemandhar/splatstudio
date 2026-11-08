"""CompGS / Compact3D (Navaneet et al., ECCV 2024) native SplatStudio port; paper fidelity unverified.

Ported from the official code https://github.com/UCDvision/compact3d (MIT licence).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .losses import DEFAULTS as REG_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2311.18159",
    "https://github.com/UCDvision/compact3d",
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_compgs.probes:{name.replace('-', '_')}", description, cuda)


def _make(name, density_overrides=None, reg_overrides=None, description=None):
    base = make_method(name)
    renderer, baseline_density, photometric = base.components
    renderer = replace(renderer, name="compgs-renderer",
                       factory="ss_compgs.renderer:create_renderer")
    dcfg = {**DEFAULTS, "base": dict(baseline_density.config), **(density_overrides or {})}
    if "base" in (density_overrides or {}):
        dcfg["base"] = {**baseline_density.config, **density_overrides["base"]}
    density = ComponentSpec("compgs-density", "density", "ss_compgs.density:create_density",
                            baseline_density.requires, (), dcfg)
    reg = ComponentSpec("compgs-opacity-reg", "loss", "ss_compgs.losses:create_opacity_reg",
                        (), (), {**REG_DEFAULTS, **(reg_overrides or {})})
    probes = (
        _probe("kmeans-official", "assign/update vs an independent one-hot Lloyd re-implementation "
               "of Quantize_kMeans (warm start, final re-assignment, empty cluster -> 0)", False),
        _probe("ste-render", "render(STE params) == render(decoded C[idx]); grads on raw params == "
               "grads w.r.t. the decoded values; differs from the unquantised render"),
        _probe("schedule", "assignment iterations 20001..24901 step 100 + 25001, centroid "
               "updates elsewhere, prune steps 16k..20k, regulariser window 15k..20k", False),
        _probe("opacity-reg", "lambda * sum sigmoid(o) inside [15000, 20000], 0 outside; "
               "gradient lambda * s(1-s)", False),
        _probe("pipeline", "tiny run of the density: opacity prune keeps optimizer aligned, "
               "assign/update, bake puts every row on its codebook, ACTIVE cleared, size formula"),
    )
    contributions = (
        ContributionSpec(
            "kmeans-vq-training",
            "C1 (§3, §4): quantisation-aware training with 4 k-means codebooks (DC 4096, SH 4096, "
            "log-scale 32768, raw quaternion 32768) from 20K; straight-through estimator; "
            "centroids re-averaged every iteration; assignment (5 Lloyd its) every 100 its "
            "until 25K, frozen afterwards",
            ("kmeans-official", "ste-render", "schedule", "pipeline"),
        ),
        ContributionSpec(
            "opacity-regularization-pruning",
            "C2 (§3): L += 1e-7 * sum sigmoid(o) for 15K..20K and pruning of sigma < 0.005 every "
            "1000 iterations in that window",
            ("opacity-reg", "schedule", "pipeline"),
        ),
        ContributionSpec(
            "decoded-compact-storage",
            "C3 (§3): only codebooks + indices (+ fp32 position/opacity) are kept; the decoded "
            "model C[idx] is baked at 30K so evaluation renders it; paper-formula size "
            "(log2 K-bit indices, RLE on the rotation index) reported",
            ("pipeline",),
        ),
    )
    return replace(
        base,
        description=description or (
            "CompGS 32K native port (3DGS + opacity-regularised pruning 15K-20K + k-means VQ "
            "of colour/covariance from 20K; decoded-model evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric, reg),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("compgs")


SMOKE_DENSITY = {
    "base": {"refine_stop_iter": 1500},
    "total_steps": 2000,
    "vq_start": 1700,
    "freeze_window": 100,
    "final_freq": 1900,
    "prune_start": 1500,
    "prune_stop": 1700,
    "prune_every": 100,
}


def smoke():
    """Diagnostic 2000-step schedule (densify to 1500, reg+prune 1500-1700, VQ 1700-2000)."""
    return _make("compgs-smoke", SMOKE_DENSITY, {"start": 1500, "stop": 1700},
                 "CompGS compressed 2000-step smoke schedule (diagnostic only)")
