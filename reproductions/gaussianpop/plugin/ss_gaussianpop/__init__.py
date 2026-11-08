"""GaussianPOP (arXiv 2602.06830) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2602.06830",
    "https://github.com/nerfstudio-project/gsplat (v1.5.3 rasterizer, adapted kernel)",
)


def _make(name, ratio):
    base = make_method("gsplat")
    renderer, baseline_density, photometric = base.components
    config = {**DEFAULTS, "prune_ratio": ratio, "base": dict(baseline_density.config)}
    density = ComponentSpec(
        "pop-density", "density", "ss_gaussianpop.density:create_density",
        baseline_density.requires, (), config,
    )
    probes = (
        ProbeSpec(
            "delta-se-oracle", "ss_gaussianpop.probes:delta_se_oracle",
            "CUDA Alg. 1 Δ_SE vs brute-force removal re-render (Eq. 4), C_render vs gsplat",
            True,
        ),
        ProbeSpec(
            "prune-cycles", "ss_gaussianpop.probes:prune_cycles",
            "C re-quantification cycles remove the lowest-error Gaussians to the compound "
            "keep schedule; fresh Adam; 3DGS pretrain/fine-tune means LR",
            True,
        ),
    )
    contributions = (
        ContributionSpec(
            "removal-error",
            "C1+C2: per-Gaussian removal error ||T_k a_k (c_k - b_{k+1})||^2 with b_{k+1} "
            "recovered from cached prefix colour/transmittance in one forward pass (CUDA)",
            ("delta-se-oracle",),
        ),
        ContributionSpec(
            "iterative-post-training-prune",
            f"C3: after 30k 3DGS, 8 re-quantification cycles prune {int(ratio * 100)}% "
            "lowest-error Gaussians, then 5k fine-tune",
            ("prune-cycles",),
        ),
    )
    return replace(
        base,
        name=name,
        description=(
            f"GaussianPOP post-training native port (P={ratio}, C=8; single 35k run: 3DGS 30k "
            "+ prune + 5k fine-tune); paper fidelity unverified"
        ),
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("gaussianpop", 0.8)


def p07():
    return _make("gaussianpop-p07", 0.7)


def p09():
    return _make("gaussianpop-p09", 0.9)
