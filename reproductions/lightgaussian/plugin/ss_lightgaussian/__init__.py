"""LightGaussian (Fan et al., NeurIPS 2024) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2311.17245",
    "https://github.com/VITA-Group/LightGaussian (6676b98; prune/distill/vectree ported)",
    "https://github.com/Kevin-2017/compress-diff-gaussian-rasterization (9893d8c; "
    "renderCUDA_count significance rules ported)",
    "https://github.com/nerfstudio-project/gsplat (v1.5.3 projection / tile lists)",
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_lightgaussian.probes:{name.replace('-', '_')}", description, cuda)


def _make(name, overrides=None, description=None):
    base = make_method("gsplat")
    renderer, baseline_density, photometric = base.components
    renderer = replace(renderer, name="lightgaussian-renderer",
                       factory="ss_lightgaussian.renderer:create_renderer")
    config = {**DEFAULTS, "base": dict(baseline_density.config), **(overrides or {})}
    density = ComponentSpec("lightgaussian-density", "density",
                            "ss_lightgaussian.density:create_density",
                            baseline_density.requires, (), config)
    photometric = replace(photometric, factory="ss_lightgaussian.losses:create_photometric",
                          config={"ssim_weight": 0.2})
    distill = ComponentSpec("lightgaussian-distill", "loss", "ss_lightgaussian.losses:create_distill",
                            ("render.rgb",), (),
                            {"ssim_weight": 0.2, "renderer": dict(renderer.config)})
    probes = (
        _probe("significance-oracle", "CUDA renderCUDA_count port (sum sigma*T, 0.99 clamp, "
               "1/255 skip, 1e-4 termination) vs brute-force per-pixel composite"),
        _probe("prune-rule", "v-score (V / sort_desc(V)[0.9N])^0.1 * score without clipping; "
               "prune v <= sort(v)[int(0.66 (N-1))]", False),
        _probe("distill-views", "pseudo views (centre + N(0,0.05^2)) on k%3!=0 only; distill "
               "loss = L1+0.2DSSIM vs teacher at the same camera; photometric gated off; "
               "zeroed degree-3 SH == degree-2 render"),
        _probe("vq-official", "weighted EMA step == official one-hot formula, kaiming init, "
               "dead-code replacement, learned codebook beats init, top-40% kept"),
        _probe("codec-roundtrip", "official extreme_saving layout (13-bit indices, packbits, "
               "fp16, xyz fp32), decode == expected, size = zip bytes / 2^20", False),
        _probe("pipeline", "tiny end-to-end run through prune, distill, VQ, VQ fine-tune, "
               "encode: counts, frozen opacity, SH-2, codebook-driven VQ rows, decode, leaves"),
        _probe("schedule", "stage boundaries 30000/35000/40000/45000, LR factors, weight "
               "decay, pseudo-view cadence, run-length guard", False),
    )
    contributions = (
        ContributionSpec(
            "global-significance-prune",
            "C1 (§3.2 Eq. 3/4): GS_j = sum over training pixels of sigma_j T * (V/V_90)^0.1; "
            "remove the 66% least significant after 3DGS 30k, 5000 co-adaptation steps "
            "without densification",
            ("significance-oracle", "prune-rule", "pipeline"),
        ),
        ContributionSpec(
            "sh-distillation",
            "C2 (§3.3 Eq. 5/6): SH 3 -> 2, 5000 steps of L1+0.2DSSIM distillation from the "
            "frozen SH-3 teacher, pseudo views with camera centre + N(0, 0.05^2) on 2/3 of "
            "the steps, opacity frozen, fresh AdamW",
            ("distill-views", "pipeline", "schedule"),
        ),
        ContributionSpec(
            "significance-vq",
            "C3 (§3.4): 8192-code significance-weighted EMA VQ of the 27-d SH of the 60% least "
            "significant Gaussians, fp16 storage, 5000-step VQ fine-tune with fixed "
            "assignment; decoded-model evaluation, official zip size",
            ("vq-official", "codec-roundtrip", "pipeline"),
        ),
    )
    return replace(
        base,
        name=name,
        description=description or (
            "LightGaussian native port (single 45k run: 3DGS 30k + significance prune 66% + 5k "
            "co-adaptation + 5k SH distillation + VQ + 5k VQ fine-tune; decoded-model "
            "evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric, distill),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("lightgaussian")


def released():
    """Released-code pipeline: no VQ fine-tune (vectree output is the final model), 40k steps."""
    return _make("lightgaussian-released", {"total_steps": 40000},
                 "LightGaussian released-code pipeline (single 40k run: 3DGS 30k + prune + "
                 "co-adaptation + SH distillation + VQ, no VQ fine-tune); paper fidelity "
                 "unverified")


def smoke():
    """Diagnostic 2000-step schedule (not a benchmark config)."""
    return _make("lightgaussian-smoke",
                 {"prune_step": 1000, "distill_step": 1300, "vq_step": 1600, "total_steps": 2000,
                  "vq_iterations": 100},
                 "LightGaussian compressed 2000-step smoke schedule (diagnostic only)")
