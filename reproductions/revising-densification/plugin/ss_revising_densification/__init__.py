"""Revising Densification in Gaussian Splatting (Rota Bulo, Porzi, Kontschieder; ECCV 2024,
arXiv 2404.06109). Native SplatStudio port re-derived from the paper (no official code was
released); paper fidelity unverified.

Contributions: error-based densification score from per-pixel errors attributed to primitives by
their alpha-compositing weights (custom CUDA), primitive growth control (global cap + 5 % per
run, top score), opacity correction on clone, opacity decrement instead of reset plus a
residual-transmittance loss, ADC active until 27k. Renderer and photometric loss are stock.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .losses import LOSS_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2404.06109",)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_revising_densification.probes:{fn}", description, cuda)


PROBES = (
    _probe("error-attribution-oracle", "error_attribution",
           "CUDA E^pi_k = sum_u Err(u) alpha T equals a brute-force per-pixel alpha-compositing "
           "loop and the paper's L_aux autograd route (grad of a rendered zero scalar e_k); "
           "uniform-error and weight-free ablations differ", True),
    _probe("error-score-max", "error_score_max",
           "Density score E_k is the max of E^pi_k over the views since the last ADC run (not "
           "sum/mean) and resets after a run", True),
    _probe("error-driven-selection", "error_selection",
           "High-error / zero-gradient Gaussian grows, zero-error / high-gradient one does not; "
           "E_k > 0.1 threshold; the 3DGS-gradient ablation selects the opposite"),
    _probe("growth-budget", "growth_budget",
           "Per run +floor(0.05 N) primitives chosen by top E_k, global cap respected, pruned "
           "primitives replaced up to the cap; no-control ablation exceeds it"),
    _probe("clone-opacity", "clone_opacity",
           "Clone source and copy get 1-sqrt(1-alpha), split children keep alpha; rendered "
           "transmittance behind two coincident clones equals 1-alpha (ablation: (1-alpha)^2)",
           True),
    _probe("opacity-regularisation", "opacity_regularisation",
           "Opacities drop by 0.001 per ADC run and are never reset; transmittance loss equals "
           "0.1 mean(1-alpha) and its gradient raises opacity; reset ablation resets", True),
    _probe("error-map", "error_map",
           "Stored per-pixel error is detached 1-SSIM(u); its interior mean matches the runtime "
           "SSIM (valid conv) and the l1 variant matches |I-I*|"),
    _probe("adc-schedule", "adc_schedule",
           "264 ADC runs at steps 600..26900, none at or after 27000"),
    _probe("primitive-cap-table", "cap_table",
           "Global cap resolves to the baseline count of each benchmark scene from its SfM point "
           "count; int override; unknown scene raises"),
)
CONTRIBUTIONS = (
    ContributionSpec("error-based-densification", "Sec. 3.2: per-pixel SSIM error attributed to "
                     "primitives by alpha-compositing weights, max over views, E_k > 0.1 "
                     "(custom CUDA)",
                     ("error-attribution-oracle", "error-score-max", "error-driven-selection",
                      "error-map")),
    ContributionSpec("clone-opacity-correction", "Sec. 3.3: alpha_hat = 1 - sqrt(1 - alpha) for "
                     "cloned primitives, split keeps opacity", ("clone-opacity",)),
    ContributionSpec("growth-control", "Sec. 3.4: global cap (baseline count) and 5 % growth per "
                     "run, highest scores first", ("growth-budget", "primitive-cap-table")),
    ContributionSpec("opacity-regularisation", "Sec. 3.5: opacity -0.001 per ADC run instead of "
                     "reset, residual-transmittance loss weight 0.1", ("opacity-regularisation",)),
    ContributionSpec("long-adc", "Sec. 4.2: ADC active until 27k of 30k iterations",
                     ("adc-schedule",)),
)


def method():
    base = make_method("revising-densification")
    density = ComponentSpec(
        "revdens-density", "density", "ss_revising_densification.density:create_density",
        ("render.rgb",), (), dict(DEFAULTS),
    )
    aux = ComponentSpec(
        "revdens-aux", "loss", "ss_revising_densification.losses:create_aux",
        ("render.rgb",), (), dict(LOSS_DEFAULTS),
    )
    return replace(
        base,
        description="Revising Densification in GS (arXiv 2404.06109) native port re-derived "
                    "from the paper (no official code); paper fidelity unverified.",
        components=(base.components[0], density, base.components[2], aux),
        probes=tuple(
            replace(p, function="ss_revising_densification.probes:renderer_gradient",
                    description=p.description + " (run through the revdens loss/density step)")
            if p.name == "renderer-gradient" else p
            for p in base.probes
        ) + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
