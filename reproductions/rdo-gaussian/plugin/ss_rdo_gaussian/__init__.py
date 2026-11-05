"""RDO-Gaussian (Wang et al., ECCV 2024) native SplatStudio port; paper fidelity unverified.

Global learned state (six ECVQ codebooks + entropy-model logits) lives in `bank.RDOBank`,
owned by the density component and read by the renderer through `bank.ACTIVE`; per-Gaussian
Gaussian / SH masks are fields. The final step encodes the official bitstream, decodes it and
replaces the parameters, so evaluation measures the decoded model (see density.py, plan.md).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import GS_MASK, SH_MASK

CITATIONS = (
    "https://arxiv.org/abs/2406.01597",
    "https://github.com/USTC-IMCL/RDO-Gaussian",
)

FIELDS = (
    FieldSpec(GS_MASK, (), learning_rate=0.01, initial_value=0.0),  # official gs_mask, lr 0.01
    FieldSpec(SH_MASK, (3,), learning_rate=0.05, initial_value=0.0),  # sh_mask, lr 0.05 (real)
)

# App. Tab. A.1-A.3 / run_real_world.sh: (lambda_GSprune, lambda_SHprune), Rate 1 = highest
RATES = {1: (0.0005, 0.005), 2: (0.002, 0.02), 3: (0.005, 0.05), 4: (0.01, 0.1),
         5: (0.02, 0.2), 6: (0.05, 0.5)}


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_rdo_gaussian.probes:{name.replace('-', '_')}", description, cuda)


def method(rate=1, name="rdo-gaussian", density_overrides=None):
    lam_gs, lam_sh = RATES[rate]
    base = make_method(name)
    _, baseline_density, photometric = base.components
    renderer = ComponentSpec(
        "rdo-renderer", "renderer", "ss_rdo_gaussian.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"), {})
    config = {**DEFAULTS, "base": dict(baseline_density.config)}
    for k, v in (density_overrides or {}).items():
        config[k] = {**config[k], **v} if isinstance(config.get(k), dict) else v
    density = ComponentSpec("rdo-density", "density", "ss_rdo_gaussian.density:create_density",
                            baseline_density.requires, (), config)
    losses = (
        ComponentSpec("rdo-rate", "loss", "ss_rdo_gaussian.losses:create_rate",
                      ("render.rgb",), (), {"weight": 1.0}),
        ComponentSpec("rdo-vq", "loss", "ss_rdo_gaussian.losses:create_vq",
                      ("render.rgb",), (), {"weight": 1.0}),
        ComponentSpec("rdo-gs-prune", "loss", "ss_rdo_gaussian.losses:create_gs_prune",
                      ("render.rgb",), (), {"weight": lam_gs}),
        ComponentSpec("rdo-sh-prune", "loss", "ss_rdo_gaussian.losses:create_sh_prune",
                      ("render.rgb",), (), {"weight": lam_sh}),
    )
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("gs-mask-ste", "Gaussians with sigmoid(phi) <= 0.1 render exactly as removed; the "
               "STE mask gradient equals (dL/ds_hat . s + dL/dalpha_hat alpha) sigmoid'(phi) "
               "from an independent chain; L_GS = mean sigmoid(phi)"),
        _probe("sh-mask-ste", "Per-degree SH masks zero exactly their 3/5/7 coefficients; L_SH "
               "equals the (2l+1)/15-weighted mean over GS-kept Gaussians (Eq. 8)"),
        _probe("ecvq", "Assignment equals an independent loop (argmin ||x-c||_2 [+ bits/lambda "
               "once rate-constrained]); rate/VQ losses = sum/N/55; index cache every 10 steps; "
               "STE: render gradient reaches x, not the codebook; codebook gradient from L_VQ, "
               "logit gradient from L_rate"),
        _probe("kmeans-reactivate", "k-means equals an independent Lloyd loop from the same "
               "initial samples, logits = log(bins/||bins||_2); reactivation revives only "
               "p <= 1e-5 codewords by copying live ones"),
        _probe("rans-codec", "Vendored C++ pmf_to_quantized_cdf equals an independent Python "
               "port; rANS encode/decode is lossless and within 1% of the ideal entropy", False),
        _probe("encode-decode", "Fixture encode -> bytes -> decode: decoded parameters equal the "
               "codewords (log scale, raw rotation, DC, SH zero on pruned degrees), SH-mask "
               "sort/boundaries equal an independent computation, sizes equal the official "
               "files written by np.savez to disk; a fresh renderer reproduces the baked model"),
        _probe("schedule", "Activation after 15000, reactivation 16000-20000, rate constraint "
               "from 20001, bank LR x0.1 after 12000 bank steps, no bank step at total_steps, "
               "encode at total_steps", False),
        _probe("rd-losses", "The four RD losses are attached zeros before activation and "
               "weight x renderer term after (Eq. 13 weights)"),
    )
    contributions = (
        ContributionSpec(
            "rd-objective",
            "C1 (§3.2 Eq. 13): L = lambda_GS L_GS + lambda_SH L_SH + L_rate + L_VQ + L_render "
            "from iteration 15001 (official normalisation /N/55)",
            ("rd-losses", "ecvq", "schedule"),
        ),
        ContributionSpec(
            "gaussian-pruning",
            "C2 (§3.2 Eq. 2-5): learnable STE Gaussian mask (threshold 0.1, lr 0.01) on scale and "
            "opacity; dynamic, removed only at the end",
            ("gs-mask-ste", "rd-losses"),
        ),
        ContributionSpec(
            "adaptive-sh-pruning",
            "C3 (§3.2 Eq. 6-8): three learnable STE masks per Gaussian for SH degrees 1-3 "
            "(threshold 0.1, lr 0.05), coefficient-weighted loss",
            ("sh-mask-ste", "rd-losses"),
        ),
        ContributionSpec(
            "ecvq",
            "C4 (§3.2 Eq. 9-11): entropy-constrained VQ of scale/rotation/DC/SH1-3 (8192 / 4096 "
            "codewords) with learned softmax entropy models, k-means init, codeword "
            "reactivation, rate-constrained assignment from 20001",
            ("ecvq", "kmeans-reactivate", "schedule"),
        ),
        ContributionSpec(
            "parameter-encoding",
            "C5 (§3.3): masked-Gaussian removal, SH-mask sorting (8 boundaries), codebook "
            "shrinking, rANS coding with the learned pmfs, fp16 positions, 8-bit opacity; "
            "official bitstream bytes, decoded-model evaluation",
            ("rans-codec", "encode-decode"),
        ),
    )
    return replace(
        base,
        description=(
            f"RDO-Gaussian native port (rate point {rate}: lambda_GS={lam_gs}, "
            f"lambda_SH={lam_sh}; Gaussian + SH masks, ECVQ, rANS bitstream, decoded-model "
            "evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric) + losses,
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )


def rate2():
    return method(2, "rdo-gaussian-rate2")


def rate3():
    return method(3, "rdo-gaussian-rate3")


def rate4():
    return method(4, "rdo-gaussian-rate4")


def rate5():
    return method(5, "rdo-gaussian-rate5")


def rate6():
    return method(6, "rdo-gaussian-rate6")


SMOKE = {"total_steps": 2000, "gs_prune_start": 1001, "sh_prune_start": 1001, "vq_start": 1001,
         "rate_constrain_start": 1401, "reactivate_every": 100, "lr_milestone": 800,
         "log_every": 200, "base": {"refine_stop_iter": 1000}}


def smoke():
    """Diagnostic 2000-step schedule with every event (not a benchmark configuration)."""
    m = method(1, "rdo-gaussian-smoke", SMOKE)
    return replace(m, description="RDO-Gaussian compressed 2000-step smoke schedule "
                   "(diagnostic only)")
