"""Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting (Armagan et al., ICASSP
2025, arXiv 2501.14534). Native SplatStudio port written from the paper (no official code is
released); each trick follows the official code of the paper it is adopted from. Headline
configuration = all eight tricks (Table 3 caption): BL, DS, SG, GM, AT, SHM, DE, SC.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES, GS_MASK, SCHEDULE_CAPABILITY, SH_MASK

CITATIONS = (
    "https://arxiv.org/abs/2501.14534",
    "https://arxiv.org/abs/2311.13681",  # Compact-3DGS (GM)
    "https://arxiv.org/abs/2311.17245",  # LightGaussian (SG)
    "https://arxiv.org/abs/2406.01597",  # RDO-Gaussian (SHM)
    "https://arxiv.org/abs/2312.04564",  # EAGLES (BL, DS)
    "https://arxiv.org/abs/2403.09413",  # RAIN-GS (SC)
    "https://arxiv.org/abs/2406.15643",  # Taming 3DGS (AT)
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_trick_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("trick-gaussian-mask", "gaussian_mask",
           "Eq. 3: render equals gsplat with hand-masked scale and opacity; STE gradient is "
           "sigmoid'; prune removes exactly sigmoid(m) <= 0.05; L_m = mean sigmoid(m)"),
    _probe("trick-sh-mask", "sh_mask",
           "Eq. 5: render equals gsplat with the masked band zeroed; no masking before "
           "sh_mask_start; L_sh equals (2l+1)/15-weighted mean"),
    _probe("trick-significance", "significance_probe",
           "CUDA hit count / sum(opacity*T) equal a brute-force per-pixel loop; (V/V90)^0.1 "
           "v-score; LightGaussian prune rule; rates 0.6*0.7^k"),
    _probe("trick-progressive", "progressive",
           "Eq. 6 sizes (linear, 1/8 -> full at 19500); blur 9x9/2.4 lowered at multiples of 100 "
           "and scaled by the DS factor; blur numerics vs dense 2-D kernel; training render at "
           "scheduled size, evaluation at full size"),
    _probe("trick-lowpass-abe", "lowpass_abe",
           "s = max(HW/(9 pi N), 0.3) reaches gsplat eps2d at step 1, recomputed every 1000 until "
           "10000, 0.3 after; ABE copies at 0.3*extent*mu only <= 10000; 20 % lowest-error init"),
    _probe("trick-accelerated", "accelerated",
           "IntervalAdam = Adam on 16-step gradient sums (shN unchanged in between); separable "
           "SSIM equals 2-D same-padded 3DGS SSIM", cuda=False),
    _probe("trick-redensify", "redensify",
           "Densify (500,15000) every 100 plus 20100..20500; mask prune every 500 from 15000; "
           "resets 3000..12000; significance steps of Fig. 3", cuda=False),
    _probe("trick-cycle", "cycle",
           "Compressed native-style run on a synthetic scene: init subset, clone/split/ABE, six "
           "significance rounds, late densification, progressive sizes, FP16 final model"),
)

CONTRIBUTIONS = (
    ContributionSpec("gm-volume-mask", "Pruning with learned volume masks (§3.2.1, Eq. 3)",
                     ("trick-gaussian-mask", "trick-redensify", "trick-cycle")),
    ContributionSpec("sg-significance-pruning", "Significance-score pruning, 6 decaying rounds "
                     "(§3.2.2, Eq. 4)", ("trick-significance", "trick-cycle")),
    ContributionSpec("shm-sh-masking", "Per-band SH masking (§3.2.3, Eq. 5)",
                     ("trick-sh-mask",)),
    ContributionSpec("bl-progressive-blur", "Progressive Gaussian blurring of training images "
                     "(§3.2.4a)", ("trick-progressive",)),
    ContributionSpec("ds-progressive-resolution", "Progressive resolution (§3.2.4b, Eq. 6)",
                     ("trick-progressive", "trick-cycle")),
    ContributionSpec("sc-progressive-scale", "Progressive low-pass scale, ABE split and 20 % "
                     "lowest-error SfM init (§3.2.4c)", ("trick-lowpass-abe",)),
    ContributionSpec("at-accelerated-training", "SH bands updated every 16 steps and separable "
                     "SSIM (§3.2.5)", ("trick-accelerated",)),
    ContributionSpec("de-late-densification", "Densification re-enabled 20K-20.5K (§4.2)",
                     ("trick-redensify", "trick-cycle")),
    ContributionSpec("fp16-storage", "FP16 parameters after training (§4.1)", ("trick-cycle",)),
)


def method():
    base = make_method("trick-gs")
    renderer = ComponentSpec(
        "trick-renderer", "renderer", "ss_trick_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"gs_threshold": DEFAULTS["gs_threshold"], "sh_threshold": DEFAULTS["sh_threshold"]},
    )
    density_cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in DEFAULTS.items()
                   if k != "sfm_error_dir"}
    density = ComponentSpec(
        "trick-density", "density", "ss_trick_gs.density:create_density",
        ("gradient.screen.signed",), (), density_cfg,
    )
    photometric = ComponentSpec(
        "photometric", "loss", "ss_trick_gs.losses:create_photometric",
        ("render.rgb", SCHEDULE_CAPABILITY), (), {"ssim_weight": 0.2},
    )
    gm = ComponentSpec("gaussian-mask", "loss", "ss_trick_gs.losses:create_gaussian_mask",
                       (SCHEDULE_CAPABILITY,), (), {"weight": 0.05})
    shm = ComponentSpec("sh-mask", "loss", "ss_trick_gs.losses:create_sh_mask",
                        (SCHEDULE_CAPABILITY,), (), {"weight": 0.05})
    fields = (
        FieldSpec(GS_MASK, (), learning_rate=0.5, initial_value=1.0),
        FieldSpec(SH_MASK, (3,), learning_rate=0.05, initial_value=0.0),
    )
    return replace(
        base,
        description="Trick-GS (arXiv 2501.14534) native port, all eight tricks, written from the "
        "paper (no official code); paper fidelity unverified.",
        components=(renderer, density, photometric, gm, shm),
        fields=fields,
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def _variant(name, description, **density_overrides):
    m = method()
    comps = tuple(replace(c, config={**c.config, **density_overrides}) if c.kind == "density"
                  else c for c in m.components)
    return replace(m, name=name, description=description, components=comps)


def iteration1():
    """Gap-analysis iteration 1 (post-hoc): SH masking starts at 15001 as in the RDO-Gaussian
    official code (`sh_prune_start_iter`), the source Trick-GS cites for SHM. With the 3500 start
    the masks close on still-untrained (near-zero, 16-step-batched) SH coefficients, whose mask
    gradient sum(c * dL/dc_hat) is then ~0, so ~99 % of Gaussians lose all SH bands
    (gap-analysis-1.md); Fig. 3 of the paper shows ~80 %."""
    return _variant("trick-gs-it1", "Trick-GS iteration 1 (post-hoc, gap analysis 1): SH "
                    "masking from 15001 (RDO-Gaussian code); paper fidelity unverified.",
                    sh_mask_start=15001)


def ablate_no_shm():
    """Diagnostic only (gap analysis 1): SH masking never activates."""
    return _variant("trick-gs-ablate-noshm", "Trick-GS diagnostic ablation without SH masking; "
                    "not a paper configuration.", sh_mask_start=10**9)
