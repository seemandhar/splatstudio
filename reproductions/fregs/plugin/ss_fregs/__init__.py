"""FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization (arXiv 2403.06908).

Native SplatStudio port; paper fidelity unverified. All hyper-parameter defaults live in
`ss_fregs.defaults.DEFAULTS`; w_l, w_h and grow_grad2d there are calibration placeholders.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .defaults import DEFAULTS, THRESHOLD_GRID

CITATION = "https://arxiv.org/abs/2403.06908"


def _make(name, *, density=None, frequency=None, count_matched=True):
    """Build a FreGS variant; overrides are stored in ComponentSpec.config (spec hash)."""
    density_cfg = {**DEFAULTS["density"], **(density or {})}
    frequency_cfg = {**DEFAULTS["frequency"], **(frequency or {})}
    base = make_method(name)
    renderer, _, photometric = base.components
    density_spec = ComponentSpec(
        "fregs-density",
        "density",
        "splatstudio.backends.gsplat:create_density",
        ("gradient.screen.signed",),
        (),
        density_cfg,
    )
    frequency_spec = ComponentSpec(
        "fregs-frequency",
        "loss",
        "ss_fregs.frequency:create_frequency_loss",
        ("render.rgb",),
        (),
        frequency_cfg,
    )
    anneal = bool(frequency_cfg["anneal"])
    probes = [
        ProbeSpec(
            "freq-oracle",
            "ss_fregs.probes:freq_oracle",
            "L_f value and dL_f/drgb vs an independent float64 explicit-DFT oracle (Eq. 2-6)",
        ),
        ProbeSpec(
            "freq-annealing" if anneal else "freq-schedule",
            "ss_fregs.probes:freq_schedule",
            "Known-frequency sinusoids: band D0 < r <= D_t grows from T0 to T; L_f = 0 after "
            "stop_iter"
            if anneal
            else "Known-frequency sinusoid seen from step 1 (no annealing); L_f = 0 after stop_iter",
        ),
        ProbeSpec(
            "fregs-high-band-gain",
            "ss_fregs.probes:high_band_gain",
            "w_h > 0 and the high band's screen-gradient gain over the w_h = 0 ablation exceeds 10x "
            "the float noise floor at t=T (t=1 without annealing)",
            True,
        ),
        ProbeSpec(
            "fregs-gradient-gain",
            "ss_fregs.probes:gradient_gain",
            "Synthetic scene: t=1 screen-gradient ratio > 1.05 vs the no-op ablation's 1.000",
            True,
        ),
    ]
    probes.append(
        ProbeSpec(
            "fregs-calibration-ratio",
            "ss_fregs.probes:calibration_ratio",
            "Real data (garden train views, cached 15k reference): t=1 ratio 1.7 +- 0.15 and t=T "
            "ratio in Fig. 2's [1.15, 2.3]; unverified without the files",
            True,
        )
    )
    contributions = [
        ContributionSpec(
            "fig2-weight-calibration",
            "Calibrated w_l, w_h reproduce Fig. 2's screen-gradient ratios on real training views",
            ("fregs-calibration-ratio",),
        ),
        ContributionSpec(
            "frequency-regularization",
            "Amplitude and phase discrepancies in Fourier space (Sec. 3.2, Eq. 2-6)",
            ("freq-oracle", "fregs-gradient-gain"),
        ),
    ]
    if anneal:
        contributions.append(
            ContributionSpec(
                "frequency-annealing",
                "Fixed low-pass plus linearly growing high-pass band from T0 to T; L_f stops "
                "with densification (Sec. 3.3, Sec. 4.1)",
                ("freq-annealing", "fregs-high-band-gain"),
            )
        )
    else:
        contributions.append(
            ContributionSpec(
                "frequency-stop",
                "All-band L_f from step 1 (Base+FR), stopped with densification (Sec. 4.1)",
                ("freq-schedule", "fregs-high-band-gain"),
            )
        )
    if count_matched:  # only the fregs-thr0002 diagnostic opts out; <= 0.0002 fails the probe
        probes.append(
            ProbeSpec(
                "fregs-threshold",
                "ss_fregs.probes:threshold",
                "Raised grow_grad2d is applied by gsplat DefaultStrategy (no growth between "
                "0.0002 and the threshold; base strategy grows)",
            )
        )
        contributions.append(
            ContributionSpec(
                "count-matched-densification",
                "Raised view-space gradient threshold keeps #Gaussians near 3DGS (Sec. 4.2)",
                ("fregs-threshold",),
            )
        )
    return replace(
        base,
        description="FreGS (CVPR 2024) native port; paper fidelity unverified. "
        "w_l, w_h, grow_grad2d are calibrated (or placeholder) values, not paper values.",
        components=(renderer, density_spec, photometric, frequency_spec),
        probes=base.probes + tuple(probes),
        contributions=base.contributions + tuple(contributions),
        citations=(CITATION,),
    )


def method():
    """FreGS: Base + FR + FA with the raised (count-matched) densification threshold."""
    return _make("fregs")


def fregs_fr():
    """Table 2 ablation Base+FR: all bands from step 1 (no annealing), same w_l, w_h, threshold."""
    return _make("fregs-fr", frequency={"anneal": False})


def fregs_thr0002():
    """Diagnostic: FreGS with the unchanged 3DGS threshold 0.0002."""
    return _make("fregs-thr0002", density={"grow_grad2d": 0.0002}, count_matched=False)


def _calib(value):
    return _make(f"fregs-calib-g{round(value * 1e4):04d}", density={"grow_grad2d": value})


def calib_g0003():
    return _calib(THRESHOLD_GRID[0])


def calib_g0004():
    return _calib(THRESHOLD_GRID[1])


def calib_g0005():
    return _calib(THRESHOLD_GRID[2])


def calib_g0006():
    return _calib(THRESHOLD_GRID[3])
