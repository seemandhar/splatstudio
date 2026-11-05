"""DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds (Chen et al., CVPR 2025,
arXiv 2503.18402). Native SplatStudio port of the 3DGS-backbone variant of the official code
https://github.com/YouyuChen0207/DashGaussian (train_dash.py, utils/schedule_utils.py,
scene/gaussian_model.py:prune_and_densify); paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES, SCHEDULE_CAPABILITY

CITATIONS = (
    "https://arxiv.org/abs/2503.18402",
    "https://github.com/YouyuChen0207/DashGaussian",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_dashgaussian.probes:{fn}", description, cuda)


PROBES = (
    _probe("dash-renderer-gradient", "renderer_gradient",
           "Built-in renderer-gradient check with the schedule built: finite nonzero screen "
           "gradients reach the density statistics at the scheduled resolution", True),
    _probe("dash-reso-schedule", "reso_schedule",
           "FFT significance schedule: white-noise analytic r_m=2 and r_i=(E_i/E)^-1/2, grid-search "
           "reference of levels, supp. Eq. 7 begin steps, floored inverse-square interpolation, "
           "monotone to r=1 at 27000; 'const' mode differs"),
    _probe("dash-render-scale", "render_scale",
           "Training render = floor(H/r) x floor(W/r) equal to gsplat with hand-scaled K; "
           "no_grad (evaluation) render is full resolution", True),
    _probe("dash-target-downsample", "target_downsample",
           "Loss target equals PIL bilinear reduce (= torch antialias bilinear); loss equals "
           "L1 + 0.2 D-SSIM against it; full-size path untouched"),
    _probe("dash-primitive-schedule", "primitive_schedule",
           "Densify rate equals Eq. 4 (code form, exponent 2 - it/27000, P_fin after 26900, cap "
           "0.2); momentum Eq. 5 with max and 1e6 cap; P_fin^(0) = 6 P_init"),
    _probe("dash-topk-densify", "topk_densify",
           "Prune-first, exactly the top-k qualified Gaussians cloned (small) or split (large), "
           "no size pruning, zero Adam moments, P_add = qualified count"),
    _probe("dash-lr-schedule", "lr_schedule",
           "Means LR held until the first level with r < 2, then log-linear to 1 % at 30000"),
    _probe("dash-cycle", "cycle",
           "Native runtime run on demo data with a compressed schedule: render scale decreases "
           "monotonically to 1, evaluation at full resolution", True),
)
CONTRIBUTIONS = (
    ContributionSpec("resolution-scheduler", "Frequency-guided resolution scheduler (Sec. 3.2, "
                     "Eq. 2-3, supp. Eq. 6-7) with antialiased GT downsampling (supp. 7.1)",
                     ("dash-reso-schedule", "dash-render-scale", "dash-target-downsample",
                      "dash-cycle")),
    ContributionSpec("primitive-scheduler", "Resolution-guided primitive growth (Eq. 4) with "
                     "top-k prune-first densification (supp. 7.4)",
                     ("dash-primitive-schedule", "dash-topk-densify")),
    ContributionSpec("momentum-budget", "Momentum-based primitive budgeting (Eq. 5)",
                     ("dash-primitive-schedule",)),
    ContributionSpec("position-lr-schedule", "Position LR held during the low-resolution stage "
                     "(supp. 7.2)", ("dash-lr-schedule",)),
)


def method():
    base = make_method("dashgaussian")
    renderer = ComponentSpec(
        "dash-renderer", "renderer", "ss_dashgaussian.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, {},
    )
    density = ComponentSpec(
        "dash-density", "density", "ss_dashgaussian.density:create_density",
        ("gradient.screen.signed",), (), dict(DEFAULTS),
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_dashgaussian.losses:create_photometric",
        ("render.rgb", SCHEDULE_CAPABILITY), (), {"ssim_weight": 0.2, "ssim_padding": "same"},
    )
    # The built-in renderer-gradient probe drives the density without training images; the
    # DashGaussian density refuses to train without its schedule, so an equivalent probe that
    # builds the schedule first replaces it.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    contributions = tuple(
        replace(c, probes=tuple("dash-renderer-gradient" if n == "renderer-gradient" else n
                                for n in c.probes))
        for c in base.contributions
    )
    return replace(
        base,
        description="DashGaussian (arXiv 2503.18402) native port on the 3DGS backbone "
        "(resolution + primitive schedules); paper fidelity unverified.",
        components=(renderer, density, loss),
        probes=base_probes + PROBES,
        contributions=contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
