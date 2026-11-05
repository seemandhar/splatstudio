"""AD-GS: Alternating Densification for Sparse-Input 3DGS (arXiv 2509.11003) — native port.

Defaults are the paper / official-code values (supp. Table 7, scripts/run_llff_ad.sh).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .renderer import CAPABILITIES

SCHEDULE = {"warmup_iters": 1500, "cycle_len": 200, "high_dens_iters": 100}


def method():
    base = make_method("ad-gs")
    renderer = ComponentSpec(
        "adgs-renderer",
        "renderer",
        "ss_ad_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        tuple(sorted(CAPABILITIES)),
        {"train_background": "random", "eval_background": "black", "sh_interval": 500,
         "max_sh_degree": 3, "eval_sh_degree": 3},
    )
    density = ComponentSpec(
        "adgs-density",
        "density",
        "ss_ad_gs.density:create_density",
        ("gradient.screen.signed",),
        (),
        dict(DENSITY_DEFAULTS),
    )
    photometric = base.components[2]  # L_ph of G1, lambda_ssim = 0.2
    depth = ComponentSpec(
        "adgs-depth",
        "loss",
        "ss_ad_gs.losses:create_depth_smoothness",
        ("render.depth.accumulated_z",),
        (),
        {**SCHEDULE, "weight": 0.01, "range_weight": 0.001},
    )
    cotrain = ComponentSpec(
        "adgs-cotrain",
        "loss",
        "ss_ad_gs.losses:create_cotraining",
        ("render.camera", "render.rgb", "render.depth.accumulated_z"),
        (),
        {**SCHEDULE, "depth_weight": 0.01, "range_weight": 0.001, "pseudo_weight": 1.0,
         "start_sample_pseudo": 500, "lambda_ssim": 0.2},
    )
    probes = (
        ProbeSpec("alternating-densification", "ss_ad_gs.probes:alternating_densification",
                  "Phase schedule and per-phase (tau, epsilon) densify/prune on a fixture"),
        ProbeSpec("phase-gated-losses", "ss_ad_gs.probes:phase_gated_losses",
                  "Depth/pseudo regularisers active only in low-densification steps after warm-up"),
        ProbeSpec("edge-aware-depth", "ss_ad_gs.probes:edge_aware_depth",
                  "Eq. 7 value and gradient against an independent per-pixel loop"),
        ProbeSpec("pseudo-view-consistency", "ss_ad_gs.probes:pseudo_view_consistency",
                  "Two-field co-training: G1 and G2 receive detached-target pseudo-view "
                  "gradients; pseudo poses near the training cameras", True),
        ProbeSpec("sh-schedule", "ss_ad_gs.probes:sh_schedule",
                  "Training renders use SH degree min(t // 500, 3); evaluation degree 3"),
    )
    contributions = (
        ContributionSpec("alternating-densification",
                         "Warm-up / high / low densification thresholds (paper §3.3)",
                         ("alternating-densification",)),
        ContributionSpec("alternating-losses",
                         "Regularisers only in the low-densification phase (Eq. 9)",
                         ("phase-gated-losses",)),
        ContributionSpec("pseudo-view-consistency",
                         "Co-trained second field, photometric consistency on pseudo views (Eq. 6)",
                         ("pseudo-view-consistency",)),
        ContributionSpec("edge-aware-depth-smoothness",
                         "Edge-aware depth smoothness + depth range term (Eq. 7)",
                         ("edge-aware-depth",)),
        ContributionSpec("official-training-protocol",
                         "Iteration 1: official SH schedule (every 500 steps)",
                         ("sh-schedule",)),
    )
    return replace(
        base,
        description="AD-GS native port (two co-trained fields, alternating densification); "
        "paper fidelity unverified.",
        components=(renderer, density, photometric, depth, cotrain),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=("https://arxiv.org/abs/2509.11003", "https://github.com/gurutvapatle/AD-GS"),
    )
