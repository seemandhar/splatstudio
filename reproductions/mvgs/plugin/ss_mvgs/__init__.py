"""MVGS: Multi-view Regulated Gaussian Splatting for Novel View Synthesis (Du, Wang, Yu;
arXiv 2410.02103). Native SplatStudio port of the 3DGS variant of the official code
https://github.com/xiaobiaodu/MVGS @ 36d073a; paper fidelity unverified.

`mvgs`          official indoor Mip-NeRF 360 recipe (run_360.py MV_normal): cross-intrinsic stages
                M = 48/24/12/8 at -r 8/4/2/-1 for 3000/3000/3000/30000 iterations (39000 steps).
`mvgs-outdoor`  official bicycle/garden/stump recipe (MV_largest): M = 3 at -r -1, 30000 steps.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, INDOOR_STAGES, OUTDOOR_STAGES
from .renderer import BATCH_CAPABILITY, CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2410.02103",
    "https://github.com/xiaobiaodu/MVGS",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_mvgs.probes:{fn}", description, cuda)


PROBES = (
    _probe("mvgs-multiview-sum", "multiview_sum",
           "A 4-view batch of distinct stack-sampled views: loss equals the SUM of independently "
           "rendered single-view 3DGS losses and its gradient the sum of per-view gradients "
           "(fails for mean, single view or repeated views)", True),
    _probe("mvgs-cig-schedule", "cig_schedule",
           "Stage boundaries 3000/3001/9000/9001/39000, M per stage, official -r sizes, stage-local "
           "densify/reset/LR restart; training render at the stage size equals gsplat with "
           "hand-scaled K and evaluation render is full resolution", True),
    _probe("mvgs-mvad", "mvad",
           "Threshold halved iff some pair of unit camera directions is > 60 degrees apart "
           "(independent arccos form); unchanged for one view or when disabled"),
    _probe("mvgs-crd", "crd",
           "Max-loss window equals a port of the official loop; pixel rays equal independent "
           "unprojection; closest-point midpoint of rays through a common point is that point; "
           "Gaussians inside the cross-ray cuboid are cloned regardless of size/gradient with zero "
           "Adam moments, others untouched, nothing cloned with crd off", True),
    _probe("mvgs-cycle", "cycle",
           "Renderer/loss/density through a compressed 2-stage schedule on synthetic frames: "
           "statistics count only the last view's visible Gaussians, densification runs, stage "
           "sizes switch, LR restarts at the stage boundary, optimizer state stays aligned", True),
)

CONTRIBUTIONS = (
    ContributionSpec("multi-view-regulated-learning",
                     "M views per iteration with summed 3DGS loss (Sec. 3.1, Eq. 4)",
                     ("mvgs-multiview-sum", "mvgs-cycle")),
    ContributionSpec("cross-intrinsic-guidance",
                     "Coarse-to-fine intrinsics/resolution stages with matched M (Sec. 3.2, Eq. 9)",
                     ("mvgs-cig-schedule", "mvgs-cycle")),
    ContributionSpec("multi-view-augmented-densification",
                     "Halved densification threshold for distinct views (Sec. 3.3.1, Eq. 10)",
                     ("mvgs-mvad",)),
    ContributionSpec("cross-ray-densification",
                     "Clone Gaussians in cross-ray cuboids of max-loss windows (Sec. 3.3.2, supp. "
                     "Alg. 1)", ("mvgs-crd",)),
)


def _build(name, stages, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "mvgs-renderer", "renderer", "ss_mvgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, {},
    )
    density = ComponentSpec(
        "mvgs-density", "density", "ss_mvgs.density:create_density",
        ("gradient.screen.signed",), (),
        {**DEFAULTS, "stages": [list(s) for s in stages]},
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_mvgs.losses:create_photometric",
        ("render.rgb", BATCH_CAPABILITY), (), {"ssim_weight": 0.2, "ssim_padding": "same"},
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, loss),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _build(
        "mvgs", INDOOR_STAGES,
        "MVGS (arXiv 2410.02103) native port, official indoor recipe (CIG 48/24/12/8 views, "
        "39000 steps; run with --steps 39000); paper fidelity unverified.",
    )


def method_outdoor():
    return _build(
        "mvgs-outdoor", OUTDOOR_STAGES,
        "MVGS (arXiv 2410.02103) native port, official outdoor recipe (3 views, 30000 steps); "
        "paper fidelity unverified.",
    )
