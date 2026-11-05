"""SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and
High-Quality Mesh Rendering (Guédon & Lepetit, CVPR 2024, arXiv 2311.12775).

Native SplatStudio port of the official pipeline https://github.com/Anttwo/SuGaR
(`train.py -r sdf --high_poly True --refinement_time long`); paper fidelity unverified.
One run of 30000 steps = vanilla 3DGS 1-7000, coarse SuGaR (SDF + normal regularization)
7001-15000, level-set + Poisson mesh extraction after step 15000, mesh-bound Gaussian
refinement 15001-30000 (R-SuGaR-15K, 1 Gaussian per triangle, 1M-triangle fg/bg meshes).
`sugar-smoke` compresses the stage boundaries for short smoke tests only.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .losses import CONSISTENCY, ENTROPY, NORMAL, SDF
from .renderer import CAPS

CITATIONS = ("https://arxiv.org/abs/2311.12775", "https://github.com/Anttwo/SuGaR")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_sugar.probes:{fn}", description, cuda)


def _make(name, description, stages):
    base = make_method(name)
    _, base_density, photometric = base.components
    renderer = ComponentSpec(
        "sugar-renderer", "renderer", "ss_sugar.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPS, {"alpha_cap": 0.999},
    )
    density_cfg = {**base_density.config, "reset_every": 3000,
                   "refine_stop_iter": stages["coarse_start"],
                   **{k: stages[k] for k in ("coarse_start", "prune_at", "extract_at")}}
    density = ComponentSpec(
        "sugar-density", "density", "ss_sugar.density:create_density",
        ("gradient.screen.signed",), (), density_cfg,
    )
    entropy = ComponentSpec(
        "sugar-entropy", "loss", "ss_sugar.losses:create_entropy", ("render.rgb",), (),
        {**ENTROPY, "start": stages["coarse_start"], "end": stages["prune_at"]},
    )
    geo = ("render.depth.accumulated", "training.camera")
    sdf = ComponentSpec("sugar-sdf", "loss", "ss_sugar.losses:create_sdf", geo, (),
                        {**SDF, "start": stages["prune_at"]})
    normal = ComponentSpec("sugar-normal", "loss", "ss_sugar.losses:create_normal", geo, (),
                           {**NORMAL, "start": stages["prune_at"]})
    consistency = ComponentSpec(
        "sugar-normal-consistency", "loss", "ss_sugar.losses:create_consistency",
        ("sugar.mesh_binding",), (), dict(CONSISTENCY),
    )
    probes = (
        _probe("sdf-field", "sdf_field",
               "Implemented f(x) equals an independent beta*sqrt(-2 log d); a flat opaque "
               "Gaussian gives the plane distance; R is 0 for a consistent depth map and grows "
               "with a depth offset"),
        _probe("normal-regularizer", "normal_regularizer",
               "R_Norm is 0 for co-planar neighbours, equals an independent per-sample loop for "
               "tilted ones and reaches the quaternions"),
        _probe("level-set", "level_set",
               "Flat Gaussian wall: lambda=0.3 points at the analytic offset s*sqrt(-2 ln 0.3) on "
               "the camera side, normals along the wall axis", True),
        _probe("poisson-mesh", "poisson",
               "Open3D Poisson+decimation of sphere level points: radius error < 2 %, triangle "
               "target honoured, vertices snapped onto the input points"),
        _probe("mesh-binding", "mesh_binding",
               "Bound means = barycentres, rotation = [normal, edge rotated by atan2(y,x)] vs "
               "Rodrigues, scales (eps,s1,s2), vertex gradient, normal consistency 0 for a "
               "plane and > 0 for a fold, vertex LR schedule"),
        _probe("stage-schedule", "stage_schedule",
               "Entropy only inside its window; prune < 0.5 at prune_at; extraction replaces "
               "the population with mesh-bound Gaussians, optimizers aligned, renderer and "
               "checkpoint fields consistent", True),
    )
    contributions = (
        ContributionSpec("sdf-regularization", "Surface-alignment SDF regularizer R with "
                         "opacity entropy and pruning (Sec. 3.1, Eq. 8, Sec. 4.1)",
                         ("sdf-field", "stage-schedule")),
        ContributionSpec("normal-regularization", "Normal regularizer R_Norm (Eq. 10)",
                         ("normal-regularizer",)),
        ContributionSpec("level-set-mesh-extraction", "Level-set points from depth maps and "
                         "Poisson reconstruction (Sec. 3.2)", ("level-set", "poisson-mesh")),
        ContributionSpec("mesh-bound-refinement", "Gaussians bound to a learnable mesh, 2D "
                         "rotation and scales, normal consistency (Sec. 3.3, supp. 1-2)",
                         ("mesh-binding", "stage-schedule")),
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric, entropy, sdf, normal, consistency),
        fields=(FieldSpec("sugar_rot2d", (2,), 0.001, 0.0),),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


PAPER = {"coarse_start": 7000, "prune_at": 9000, "extract_at": 15000}
SMOKE = {"coarse_start": 1000, "prune_at": 1300, "extract_at": 2000}


def method():
    return _make("sugar", "SuGaR (arXiv 2311.12775) native port of the official sdf pipeline "
                 "(run 30000 steps = R-SuGaR-15K); paper fidelity unverified.", PAPER)


def smoke():
    return _make("sugar-smoke", "SuGaR with compressed stage boundaries (3DGS 1000, coarse to "
                 "2000, refine after) for smoke tests only; not the paper protocol.", SMOKE)
