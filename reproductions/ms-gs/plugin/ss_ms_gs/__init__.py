"""Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering (Yan, Low, Chen, Lee, CVPR 2024,
arXiv 2311.17089). Native SplatStudio port of the official code
https://github.com/JokerYan/MS-GS @ 50e5040 (train.py --ms_train --filter_small --insert_large,
scene/gaussian_model.py) and its rasterizer fork https://github.com/JokerYan/MS-GS-rasterizer
@ e7a2b4f (pixel_sizes / selective culling in cuda_rasterizer/forward.cu); paper fidelity
unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .renderer import CAPABILITIES, SELECTION_CAPABILITY
from .renderer import DEFAULTS as RENDER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2311.17089",
    "https://github.com/JokerYan/MS-GS",
)

# Non-trainable per-Gaussian metadata. They never enter the autograd graph, so Adam never
# updates them; the contract requires a positive learning rate, hence the nominal 1e-8.
FIELDS = (
    FieldSpec("ms_level", (), learning_rate=1e-8, initial_value=0.0),
    FieldSpec("ms_max_px", (), learning_rate=1e-8, initial_value=-1.0),
    FieldSpec("ms_min_px", (), learning_rate=1e-8, initial_value=-1.0),
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_ms_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("msgs-pixel-coverage", "pixel_coverage_probe",
           "S_k = min half-extent of the undilated 2D Gaussian at the 1/255 level set equals an "
           "independent float64 derivation (3DGS computeCov2D) and the analytic fronto-parallel "
           "case; 0 for z <= 0.2"),
    _probe("msgs-selection-rule", "selection_rule_probe",
           "Culling truth table of forward.cu: rel_min < 0.5 and S < 2 px, or rel_max > 2.0; "
           "uninitialised ranges never cull; large filter nested in small filter"),
    _probe("msgs-range-update", "range_update_probe",
           "Eq. 6 code form: max <- max(0.95 max, S) for level > 0, min <- min(1.05 min, S) "
           "below the coarsest index, only for visible Gaussians of the rendered level"),
    _probe("msgs-aggregation", "aggregation_probe",
           "Voxel average pooling (floor(p / (0.02 r / 4)) after one-sided 2 - 1/x contraction) "
           "equals an independent grouping loop; scale x 2 / clip(S_avg, 0.25, 2); new level, "
           "range -1"),
    _probe("msgs-schedule", "ms_schedule_probe",
           "Per-epoch resolution: 1x w.p. 0.75 else least-trained enabled level, levels 2-3/4-5/"
           "6-7 after 1010/1020/1030, 2x never; loss x 0.1 off 1x; position LR held after "
           "30000"),
    _probe("msgs-densify-fields", "densify_fields_probe",
           "Only level-0 Gaussians clone/split/prune/opacity-reset; split children get S range "
           "/ 1.6"),
    _probe("msgs-selective-render", "selective_render",
           "CUDA: aggregation inserts coarse Gaussians for tiny ones; at 1x the coarse ones are "
           "culled, at 4x the tiny fine ones are culled and the coarse ones drawn", True),
    _probe("msgs-cuda-kernel", "cuda_kernel_probe",
           "Fused CUDA coverage+selection kernel (csrc/coverage.cu) equals the torch reference "
           "(S rel. error <= 1e-3, <= 5 mask flips of 20000)", True),
)
CONTRIBUTIONS = (
    ContributionSpec("pixel-coverage", "Pixel coverage of a splatted Gaussian (Sec. 4.1, Fig. 3)",
                     ("msgs-pixel-coverage", "msgs-cuda-kernel", "msgs-selective-render")),
    ContributionSpec("aggregate-insert", "Aggregate small Gaussians into coarse levels "
                     "(Alg. 1, supp. Sec. 2)", ("msgs-aggregation", "msgs-selective-render")),
    ContributionSpec("selective-rendering", "Selective rendering by relative pixel coverage with "
                     "per-Gaussian S_max/S_min (Eq. 6-7, Alg. 2)",
                     ("msgs-selection-rule", "msgs-range-update", "msgs-selective-render")),
    ContributionSpec("multiscale-training", "Multi-scale training schedule with level-0-only "
                     "density control (Sec. 4.1, official train.py)",
                     ("msgs-schedule", "msgs-densify-fields")),
)


def method():
    base = make_method("ms-gs")
    renderer = ComponentSpec(
        "ms-gs-renderer", "renderer", "ss_ms_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, dict(RENDER_DEFAULTS),
    )
    density = ComponentSpec(
        "ms-gs-density", "density", "ss_ms_gs.density:create_density",
        ("gradient.screen.signed",), (),
        {**DENSITY_DEFAULTS, "scale_clip": list(DENSITY_DEFAULTS["scale_clip"]),
         "render": dict(RENDER_DEFAULTS)},
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_ms_gs.losses:create_photometric",
        ("render.rgb", SELECTION_CAPABILITY), (), {"ssim_weight": 0.2, "low_res_weight": 0.1},
    )
    return replace(
        base,
        description="Multi-Scale 3DGS (arXiv 2311.17089) native port: pixel-coverage selective "
        "rendering, aggregated coarse levels, multi-scale training; paper fidelity unverified.",
        components=(renderer, density, loss),
        fields=FIELDS,
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
