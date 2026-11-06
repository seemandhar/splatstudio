"""FisherRF (ECCV 2024, arXiv 2311.17874) — native SplatStudio port of the Gaussian-splatting
code at github.com/JiangWenPL/FisherRF (commit b747328): Fisher-information view scoring with the
official squared-gradient CUDA kernel, and the official active view-selection training loop.

Registered variants (Blender protocol defaults, `scripts/blender_seq*.sh`):
  fisherrf              sequential EIG selection, 4 init -> 20 views, 1 view / 100 epochs
  fisherrf-random       same schedule, official random selector (paper "3D Gaussian + Random")
  fisherrf-batch        batch EIG selection, 4 views / 300 epochs (Table 3)
  fisherrf-batch-random batch random baseline
  fisherrf-v10          2 init -> 10 views, 1 view / 100 epochs (Table 4)
  fisherrf-v10-random   10-view random baseline
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .active import DEFAULTS as ACTIVE_DEFAULTS
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2311.17874", "https://github.com/JiangWenPL/FisherRF")

SCHEMAS = {
    "seq1": {"total_views": 20, "views_per_round": 1, "num_init_views": 4, "interval_epochs": 100},
    "seq4": {"total_views": 20, "views_per_round": 4, "num_init_views": 4, "interval_epochs": 300},
    "v10": {"total_views": 10, "views_per_round": 1, "num_init_views": 2, "interval_epochs": 100},
}


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_fisherrf.probes:{fn}", desc, cuda)


def build(name, selector="fisher", schema="seq1", **overrides):
    base = make_method(name)
    active = {**ACTIVE_DEFAULTS, **SCHEMAS[schema], "selector": selector, **overrides}
    comps = (
        ComponentSpec("fisherrf-renderer", "renderer", "ss_fisherrf.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, active),
        ComponentSpec("fisherrf-density", "density", "ss_fisherrf.density:create_density",
                      DENSITY_REQUIRES, (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("photometric", "loss", "ss_fisherrf.losses:create_photometric",
                      ("render.rgb",), (), {"ssim_weight": 0.2}),
    )
    probes = (
        _probe("fisher-kernel", "fisher_kernel",
               "Vendored FisherRF kernel diag H equals sum over pixels of the squared per-pixel "
               "gradient of (R+G+B) from per-pixel autograd of gsplat (independent "
               "rasterizer), chained once through sigmoid/exp to raw params; differs from the "
               "square of the summed gradient", True),
        _probe("eig-selection", "eig_selection",
               "EIG score sum H_c/(H_train+lambda) vs numpy; greedy batch == sequential "
               "re-scoring; random selector == random.Random(seed) shuffle of sorted pool"),
        _probe("active-schedule", "active_schedule",
               "Official VNSeqMInplace load_its (seq1/seq4/v10) and farthest-point init views"),
        _probe("active-loop", "active_loop",
               "Mini runtime loop: only selected views update parameters, initial Gaussians and "
               "fresh optimizers restored at each acquisition, pool = training cameras only, "
               "budget reached, early evaluation refused", True),
    )
    contribs = (
        ContributionSpec("fisher-statistic-rendering",
                         "Diagonal Fisher information diag(J^T J) of the Gaussian parameters "
                         "per view via the official squared-gradient CUDA backward (Eq. 7-8)",
                         ("fisher-kernel",)),
        ContributionSpec("eig-view-selection",
                         "Next-best-view / greedy batch by tr(H_cand H_train^-1) (Eq. 5, Alg. 1) "
                         "with the official random baseline", ("eig-selection",)),
        ContributionSpec("active-training-loop",
                         "Official acquisition schedule, initial views, restore-to-init after "
                         "each acquisition, view budget (Sec. 4.1, active_train.py)",
                         ("active-schedule", "active-loop")),
    )
    return replace(
        base,
        description="FisherRF (ECCV 2024) native port: official Fisher CUDA kernel + active "
                    f"view selection ({selector}, {schema}); paper fidelity unverified.",
        components=comps,
        probes=base.probes + probes,
        contributions=base.contributions + contribs,
        citations=CITATIONS,
    )


def method():
    return build("fisherrf")


def random_baseline():
    return build("fisherrf-random", selector="random")


def batch():
    return build("fisherrf-batch", schema="seq4")


def batch_random():
    return build("fisherrf-batch-random", selector="random", schema="seq4")


def v10():
    return build("fisherrf-v10", schema="v10")


def v10_random():
    return build("fisherrf-v10-random", selector="random", schema="v10")
