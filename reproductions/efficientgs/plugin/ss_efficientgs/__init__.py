"""EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene
Representation (arXiv 2404.12777). Native SplatStudio port of the official code
https://github.com/wenkaier/EfficientGS; paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, REQUIRES
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2404.12777", "https://github.com/wenkaier/EfficientGS")

# Per-Gaussian SH degree (Sec. 3.3). Not part of the loss graph, so Adam never updates it; the
# tiny learning rate only satisfies the FieldSpec contract. Density ops copy/drop it with the
# population and the checkpoint stores it for evaluation.
SH_ORDER = FieldSpec("sh_order", (1,), learning_rate=1e-12, initial_value=0.0)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_efficientgs.probes:{fn}", description, cuda)


def method():
    base = make_method("efficientgs")
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "egs-renderer",
        "renderer",
        "ss_efficientgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        CAPABILITIES,
        {},
    )
    density = ComponentSpec(
        "egs-density",
        "density",
        "ss_efficientgs.density:create_density",
        REQUIRES,
        (),
        dict(DEFAULTS),
    )
    probes = (
        _probe("egs-gradnorm-kernel", "gradnorm_kernel",
               "CUDA Eq. 6 statistic equals a brute-force per-pixel autograd sum of NDC gradient "
               "norms; summed vectors equal gsplat's means2d.grad; cancellation case S >> |sum|"),
        _probe("egs-selective-densify", "selective_densify",
               "A Gaussian whose per-pixel gradients cancel is densified under Eq. 6 but not "
               "under the 3DGS statistic (Eq. 5) at the same threshold"),
        _probe("egs-stats-kernel", "stats_kernel",
               "CUDA top-K flags (Eq. 8) and colour-error weights (Eq. 9) equal a brute-force "
               "per-pixel blend, for both weight conventions and K = 1, 2"),
        _probe("egs-topk-prune", "topk_prune",
               "Pruning removes a Gaussian that is never a pixel's top-1 weight (K=1) and keeps "
               "it at K=2; dominant Gaussians survive"),
        _probe("egs-sparse-sh", "sparse_sh",
               "Per-Gaussian SH degree render equals gsplat at that degree; Eq. 9 selection "
               "raises the top r_s by one degree, capped at 3"),
        _probe("egs-schedule", "schedule",
               "Event steps (prune 15k/18k/21k, SH alloc 15.5k, increments 16k/17k/18k), "
               "shN lr 5e-4, degree 0 before the first increment", False),
    )
    contributions = (
        ContributionSpec("selective-densification",
                         "Densify by the per-pixel gradient-norm sum S_g > 0.0007 (Sec. 3.1, Eq. 6)",
                         ("egs-gradnorm-kernel", "egs-selective-densify")),
        ContributionSpec("gaussian-pruning",
                         "Keep only Gaussians that are a top-K blending weight of some pixel "
                         "(Sec. 3.2, Eq. 7-8)", ("egs-stats-kernel", "egs-topk-prune")),
        ContributionSpec("sparse-sh-order-increment",
                         "SH degree 0 until after pruning, then +1 degree for the 20% Gaussians "
                         "with the largest colour error (Sec. 3.3, Eq. 9)",
                         ("egs-stats-kernel", "egs-sparse-sh")),
        ContributionSpec("pipeline-schedule", "Official train.sh schedule and SH learning rate",
                         ("egs-schedule",)),
    )
    return replace(
        base,
        description="EfficientGS (arXiv 2404.12777) native port; paper fidelity unverified.",
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        fields=(SH_ORDER,),
        citations=CITATIONS,
    )
