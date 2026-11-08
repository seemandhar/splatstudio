"""BOGausS: Better Optimized Gaussian Splatting (Pateux et al., arXiv 2504.01844).

Native SplatStudio port written from the paper (no official code is available); paper
fidelity unverified. Method `bogauss` = "Ours (Big)": final count = the vanilla 3DGS count of
the scene (Taming 3DGS big budgets, looked up from the initial SfM point count; set the density
config `budget` for other scenes).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import TARGET_CAPABILITY

CITATIONS = ("https://arxiv.org/abs/2504.01844",)
RENDERER_CAPS = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                 TARGET_CAPABILITY)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_bogauss.probes:{fn}", description, cuda)


PROBES = (
    _probe("bogauss-precision", "precision",
           "Delta_i equals alpha*tr(sum_c pinv lateral precision)^-1/2 from explicit per-camera "
           "matrices; halves by sqrt(2) with duplicated cameras, scales linearly with depth"),
    _probe("bogauss-unbiased-sparse-adam", "unbiased_sparse_adam",
           "Per-Gaussian lifespan t counts only visible updates; alpha=max(1/t,1-beta) gives "
           "unbiased first steps; invisible rows frozen; Delta-scaled; differs from dense Adam"),
    _probe("bogauss-linear-size", "linear_size",
           "Size update is linear in sigma (grad/sigma, step eta*Delta), clamped at the "
           "minimal size Delta; differs from a log-space update"),
    _probe("bogauss-state-inheritance", "state_inheritance",
           "Split children inherit t*1, 0.2*m, 0.04*v; untouched Gaussians keep their states; "
           "parameters stay trainable leaves"),
    _probe("bogauss-stats-kernel", "stats_kernel",
           "CUDA per-Gaussian sums of T*alpha, T*alpha*E(p) and footprint equal a brute-force "
           "per-pixel loop over gsplat's tile lists; blend weights sum to gsplat's alpha", True),
    _probe("bogauss-prune", "prune",
           "Effective-opacity pruning removes exactly the lowest P_prune < 0.02 Gaussians, "
           "capped at 1 % of the population"),
    _probe("bogauss-split", "split",
           "Split-only densification along the longest axis at +-0.3 sigma; SplitAdaptation "
           "children reproduce the mother's alpha profile far better than unadapted or 3DGS "
           "(/1.6) children"),
    _probe("bogauss-split-priority", "split_priority",
           "Split candidates are the top P_SNR = sum_c SE_ci / SE_c, which differs from the "
           "absolute-error ranking"),
    _probe("bogauss-schedule", "schedule",
           "Densification every 500 steps from 1000 to 15000, growth targets reach the budget, "
           "no opacity reset"),
)
CONTRIBUTIONS = (
    ContributionSpec("precision-confidence", "Reconstruction confidence Delta_i from the camera "
                     "setup (Eq. precision) and the derived minimal Gaussian size",
                     ("bogauss-precision", "bogauss-linear-size")),
    ContributionSpec("unbiased-sparse-adam", "Unbiased sparse Adam with per-Gaussian lifespan, "
                     "precision-aware (Delta-scaled) position and linear-size updates "
                     "(Eq. adamNewState)", ("bogauss-unbiased-sparse-adam", "bogauss-linear-size")),
    ContributionSpec("state-inheritance", "Optimizer states inherited by split children "
                     "(alpha_t=1, alpha_g=0.2)", ("bogauss-state-inheritance",)),
    ContributionSpec("effective-opacity-pruning", "max_c sum_p w / S < 0.02 pruning, <= 1 % per "
                     "call, replacing opacity reset", ("bogauss-stats-kernel", "bogauss-prune",
                                                       "bogauss-schedule")),
    ContributionSpec("density-preserving-split", "Split-only densification along the longest "
                     "axis with offline SplitAdaptation of size and opacity", ("bogauss-split",)),
    ContributionSpec("snr-split-priority", "Distortion-based split prioritisation P_SNR on a "
                     "budget growth curve", ("bogauss-stats-kernel", "bogauss-split-priority",
                                             "bogauss-schedule")),
)


def method():
    base = make_method("bogauss")
    renderer = ComponentSpec(
        "bogauss-renderer", "renderer", "ss_bogauss.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    density = ComponentSpec(
        "bogauss-density", "density", "ss_bogauss.density:create_density",
        (TARGET_CAPABILITY,), (), dict(DEFAULTS),
    )
    return replace(
        base,
        description="BOGausS (arXiv 2504.01844) native port written from the paper (no "
                    "official code), 'Ours (Big)'; paper fidelity unverified.",
        components=(renderer, density, base.components[2]),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
