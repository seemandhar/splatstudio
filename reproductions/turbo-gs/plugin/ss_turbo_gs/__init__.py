"""Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields (Lu et al.,
CVPR 2026, arXiv 2412.13547), Turbo-3DGS variant. Native SplatStudio port written from the
paper (the official repository https://github.com/inspirelt/Turbo-GS holds no code);
paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS, LOSS_VALUE
from .renderer import CAPABILITIES, COLORS, DILATED

CITATIONS = (
    "https://arxiv.org/abs/2412.13547",
    "https://github.com/inspirelt/Turbo-GS",
    "https://github.com/humansensinglab/taming-3dgs (budget table full_eval.py big_budgets)",
)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_turbo_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("turbo-renderer-gradient", "renderer_gradient",
           "Built-in renderer-gradient check against the dilated target subset: finite nonzero "
           "screen and colour gradients reach the density statistics", True),
    _probe("turbo-dilated-merge", "dilated_merge",
           "4 interleaved d=2 dilated renders reproduce the full render (>50 dB, >10 dB above "
           "naive half-res intrinsics); training size H/2 x W/2; eval full size; all offsets",
           True),
    _probe("turbo-dilated-target", "dilated_target",
           "Loss supervises target[oy::2, ox::2] with L1 + 0.2 D-SSIM; detached value exported"),
    _probe("turbo-budget-schedule", "budget_schedule",
           "Eq. 6 endpoints, alpha=1 linear warm-up, power-law fit alpha, plateau raises alpha, "
           "M_adaptive grow/shrink and clip (Alg. 1)"),
    _probe("turbo-color-densify", "color_densify",
           "Colour-only qualifiers densify only when the 20 % branch fires; tau_c = 0.01 tau_p; "
           "budget cap keeps top scores"),
    _probe("turbo-refine-allowance", "refine_allowance",
           "refine() on a real event densifies exactly B(t) - n with t = 100 step/total_steps; "
           "uncapped ablation densifies all; densify-window t gives a different target"),
    _probe("turbo-color-grad", "color_grad",
           "Accumulated colour statistic equals independent autograd ||dL/dsh0|| on visible "
           "Gaussians", True),
)
CONTRIBUTIONS = (
    ContributionSpec("dilated-rendering", "Dilated rendering of a d=2 pixel sub-lattice with "
                     "high-resolution footprints (Sec. 4.1)",
                     ("turbo-dilated-merge", "turbo-dilated-target")),
    ContributionSpec("convergence-aware-budget", "Power-law convergence-aware budget control "
                     "(Sec. 4.2, Eq. 5-6, supp. Alg. 1)",
                     ("turbo-budget-schedule", "turbo-refine-allowance")),
    ContributionSpec("position-appearance-densification", "Densification from position and "
                     "colour gradients (Sec. 4.3)",
                     ("turbo-color-densify", "turbo-color-grad")),
)


def method():
    base = make_method("turbo-gs")
    renderer = ComponentSpec(
        "turbo-dilated-renderer", "renderer", "ss_turbo_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"dilation": 2, "eps2d": 0.3, "seed": 0},
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_turbo_gs.losses:create_photometric",
        ("render.rgb", DILATED), (LOSS_VALUE,), {"ssim_weight": 0.2},
    )
    density = ComponentSpec(
        "turbo-density", "density", "ss_turbo_gs.density:create_density",
        ("gradient.screen.signed", COLORS), (), dict(DEFAULTS),
    )
    # splatstudio's renderer-gradient probe uses a full-size target, which cannot match the
    # dilated training render; an equivalent dilation-aware probe replaces it.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    contributions = tuple(
        replace(c, probes=tuple("turbo-renderer-gradient" if n == "renderer-gradient" else n
                                for n in c.probes))
        for c in base.contributions
    )
    return replace(
        base,
        description="Turbo-GS (arXiv 2412.13547) Turbo-3DGS native port: dilated rendering, "
        "convergence-aware budget, position+colour densification; paper fidelity unverified.",
        components=(renderer, loss, density),
        probes=base_probes + PROBES,
        contributions=contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
