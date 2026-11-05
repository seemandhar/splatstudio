"""DeblurGS (Oh, Chung, Lee, Lee; ECCV 2024, arXiv 2404.11358) -- native SplatStudio port of
the official code (github.com/taekkii/deblurgs @e0587475).

Per-image Bezier se(3) camera trajectories and sub-frame alignment parameters live in
``ss_deblurgs.bank.ACTIVE`` during training (owned by ``deblurgs-density``) and are exported to
the checkpoint's ``density_state["bank"]``. Evaluation renders sharp; the paper's protocol
(test-pose post-optimisation) is ``python -m ss_deblurgs.evaluate --run RUN``.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import HINGE_DEFAULTS, RGB_DEFAULTS, SMOOTH_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2404.11358", "https://github.com/taekkii/deblurgs")
LOSS_REQUIRES = ("render.motion_blur",)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_deblurgs.probes:{fn}", desc, cuda)


def method(name="deblurgs", alignment_lr=3e-3):
    base = make_method(name)
    comps = (
        ComponentSpec("deblurgs-renderer", "renderer", "ss_deblurgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("deblurgs-density", "density", "ss_deblurgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, "bank_curve_alignment_lr": alignment_lr}),
        ComponentSpec("deblurgs-rgb", "loss", "ss_deblurgs.losses:create_rgb", LOSS_REQUIRES,
                      (), dict(RGB_DEFAULTS)),
        ComponentSpec("deblurgs-smooth", "loss", "ss_deblurgs.losses:create_smoothness",
                      LOSS_REQUIRES, (), dict(SMOOTH_DEFAULTS)),
        ComponentSpec("deblurgs-hinge", "loss", "ss_deblurgs.losses:create_hinge",
                      LOSS_REQUIRES, (), dict(HINGE_DEFAULTS)),
    )
    probes = (
        _probe("bezier-se3", "bezier_se3",
               "Bezier curve equals de Casteljau; se(3) exp equals matrix_exp of the twist "
               "(pytorch3d row convention); initial trajectory reproduces the COLMAP pose; "
               "curve LR halves every 15k from step 1000"),
        _probe("blur-synthesis", "blur_synthesis",
               "Training render = mean of independent gsplat renders at the 21 bank poses over "
               "one background; differs from the sharp render once the trajectory spreads; "
               "single sub-frame at t=0 before step 1000; curve gets gradients only in "
               "[1000, 100000)", True),
        _probe("alignment", "alignment",
               "Sub-frame times sorted with fixed end points, initialised to linspace; nu lr 0 "
               "until 30000 then decays to 1e-7 at 150000; nu receives gradient and moves "
               "after 30000 (not in deblurgs-nu0)", True),
        _probe("gamma", "gamma",
               "Loss target = GT^2.2 and evaluation image = render^(1/2.2) (independent "
               "formula); without the tone curve the loss changes", True),
        _probe("smoothness", "smoothness",
               "Temporal smoothness equals mean |sub[j+1]-sub[j]|, weight 1e-3 -> 1e-5 "
               "log-linear over 150k; hinge equals the official formula"),
        _probe("densify-annealing", "densify_annealing",
               "theta(1)=4e-4, theta(12500)=sqrt(8e-8), theta(>=25000)=2e-4; a Gaussian with "
               "grad 3e-4 clones at step 25000 but not at 600; reset = min(o, 0.1), prune "
               "opacity < 0.005 only; official extent formula", True),
        _probe("test-pose-opt", "test_pose_opt",
               "Official test-pose post-optimisation recovers a perturbed camera on a "
               "synthetic scene (PSNR rises, translation error falls)", True),
    )
    contribs = (
        ContributionSpec("camera-trajectory", "Per-image 9th-order Bezier trajectory in se(3) "
                         "and averaged sub-frame rendering (Eq. 4-7)",
                         ("bezier-se3", "blur-synthesis")),
        ContributionSpec("subframe-alignment", "Learnable sub-frame alignment nu (Eq. 5-6)",
                         ("alignment",)),
        ContributionSpec("gamma-tonemap", "Gamma tone mapping of the synthesised blur (Eq. 7)",
                         ("gamma",)),
        ContributionSpec("temporal-smoothness", "Temporal smoothness loss with annealed weight "
                         "(Eq. 9-10)", ("smoothness",)),
        ContributionSpec("densification-annealing", "Gaussian Densification Annealing of the "
                         "clone/split threshold (Sec. 3.3)", ("densify-annealing",)),
        ContributionSpec("test-pose-alignment", "Sharp rendering with test-pose "
                         "post-optimisation (Sec. 4.2)", ("test-pose-opt",)),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description=f"DeblurGS native port [{name}: nu lr {alignment_lr}] (per-image Bezier "
                    "se(3) trajectories, 21 averaged sub-frames, gamma, temporal smoothness, "
                    "densification annealing, official clamp opacity / SH 2); evaluated sharp "
                    "with test-pose post-optimisation (ss_deblurgs.evaluate); paper fidelity "
                    "unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        citations=CITATIONS,
    )


def method_nu0():
    """Literal released-code default: curve_alignment_lr = 0.0 (nu never updated)."""
    return method("deblurgs-nu0", alignment_lr=0.0)
