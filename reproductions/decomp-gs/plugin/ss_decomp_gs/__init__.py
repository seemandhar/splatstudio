"""DecompGS: Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction
(Huang, Liu, Wong, arXiv 2507.20239). Native SplatStudio port written from the paper (no
official code released); renderer/loss resolution plumbing adapted from the DashGaussian port
(reproductions/dashgaussian). Paper fidelity unverified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import CAPABILITIES, SCHEDULE_CAPABILITY

CITATIONS = ("https://arxiv.org/abs/2507.20239",)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_decomp_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("decomp-renderer-gradient", "renderer_gradient",
           "Built-in renderer-gradient check with the C2F schedule active: finite nonzero screen "
           "gradients reach the density statistics at the reduced resolution", True),
    _probe("decomp-c2f-schedule", "c2f_schedule",
           "Eq. 2-3 energies/end iterations equal a numpy-FFT + PIL-resize reference; scale r "
           "spans [T_{r+1}, T_r); flat image gives no low-res phase; c2f off renders at r=1"),
    _probe("decomp-render-scale", "render_scale",
           "Training render = floor(H/r) x floor(W/r) equal to gsplat with scaled K; no_grad "
           "(evaluation) render at full resolution", True),
    _probe("decomp-target-downsample", "target_downsample",
           "Loss target is the antialiased bilinear D_r (PIL reference), not an aliased subsample"),
    _probe("decomp-g2l-phases", "g2l_phases",
           "Before T_2: large qualified Gaussians split, clone candidates untouched; after T_2 "
           "(or with g2l off): 3DGS clone + split; optimizer/field alignment kept", True),
    _probe("decomp-adaptive-prune", "adaptive_prune",
           "tau = min(sorted(alpha)[floor(N p)], tau_u) vs numpy; tau_u cap; density prunes "
           "alpha < tau, fixed 0.005 prunes fewer", True),
    _probe("decomp-lr-schedule", "lr_schedule",
           "Position LR constant until full resolution (T_2), then log-linear to 1 % at 30000"),
    _probe("decomp-cycle", "cycle",
           "Native runtime run on demo data, compressed schedule: resolution decreases "
           "monotonically to r=1, no clones before T_2, pruning active", True),
)
CONTRIBUTIONS = (
    ContributionSpec("global-to-local-densification",
                     "G2L (Sec. 3.2): split-only global spread until T_2, then split + clone",
                     ("decomp-g2l-phases", "decomp-cycle")),
    ContributionSpec("energy-coarse-to-fine",
                     "Energy-guided coarse-to-fine multi-resolution training (Sec. 3.3, Eq. 1-3) "
                     "with position LR held at reduced resolution (Impl. details)",
                     ("decomp-c2f-schedule", "decomp-render-scale", "decomp-target-downsample",
                      "decomp-lr-schedule", "decomp-cycle")),
    ContributionSpec("adaptive-opacity-pruning",
                     "Adaptive opacity threshold tau = min(tau_k, tau_u) (Sec. 3.4)",
                     ("decomp-adaptive-prune",)),
)


def method():
    base = make_method("decomp-gs")
    renderer = ComponentSpec(
        "decomp-renderer", "renderer", "ss_decomp_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, {},
    )
    density = ComponentSpec(
        "decomp-density", "density", "ss_decomp_gs.density:create_density",
        ("gradient.screen.signed",), (), dict(DEFAULTS),
    )
    loss = ComponentSpec(
        "photometric", "loss", "ss_decomp_gs.losses:create_photometric",
        ("render.rgb", SCHEDULE_CAPABILITY), (), {"ssim_weight": 0.2, "ssim_padding": "same"},
    )
    # The built-in renderer-gradient probe drives the density without training images; the
    # C2F density needs its schedule, so an equivalent probe that builds it replaces it.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    contributions = tuple(
        replace(c, probes=tuple("decomp-renderer-gradient" if n == "renderer-gradient" else n
                                for n in c.probes))
        for c in base.contributions
    )
    return replace(
        base,
        description="DecompGS (arXiv 2507.20239) native port on the 3DGS backbone: G2L "
        "densification, energy-guided C2F resolution, adaptive opacity pruning; written from "
        "the paper (no official code); paper fidelity unverified.",
        components=(renderer, density, loss),
        probes=base_probes + PROBES,
        contributions=contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )
