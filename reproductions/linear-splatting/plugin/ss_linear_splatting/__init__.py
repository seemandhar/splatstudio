"""Beyond Gaussians: Fast and High-Fidelity 3D Splatting with Linear Kernels (3DLS; Chen et al.,
arXiv 2411.12440). Native SplatStudio port; paper fidelity unverified.

No official code exists (github.com/hche8927/3DLS is the project page only). The linear-kernel
CUDA rasterizer (forward + backward with Adaptive Gradient Scaling) is adapted from gsplat
1.5.3's RasterizeToPixels3DGS{Fwd,Bwd}.cu; projection, SH and tiling are gsplat's.

Methods
- `3dls`: linear kernel + Distribution Alignment (lambda 2.5) + AGS, 0.6 L1 + 0.2 L2 + 0.2 D-SSIM,
  3DGS densification with grow_scale3d 0.006, prune_scale3d 0.4, prune_opa 0.025.
- `3dls-noags`: ablation row "LK + DA" (Table 2), AGS off.
- `3dls-aags`: inferred implementation detail (not a reading of the text): AGS on the
  DA-aligned distance, omega = exp(-(D/lam)^2). Fig. 3b confirms the literal Eq. 13 form
  omega = exp(-D^2) on unaligned D, and that literal form fails at any scope: it lowers the mean
  densification statistic 8.5x, but even with an unscaled statistic or omega on conics only it
  trains to ~21.1 dB vs 23.07 at 2k (reviews/review-1.md, evidence/reviewer_scratch/).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2411.12440",
    "https://hche8927.github.io/3DLS/",
    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3",
)

# supp. Table 6 (3DLS row); others are the gsplat/3DGS defaults the paper keeps.
DENSITY = {
    "grow_grad2d": 0.0002,
    "grow_scale3d": 0.006,
    "prune_scale3d": 0.4,
    "prune_opa": 0.025,
    "grow_scale2d": 0.05,
    "prune_scale2d": 0.15,
}
LOSS_WEIGHTS = {"l1": 0.6, "l2": 0.2, "ssim": 0.2}


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_linear_splatting.probes:{fn}", description, cuda)


PROBES = (
    _probe("linear-forward-vs-dense", "linear_forward_vs_dense",
           "CUDA linear-kernel render (RGB, alpha, depth) equals a dense float64 torch "
           "reference of max(0,1-D/lam) blending and differs from gsplat's Gaussian render"),
    _probe("linear-backward-vs-dense", "linear_backward_vs_dense",
           "With AGS off, CUDA gradients (means2d, conics, colours, opacities) equal autograd "
           "of the dense reference"),
    _probe("alignment-footprint", "alignment_footprint",
           "A single splat's rendered support radius is lam*sigma (lam 2.5 and 1.0), and the "
           "lam-derived tile boxes never clip the alpha>=1/255 support"),
    _probe("ags-gradient-scaling", "ags_gradient_scaling",
           "With AGS on, geometry gradients equal the reference with dk/dD scaled by "
           "exp(-D^2); opacity/colour gradients are unscaled; means2d gradients differ from "
           "AGS off"),
    _probe("loss-weights", "loss_weights",
           "Summed loss components equal 0.6 L1 + 0.2 L2 + 0.2 (1-SSIM)", False),
    _probe("density-thresholds", "density_thresholds",
           "Built density controller uses the paper's Table 6 thresholds", False),
)


def _make(name, ags, description, ags_coef=1.0):
    base = make_method(name)
    renderer = ComponentSpec(
        "linear-renderer", "renderer", "ss_linear_splatting.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"lam": 2.5, "ags": ags, "ags_coef": ags_coef, "tile_size": 16, "eps2d": 0.3})
    density = replace(base.components[1], config=dict(DENSITY))
    losses = tuple(
        ComponentSpec(f"loss-{term}", "loss", "ss_linear_splatting.losses:create_loss",
                      ("render.rgb",), (), {"term": term, "weight": w})
        for term, w in LOSS_WEIGHTS.items())
    probes = PROBES if ags else tuple(p for p in PROBES if p.name != "ags-gradient-scaling")
    contributions = [
        ContributionSpec("linear-kernel",
                         "Gaussian falloff replaced by max(0, 1 - D_M) of the 2D Mahalanobis "
                         "distance; custom CUDA forward/backward (Sec. 3.2, Eq. 9-11)",
                         ("linear-forward-vs-dense", "linear-backward-vs-dense")),
        ContributionSpec("distribution-alignment",
                         "D_M scaled by lam = 2.5 to match the Gaussian footprint; bounding "
                         "radius derived from lam (Sec. 3.3.1, Eq. 12)",
                         ("alignment-footprint",)),
        ContributionSpec("training-recipe",
                         "0.6 L1 + 0.2 L2 + 0.2 SSIM loss and densification thresholds "
                         "0.006 / 0.4 / 0.025 (Sec. 4.1, supp. A)",
                         ("loss-weights", "density-thresholds")),
    ]
    if ags:
        contributions.insert(2, ContributionSpec(
            "adaptive-gradient-scaling",
            "Kernel gradient scaled by omega = exp(-D_M^2) in the backward pass (Sec. 3.3.2, "
            "Eq. 13-14)", ("ags-gradient-scaling",)))
    # the base photometric loss is replaced, so its probe/contribution go (loss-weights covers it)
    base_probes = tuple(p for p in base.probes if p.name != "photometric-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "photometric-objective")
    return replace(
        base, description=description,
        components=(renderer, density) + losses,
        probes=base_probes + probes, contributions=base_contribs + tuple(contributions),
        citations=CITATIONS)


def linear_splatting():
    return _make("3dls", True,
                 "3D Linear Splatting (arXiv 2411.12440): linear kernel + DA + AGS; native port, "
                 "paper fidelity unverified.")


def linear_splatting_noags():
    return _make("3dls-noags", False,
                 "3DLS ablation 'LK + DA' (no AGS, arXiv 2411.12440 Table 2); native port, "
                 "paper fidelity unverified.")


def linear_splatting_aligned_ags():
    return _make("3dls-aags", True,
                 "3DLS (arXiv 2411.12440) with AGS on the lam-aligned distance, omega = "
                 "exp(-(D/lam)^2) (interpretation variant); native port, paper fidelity "
                 "unverified.", ags_coef=1.0 / 2.5**2)


method = linear_splatting
