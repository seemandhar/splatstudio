"""Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting (MSAA-GS;
Zhou, Zhang, Xiong, Xia; arXiv 2508.10507). Native SplatStudio port; paper fidelity unverified.

No official code was found. The multi-sample CUDA rasterizer (forward + backward) is adapted from
gsplat 1.5.3's RasterizeToPixels3DGS{Fwd,Bwd}.cu (via this repository's 3DLS port); projection,
SH and tiling are gsplat's. Density control is 3DGS's (gsplat DefaultStrategy + opacity reset).

Methods
- `msaa-gs`: "Ours(3DGS)" — 4x MSAA rasterizer + 0.8 L_w (AWS) + 0.2 D-SSIM + 0.1 L_grad (GDC).
- `msaa-gs-msaa-only`: ablation row "3DGS+MSAA" (Table 3), 3DGS loss.
- `msaa-gs-aws-gdc-only`: ablation row "3DGS+AWS+GDC" (Table 3), gsplat single-sample renderer.
- `msaa-gs-it1`: iteration 1 (post-hoc, not in the paper): AWS weights mean-normalised.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .losses import AWS_DEFAULTS, GDC_DEFAULTS
from .renderer import CAPABILITIES, OPTIONS

CITATIONS = (
    "https://arxiv.org/abs/2508.10507",
    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3",
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_msaa_gs.probes:{fn}", description, cuda)


MSAA_PROBES = (
    _probe("msaa-forward-vs-dense", "msaa_forward_vs_dense",
           "CUDA render (RGB, depth, alpha) equals a dense float64 mean of 4 independent "
           "composites at the sub-sample positions and differs from a single-sample render"),
    _probe("msaa-backward-vs-dense", "msaa_backward_vs_dense",
           "CUDA gradients (means2d, conics, colours, opacities) equal autograd of the dense "
           "4-sample reference"),
    _probe("msaa-single-sample-equals-gsplat", "single_sample_equals_gsplat",
           "With one centre sample the kernel reproduces gsplat rasterize_to_pixels forward and "
           "backward; with 4 samples it does not"),
)
LOSS_PROBES = (
    _probe("aws-weighting", "aws_weighting",
           "Recon loss equals 0.8 sum(w|e|)/HW + 0.2 D-SSIM with w = 0.5 + 0.5 e/(max e + eps), "
           "differs from plain L1, up-weights a high-error region", False),
    _probe("gdc-gradient-difference", "gdc_gradient_difference",
           "GDC equals 0.1 |a-b| on analytic ramps, ignores constant offsets, and its gradient "
           "is localised at an edge", False),
    _probe("loss-weights", "loss_weights",
           "Summed loss components equal 0.8 L_w + 0.2 D-SSIM + 0.1 L_grad (Eq. 13)", False),
)

MSAA_CONTRIB = ContributionSpec(
    "msaa-rasterization",
    "4x multi-sample anti-aliasing inside the rasterizer: per-sample alpha compositing at "
    "rotated-grid sub-pixel positions, averaged (Sec. 3.2, Eq. 4-5, 14); custom CUDA fwd/bwd",
    ("msaa-forward-vs-dense", "msaa-backward-vs-dense", "msaa-single-sample-equals-gsplat"))
LOSS_CONTRIBS = (
    ContributionSpec("adaptive-weighting",
                     "Error-adaptive weighted L1 replaces L1 (Sec. 3.3, Eq. 6-9)",
                     ("aws-weighting", "loss-weights")),
    ContributionSpec("gradient-difference-constraint",
                     "Forward-difference image-gradient l1 loss, lambda_3 = 0.1 "
                     "(Sec. 3.4, Eq. 10-13)", ("gdc-gradient-difference", "loss-weights")),
)


def _make(name, msaa, constraints, description, aws_overrides=None):
    base = make_method(name)
    renderer, density, photometric = base.components
    if msaa:
        renderer = ComponentSpec(
            "msaa-renderer", "renderer", "ss_msaa_gs.renderer:create_renderer",
            ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, dict(OPTIONS))
    probes, contribs = tuple(base.probes), tuple(base.contributions)
    if constraints:
        losses = (
            ComponentSpec("aws-recon", "loss", "ss_msaa_gs.losses:create_recon",
                          ("render.rgb",), (), {**AWS_DEFAULTS, **(aws_overrides or {})}),
            ComponentSpec("gdc", "loss", "ss_msaa_gs.losses:create_gdc", ("render.rgb",), (),
                          dict(GDC_DEFAULTS)),
        )
        # the base photometric loss is replaced (loss-weights covers the new objective)
        probes = tuple(p for p in probes if p.name != "photometric-gradient") + LOSS_PROBES
        contribs = tuple(c for c in contribs if c.name != "photometric-objective") + LOSS_CONTRIBS
    else:
        losses = (photometric,)
    if msaa:
        probes += MSAA_PROBES
        contribs += (MSAA_CONTRIB,)
    return replace(base, description=description, components=(renderer, density) + losses,
                   probes=probes, contributions=contribs, citations=CITATIONS)


def msaa_gs():
    return _make("msaa-gs", True, True,
                 "MSAA-GS (arXiv 2508.10507) Ours(3DGS): 4x MSAA rasterizer + AWS + GDC; "
                 "native port, paper fidelity unverified.")


def msaa_only():
    return _make("msaa-gs-msaa-only", True, False,
                 "MSAA-GS ablation '3DGS+MSAA' (arXiv 2508.10507 Table 3); native port, paper "
                 "fidelity unverified.")


def aws_gdc_only():
    return _make("msaa-gs-aws-gdc-only", False, True,
                 "MSAA-GS ablation '3DGS+AWS+GDC' (arXiv 2508.10507 Table 3); native port, "
                 "paper fidelity unverified.")


def msaa_gs_it1():
    """Iteration 1 (post-hoc, gap-analysis-1.md): AWS weights divided by their image mean."""
    return _make("msaa-gs-it1", True, True,
                 "MSAA-GS (arXiv 2508.10507) iteration 1 (post-hoc): as msaa-gs with the AWS "
                 "weight map mean-normalised (not in the paper) to keep 3DGS's L1 scale; native "
                 "port, paper fidelity unverified.", {"normalize_mean": True})


method = msaa_gs
