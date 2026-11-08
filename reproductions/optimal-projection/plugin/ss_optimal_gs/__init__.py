"""On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy (Huang et
al., ECCV 2024, arXiv 2402.00752). Native SplatStudio port; paper fidelity unverified.

Ported from the official code https://github.com/LetianHuang/op43dgs (pinhole rasterizer,
Inria Gaussian-Splatting licence): tangent-plane projection (forward.cu computeCov2D and the
official closed-form gradient of backward.cu computeCov2DCUDA) and the unit-sphere rasterizer
(renderCUDA forward/backward) as a JIT CUDA extension on gsplat 1.5.3 tile lists. Training is
unmodified 3DGS (base gsplat density with the 3DGS opacity reset, L1 + 0.2 D-SSIM).

Methods
- `optimal-gs`: the paper's method.
- `optimal-gs-ewa`: control — same rasterizer and Inria blending rules with the standard EWA
  projection / planar evaluation, isolating the projection change.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES, OPTIONS

CITATIONS = (
    "https://arxiv.org/abs/2402.00752",
    "https://github.com/LetianHuang/op43dgs",
    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3",
)


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_optimal_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("tangent-jacobian", "tangent_jacobian_probe",
           "Projection Jacobian equals the official computeCov2D formula and the paper's Q J_p "
           "(Eq. 14, 18); equals EWA on the optical axis and differs off axis", False),
    _probe("cov-gradient", "cov_gradient_probe",
           "Official closed-form dJ/dt: 17 coefficients equal autograd, j12_x carries the "
           "official typo; the renderer's custom backward uses them", False),
    _probe("raster-forward-vs-dense", "raster_forward_probe",
           "CUDA unit-sphere render (RGB, depth, alpha) equals a float64 dense reference written "
           "from the official renderCUDA trig formulas; differs from the EWA control"),
    _probe("raster-backward-vs-dense", "raster_backward_probe",
           "CUDA gradients (means2d, conics, colours, opacities) equal autograd of the dense "
           "reference, optimal and EWA modes"),
    _probe("projection-error", "projection_error_probe",
           "Off-axis splats under a wide FOV: optimal-projection alpha is closer to the exact "
           "ray maximum of the 3D Gaussian than EWA (< 0.5x the error); both exact on axis"),
)


def _make(name, projection, description):
    base = make_method(name)
    cfg = dict(OPTIONS)
    cfg["projection"] = projection
    renderer = ComponentSpec(
        "optimal-renderer", "renderer", "ss_optimal_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES, cfg)
    contributions = (
        ContributionSpec(
            "optimal-projection",
            "Each Gaussian projected onto the plane tangent to the unit sphere at its mean's "
            "ray: J = Q J_p (Sec. 5.1, Eq. 11-15, 18) with the official gradient",
            ("tangent-jacobian", "cov-gradient", "projection-error")),
        ContributionSpec(
            "unit-sphere-rasterizer",
            "Pixel rays evaluated in each Gaussian's tangent-plane coordinates (Sec. 5.2, "
            "Eq. 16-17); custom CUDA forward/backward",
            ("raster-forward-vs-dense", "raster-backward-vs-dense", "projection-error")),
    )
    return replace(
        base, description=description,
        components=(renderer,) + tuple(base.components[1:]),
        probes=tuple(base.probes) + PROBES,
        contributions=tuple(base.contributions) + contributions,
        citations=CITATIONS)


def optimal_gs():
    return _make("optimal-gs", "optimal",
                 "Optimal Gaussian Splatting (arXiv 2402.00752): tangent-plane projection + "
                 "unit-sphere rasterizer, ported from the official CUDA; native port, paper "
                 "fidelity unverified.")


def optimal_gs_ewa():
    return _make("optimal-gs-ewa", "ewa",
                 "Control for optimal-gs: same CUDA rasterizer and Inria blending rules with the "
                 "standard EWA projection (not a paper method).")


method = optimal_gs
