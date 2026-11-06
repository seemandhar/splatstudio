"""Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images (Niedermayr, Neuhauser,
Westermann, ICCV 2025, arXiv 2503.14171). Native SplatStudio port; paper fidelity unverified.

The official repository (github.com/KeKsBoTer/upscale3dgs) has no code; the CUDA jet rasterizer
here is derived from the paper (forward recurrences as in the authors' brush viewer shader,
backward by reverse mode with the inversion trick) on gsplat's projection and tile lists.

Methods
- `gradient-upscaling-2x` / `-4x`: "Spline (Ours)" - train with rendering at W/s + analytic
  image derivatives + gradient-aware spline upscaling, loss at full resolution.
- `bicubic-upscaling-2x` / `-4x`: "Bicubic" - same, with PyTorch bicubic upscaling.
- `gradient-upscaling-3dgs`: "3DGS" control - stock gsplat renderer with the same Taming
  densification; train with `--downscale s`, evaluate at full resolution (and "Spline (3DGS)")
  with `apply_upscaling.py`.
All use Taming-3DGS budgeted densification to 1 M Gaussians (paper Sec. 6.2, Tab. 1 caption).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES, TARGET_CAPABILITY

CITATIONS = (
    "https://arxiv.org/abs/2503.14171",
    "https://github.com/KeKsBoTer/upscale3dgs",
    "https://arxiv.org/abs/2406.15643",
)

# Taming 3DGS densification only (installed port of plan #30): fixed final count reached by the
# predefined growth curve; the other Taming changes (batched SH, opacity freeze, lr) are off.
DENSITY = {
    "mode": "final_count",
    "budget": 1000000,
    "densification_interval": 100,
    "high_opacity": "off",
    "sh_update_every": 1,
    "lr": {},
}


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_gradient_upscaling.probes:{fn}", description, cuda)


JET_PROBES = (
    _probe("jet-value-equals-gsplat", "jet_value_equals_gsplat",
           "Value channel of the CUDA jet rasterizer equals gsplat rasterization (white bg)"),
    _probe("jet-derivs-vs-finite-diff", "jet_derivs_vs_finite_diff",
           "Analytic I_x, I_y, I_xy equal central differences of the render under sub-pixel "
           "shifts of the sampling grid"),
    _probe("jet-vs-torch-autograd", "jet_vs_torch_autograd",
           "Forward jets and backward gradients (means2d, conics, colours, opacities) equal an "
           "independent float64 compositing reference differentiated by torch autograd"),
)
SPLINE_PROBES = (
    _probe("spline-equals-paper-matrix", "spline_equals_paper_matrix",
           "Separable Hermite upscaler equals p=[1 x x2 x3] C^-1 F C^-T [1 y y2 y3]^T", False),
    _probe("spline-equals-catmull-rom", "spline_equals_catmull_rom",
           "With central-difference derivatives the spline equals Keys/Catmull-Rom bicubic; "
           "wrong derivative scaling is detected", False),
    _probe("analytic-beats-fd", "analytic_beats_fd",
           "At 4x, spline with analytic derivatives reconstructs the native render better than "
           "the same spline with finite differences and than PyTorch bicubic"),
)
E2E_PROBE = (
    _probe("renderer-end-to-end", "renderer_end_to_end",
           "Configured upscaling renderer outputs target size, finite gradients, screen-space "
           "gradient, and d(loss)/d(opacity) matches finite differences"),
)


def _density():
    return ComponentSpec(
        "taming-density", "density", "ss_taming_3dgs.density:create_density",
        ("gradient.screen.signed", TARGET_CAPABILITY), (), dict(DENSITY))


def _make(name, factor, upscaler, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "upscale-renderer", "renderer", "ss_gradient_upscaling.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"factor": float(factor), "upscaler": upscaler, "tile_size": 16})
    if upscaler == "spline":
        probes = JET_PROBES + SPLINE_PROBES + E2E_PROBE
        contributions = (
            ContributionSpec("gradient-aware-spline-upscaling",
                             "Bicubic spline per low-res cell from rendered values and analytic "
                             "derivatives (Sec. 5, supp. 'Spline Image Interpolation')",
                             ("spline-equals-paper-matrix", "spline-equals-catmull-rom",
                              "analytic-beats-fd")),
            ContributionSpec("analytic-image-gradients",
                             "I_x, I_y, I_xy computed in the 3DGS blending loop (Sec. 5, supp. "
                             "'Forward pass'); custom CUDA",
                             ("jet-value-equals-gsplat", "jet-derivs-vs-finite-diff")),
            ContributionSpec("training-with-upscaling",
                             "Back-propagation through upscaler and analytic gradients, loss at "
                             "full resolution (Sec. 5 'Differentiable Image Gradients'; supp. "
                             "'Backward Pass', inversion trick)",
                             ("jet-vs-torch-autograd", "renderer-end-to-end")),
        )
    else:
        probes = E2E_PROBE
        contributions = (
            ContributionSpec("training-with-bicubic-upscaling",
                             "Paper's 'Bicubic' baseline: PyTorch bicubic upscaling in the "
                             "training loop", ("renderer-end-to-end",)),
        )
    return replace(
        base, description=description, components=(renderer, _density(), base.components[2]),
        probes=base.probes + probes, contributions=base.contributions + contributions,
        citations=CITATIONS)


def spline2x():
    return _make("gradient-upscaling-2x", 2, "spline",
                 "Gradient-aware spline upscaling (arXiv 2503.14171) 'Spline (Ours)' 2x, Taming "
                 "1M densification; native port, paper fidelity unverified.")


def spline4x():
    return _make("gradient-upscaling-4x", 4, "spline",
                 "Gradient-aware spline upscaling (arXiv 2503.14171) 'Spline (Ours)' 4x, Taming "
                 "1M densification; native port, paper fidelity unverified.")


def bicubic2x():
    return _make("bicubic-upscaling-2x", 2, "bicubic",
                 "'Bicubic' baseline of arXiv 2503.14171 (PyTorch bicubic upscaling in training) "
                 "2x, Taming 1M densification; paper fidelity unverified.")


def bicubic4x():
    return _make("bicubic-upscaling-4x", 4, "bicubic",
                 "'Bicubic' baseline of arXiv 2503.14171 (PyTorch bicubic upscaling in training) "
                 "4x, Taming 1M densification; paper fidelity unverified.")


def control3dgs():
    """'3DGS' row: stock renderer + same densification; use with ss-train --downscale s."""
    from ss_taming_3dgs.renderer import TARGET_CAPABILITY as TC

    base = make_method("gradient-upscaling-3dgs")
    renderer = ComponentSpec(
        "taming-renderer", "renderer", "ss_taming_3dgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed", TC),
        {"absgrad": False, "antialiased": False})
    return replace(
        base,
        description=("'3DGS' control of arXiv 2503.14171: gsplat renderer + Taming 1M "
                     "densification, trained with --downscale s and evaluated at full "
                     "resolution by apply_upscaling.py; paper fidelity unverified."),
        components=(renderer, _density(), base.components[2]), citations=CITATIONS)


method = spline4x
