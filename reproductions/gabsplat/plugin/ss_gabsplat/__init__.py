"""3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance Field Rendering (Zhou et al.,
ACM MM 2025, arXiv 2508.05343). Native SplatStudio port; paper fidelity unverified.

No official code exists (github.com/Zhjy1/3DGabSplat holds a README only, checked 2026-11-07).
The 2D Gabor CUDA rasterizer (forward + backward) is adapted from this repo's linear-splatting
kernel, itself adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu; projection, SH and
tiling are gsplat's.

Methods
- `gabsplat`: Gabor primitive (F = 2) + frequency-adaptive density; Mip-NeRF 360 recipe of
  supp. B (init f 0.05, w 0.01; lr f 0.02, w 0.01, opacity 0.025; prune_opa 0.01, opacity reset
  0.02; child reset f 0.001 / w 0.01; weight reset 0.01 every 3000).
- `gabsplat-noreset`: ablation without the frequency-adaptive resets (diagnostics only).
"""

import math
from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES, FREQ, WEIGHT

CITATIONS = (
    "https://arxiv.org/abs/2508.05343",
    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3",
)

# supp. B, Mip-NeRF 360 values (main-text Sec. 4.1 gives f0 0.001, lr_f 0.01, lr_w 0.02)
NUM_FREQS = 2
INIT_FREQ = 0.05
INIT_WEIGHT = 0.01
LR_FREQ = 0.02
LR_WEIGHT = 0.01
DENSITY = {
    "grow_grad2d": 0.0002,
    "prune_opa": 0.01,  # supp. B; gsplat's reset value 2 * prune_opa = 0.02 (supp. B)
    "opacity_lr": 0.025,
    "child_freq": 0.001,
    "child_weight": 0.01,
    "reset_weight": 0.01,
}


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_gabsplat.probes:{fn}", description, cuda)


PROBES = (
    _probe("gabor-forward-vs-dense", "gabor_forward_vs_dense",
           "CUDA Gabor render (RGB, depth channel, alpha) equals a dense float64 torch reference "
           "of Eq. 12-13 and differs from the pure-Gaussian render"),
    _probe("gabor-backward-vs-dense", "gabor_backward_vs_dense",
           "CUDA gradients w.r.t. means2d, conics, colours, opacities, 2D frequencies and weights "
           "equal autograd of the dense reference"),
    _probe("frequency-projection", "frequency_projection",
           "2D frequency equals paper Eq. 9-10 (two Jacobian completions) and numerical "
           "z-integration of the 3D Gabor; dropping the z-correction fails", False),
    _probe("renderer-frequency", "renderer_frequency",
           "Frequencies used by the renderer equal (Sigma2d + 0.3 I)^-1 Sigma2d f2d_paper per "
           "Gaussian on a projected scene; gradients reach gabor_freq and gabor_weight"),
    _probe("frequency-adaptive-density", "frequency_adaptive_density",
           "Densification children get f = 0.001, w = 0.01 while parents keep theirs; all w "
           "reset to 0.01 with the opacity reset (<= 0.02) at step 3000; fields stay trainable"),
    _probe("recipe", "recipe",
           "Fields, learning rates, initial values and density thresholds equal supp. B", False),
)


def _make(name, resets, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "gabor-renderer", "renderer", "ss_gabsplat.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"num_freqs": NUM_FREQS, "tile_size": 16, "eps2d": 0.3})
    density = ComponentSpec(
        "gabor-density", "density", "ss_gabsplat.density:create_density",
        ("gradient.screen.signed",), (),
        {**DENSITY, "child_reset": resets, "weight_reset": resets})
    fields = (
        FieldSpec(FREQ, (NUM_FREQS, 3), LR_FREQ, INIT_FREQ),
        FieldSpec(WEIGHT, (NUM_FREQS,), LR_WEIGHT, math.log(INIT_WEIGHT / (1 - INIT_WEIGHT))),
    )
    probes = PROBES if resets else tuple(p for p in PROBES
                                         if p.name != "frequency-adaptive-density")
    contributions = [
        ContributionSpec("gabor-primitive",
                         "Gaussian envelope x [(1 - sum w) + sum w cos(2 pi f.(x - mu))], F = 2 "
                         "learnable 3D frequencies and sigmoid weights (Sec. 3.2, Eq. 7)",
                         ("gabor-forward-vs-dense", "recipe")),
        ContributionSpec("gabor-splatting",
                         "Frequency projection (JW)^-T f and z-integrated 2D frequency; custom "
                         "CUDA Gabor rasterizer forward/backward (Sec. 3.3, Eq. 9-13, supp. A)",
                         ("frequency-projection", "renderer-frequency",
                          "gabor-backward-vs-dense")),
    ]
    if resets:
        contributions.append(ContributionSpec(
            "frequency-adaptive-optimization",
            "Child frequency/weight reset at densification and weight reset every 3000 iters "
            "with the opacity reset (Sec. 3.4.2, supp. B)", ("frequency-adaptive-density",)))
    return replace(
        base, description=description, components=(renderer, density, base.components[2]),
        fields=fields, probes=base.probes + probes,
        contributions=base.contributions + tuple(contributions), citations=CITATIONS)


def gabsplat():
    return _make("gabsplat", True,
                 "3DGabSplat (arXiv 2508.05343): 3D Gabor primitives (F=2) + frequency-adaptive "
                 "density; native port, paper fidelity unverified.")


def gabsplat_noreset():
    return _make("gabsplat-noreset", False,
                 "3DGabSplat ablation without frequency-adaptive resets (arXiv 2508.05343); "
                 "native port, paper fidelity unverified.")


method = gabsplat
