"""FlashGS (arXiv 2408.07967) for SplatStudio.

FlashGS is a render-only CUDA rasterizer (the official release has no backward pass). The
`flashgs` method therefore trains exactly like the `gsplat` baseline and renders with the
vendored, unmodified FlashGS kernels whenever no gradient is required (validation / ss-eval).
The paper's actual protocol - rendering an already trained 3DGS model - is the separate tool
`apply_flashgs.py`, which works on any finished baseline run.
"""

from dataclasses import replace

from splatstudio.methods import make_method
from splatstudio.specs import ComponentSpec, ContributionSpec, ProbeSpec

P = "ss_flashgs.probes:"


def method():
    base = make_method("flashgs")
    renderer = ComponentSpec(
        "flashgs-renderer",
        "renderer",
        "ss_flashgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        base.components[0].provides,
        {"inference_backend": "flashgs", "tile": 16},
    )
    probes = base.probes + (
        ProbeSpec("flashgs-parity", P + "parity",
                  "FlashGS image matches the original 3DGS and gsplat forwards; model conversion "
                  "equals upstream loadPly", True),
        ProbeSpec("precise-intersection", P + "precise_intersection",
                  "No visible tile dropped; fewer key/value pairs than 3DGS AABB and coarse rect",
                  True),
        ProbeSpec("render-speed", P + "speed",
                  "FlashGS forward faster than the original 3DGS forward at 1080p", True),
        ProbeSpec("inference-switch", P + "inference_switch",
                  "gsplat used for differentiable renders, FlashGS for no-grad renders", True),
    )
    contributions = base.contributions + (
        ContributionSpec("opacity-aware-precise-intersection",
                         "Eq. 3 opacity-aware extent + exact ellipse/tile test (Sec. 4.1)",
                         ("precise-intersection", "flashgs-parity")),
        ContributionSpec("adaptive-scheduling-and-pipelined-render",
                         "Warp-cooperative intersection, fused preprocess, pipelined render "
                         "(Sec. 5); performance-only, output unchanged",
                         ("render-speed", "flashgs-parity")),
        ContributionSpec("render-only-integration",
                         "No backward in the release: gsplat trains, FlashGS renders inference",
                         ("inference-switch",)),
    )
    return replace(
        base,
        components=(renderer,) + base.components[1:],
        probes=probes,
        contributions=contributions,
        description=("FlashGS native port (official CUDA, render-only; training = gsplat "
                     "baseline); paper fidelity unverified."),
        citations=("https://arxiv.org/abs/2408.07967", "https://github.com/InternLandMark/FlashGS"),
    )
