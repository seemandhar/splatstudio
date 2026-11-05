"""CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians (ECCV 2024,
arXiv 2403.19495) — native SplatStudio port.

Defaults are the official-code values (github.com/avinashpaliwal/CoherentGS, train.py,
scene/gaussian_model.py) for LLFF with 3 input views; paper/code differences are listed in
reproductions/coherentgs/paper.md and the README. Run with
`ss-train coherentgs --data <llff-fewview3-cgs scene> --steps 13000 --num-gaussians 10000000
 --max-gaussians 12000000 --recipe splatstudio`.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import FLOW_DEFAULTS, PHOTO_DEFAULTS, SMOOTH_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2403.19495",
    "https://github.com/avinashpaliwal/CoherentGS",
    "https://people.engr.tamu.edu/nimak/Papers/CoherentGS/index.html",
)

P = "ss_coherentgs.probes:"


def method():
    base = make_method("coherentgs")
    components = (
        ComponentSpec("coherent-renderer", "renderer", "ss_coherentgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("coherent-density", "density", "ss_coherentgs.density:create_density",
                      (), (), dict(DENSITY_DEFAULTS)),
        ComponentSpec("photometric", "loss", "ss_coherentgs.losses:create_photometric",
                      ("render.rgb",), (), dict(PHOTO_DEFAULTS)),
        ComponentSpec("coherent-smooth", "loss", "ss_coherentgs.losses:create_smooth",
                      ("render.coherent.disparity",), (), dict(SMOOTH_DEFAULTS)),
        ComponentSpec("coherent-flow", "loss", "ss_coherentgs.losses:create_flow",
                      ("render.coherent.xyz",), (), dict(FLOW_DEFAULTS)),
    )
    probes = (
        ProbeSpec("ray-structure", P + "ray_structure",
                  "Eq. 3: every Gaussian projects to its own integer pixel at D_init + residual"),
        ProbeSpec("decoder-architecture", P + "decoder_architecture",
                  "Official decoder layer spec / output size / posenc; 3-view capacities 15/10"),
        ProbeSpec("masked-residual", P + "masked_residual",
                  "Residual 50 chsum(tanh(f) S/sum S), opacity sigmoid(logit 0.5 + 100 chsum)"),
        ProbeSpec("alignment", P + "alignment",
                  "Eq. 8 flow-based scale/offset alignment recovers depth on a 2-view plane"),
        ProbeSpec("radius", P + "radius",
                  "Eq. 9 radius 1.2 D/(2f); reset every 100 official its before 15000; base LRs"),
        ProbeSpec("tv-loss", P + "tv_loss",
                  "Eq. 4-5 L_multi vs finite differences; gamma 0.35005 -> 1; beta_m 5"),
        ProbeSpec("flow-loss", P + "flow_loss",
                  "Eq. 6 warping loss vs an independent bilinear warp; effective weight 0.01"),
        ProbeSpec("photometric-terms", P + "photometric_terms",
                  "0.8 ML1 + 0.2 (1 - SSIM_same) + 0.05/32 L1(VGG19 relu1_1) vs independent formulas"),
        ProbeSpec("multisample-render", P + "multisample_render",
                  "s in randint(1,4) supersampling + AvgPool after official it 12000; SH schedule",
                  True),
        ProbeSpec("persistence", P + "persistence",
                  "Derived Gaussians written to params; gradients only via decoders; params-only "
                  "render equals the training render", True),
    )
    contributions = (
        ContributionSpec("pixel-gaussians", "One Gaussian per input pixel constrained to its ray, "
                         "positioned by residual depth (Sec. 4.1, Eq. 3)",
                         ("ray-structure", "persistence")),
        ContributionSpec("implicit-decoder", "Single-view constraint: implicit decoders for "
                         "residual depth and opacity with 5-channel depth masks (Sec. 4.1, supp. 1)",
                         ("decoder-architecture", "masked-residual")),
        ContributionSpec("multiview-tv", "L_multi = (1-l) TV + l MTV on rendered disparity "
                         "(Eq. 4-5)", ("tv-loss",)),
        ContributionSpec("flow-regularization", "Flow consistency of corresponding Gaussian "
                         "positions (Sec. 4.2, Eq. 6)", ("flow-loss",)),
        ContributionSpec("multisampling", "Several samples per pixel, averaged (Sec. 4.2)",
                         ("multisample-render",)),
        ContributionSpec("initialization", "Flow-based coarse alignment of monocular depth "
                         "(Eq. 8) and depth-dependent isotropic radius (Eq. 9)",
                         ("alignment", "radius")),
        ContributionSpec("official-objective", "Official photometric loss (multiscale L1, SSIM, "
                         "VGG)", ("photometric-terms",)),
    )
    keep = {"state-transitions", "photometric-gradient"}
    return replace(
        base,
        description="CoherentGS native port (pixel-structured Gaussians, implicit residual-depth/"
                    "opacity decoders, TV + flow regularisation, multisampling); paper fidelity "
                    "unverified.",
        components=components,
        probes=tuple(p for p in base.probes if p.name in keep) + probes,
        contributions=tuple(c for c in base.contributions if set(c.probes) <= keep) + contributions,
        citations=CITATIONS,
    )
