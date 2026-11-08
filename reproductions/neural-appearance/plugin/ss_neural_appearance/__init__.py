"""Compact Neural Appearance Models for Efficient Gaussian Splatting (Hahlbohm et al., arXiv
2609.05255) — native SplatStudio port of the paper's Neural appearance model and of its paired SH
baseline, both inside the paper's shared pipeline (Faster-GS rasterizer / FusedAdam / Morton from
reproductions/faster-gs, MCMC densification and PPISP from the official FasterGSVDA code,
github.com/nerficg-project/efficient-gaussian-appearance, Apache-2.0; PPISP vendored from
github.com/nv-tlabs/ppisp, Apache-2.0).

Entry points ``neural-appearance-<scene>`` / ``neural-appearance-sh-<scene>``: the scene suffix
selects the official MCMC primitive cap (paper_configs/default/<appearance>/<scene>.yaml). Train
with ``--steps 35000`` = 30000 + 5000 PPISP controller-distillation steps.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, MethodSpec, ProbeSpec

from .appearance import SH_C0
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2609.05255",
    "https://github.com/nerficg-project/efficient-gaussian-appearance",
    "https://arxiv.org/abs/2602.09999 (Faster-GS; kernels via reproductions/faster-gs)",
    "https://arxiv.org/abs/2404.09591 (3DGS-MCMC)",
    "https://arxiv.org/abs/2601.18336 (PPISP; github.com/nv-tlabs/ppisp)",
)

# MAX_PRIMITIVES of paper_configs/default/{neural,sh}/<scene>.yaml (identical for both)
CAPS = {"bicycle": 6_000_000, "flowers": 3_000_000, "garden": 5_000_000, "stump": 4_500_000,
        "treehill": 3_500_000, "bonsai": 1_500_000, "counter": 1_500_000, "kitchen": 1_500_000,
        "room": 1_500_000, "truck": 2_500_000, "train": 1_000_000, "drjohnson": 3_500_000,
        "playroom": 2_500_000}
REGISTERED = ("garden", "room", "bicycle", "truck")


def _p(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_neural_appearance.probes:{fn}", desc, cuda)


PROBES = (
    _p("na-neural-decoder", "neural_decoder",
       "MLP input = [sin/cos features, SH basis (== 3DGS eval_sh)], degree gating, 816 params, zero "
       "output init, residual off at degree 0, A = tanh(MLP(x)) vs an explicit matrix formula"),
    _p("na-render-composition", "render_composition",
       "Rendered image (rasterizer degree-0 path fed sh0 + A/Y0) equals an independent float64 dense "
       "compositor of relu(0.5 + c0 + A(d)); the residual changes the image", True),
    _p("na-direction-gradient", "direction_gradient",
       "dL/dmu differs from the detached-direction gradient; features and MLP get gradients", True),
    _p("na-mcmc", "mcmc",
       "3DGS-MCMC relocation formula vs exact binomial sum; dead Gaussians relocated onto alive "
       "ones, +5% growth capped, sampled moments reset; opacity-gated noise"),
    _p("na-schedules", "schedules",
       "Cosine 7k->37k, MLP warm-up/decay policy, means policy, neural and SH learning rates"),
    _p("na-ppisp", "ppisp",
       "Vendored CUDA PPISP == ppisp torch reference; per-frame parameters used for training views; "
       "na_blob round trip reproduces the evaluation render (ss-eval restores MLP + PPISP)", True),
)

CONTRIBUTIONS = (
    ContributionSpec("neural-appearance", "C1 (§4, Eq. 5-7): 8 latent features + base colour per "
                     "Gaussian, shared bias-free 32-16-16-3 MLP, tanh residual, SH-schedule gating",
                     ("na-neural-decoder", "na-render-composition", "na-direction-gradient")),
    ContributionSpec("common-interface", "C2 (§3.1 Eq. 1-2): c = ReLU(0.5 + c0 + A(d)), base colour "
                     "kept; SH baseline row = degree-3 SH residual", ("na-render-composition",
                                                                       "na-schedules")),
    ContributionSpec("shared-pipeline", "C3 (§5.1, supp. §B.1): Faster-GS + MCMC (scene caps) + "
                     "PPISP with controller distillation", ("na-mcmc", "na-ppisp", "na-schedules")),
)

FIELDS = (
    FieldSpec("na_features", (8,), 0.008, 0.0),  # NEURAL.FEATURE_DIM, zero init
    FieldSpec("na_blob", (4,), 1e-12, 0.0),  # packed MLP + PPISP state (never has a gradient; lr set to 0)
)


def build(appearance, scene):
    if scene not in CAPS:
        raise ValueError(f"unknown scene {scene}")
    neural = appearance == "neural"
    density = {**DENSITY_DEFAULTS, "appearance": appearance, "cap_max": CAPS[scene]}
    if not neural:
        density.update(lr_base=0.0025 * SH_C0, lr_residual=0.000125, cosine_decay=False)
    density["sh_degrees"] = list(density["sh_degrees"])
    renderer = {"appearance": appearance, "background": [0.0, 0.0, 0.0], "near": 0.2,
                "far": 10000.0}
    comps = (
        ComponentSpec("na-renderer", "renderer", "ss_neural_appearance.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      renderer),
        ComponentSpec("na-mcmc", "density", "ss_neural_appearance.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), density),
        ComponentSpec("photometric", "loss", "splatstudio.losses:create_photometric",
                      ("render.rgb",), (), {}),
        ComponentSpec("na-regularizers", "loss", "ss_neural_appearance.losses:create_regularizers",
                      ("render.rgb",), (), dict(LOSS_DEFAULTS)),
    )
    label = "Neural" if neural else "SH (paper's paired baseline row)"
    name = f"neural-appearance-{scene}" if neural else f"neural-appearance-sh-{scene}"
    return MethodSpec(
        name,
        f"Compact Neural Appearance (arXiv 2609.05255) native port, {label} appearance on the "
        f"paper's Faster-GS + MCMC (cap {CAPS[scene]:,}) + PPISP pipeline; train --steps 35000; "
        "paper fidelity unverified.",
        comps, PROBES, CONTRIBUTIONS, FIELDS, CITATIONS)


def _factory(appearance, scene):
    def method():
        return build(appearance, scene)

    return method


for _scene in REGISTERED:
    globals()[f"method_{_scene}"] = _factory("neural", _scene)
    globals()[f"method_sh_{_scene}"] = _factory("sh", _scene)

method = _factory("neural", "garden")
__all__ = ["build", "method", "CAPS", "replace"]
