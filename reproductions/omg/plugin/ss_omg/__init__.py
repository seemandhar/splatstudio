"""Optimized Minimal 3D Gaussian Splatting (OMG; Lee, Ko, Park, NeurIPS 2025, arXiv 2503.16924).
This is a native SplatStudio port of the official code at https://github.com/maincold2/OMG. That
repository has no licence file, so it was read and re-implemented rather than copied. Paper
fidelity is unverified.

The base is our matched Mini-Splatting port (ms_density / ms_core / kernels / csrc, copied from
reproductions/mini-splatting). OMG adds local-distinctiveness pruning (C1), the space-feature
attribute representation (C2), sub-vector quantisation (C3), and G-PCC/Huffman/LZMA storage with
evaluation of the decoded model (C4).

Methods:
- `omg`: OMG-XS, tau 0.96, outdoor importance metric (Mip-NeRF 360 outdoor, T&T).
- `omg-indoor`: OMG-XS with the indoor metric (Mip-NeRF 360 indoor, DB).
- `omg-m`: OMG-M, tau 0.99, outdoor.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import OMG_DEFAULTS
from .ms_density import DEFAULTS as MS_DEFAULTS
from .renderer import STATIC, VIEW

CITATIONS = (
    "https://arxiv.org/abs/2503.16924",
    "https://github.com/maincold2/OMG",
    "https://arxiv.org/abs/2403.14166",
)
RENDERER_CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    "training.target_image",
)
FIELDS = (
    # official training_setup: f_static / f_view use feature_lr 0.0025
    FieldSpec(STATIC, (3,), learning_rate=0.0025, initial_value=0.0),
    FieldSpec(VIEW, (3,), learning_rate=0.0025, initial_value=0.0),
)


def _probe(name, fn, description, cuda=True, module="probes"):
    return ProbeSpec(name, f"ss_omg.{module}:{fn}", description, cuda)


def _make(name, metric, tau, label):
    base = make_method(name)
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "omg-renderer", "renderer", "ss_omg.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False})
    density = ComponentSpec(
        "omg-density", "density", "ss_omg.density:create_density",
        ("gradient.screen.signed", "training.target_image"), (),
        {**MS_DEFAULTS, **OMG_DEFAULTS, "imp_metric": metric, "importance_thresh": tau})
    ms = "ms_probes"
    probes = (
        _probe("ms-stats-kernel", "stats_kernel", "Mini-Splatting CUDA blend statistics equal a "
               "brute-force per-pixel loop (copied MS probe)", module=ms),
        _probe("ms-depth-midpoint", "depth_midpoint", "Depth point = density maximum of the "
               "argmax-weight Gaussian along the ray (copied MS probe)", module=ms),
        _probe("ms-blur-split", "blur_split", "Blur split of a zero-gradient covering Gaussian "
               "(copied MS probe)", module=ms),
        _probe("ms-depth-reinit", "depth_reinit", "Depth re-initialisation: points on the surface, "
               "GT colours, opacity 0.1, fresh Adam (copied MS probe)", module=ms),
        _probe("ms-simplification", "simplification", "Intersection preserving + importance "
               "sampling int(0.5 nnz) at 15k (copied MS probe)", module=ms),
        _probe("ms-schedule", "schedule", "Means LR restart and SH 0 -> 3 schedule after 15k "
               "(copied MS probe)", cuda=False, module=ms),
        _probe("omg-ld-scoring", "ld_scoring", "20k prune keeps exactly the Gaussians above the "
               "tau-CDF split of Ibar * LD^lambda, with LD from bit-loop Morton codes and the "
               "official neighbour construction; differs from the plain CDF prune; fresh Adam"),
        _probe("omg-space-feature", "space_feature", "sh0/shN/opacity = MLP_t/v/o([MLP_s(freq16("
               "contract(p))), T or V]) vs explicit sin/cos + matmul reference; tcnn layouts "
               "(15360 params); position changes F, V changes only shN", cuda=False),
        _probe("omg-train-render", "train_render", "Live render equals an independent "
               "rasterisation of the reference attributes at the official SH degree; gradients "
               "reach MLPs, T, V, scale, rotation, means and not the unused base fields"),
        _probe("omg-net-schedule", "net_schedule", "Closed-form net LR equals torch "
               "ChainedScheduler(LinearLR, MultiStepLR) replay; event steps 15k/20k/29k/30k",
               cuda=False),
        _probe("omg-svq", "svq", "SVQ layout 3x64x1 / 2x512x2 / 3x1024x2, nearest-code indices, "
               "render uses codebooks only (codebook gradients, no per-Gaussian geometry/feature "
               "gradients), indices frozen"),
        _probe("omg-codec", "codec_roundtrip", "Huffman lossless with optimal cost, fp16<->uint16 "
               "lossless, G-PCC + container encode -> decode -> bake; a bank-free renderer of the "
               "decoded state matches the live SVQ model within fp16 error"),
    )
    contributions = (
        ContributionSpec("mini-splatting-base", "Mini-Splatting densification, depth reinit "
                         "and importance sampling up to 15k (OMG is built on it)",
                         ("ms-stats-kernel", "ms-depth-midpoint", "ms-blur-split",
                          "ms-depth-reinit", "ms-simplification", "ms-schedule")),
        ContributionSpec("local-distinctiveness", "C1 (§3.3): Ibar * ((1/D) sum_K |T_i-T_j|_1)"
                         "^lambda with Morton-order neighbours, tau-CDF prune at 20k",
                         ("omg-ld-scoring",)),
        ContributionSpec("space-feature-attributes", "C2 (§3.1, Eq. 3-4): per-Gaussian T, V + "
                         "tiny positional-encoding field F; MLPs give SH DC, opacity, SH rest",
                         ("omg-space-feature", "omg-train-render", "omg-net-schedule")),
        ContributionSpec("sub-vector-quantisation", "C3 (§3.2): SVQ of scale, rotation and [T,V] "
                         "in the last 1k iterations, K-means init, frozen indices",
                         ("omg-svq",)),
        ContributionSpec("storage", "C4 (§4.1): fp16 positions via G-PCC, Huffman indices, fp16 "
                         "codebooks/MLPs, LZMA; decoded-model evaluation", ("omg-codec",)),
    )
    return replace(
        base,
        description=f"OMG ({label}) native port on the Mini-Splatting port; decoded-model "
                    "evaluation; paper fidelity unverified",
        components=(renderer, density, photometric),
        fields=FIELDS,
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("omg", "outdoor", 0.96, "OMG-XS, outdoor metric")


def indoor():
    return _make("omg-indoor", "indoor", 0.96, "OMG-XS, indoor metric")


def medium():
    return _make("omg-m", "outdoor", 0.99, "OMG-M, outdoor metric")
