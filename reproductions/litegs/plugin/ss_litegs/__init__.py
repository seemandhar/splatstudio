"""LiteGS: A High-performance Framework to Train 3DGS in Subminutes via System and Algorithm
Codesign (arXiv 2503.01199). Native SplatStudio port of the algorithmic parts of the official code
https://github.com/MooreThreads/LiteGS (Gaussian-Splatting licence); paper fidelity unverified.

`litegs` = LiteGS-quality (official full_eval.py): per-scene target = 3DGS primitive count, 30K
iterations. The speed-only system parts (warp-based fp16 rasteriser, Morton cluster-cull-compact)
are not ported; gsplat renders.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import META_CAPABILITY

CITATIONS = (
    "https://arxiv.org/abs/2503.01199",
    "https://github.com/MooreThreads/LiteGS",
)
RENDERER_CAPS = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                 META_CAPABILITY)


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_litegs.probes:{fn}", description, cuda)


PROBES = (
    _probe("litegs-stats-kernel", "stats_kernel",
           "CUDA per-Gaussian S=sum g^2, M=sum g, C, sum T*alpha of the per-pixel opacity gradient "
           "equal per-pixel autograd of v.I(x,y) through gsplat; M equals the summed opacity grad",
           True),
    _probe("litegs-variance-score", "variance_score_probe",
           "Score = clamp(S/(C+1)-(M/(C+1))^2,0)*C*o^2 (independent formula); conflicting per-pixel "
           "gradients outrank consistent ones with the same sum; magnitude ablation ranks the "
           "other way"),
    _probe("litegs-densify", "densify_probe",
           "Budget arithmetic, sampling only score>0, split keeps the parent and adds a child with "
           "scale/1.6, clone copies, zero Adam moments, weight prune of old W==0 only, fields stay "
           "trainable", True),
    _probe("litegs-decay-schedule", "decay_schedule",
           "Opacity decay halves sigmoid(o) with a 1/128 floor and clears all Adam states; "
           "densify/decay epochs, until and statistics window for n_train=161 (garden)"),
    _probe("litegs-adam", "adam_probe",
           "LiteAdam equals an independent bias-free Adam on visible rows; invisible rows and their "
           "moments are untouched; differs from torch Adam"),
    _probe("litegs-sh-schedule", "sh_schedule",
           "Training SH degree = min(epoch//5,3) via the shared schedule; eval uses the runtime "
           "degree; training rgb clamped to [0,1]", True),
)
CONTRIBUTIONS = (
    ContributionSpec(
        "opacity-gradient-variance-densification",
        "Densification metric C*Var of per-pixel opacity gradients (Sec. 3.3), custom CUDA "
        "moments, budgeted multinomial clone/split and weight pruning (official "
        "DensityControllerTamingGS)",
        ("litegs-stats-kernel", "litegs-variance-score", "litegs-densify"),
    ),
    ContributionSpec(
        "opacity-decay-schedule",
        "Opacity decay (x0.5) every 10 epochs, densify every 5 epochs from epoch 3 to 80% of "
        "training (Sec. 3.3, Sec. 4 implementation details)",
        ("litegs-decay-schedule",),
    ),
    ContributionSpec(
        "litegs-training-recipe",
        "Official training recipe: visible-only Adam without bias correction, TamingGS learning "
        "rates, SH degree every 5 epochs, rgb clamp before the loss",
        ("litegs-adam", "litegs-sh-schedule"),
    ),
)


def _make(name, density_config, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "litegs-renderer", "renderer", "ss_litegs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False, "sh_epoch_interval": 5,
         "clamp_training_rgb": True},
    )
    density = ComponentSpec(
        "litegs-density", "density", "ss_litegs.density:create_density",
        (META_CAPABILITY,), (), {**DEFAULTS, **density_config},
    )
    photometric = replace(base.components[2], config={"ssim_weight": 0.2})
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base.probes + PROBES,
        contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _make("litegs", {},
                 "LiteGS (arXiv 2503.01199) native port of the algorithmic parts, LiteGS-quality "
                 "budgets; paper fidelity unverified.")
