"""FastGS: Training 3D Gaussian Splatting in 100 Seconds (CVPR 2026, arXiv 2511.04283) —
native SplatStudio port of github.com/fastgs/FastGS @ 44e02a5 with its CUDA rasterizer
vendored unmodified (raster.py).

Methods
- `fast-gs`:        paper main row ("FastGS", train_base.sh: densification every 500 steps),
                    Compact Box beta (mult) 0.5 = official Mip-NeRF 360 setting.
- `fast-gs-big`:    "FastGS-Big" (train_big.sh: densification every 100 steps), mult 0.5.
- `fast-gs-tdb`, `fast-gs-big-tdb`: same with mult 0.7 (official T&T / Deep Blending scripts).
Per-scene arguments (grad_abs_thresh, dense, SH learning rates, loss_thresh) are looked up
from the initial SfM point count (scenes.py).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2511.04283", "https://github.com/fastgs/FastGS",
             "https://fastgs.github.io")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_fast_gs.probes:{fn}", desc, cuda)


PROBES = (
    _probe("fastgs-metric-kernel", "metric_kernel",
           "Vendored rasterizer: image equals an independent float64 dense compositor (mult "
           "large, no tile culling); per-Gaussian metric counts equal a brute-force count of "
           "masked contributing pixels; CB mult 0.5 removes tile pairs and only lowers counts",
           True),
    _probe("fastgs-scores", "scores",
           "Multi-view scores with an injected renderer equal an independent Eq. 1-6 "
           "computation (channel-mean L1, min-max, tau, floor(sum/K), sum count*E_photo, "
           "min-max)"),
    _probe("fastgs-densify-gate", "densify_gate",
           "Only Gaussians with s_d > 5 and the gradient criterion are cloned (3DGS grad, "
           "scale <= dense*extent) or split (AbsGS grad, scale > dense*extent); opacity <= 0.8 "
           "afterwards; fields stay trainable"),
    _probe("fastgs-prune", "prune",
           "Early prune: removed set is a subset of the 3DGS candidates and <= int(0.5 #cand), "
           "drawn by 1/(1e-6+1-s_p); late prune removes exactly o<0.1 or s_p>0.9"),
    _probe("fastgs-schedule", "schedule",
           "Densify at 1000..14500 step 500 (base) / 600..14900 step 100 (big), opacity reset "
           "3000..12000, late prune 18000/21000/24000/27000, all wired through after_backward"),
    _probe("fastgs-optimizer-schedule", "optimizer_schedule",
           "ScheduledAdam steps every iteration <=15k (SH every 16), every 32 to 20k, every 64 "
           "after, on summed gradients (differs from per-step Adam); lr overrides applied"),
    _probe("fastgs-scene-table", "scene_table",
           "Per-scene official arguments resolved from the SfM count (garden/counter base and "
           "big), defaults for unknown scenes, refusal of a CB mult mismatch"),
)

CONTRIBUTIONS = (
    ContributionSpec("vcd", "Multi-view consistent densification: average high-error-pixel "
                     "count in the footprint over K=10 views > tau_d gates clone/split "
                     "(Sec. 3.2, Eq. 1-4)",
                     ("fastgs-metric-kernel", "fastgs-scores", "fastgs-densify-gate",
                      "fastgs-schedule")),
    ContributionSpec("vcp", "Multi-view consistent pruning score (Eq. 5-6): score-sampled 3DGS "
                     "pruning before 15k, o<0.1 or s_p>0.9 every 3k after (supp. Sec. 8.1)",
                     ("fastgs-scores", "fastgs-prune", "fastgs-schedule")),
    ContributionSpec("compact-box", "Compact Box: SnugBox/AccuTile tile assignment with "
                     "Mahalanobis threshold scaled by beta (Sec. 3.4, supp. Eq. 15), official "
                     "kernel", ("fastgs-metric-kernel",)),
    ContributionSpec("3dgs-accel", "Baseline 3DGS-accel optimizer step schedule and SH "
                     "batching; AbsGS gradient for split; per-scene official arguments "
                     "(supp. Sec. 8.1)",
                     ("fastgs-optimizer-schedule", "fastgs-scene-table")),
)


def _make(name, table, interval, mult, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "fastgs-renderer", "renderer", "ss_fast_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        {"background": list(RENDERER_DEFAULTS["background"]), "mult": mult},
    )
    density = ComponentSpec(
        "fastgs-density", "density", "ss_fast_gs.density:create_density",
        tuple(sorted(DENSITY_REQUIRES)), (),
        {**DENSITY_DEFAULTS, "lr": dict(DENSITY_DEFAULTS["lr"]),
         "background": list(DENSITY_DEFAULTS["background"]), "scene_args": {},
         "table": table, "densification_interval": interval, "mult": mult},
    )
    loss = ComponentSpec("photometric", "loss", "ss_fast_gs.losses:create_photometric",
                         ("render.rgb",), (), {"ssim_weight": 0.2})
    return replace(
        base, description=description, components=(renderer, density, loss),
        probes=base.probes + PROBES, contributions=base.contributions + CONTRIBUTIONS,
        citations=CITATIONS,
    )


def method():
    return _make("fast-gs", "base", 500, 0.5,
                 "FastGS (arXiv 2511.04283) native port, main 'FastGS' row (densify every 500, "
                 "CB mult 0.5); official rasterizer vendored; paper fidelity unverified.")


def big():
    return _make("fast-gs-big", "big", 100, 0.5,
                 "FastGS-Big (arXiv 2511.04283) native port (densify every 100, CB mult 0.5); "
                 "paper fidelity unverified.")


def tdb():
    return _make("fast-gs-tdb", "base", 500, 0.7,
                 "FastGS (arXiv 2511.04283) native port for T&T / Deep Blending (CB mult 0.7); "
                 "paper fidelity unverified.")


def big_tdb():
    return _make("fast-gs-big-tdb", "big", 100, 0.7,
                 "FastGS-Big (arXiv 2511.04283) native port for T&T / Deep Blending (CB mult "
                 "0.7); paper fidelity unverified.")
