"""GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and
High-Quality 3D Gaussian Splatting (CVPR 2025, arXiv 2411.06019). Native SplatStudio port of
https://github.com/noodle-lab/GaussianSpa; paper fidelity unverified.

Methods:
  gaussianspa           opacity criterion (train_opacity.sh defaults r1=0.50, r2=0.80)
  gaussianspa-imp       importance criterion (train_imp_score.sh defaults r1=0.50, r2=0.70)
  gaussianspa-<scene>   README "Recommend Arguments Setting for Evaluation" presets
The Mini-Splatting stages (blur split, depth reinit, 15k simplification, schedules) are copied
from the SPLATIFY Mini-Splatting port (see ms_density.py); everything runs 40000 steps.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .losses import PENALTY_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2411.06019",
    "https://github.com/noodle-lab/GaussianSpa",
    "https://arxiv.org/abs/2403.14166",
)
FIELD_LR = 1e-12
RENDERER_CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    "training.target_image",
)
# official README table: scene -> (criterion, prune_ratio1, prune_ratio2)
PRESETS = {
    "drjohnson": ("importance", 0.75, 0.50),
    "playroom": ("opacity", 0.50, 0.77),
    "bicycle": ("importance", 0.50, 0.72),
    "bonsai": ("importance", 0.50, 0.80),
    "counter": ("opacity", 0.60, 0.50),
    "flowers": ("opacity", 0.50, 0.70),
    "garden": ("importance", 0.60, 0.50),
    "kitchen": ("opacity", 0.40, 0.80),
    "room": ("opacity", 0.50, 0.80),
    "stump": ("importance", 0.65, 0.70),
    "treehill": ("opacity", 0.40, 0.75),
    "train": ("opacity", 0.50, 0.80),
    "truck": ("opacity", 0.62, 0.80),
}


def _probe(name, fn, description, cuda=True, module="probes"):
    return ProbeSpec(name, f"ss_gaussianspa.{module}:{fn}", description, cuda)


def _make(name, criterion, r1, r2, description):
    base = make_method(name)
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "spa-renderer", "renderer", "ss_gaussianspa.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), RENDERER_CAPS,
        {"absgrad": False, "antialiased": False},
    )
    window = {k: DEFAULTS[k] for k in ("spa_start", "spa_stop", "spa_interval")}
    density = ComponentSpec(
        "spa-density", "density", "ss_gaussianspa.density:create_density",
        ("gradient.screen.signed", "training.target_image"), (),
        {**DEFAULTS, "criterion": criterion, "prune_ratio1": r1, "prune_ratio2": r2},
    )
    penalty = ComponentSpec(
        "spa-penalty", "loss", "ss_gaussianspa.losses:create_penalty",
        ("state.gaussian3d",), (),
        {**PENALTY_DEFAULTS, **window, "stop_inclusive": criterion == "opacity"},
    )
    probes = (
        _probe("spa-penalty", "penalty_probe",
               "Penalty = delta/2||a-z+u||^2 and d/dlogit = delta(a-z+u)a(1-a) (independent "
               "formula); zero off the every-50-steps window (15200, 25200]", False),
        _probe("spa-projection", "projection_probe",
               "prox_h(a+u) equals an independent numpy top-kappa reference (opacity or "
               "importance ranking); uses a+u, not a", False),
        _probe("spa-admm-cycle", "admm_cycle_probe",
               "Density: z=prox(a), u=0 at 15200; z=prox(a+u), u+=a-z at 15250; idle off-"
               "interval; a fresh density on checkpointed params reproduces the next update"),
        _probe("spa-hard-prune", "hard_prune_probe",
               "At 25200 exactly the int((1-r2)N) top-ranked Gaussians survive with their "
               "Adam moments"),
        _probe("spa-sampling", "sampling_probe",
               "15k sampling keeps int(N(1-r1)) with p ∝ imp+1 (zero-importance Gaussians can "
               "survive), then reinit with z=u=0"),
        _probe("ms-stats-kernel", "stats_kernel",
               "Copied Mini-Splatting CUDA blend statistics equal a brute-force blend",
               module="ms_probes"),
        _probe("ms-depth-midpoint", "depth_midpoint",
               "Copied Mini-Splatting depth kernel = density maximum along the ray",
               module="ms_probes"),
        _probe("ms-blur-split", "blur_split",
               "Copied Mini-Splatting blur split (Eq. 1) fires only with theta_blur",
               module="ms_probes"),
        _probe("ms-depth-reinit", "depth_reinit",
               "Copied Mini-Splatting depth reinit: surface points, GT colours, opacity 0.1",
               module="ms_probes"),
        _probe("ms-schedule", "schedule",
               "Means LR lr(it-10000) after 15k; SH 0 until 15k then +1 per 1000", False,
               module="ms_probes"),
    )
    contributions = (
        ContributionSpec("optimizing-step", "Augmented-Lagrangian penalty pulling opacity "
                         "toward the sparse auxiliary z (Sec. 3.3, Eq. 11-14)", ("spa-penalty",)),
        ContributionSpec("sparsifying-step", "Proximal l0 projection z = prox_h(a + lambda), "
                         f"{criterion}-ranked (Eq. 15-16, Sec. 5.2)", ("spa-projection",)),
        ContributionSpec("multiplier-update", "lambda <- lambda + a - z every 50 steps in "
                         "(15200, 25200] (Eq. 17, Algorithm 1)", ("spa-admm-cycle",)),
        ContributionSpec("zero-gaussian-removal", "Hard prune of the 'zero' Gaussians at 25200, "
                         "then light tuning to 40k (Sec. 3.4)", ("spa-hard-prune",)),
        ContributionSpec("mini-splatting-pipeline", "Mini-Splatting stages before 15k with "
                         "GaussianSpa's 15k sampling (official train_*.py)",
                         ("spa-sampling", "ms-stats-kernel", "ms-depth-midpoint", "ms-blur-split",
                          "ms-depth-reinit", "ms-schedule")),
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric, penalty),
        # ADMM state; never receives a gradient (the penalty detaches z and u), so Adam skips it.
        # The contract requires lr > 0; the tiny value would be inert even if a grad appeared.
        fields=(FieldSpec("spa_z", (), FIELD_LR, 0.0), FieldSpec("spa_u", (), FIELD_LR, 0.0)),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("gaussianspa", "opacity", 0.50, 0.80,
                 "GaussianSpa (CVPR 2025) native port, opacity criterion (train_opacity.sh "
                 "defaults); paper fidelity unverified.")


def importance():
    return _make("gaussianspa-imp", "importance", 0.50, 0.70,
                 "GaussianSpa (CVPR 2025) native port, importance criterion "
                 "(train_imp_score.sh defaults); paper fidelity unverified.")


def _preset(scene):
    criterion, r1, r2 = PRESETS[scene]

    def factory():
        return _make(f"gaussianspa-{scene}", criterion, r1, r2,
                     f"GaussianSpa (CVPR 2025) native port, official README preset for {scene} "
                     f"({criterion}, prune_ratio1={r1}, prune_ratio2={r2}); paper fidelity "
                     "unverified.")

    factory.__name__ = f"preset_{scene}"
    return factory


for _scene in PRESETS:
    globals()[f"preset_{_scene}"] = _preset(_scene)
