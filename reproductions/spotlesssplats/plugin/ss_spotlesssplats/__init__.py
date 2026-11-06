"""SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting (Sabour, Goli et al., ACM TOG,
arXiv 2406.20055) — native SplatStudio port of the official gsplat trainer
(github.com/lilygoli/SpotLessSplats, examples/spotless_trainer.py, commit 0caae3c).

* ``spotlesssplats``        SLS-mlp with utilisation-based pruning (the paper's headline rows).
* ``spotlesssplats-noubp``  SLS-mlp without UBP (paper Fig. 8 "SLS-mlp w/o UBP").
* ``spotlesssplats-agg``    SLS-agg: agglomerative-cluster vote, no UBP (paper Fig. 8 "SLS-agg";
                            official ``--cluster``, default ``ubp=False``).

Needs per-training-image Stable Diffusion 2.1 features ``<data>/SD/<stem>.npy`` (and for agg the
cluster maps ``<data>/SLS_clusters/<stem>.npy``); see README.md.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import requirements
from .losses import MLP_DEFAULTS, ROBUST_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2406.20055",
    "https://github.com/lilygoli/SpotLessSplats",
)
P = "ss_spotlesssplats.probes:"


def _build(name, mask_mode, ubp, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "sls-renderer", "renderer", "ss_spotlesssplats.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
        dict(RENDERER_DEFAULTS))
    density = ComponentSpec(
        "sls-density", "density", "ss_spotlesssplats.density:create_density",
        tuple(sorted(requirements(ubp))), (), {**DENSITY_DEFAULTS, "ubp": ubp})
    robust = ComponentSpec(
        "sls-robust", "loss", "ss_spotlesssplats.losses:create_robust", ("render.rgb",), (),
        {**ROBUST_DEFAULTS, "mask_mode": mask_mode})
    components = [renderer, density, robust]
    if mask_mode == "mlp":
        components.append(ComponentSpec(
            "sls-mlp", "loss", "ss_spotlesssplats.losses:create_mlp_loss", ("render.rgb",), (),
            dict(MLP_DEFAULTS)))
    keep = {"state-transitions", "renderer-gradient"}
    probes = [p for p in base.probes if p.name in keep] + [
        ProbeSpec("robust-mask", P + "robust_mask_probe",
                  "Eq. 2 / code robust_mask vs NumPy reference; distractor blob masked"),
        ProbeSpec("quantile-histogram", P + "histogram_probe",
                  "Discounted 10000-bin residual histogram quantiles vs NumPy (§4.2.2)"),
        ProbeSpec("warmup-schedule", P + "warmup_probe",
                  "Supp. B staircase alpha; alpha=1 uses all pixels, late masking drops outliers"),
        ProbeSpec("pipeline-step", P + "pipeline_probe",
                  "Renderer -> robust loss (+MLP) -> density step on a synthetic scene", True),
    ]
    contributions = [c for c in base.contributions
                     if c.name in {"state-lifecycle", "rasterizer-statistics"}] + [
        ContributionSpec("robust-masked-l1",
                         "Masked L1 with RobustNeRF-style spatial inlier mask (Eq. 1-2), "
                         "replacing L1 + D-SSIM (official ssim_lambda = 0)",
                         ("robust-mask", "pipeline-step")),
        ContributionSpec("histogram-quantile",
                         "Residual quantiles tracked with a discounted histogram (§4.2.2)",
                         ("quantile-histogram",)),
        ContributionSpec("bernoulli-warmup",
                         "Bernoulli scheduled-sampling warm-up of the mask (Eq. 9, supp. B)",
                         ("warmup-schedule",)),
        ContributionSpec("no-opacity-reset-sh-reset",
                         "No opacity reset, no scale prune; shN clamped to 0.001 at step 8002",
                         ("density-schedule",)),
    ]
    probes.append(ProbeSpec("density-schedule", P + "density_probe",
                            "Official refine: duplicate/split/prune, UBP prune + new-Gaussian "
                            "utilisation 1, SH reset clamp(max) + zeroed moments"))
    if mask_mode == "mlp":
        probes += [
            ProbeSpec("mlp-equivalence", P + "mlp_probe",
                      "Low-res first layer == official upsample-then-MLP; hinge signs (Eq. 7)"),
            ProbeSpec("mlp-loss-order", P + "mlp_loss_order_probe",
                      "sls-mlp refuses to run without sls-robust's step-local values"),
        ]
        contributions.append(ContributionSpec(
            "sls-mlp-classifier",
            "SD-feature + PE MLP inlier classifier co-trained with hinge supervision between the "
            "tau=.5 / .9 residual masks (Eq. 5-8)",
            ("mlp-equivalence", "mlp-loss-order", "pipeline-step")))
    if mask_mode == "agg":
        probes.append(ProbeSpec("agg-vote", P + "agg_probe",
                                "Cluster inlier-fraction vote (Eq. 3-4) vs loop reference"))
        contributions.append(ContributionSpec(
            "sls-agg-clustering", "Agglomerative SD-feature superpixels vote (Eq. 3-4)",
            ("agg-vote", "pipeline-step")))
    if ubp:
        probes.append(ProbeSpec("ubp-kernel", P + "ubp_kernel_probe",
                                "UBP CUDA statistic vs brute-force per-pixel autograd", True))
        contributions.append(ContributionSpec(
            "utilization-pruning",
            "Utilisation-based pruning: per-pixel squared screen-mean gradients summed over 100 "
            "steps, prune < 1e-14, every 100 steps for 500 < t < 15000 (Eq. 10)",
            ("ubp-kernel", "density-schedule", "pipeline-step")))
    return replace(base, description=description, components=tuple(components),
                   probes=tuple(probes), contributions=tuple(contributions),
                   citations=CITATIONS)


def method():
    return _build("spotlesssplats", "mlp", True,
                  "SpotLessSplats SLS-mlp + UBP native port (official gsplat trainer numerics, "
                  "no GLO); paper fidelity unverified.")


def method_noubp():
    return _build("spotlesssplats-noubp", "mlp", False,
                  "SpotLessSplats SLS-mlp without utilisation-based pruning; paper fidelity "
                  "unverified.")


def method_agg():
    return _build("spotlesssplats-agg", "agg", False,
                  "SpotLessSplats SLS-agg (agglomerative SD-feature clusters) native port; paper "
                  "fidelity unverified.")
