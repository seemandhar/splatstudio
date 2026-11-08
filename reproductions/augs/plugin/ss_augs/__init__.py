"""AuGS -- Augmented Radiance Field (Yang et al., ICLR 2026, arXiv 2602.19916) as a native
SplatStudio method. Official code: github.com/XiaoXinyyx/AuGS_Code (read, not executed)."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CAPS = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")


def _method(name, density_overrides=None, description=None):
    base = make_method(name)
    renderer = ComponentSpec(
        "augs-renderer", "renderer", "ss_augs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPS, {})
    density = ComponentSpec(
        "augs-density", "density", "ss_augs.density:create_density",
        ("gradient.screen.signed",), (), {**DEFAULTS, **(density_overrides or {})})
    probes = (
        ProbeSpec("lobe-opacity", "ss_augs.probes:lobe_probe",
                  "Renderer opacity of augmented Gaussians equals an independent Eq. 1 "
                  "evaluation (zero outside T*pi), originals untouched, lobe params get gradients"),
        ProbeSpec("median-depth", "ss_augs.probes:median_depth_probe",
                  "CUDA T-threshold depth picks the Gaussian an independent compositing "
                  "reference picks (front/back switch with opacity) at its max-response depth",
                  True),
        ProbeSpec("error-driven-2d", "ss_augs.probes:fit2d_probe",
                  "2D kernels are sampled in the error region and reduce the image loss over "
                  "the rendered background", True),
        ProbeSpec("inverse-splatting", "ss_augs.probes:inverse_probe",
                  "Back-projected kernel reprojects to the target 2D covariance, lies on the "
                  "near surface across a depth discontinuity, faces the camera; T init rule"),
        ProbeSpec("joint-optimisation", "ss_augs.probes:joint_probe",
                  "Sparse Adam without bias correction: augmented rows update, original rows "
                  "frozen except opacity of used Gaussians"),
    )
    contributions = (
        ContributionSpec("view-dependent-opacity", "Phong-like opacity lobe (Eq. 1)",
                         ("lobe-opacity",)),
        ContributionSpec("error-driven-2d-training",
                         "2D Gaussians fitted over renders where the error is large (§3.2), "
                         "T-threshold depth maps", ("error-driven-2d", "median-depth")),
        ContributionSpec("inverse-gaussian-splatting",
                         "SLINK clustering + WPCA + scale calibration + lobe init (§3.3)",
                         ("inverse-splatting",)),
        ContributionSpec("joint-optimisation", "Augmented kernels fully optimised, originals "
                         "opacity-only (§3.4)", ("joint-optimisation",)),
    )
    return replace(
        base,
        components=(renderer, density, base.components[2]),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        fields=(
            FieldSpec("lobe_dir", (3,), learning_rate=0.001, initial_value=0.0),
            FieldSpec("lobe_t", (), learning_rate=0.0002, initial_value=0.0),
            FieldSpec("lobe_beta", (), learning_rate=0.002, initial_value=0.0),
            # 1 marks augmented Gaussians; never receives gradients (not in the graph).
            FieldSpec("aug_flag", (), learning_rate=1e-12, initial_value=0.0),
        ),
        description=description or (
            "AuGS (3DGS base, sh=3) native port: 30k-step 3DGS base, error-driven 2D Gaussians, "
            "inverse splatting, view-dependent opacity lobes, joint optimisation; paper fidelity "
            "unverified"),
        citations=("https://arxiv.org/abs/2602.19916", "https://github.com/XiaoXinyyx/AuGS_Code"),
    )


def method():
    return _method("augs")


def method_smoke():
    """Short schedule for smoke tests only (not a paper configuration)."""
    return _method("augs-smoke", {"base_steps": 1500, "refine_iters_per_view": 3,
                                  "coarse_steps": 20, "finetune_steps": 50},
                   "AuGS smoke-test schedule (1500 base steps, 3 joint steps/view, 20/50 2D "
                   "steps); not a paper configuration")
