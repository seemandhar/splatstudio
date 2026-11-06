"""Deblurring 3D Gaussian Splatting (Lee, Lee, Sun, Ali, Park; ECCV 2024, arXiv 2401.00834) --
native SplatStudio port of github.com/benhenryL/Deblurring-3D-Gaussian-Splatting.

The blur MLP F_theta and its Adam live in ``ss_deblurring_3dgs.bank.ACTIVE`` during training
(owned by ``deblur3dgs-density``, exported to the checkpoint's ``density_state["bank"]``).
Training renders are blurred by the MLP-transformed Gaussians; ``ss-eval`` renders the sharp
Gaussians. Two configurations follow the official configs/real_defocus.txt and
configs/real_camera_motion.txt (20000 iterations, ``--recipe 3dgs``).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2401.00834",
             "https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting")

CONFIGS = {  # official configs/*.txt (real sets)
    "defocus": {"densify_grad_threshold": 2e-4, "densify_prune_threshold": 5e-3,
                "bank_mode": "defocus", "bank_lambda_s": 0.01},
    "motion": {"densify_grad_threshold": 5e-4, "densify_prune_threshold": 1e-2,
               "bank_mode": "motion", "bank_lambda_s": 0.01, "bank_lambda_p": 0.01},
}


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_deblurring_3dgs.probes:{fn}", desc, cuda)


def method(mode="defocus"):
    name = f"deblurring-3dgs-{mode}"
    base = make_method(name)
    comps = (
        ComponentSpec("deblur3dgs-renderer", "renderer",
                      "ss_deblurring_3dgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("deblur3dgs-density", "density", "ss_deblurring_3dgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), {**DENSITY_DEFAULTS, **CONFIGS[mode]}),
        ComponentSpec("deblur3dgs-photometric", "loss",
                      "ss_deblurring_3dgs.losses:create_photometric", ("render.rgb",), (),
                      dict(LOSS_DEFAULTS)),
    )
    probes = (
        _probe("mlp-architecture", "mlp_architecture",
               "GTnet: 91 input channels (PE without pi, pos 3 / view 10 + s + q), 3 x 64 "
               "shared ReLU trunk, linear heads with official xavier gains, deltas clamped to "
               "[1, 1.1] after lambda_s scaling, detached inputs"),
        _probe("defocus-blur", "defocus_blur",
               "Training render equals gsplat with Sigma = R(q*dr) diag(s*ds)^2 R^T for the "
               "unnormalised quaternion (independent I + 2wK + 2K^2 formula); eval render equals "
               "plain 3DGS; MLP gets gradients; identity deltas reproduce the sharp render", True),
        _probe("motion-blur", "motion_blur",
               "Training render equals the mean of 5 independent renders (one unshifted, four "
               "at x + lambda_p dx_i); the position head gets gradients", True),
        _probe("extra-points", "extra_points",
               "add_points equals a brute-force loop of the official algorithm (N_p, bounded "
               "uniform box, K=4 neighbours, t_d rejection, distance-weighted SH) and "
               "re-initialises scales/rotations/opacities with fresh Adam states"),
        _probe("depth-pruning", "depth_pruning",
               "Opacity prune threshold divided by the relative-depth weight in [1, 3]: a far "
               "low-opacity Gaussian survives that plain pruning removes"),
        _probe("schedule", "schedule",
               "Official thresholds and densify schedule, xyz lr log-linear over 30000, no "
               "opacity reset at step 3000"),
        _probe("photometric", "photometric",
               "0.8 L1 + 0.2 (1 - SSIM) with the zero-padded Inria SSIM (unfold reference)"),
    )
    contribs = (
        ContributionSpec("defocus-blur-mlp", "Per-Gaussian (dr, ds) from a small MLP enlarge "
                         "the covariance for blurry training renders (Eq. 7-10)",
                         ("mlp-architecture", "defocus-blur")),
        ContributionSpec("camera-motion-blur", "M = 5 MLP-shifted Gaussian sets averaged "
                         "(Eq. 11-12)", ("mlp-architecture", "motion-blur")),
        ContributionSpec("extra-points", "KNN-coloured extra points at step 2500 (Alg. 1)",
                         ("extra-points",)),
        ContributionSpec("depth-based-pruning", "Depth-dependent opacity pruning (Sec. 3.3)",
                         ("depth-pruning",)),
        ContributionSpec("sharp-inference", "Test renders use the untransformed Gaussians",
                         ("defocus-blur",)),
        ContributionSpec("training-recipe", "3DGS objective and official schedule (no opacity "
                         "reset, thresholds, 20k iterations)", ("schedule", "photometric")),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description=f"Deblurring 3DGS native port ({mode} blur config: blur MLP on Gaussian "
                    "covariances" + (" and positions, 5 averaged renders" if mode == "motion"
                                     else "") + ", extra KNN points at 2500, depth-based "
                    "pruning, 20k iterations, black background); evaluated sharp; paper "
                    "fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        citations=CITATIONS,
    )


def method_defocus():
    return method("defocus")


def method_motion():
    return method("motion")
