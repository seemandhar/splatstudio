"""3D Student Splatting and Scooping (Zhu et al., CVPR 2025, arXiv 2503.10148).
Native SplatStudio port; paper fidelity unverified.

Ported from the official code https://github.com/realcrane/3D-student-splating-and-scooping
(GPL-2.0; rasterizer under the Inria Gaussian-Splatting research licence): Student's t /
signed-opacity rasterizer (diff-t-rasterization renderCUDA) as a JIT CUDA extension
(csrc/t_raster.cu) on an Inria-style torch projection and gsplat 1.5.3 tiling/SH; AdamSGHMC means
sampler (utils/sghmc.py) and t-relocation recycling (scene/nt_model.py, utils.cu).

Methods = official per-scene configs (configs/<scene>.json):
  student-splatting          cap 3M,   C_burnin 5e5, C 120, burn-in 7000  (common config, 40k)
  student-splatting-bicycle  cap 3M,   C_burnin 5e8, C 120, burn-in 15000 (30k iterations)
  student-splatting-room     cap 1.5M, C_burnin 5e5, C 120, burn-in 7000  (40k iterations)
  student-splatting-truck    cap 2.6M, C_burnin 5e5, C 120, burn-in 7000  (40k iterations)
Iteration counts are passed with --steps.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2503.10148",
    "https://github.com/realcrane/3D-student-splating-and-scooping",
)
CONFIGS = {
    "student-splatting": {"cap_max": 3_000_000, "C_burnin": 5e5, "C": 1.2e2,
                          "burnin_iterations": 7000},
    "student-splatting-bicycle": {"cap_max": 3_000_000, "C_burnin": 5e8, "C": 1.2e2,
                                  "burnin_iterations": 15000},
    "student-splatting-room": {"cap_max": 1_500_000, "C_burnin": 5e5, "C": 1.2e2,
                               "burnin_iterations": 7000},
    "student-splatting-truck": {"cap_max": 2_600_000, "C_burnin": 5e5, "C": 1.2e2,
                                "burnin_iterations": 7000},
}
NU_INIT = 100.0  # configs/*.json nu_degree


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_student_splatting.probes:{fn}", description, cuda)


PROBES = (
    _probe("t-forward-vs-dense", "t_forward_vs_dense",
           "CUDA t render (signed alpha, +-0.99 clamp, 1/255 skip, T<1e-4 stop, tile-rect "
           "truncation) equals a dense float64 reference (itself checked against gsplat's "
           "Gaussian rasterizer) and differs from the Gaussian and from |o|"),
    _probe("t-backward-vs-dense", "t_backward_vs_dense",
           "CUDA gradients (means2d, conics, colours, opacities, nu) equal autograd of the dense "
           "reference with the official straight-through clamp and partial dnu; exact_nu_grad "
           "matches the exact derivative and differs from the official one"),
    _probe("t-projection", "t_projection",
           "Torch EWA projection = gsplat EWA with eps2d 0 (no low-pass) and != eps2d 0.3; "
           "radius ceil(e(nu) sqrt(lambda_max)) with the official critical-value table; Inria "
           "conic backward 1/(det^2+1e-7); near cull 0.2"),
    _probe("scooping-sign", "scooping_sign",
           "Two-splat analytic composite: a negative front component subtracts its colour and "
           "raises T; opacity activation is tanh"),
    _probe("sghmc-step", "sghmc_step",
           "AdamSGHMC means update equals the formula in burn-in (no friction, covariance noise), "
           "at the momentum reset B+1 and after (friction lr C, noise sqrt(2 lr C)); sigmoid gate "
           "leaves opaque components with -lr^2 g; lr and nu-lr schedules; Adam lr for means 0; "
           "initial opacity 0.5 and 0.1x scales", False),
    _probe("t-relocation", "t_relocation",
           "Relocation equals the float64 lgamma/beta formula (signed o, N=1 identity); recycling "
           "caps at 5 %, zeroes source moments and dead momentum; growth to min(cap, 1.05N) with "
           "aligned, trainable fields", False),
    _probe("regularizer", "regularizer_probe",
           "Loss adds 0.01 mean|tanh o| + 0.01 mean exp(s) at every step", False),
)


def _make(name):
    base = make_method(name)
    renderer = ComponentSpec(
        "t-renderer", "renderer", "ss_student_splatting.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"background": 0.0, "tile_size": 16, "near_plane": 0.2, "fov_clamp": 1.3,
         "alpha_clip": 0.99, "exact_nu_grad": False, "nu_min": 1.0, "nu_max": 10000.0})
    density = ComponentSpec(
        "sghmc-recycling", "density", "ss_student_splatting.density:create_density", (), (),
        {**CONFIGS[name], "densify_from": 500, "densify_until": 25_000, "interval": 100,
         "dead_opacity": 0.005, "recycle_fraction": 0.05, "growth": 1.05, "sig_k": 100.0,
         "sig_x0": 0.995, "lr_init_base": 0.001226442339330813,
         "lr_final_base": 1.226630619638022e-05, "lr_reference_scale": 7.45176315307617,
         "lr_max_steps": 30_000, "opacity_lr": 0.005, "nu_lr": 10.0, "nu_lr_step": 5000,
         "nu_lr_gamma": 0.5, "nu_lr_min": 0.01, "init_opacity": 0.5, "init_scale_factor": 0.1})
    regularizer = ComponentSpec(
        "mcmc-regularizer", "loss", "ss_student_splatting.losses:create_regularizer",
        ("render.rgb",), (), {"opacity_reg": 0.01, "scale_reg": 0.01})
    fields = (FieldSpec("degrees", (), learning_rate=10.0, initial_value=NU_INIT),)
    contributions = (
        ContributionSpec("student-t-kernel",
                         "Unnormalised Student's t primitive with learnable nu in [1, 1e4], no "
                         "low-pass, critical-value truncation; custom CUDA fwd/bwd (Sec. 3.2)",
                         ("t-forward-vs-dense", "t-backward-vs-dense", "t-projection")),
        ContributionSpec("scooping",
                         "Signed opacity tanh(o) in (-1, 1): negative components subtract colour "
                         "(Sec. 3.3)", ("scooping-sign", "t-forward-vs-dense")),
        ContributionSpec("sghmc-sampler",
                         "SGHMC for means with Adam-preconditioned gradient, sigmoid-gated "
                         "friction/noise, burn-in with covariance noise; Adam for the rest; MCMC "
                         "regularisers (Sec. 3.4.1)", ("sghmc-step", "regularizer")),
        ContributionSpec("t-recycling",
                         "Recycle |o| < 0.005 (<= 5 %) and add 5 % per 100 steps with the t "
                         "relocation formula (Sec. 3.4.2, Eq. relocation)", ("t-relocation",)),
    )
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
    cfg = CONFIGS[name]
    return replace(
        base,
        description=("3D Student Splatting and Scooping (arXiv 2503.10148) native port of the "
                     f"official code, cap_max {cfg['cap_max']}, C_burnin {cfg['C_burnin']:g}, "
                     f"burn-in {cfg['burnin_iterations']}; paper fidelity unverified."),
        components=(renderer, density, base.components[2], regularizer),
        fields=fields, probes=base_probes + PROBES,
        contributions=base_contribs + contributions, citations=CITATIONS)


def student_splatting():
    return _make("student-splatting")


def student_splatting_bicycle():
    return _make("student-splatting-bicycle")


def student_splatting_room():
    return _make("student-splatting-room")


def student_splatting_truck():
    return _make("student-splatting-truck")


method = student_splatting
