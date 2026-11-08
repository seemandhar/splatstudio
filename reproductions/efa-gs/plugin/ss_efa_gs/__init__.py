"""EFA-GS (Wang et al., Cyberworlds 2025, arXiv 2508.02493) as a native SplatStudio method.

Default variant EFA-GS(Mip): Mip-Splatting (3D + 2D filters, GOF absolute-gradient
densification as in the official EFA-GS `Mip-Splatting/` code) with the LFCF densification.
`efa-gs-base` is the same code path with LFCF off (official interval_times = infinity), the
paper's "Mip-splatting" control.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES

# Official EFA-GS commit 57f330f: Mip-Splatting/scripts/run_mipnerf360.py, train.py defaults,
# arguments/__init__.py, scene/gaussian_model.py.
RENDERER_CONFIG = {"kernel_size_2d": 0.1, "background": [0.0, 0.0, 0.0]}  # official black bg
DENSITY_CONFIG = {
    # Mip-Splatting / 3DGS densification (arguments/__init__.py, train.py).
    "grad_threshold": 0.0002,
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "percent_dense": 0.01,
    "prune_opacity": 0.005,
    "prune_world": 0.1,
    "opacity_reset_every": 3000,
    "opacity_reset_value": 0.01,
    # Mip-Splatting 3D filter.
    "filter_coef": 0.2,
    "near": 0.2,
    "frustum_margin": 0.15,
    "filter_interval": 100,
    "total_steps": 30000,
    # EFA-GS LFCF (run_mipnerf360.py: scaler_max 1.5, scaler_min 1.0, interval_time 2,
    # diffscale True; train.py defaults for the rest).
    "lfcf": True,
    "interval_times": 2,
    "c_max": 1.5,
    "c_min": 1.0,
    "c_last": 1.0,
    "pow": 1.0,
    "splitting_ub": 1.0,
    "split_multiplier": 2.0,
    "tolerance": 1e-5,
    "diffscale": True,
    "depth_strategy": True,
    "lfcf_children": 1,
}

PROBES = (
    ProbeSpec("lfcf-decision", "ss_efa_gs.probes:lfcf_decision",
              "Enlarge/shrink masks equal an independent per-Gaussian loop of Alg. 1 with the "
              "official selection (signed or quantile abs gradient) and tolerance rules"),
    ProbeSpec("depth-factor", "ss_efa_gs.probes:depth_factor",
              "Enlarging factor = theta c_min + (1-theta) c_max with theta = minmax log sampling "
              "interval (nearest -> c_max, deepest -> c_min); disabling it is detected"),
    ProbeSpec("scale-strategy", "ss_efa_gs.probes:scale_strategy",
              "Volume-preserving per-axis enlarge (c, c^-1/3, c^-2/3) and shrink "
              "-(log c + ln 2)/2 * (1, 1/2, 0) against hand-computed values"),
    ProbeSpec("anneal-schedule", "ss_efa_gs.probes:anneal_schedule",
              "log c multiplier 1 -> 0.5 -> 0 over 500..15000 and LFCF at 600, 800, ..., 14800"),
    ProbeSpec("filter3d", "ss_efa_gs.probes:filter3d",
              "Mip-Splatting 3D filter sqrt(0.2) d / f on an analytic case"),
    ProbeSpec("gof-densify", "ss_efa_gs.probes:gof_densify",
              "Non-LFCF step clones/splits exactly the Gaussians selected by g >= tau or "
              "g_abs >= quantile(g_abs, 1 - ratio), keeping fields trainable"),
    ProbeSpec("lfcf-gpu-step", "ss_efa_gs.probes:lfcf_gpu_step",
              "Driving the controller over steps 501-800 runs LFCF at 600/800 and GOF at 700, edits "
              "log-scales in place by the strategy deltas with Adam moments kept, adds one child "
              "per split; efa-gs-base runs no LFCF", True),
    ProbeSpec("renderer-absgrad", "ss_efa_gs.probes:renderer_absgrad",
              "Renderer gives absolute screen gradients and the density accumulates "
              "|ax| W/2 + |ay| H/2 (official rasterizer grad[:, 2]) and ||g|| per visible Gaussian",
              True),
)
CONTRIBUTIONS = (
    ContributionSpec("lfcf", "Low-Frequency-Come-First: enlarge Gaussians whose gradient rose since "
                     "the previous LFCF step, shrink + split the others (Alg. 1)",
                     ("lfcf-decision", "lfcf-gpu-step")),
    ContributionSpec("depth-strategy", "Depth-based enlarging factor between c_max and c_min",
                     ("depth-factor", "lfcf-gpu-step")),
    ContributionSpec("scale-strategy", "Volume-preserving per-axis enlarging/shrinking",
                     ("scale-strategy", "lfcf-gpu-step")),
    ContributionSpec("strategies-1-3", "LFCF every r = 2 densifications, linear log-factor "
                     "annealing, depth-dependent split probability",
                     ("anneal-schedule", "lfcf-gpu-step")),
    ContributionSpec("mip-gof-base", "Mip-Splatting base with 3D/2D filters and GOF "
                     "absolute-gradient densification (official EFA-GS base code)",
                     ("filter3d", "gof-densify", "renderer-absgrad")),
)


def _build(name, lfcf, description):
    base = make_method(name)
    renderer = ComponentSpec("efa-mip-gsplat", "renderer", "ss_efa_gs.renderer:create_renderer",
                             ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                             dict(RENDERER_CONFIG))
    density = ComponentSpec("efa-density", "density", "ss_efa_gs.density:create_density",
                            ("gradient.screen.absolute", "gradient.screen.signed"), (),
                            {**DENSITY_CONFIG, "lfcf": lfcf})
    photometric = base.components[2]
    # The built-in renderer-gradient probe skips set_training_cameras; renderer-absgrad covers it.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(
        replace(c, probes=("renderer-absgrad",)) if c.name == "rasterizer-statistics" else c
        for c in base.contributions
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base_probes + PROBES,
        contributions=base_contribs + CONTRIBUTIONS,
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=("https://arxiv.org/abs/2508.02493", "https://github.com/jcwang-gh/EFA-GS",
                   "https://arxiv.org/abs/2311.16493"),
    )


def method():
    return _build("efa-gs", True,
                  "EFA-GS(Mip) native port (LFCF + depth/scale strategies on Mip-Splatting with "
                  "GOF densification); paper fidelity unverified.")


def method_base():
    return _build("efa-gs-base", False,
                  "EFA-GS base control: official EFA-GS Mip-Splatting code path with LFCF off "
                  "(GOF abs-grad densification, 3D/2D filters); paper fidelity unverified.")
