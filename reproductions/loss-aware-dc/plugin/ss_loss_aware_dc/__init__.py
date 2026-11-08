"""Unified Loss-Aware Density Control for 3D Gaussian Splatting (Kim, Chung, Heo, NeurIPS 2026,
arXiv 2611.02044) — native SplatStudio port written from the paper text (no official code is
released). Builds on SteepGS (arXiv 2505.05587): the splitting-matrix idea and the uncertain /
stationary filters follow the SteepGS formulation; the statistics kernel is new (csrc/stats.cu).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import TARGET_CAPABILITY
from .renderer import CAPABILITIES

CITATIONS = ("https://arxiv.org/abs/2611.02044", "https://arxiv.org/abs/2505.05587")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_loss_aware_dc.probes:{fn}", desc, cuda)


def method(name="loss-aware-dc", density_overrides=None):
    base = make_method(name)
    comps = (
        ComponentSpec("lossaware-gsplat", "renderer", "ss_loss_aware_dc.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {"absgrad": False, "antialiased": False}),
        ComponentSpec("lossaware-density", "density", "ss_loss_aware_dc.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, **(density_overrides or {})}),
        ComponentSpec("photometric", "loss", "ss_loss_aware_dc.losses:create_photometric",
                      ("render.rgb",), (TARGET_CAPABILITY,), {"ssim_weight": 0.2}),
    )
    probes = (
        _probe("lad-pipeline-step", "pipeline_step",
               "Renderer screen gradients + statistics + a densification step through the "
               "method's own loss (replaces the built-in renderer-gradient probe)", True),
        _probe("lad-stats-kernel", "stats_kernel",
               "CUDA kernel g_z, H_z, g/h_dup, g/h_dec vs an independent float64 compositor over "
               "gsplat's tile lists; g_z = gsplat means2d/conic grads; z(phi) = gsplat projection",
               True),
        _probe("lad-joint-split-matrix", "joint_split_matrix",
               "S_phi (Jz^T Hz Jz + 2nd-order chain) = brute-force Hessian in (p, log s); "
               "dropped mixed term only in S_ps; normalised eigmin under the Gaussian metric; "
               "children p +- L v_p, log s +- v_s/sqrt2"),
        _probe("lad-joint-split-cuda", "joint_split_cuda",
               "CUDA hyper-dual S_phi accumulation used in training = torch functorch reference",
               True),
        _probe("lad-decisions", "decisions",
               "Split (stationary + lambda<tau + uncertain) -> clone (eta in (1e-6,1e-2], o*eta) "
               "-> decay ((1-eta) o) disjoint sets and values; each ablation flag changes them"),
        _probe("lad-schedule", "schedule",
               "Densify at s>500, s%100==0, s<15000; no global opacity reset"),
        _probe("lad-surrogate-first-order", "surrogate_first_order",
               "Kernel g_dup/g_dec = re-rendered first-order loss change of a real clone/decay",
               True),
        _probe("lad-proxy-curvature", "proxy_curvature",
               "omega = 0.8 Charbonnier rho'' + 0.2*1e-4 diagonal SSIM proxy (explicit window "
               "sums, autograd rho'')"),
    )
    contribs = (
        ContributionSpec("joint-position-scale-split",
                         "Normalised joint (p, log s) splitting matrix, eigmin direction and "
                         "children (Sec. 4.2, Eqs. 5-9, App. A.4)",
                         ("lad-stats-kernel", "lad-joint-split-matrix", "lad-joint-split-cuda",
                          "lad-decisions")),
        ContributionSpec("surrogate-clone", "Taylor-surrogate cloning with opacity eta*o "
                         "(Sec. 4.3, Eqs. 10-13)",
                         ("lad-stats-kernel", "lad-surrogate-first-order", "lad-decisions")),
        ContributionSpec("surrogate-opacity-decay", "Selective soft opacity decay replacing the "
                         "global reset (Sec. 4.4, Eqs. 14-17)",
                         ("lad-stats-kernel", "lad-surrogate-first-order", "lad-decisions",
                          "lad-schedule")),
        ContributionSpec("proxy-curvature", "Charbonnier + diagonal SSIM curvature proxy "
                         "(App. A.5)", ("lad-proxy-curvature",)),
    )
    return replace(
        base,
        description="Unified Loss-Aware Density Control (NeurIPS 2026) native port from the "
                    "paper text (no official code); paper fidelity unverified.",
        components=comps,
        probes=tuple(p for p in base.probes if p.name != "renderer-gradient") + probes,
        contributions=tuple(
            replace(c, probes=("lad-pipeline-step",)) if c.name == "rasterizer-statistics" else c
            for c in base.contributions) + contribs,
        citations=CITATIONS,
    )
