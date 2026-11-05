"""Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration (ECCV 2024,
arXiv 2403.11056) — native SplatStudio port of github.com/lzhnb/Analytic-Splatting (commit
a905939, `train.py` without `--filter3d`/`--dense`, the paper's "Ours" row) with its CUDA
rasterizer vendored unmodified.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2403.11056", "https://github.com/lzhnb/Analytic-Splatting")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_analytic_splatting.probes:{fn}", desc, cuda)


def method(name="analytic-splatting"):
    base = make_method(name)
    photometric = base.components[2]
    comps = (
        ComponentSpec("analytic-renderer", "renderer",
                      "ss_analytic_splatting.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("analytic-density", "density", "ss_analytic_splatting.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        photometric,
    )
    probes = (
        _probe("analytic-forward-integral", "forward_integral",
               "Vendored kernel's pixel response = independent float64 Eq. 13 (rel < 2e-3), "
               "within 5 % of a brute-force pixel-area quadrature, and for a sub-pixel "
               "Gaussian < 0.5x the error of the 3DGS centre sample", True),
        _probe("analytic-backward", "backward",
               "Kernel gradients (means, log-scales, quats, opacity, sh0, means2D) = float64 "
               "autograd of the independent reference (rel < 1e-3) and central finite "
               "differences of the kernel (rel < 5e-3)", True),
        _probe("analytic-logistic-cdf", "logistic_cdf",
               "Eq. 7 conditioned logistic: |S - Phi| < 5e-4 (paper Fig. 4: ~4e-4), "
               "dS/dx = (1.6 + 0.21x^2)S(1-S), 1D window error < 1e-3 for sigma in [0.3, 6.6]; cubic term essential"),
        _probe("analytic-density-schedule", "density_schedule",
               "3DGS ADC of the official train.py: 100-step cadence from 500, clone vs split "
               "by percent_dense, no screen-size prune, filter-aware opacity reset "
               "min(o, 0.01/coef), nothing after 15000"),
    )
    contribs = (
        ContributionSpec("analytic-pixel-integral",
                         "Replace the centre sample g2D by the pixel-window integral, factorised "
                         "on the eigen-axes of Sigma2D (Sec. 4.2, Eq. 10-13), forward and "
                         "backward in the official CUDA kernel",
                         ("analytic-forward-integral", "analytic-backward")),
        ContributionSpec("conditioned-logistic-cdf",
                         "Gaussian CDF approximated by S(x)=1/(1+exp(-1.6x-0.07x^3)) with a "
                         "closed-form derivative (Sec. 4.1, Def. 1, Eq. 7-9)",
                         ("analytic-logistic-cdf", "analytic-forward-integral",
                          "analytic-backward")),
        ContributionSpec("3dgs-training", "3DGS training kept unchanged (official train.py "
                         "density control and opacity reset)", ("analytic-density-schedule",)),
    )
    return replace(
        base,
        description="Analytic-Splatting (ECCV 2024) native port with the official CUDA "
                    "rasterizer; paper fidelity unverified.",
        components=comps,
        probes=base.probes + probes,
        contributions=base.contributions + contribs,
        citations=CITATIONS,
    )
