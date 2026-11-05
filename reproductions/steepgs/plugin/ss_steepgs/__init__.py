"""SteepGS: Steepest Descent Density Control for Compact 3D Gaussian Splatting (CVPR 2025,
arXiv 2505.05587) — native SplatStudio port of github.com/facebookresearch/SteepGS (commit
63eee45, `--densify_strategy steepest --S_estimator inv_cov`) with its CUDA rasterizer vendored.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2505.05587", "https://github.com/facebookresearch/SteepGS")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_steepgs.probes:{fn}", desc, cuda)


def method(name="steepgs", density_overrides=None):
    base = make_method(name)
    photometric = base.components[2]
    comps = (
        ComponentSpec("steepgs-renderer", "renderer", "ss_steepgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("steepgs-density", "density", "ss_steepgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, "options": list(DENSITY_DEFAULTS["options"]),
                       **(density_overrides or {})}),
        photometric,
    )
    probes = (
        _probe("steepgs-dLdG", "dldg_kernel",
               "Vendored kernel's splitting buffer [0,0] = sum over pixels of o*dL/dalpha from "
               "an independent float64 PyTorch Inria compositor; screen gradient matches too",
               True),
        _probe("steepgs-splitting-matrix", "splitting_matrix",
               "Density accumulates (sum dL/dG) * Sigma^-1 averaged over visible views "
               "(independent rotation formula); lambda_min = c/s_min^2 along the shortest axis"),
        _probe("steepgs-saddle-criterion", "saddle_criterion",
               "Only Gaussians with lambda_min < -1e-6 AND screen gradient >= 2e-4 densify; "
               "dropping the eigen condition changes the selection"),
        _probe("steepgs-split-direction", "split_direction",
               "Split offspring at p +- |n| v_min (official repeat sign pattern), scale s/1.6, "
               "opacity copied; clone copy and original at p + lr v_min"),
        _probe("steepgs-schedule", "schedule",
               "Densify only at s>500, s%100==0, s<15000; opacity reset 0.01 at 3000; world "
               "prune only after 3000; no screen-size prune"),
    )
    contribs = (
        ContributionSpec("splitting-matrix", "Per-Gaussian splitting matrix from the CUDA "
                         "backward (Thm. 1, §3.4; inv_cov estimator of the official code)",
                         ("steepgs-dLdG", "steepgs-splitting-matrix")),
        ContributionSpec("saddle-split-condition", "Split only if lambda_min(S) < eps_split "
                         "together with the ADC gradient condition (Thm. 2 (i), Alg. 1)",
                         ("steepgs-saddle-criterion",)),
        ContributionSpec("steepest-descent-offsets", "Two offspring displaced along "
                         "+-v_min(S) (Thm. 2 (iv)); opacity halving (iii) is off in the official "
                         "default and here", ("steepgs-split-direction",)),
        ContributionSpec("3dgs-density-schedule", "3DGS cadence / prune / opacity reset as in "
                         "the official train.py", ("steepgs-schedule",)),
    )
    return replace(
        base,
        description="SteepGS (CVPR 2025) native port with the official CUDA rasterizer; "
                    "paper fidelity unverified.",
        components=comps,
        probes=base.probes + probes,
        contributions=base.contributions + contribs,
        citations=CITATIONS,
    )


def adc_control():
    """Control: same vendored rasterizer and code base, official `--densify_strategy adc`
    (the paper's 3DGS row). Not SteepGS; used to separate renderer from density effects."""
    m = method("steepgs-adc", {"strategy": "adc"})
    return replace(m, description="Official SteepGS code base with --densify_strategy adc "
                   "(3DGS ADC control, not SteepGS); paper fidelity unverified.",
                   probes=m.probes[:3], contributions=m.contributions[:3])
