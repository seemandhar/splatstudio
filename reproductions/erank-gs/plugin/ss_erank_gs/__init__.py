"""Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting
(Hyung et al., NeurIPS 2024, arXiv 2406.11672) — native SplatStudio port of the official code
(github.com/junhahyung/erank_gs @99fd128).

erank-gs       paper "+a+b" / "3DGS+e": fixed ADC (GOF abs-gradient densification) + erank loss
erank-gs-adc   paper "+a": fixed ADC only (ablation)
erank-gs-3dgs  paper "3DGS" row under the same protocol (Inria ADC, prune 0.005, black bg)
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEFAULTS as ERANK_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2406.11672",
    "https://github.com/junhahyung/erank_gs",
    "https://arxiv.org/abs/2404.10772",
)
INRIA_ADC = {"abs_gradient": False, "clone_resample": False, "min_opacity": 0.005}


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_erank_gs.probes:{fn}", desc, cuda)


def _build(name, *, erank, fixed_adc, description):
    base = make_method(name)
    density_cfg = dict(DENSITY_DEFAULTS)
    if not fixed_adc:
        density_cfg.update(INRIA_ADC)
    comps = [
        ComponentSpec("erank-renderer", "renderer", "ss_erank_gs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("erank-density", "density", "ss_erank_gs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), density_cfg),
        ComponentSpec("photometric", "loss", "splatstudio.losses:create_photometric",
                      ("render.rgb",), (), {}),
    ]
    probes = [p for p in base.probes if p.name == "state-transitions"]
    contribs = [c for c in base.contributions if c.name == "state-lifecycle"]
    probes += [
        _probe("absgrad-units", "absgrad_units",
               "Renderer supplies gsplat absgrad; NDC-scaled L1 of it equals an independent "
               "per-pixel sum of |dL/dx|+|dL/dy| (official kernel channel) and >= signed L2", True),
        _probe("density-recipe", "density_recipe",
               "Clone (resampled or not) / split s/1.6 / prune (min opacity, 0.1 extent after "
               "3000) / reset to 0.01 every 3000 / trainable leaves after every op"),
    ]
    contribs.append(ContributionSpec(
        "density-control", "Official densification recipe (3DGS or fixed ADC as configured)",
        ("absgrad-units", "density-recipe")))
    if fixed_adc:
        probes.append(_probe(
            "adc-abs-metric", "adc_abs_metric",
            "A Gaussian whose per-pixel gradients cancel (small ||sum||, large sum ||.||) is "
            "densified only through the quantile-matched abs criterion (App. A.4)"))
        contribs.append(ContributionSpec(
            "fixed-adc", "Densify on the sum of per-pixel gradient norms (Sec. 3.2, App. A.4)",
            ("adc-abs-metric",)))
    if erank:
        comps.append(ComponentSpec("erank", "loss", "ss_erank_gs.losses:create_erank_loss",
                                   ("state.gaussian3d",), (), dict(ERANK_DEFAULTS)))
        probes.append(_probe(
            "erank-formula", "erank_formula",
            "erank(sphere)=3, erank(disk)=2, erank(needle)->1; loss equals an independent "
            "numpy formula (Eq. 4, code form); zero before step 7000; needle gradient raises "
            "its minor scales"))
        contribs.append(ContributionSpec(
            "erank-regularization", "Effective-rank barrier + smallest-scale penalty "
            "(Sec. 3.2, Eq. 4)", ("erank-formula",)))
    return replace(base, description=description, components=tuple(comps),
                   probes=tuple(probes), contributions=tuple(contribs), citations=CITATIONS)


def method():
    return _build("erank-gs", erank=True, fixed_adc=True,
                  description="Effective-rank regularization + fixed ADC on 3DGS (paper '+a+b') "
                              "native port; paper fidelity unverified.")


def method_adc():
    return _build("erank-gs-adc", erank=False, fixed_adc=True,
                  description="erank-gs ablation '+a' (fixed ADC only); paper fidelity "
                              "unverified.")


def method_3dgs():
    return _build("erank-gs-3dgs", erank=False, fixed_adc=False,
                  description="3DGS baseline in the erank-gs protocol (Inria ADC, black "
                              "background); paper fidelity unverified.")
