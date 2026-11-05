"""AGS-Mesh: Adaptive Gaussian Splatting and Meshing with Geometric Priors for Indoor Room
Reconstruction Using Smartphones (Ren et al., 3DV 2025, arXiv 2411.19271) — native SplatStudio
port of the official Splatfacto variant (``ns-train ags-mesh``, github.com/maturk/dn-splatter
commit 97588b4: AGSMeshRegularization on the DN-Splatter / Splatfacto 1.1.3 pipeline).

``ags-mesh``      photometric + DNC-filtered edge-aware sensor-depth loss (C1) + adaptive normal
                  regularization (C2: edge-masked then ANR-masked surface-normal term and the
                  Gaussian-normal term, after 7k) + min-scale loss; DN-Splatter renderer with
                  Gaussian normals, Splatfacto density, sensor-depth TSDF init (sidecars from
                  tools/prepare_mushroom.py + tools/prepare_dnc.py).
``ags-mesh-rgb``  same renderer, density and init, photometric loss only (Splatfacto control).

Reused from reproductions/dn-splatter (ss_dnsplatter): density.py, sidecar.py (extended),
renderer.py (extended with surface normals), four probes. Meshing (adaptive TSDF + IsoOctree)
is not implemented.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import DEPTH_DEFAULTS, NORMAL_DEFAULTS, SCALE_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2411.19271", "https://github.com/maturk/dn-splatter")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_agsmesh.probes:{fn}", desc, cuda)


def _build(name, priors, description):
    base = make_method("gsplat")
    comps = [
        ComponentSpec("ags-render", "renderer", "ss_agsmesh.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("splatfacto-density", "density", "ss_agsmesh.density:create_density",
                      ("gradient.screen.absolute",), (), dict(DENSITY_DEFAULTS)),
        base.components[2],  # photometric 0.8 L1 + 0.2 D-SSIM (= Splatfacto main loss)
    ]
    probes = list(base.probes) + [
        _probe("splatfacto-density", "density_recipe",
               "Splatfacto ADC on synthetic stats (copied from the DN-Splatter port)", True),
        _probe("sidecar-init", "sidecar_init",
               "Sidecar log-scales/quats applied when means match; -inf -> exact zero scale"),
        _probe("gaussian-normals", "gaussian_normals",
               "Flat Gaussian: rendered normal = R_w2c R e_argmin(s), flips with camera side, "
               "gradient reaches quats not means2d; official surface normals are detached", True),
        _probe("surface-normals", "surface_normals",
               "Normals of a rendered planar depth = analytic GL-frame plane normal, zero border"),
    ]
    contribs = list(base.contributions) + [
        ContributionSpec("splatfacto-adc", "Splatfacto 1.1.3 density control (official base)",
                         ("splatfacto-density",)),
        ContributionSpec("dense-oriented-init", "DN-Splatter sensor-depth init (official base)",
                         ("sidecar-init",)),
        ContributionSpec("gaussian-normals", "DN-Splatter Gaussian normals (official base)",
                         ("gaussian-normals",)),
    ]
    if priors:
        comps += [
            ComponentSpec("ags-depth", "loss", "ss_agsmesh.losses:create_depth",
                          ("render.depth.expected_z", "prior.sensor_depth",
                           "prior.depth_confidence"), (), dict(DEPTH_DEFAULTS)),
            ComponentSpec("ags-normal", "loss", "ss_agsmesh.losses:create_normal",
                          ("render.normal.camera", "render.normal.surface", "prior.mono_normal"),
                          (), dict(NORMAL_DEFAULTS)),
            ComponentSpec("ags-scale", "loss", "ss_agsmesh.losses:create_scale",
                          ("state.gaussian3d",), (), dict(SCALE_DEFAULTS)),
        ]
        probes += [
            _probe("dnc-filter", "dnc_filter",
                   "Synthetic frame: DNC keeps the consistent wall, rejects corrupted sensor "
                   "depth and a 35-deg-wrong prior; KNN-200 normals = independent PCA"),
            _probe("depth-schedule", "depth_schedule",
                   "0.2 x edge-aware log-L1 on raw depth before ns 7000, on D_f afterwards "
                   "(zero gradient at rejected pixels) vs an independent loop"),
            _probe("normal-anr", "normal_anr",
                   "Normal loss off until ns > 7000, edge-masked surface + Gaussian terms, ANR "
                   "angle mask (0.1 rad) from ns 15000, vs independent loops"),
            _probe("scale-loss", "scale_loss", "Min-scale loss and its analytic gradient"),
        ]
        contribs += [
            ContributionSpec("dnc", "C1: depth-normal consistency filtering of sensor depth",
                             ("dnc-filter", "depth-schedule")),
            ContributionSpec("anr", "C2: adaptive normal regularization",
                             ("normal-anr", "surface-normals")),
            ContributionSpec("scale-flatness", "DN-Splatter min-scale loss (official AGS loss)",
                             ("scale-loss",)),
        ]
    return replace(base, name=name, description=description, components=tuple(comps),
                   probes=tuple(probes), contributions=tuple(contribs), citations=CITATIONS)


def method():
    return _build("ags-mesh", True,
                  "AGS-Mesh native port (Splatfacto variant, official code 97588b4); "
                  "paper fidelity unverified.")


def method_rgb():
    return _build("ags-mesh-rgb", False,
                  "AGS-Mesh pipeline without depth/normal/scale losses (Splatfacto control); "
                  "not a paper method.")
