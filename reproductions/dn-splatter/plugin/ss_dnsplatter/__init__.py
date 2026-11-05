"""DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing (Turkulainen et al.,
WACV 2025, arXiv 2403.17822) — native SplatStudio port of the official Nerfstudio plugin
(github.com/maturk/dn-splatter, commit 97588b4; Splatfacto 1.1.3 + gsplat 1.0.0).

``dn-splatter``      full method: photometric + edge-aware sensor-depth log-L1 + Omnidata normal
                     L1 + normal TV + min-scale loss, normals from the Gaussians' shortest axis,
                     Splatfacto density control, sensor-depth TSDF init with normal-aligned flat
                     Gaussians (sidecar from tools/prepare_mushroom.py).
``dn-splatter-splatfacto``  paper baseline control = vanilla ``ns-train splatfacto`` on the same
                     1M sensor-depth points (eval/batch_run.py:186-207, load_pcd_normals False):
                     Splatfacto density + photometric loss, isotropic log(mean 3-NN) scales and
                     random quaternions (splatfacto.py:214-220), num_downscales 2 /
                     resolution_schedule 3000 (1/4 -> 1/2 -> full res), no normal pass.
``dn-splatter-rgb``  priors-off ablation (NOT the paper baseline): DN-Splatter's renderer, density,
                     full-res training and oriented flat init, photometric loss only.
Meshing (Poisson on back-projected depth/normal maps) is not implemented.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import DEPTH_DEFAULTS, NORMAL_DEFAULTS, SCALE_DEFAULTS
from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .renderer import capabilities

CITATIONS = ("https://arxiv.org/abs/2403.17822", "https://github.com/maturk/dn-splatter")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_dnsplatter.probes:{fn}", desc, cuda)


def _build(name, variant, description):
    """variant: "dn" (full method), "rgb" (priors-off ablation), "splatfacto" (paper baseline)."""
    base = make_method("gsplat")
    splatfacto = variant == "splatfacto"
    render_cfg = dict(RENDERER_DEFAULTS)
    density_cfg = dict(DENSITY_DEFAULTS)
    photometric = base.components[2]  # photometric 0.8 L1 + 0.2 D-SSIM (= Splatfacto main loss)
    if splatfacto:
        render_cfg.update(normal_pass=False, num_downscales=2, resolution_schedule=3000)
        density_cfg.update(sidecar_init="isotropic")
        photometric = ComponentSpec("photometric", "loss", "ss_dnsplatter.losses:create_photometric",
                                    ("render.rgb",), (), dict(photometric.config))
    provides = tuple(sorted(capabilities(render_cfg["normal_pass"])))
    comps = [
        ComponentSpec("dn-render", "renderer", "ss_dnsplatter.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), provides,
                      render_cfg),
        ComponentSpec("splatfacto-density", "density", "ss_dnsplatter.density:create_density",
                      ("gradient.screen.absolute",), (), density_cfg),
        photometric,
    ]
    probes = list(base.probes)
    if splatfacto:  # base probe assumes a full-resolution first render
        probes = [p if p.name != "renderer-gradient" else
                  _probe("renderer-gradient", "renderer_gradient_scheduled",
                         p.description + " (run with the resolution schedule off, plus a "
                         "1/4-resolution scheduled render)", True) for p in probes]
    probes += [
        _probe("splatfacto-density", "density_recipe",
               "Splatfacto ADC on synthetic stats: absgrad*0.5*max(H,W) threshold, split/dup "
               "(dup after in-place /1.6), cull, densify window after reset, opacity reset "
               "to 0.2 with zeroed Adam moments, unscaled means LR", True),
    ]
    contribs = list(base.contributions) + [
        ContributionSpec("splatfacto-adc", "Splatfacto 1.1.3 density control (official base)",
                         ("splatfacto-density",)),
    ]
    if splatfacto:
        probes += [
            _probe("isotropic-init", "isotropic_init",
                   "Vanilla Splatfacto init on the sidecar points: log(mean 3-NN) on all axes "
                   "(-inf -> exact zero), Shoemake random quats"),
            _probe("resolution-schedule", "resolution_schedule",
                   "num_downscales 2 / schedule 3000: 1/4, 1/2, full-res training renders with "
                   "scaled intrinsics, full-res eval, area-downscaled photometric target", True),
        ]
        contribs += [
            ContributionSpec("splatfacto-init", "Splatfacto baseline: isotropic kNN scales, "
                             "random quats from the sensor-depth points", ("isotropic-init",)),
            ContributionSpec("splatfacto-resolution", "Splatfacto baseline: 1/4 -> 1/2 -> 1 "
                             "resolution schedule", ("resolution-schedule",)),
        ]
    else:
        probes += [
            _probe("sidecar-init", "sidecar_init",
                   "Sidecar log-scales/quats applied when means match; -inf -> exact zero scale"),
            _probe("gaussian-normals", "gaussian_normals",
                   "Flat Gaussian: rendered normal = R_w2c R e_argmin(s), flips with camera side, "
                   "partial-alpha pixel over the official GL-frame white background, gradient "
                   "reaches quats but not means2d", True),
        ]
        contribs += [
            ContributionSpec("dense-oriented-init",
                             "C5: dense sensor-depth init with normal-aligned flattened Gaussians",
                             ("sidecar-init",)),
            ContributionSpec("gaussian-normals",
                             "C2: per-Gaussian normals from the shortest axis, alpha-composited",
                             ("gaussian-normals",)),
        ]
    if variant == "dn":
        comps += [
            ComponentSpec("dn-depth", "loss", "ss_dnsplatter.losses:create_depth",
                          ("render.depth.expected_z", "prior.sensor_depth"), (),
                          dict(DEPTH_DEFAULTS)),
            ComponentSpec("dn-normal", "loss", "ss_dnsplatter.losses:create_normal",
                          ("render.normal.camera", "prior.mono_normal"), (),
                          dict(NORMAL_DEFAULTS)),
            ComponentSpec("dn-scale", "loss", "ss_dnsplatter.losses:create_scale",
                          ("state.gaussian3d",), (), dict(SCALE_DEFAULTS)),
        ]
        probes += [
            _probe("depth-loss", "depth_loss",
                   "Edge-aware log-L1 vs an independent formula; effective weight 1.2; "
                   "constant image = plain masked log-L1"),
            _probe("normal-loss", "normal_loss", "Normal L1 + TV vs an independent formula"),
            _probe("scale-loss", "scale_loss", "Min-scale loss and its analytic gradient"),
        ]
        contribs += [
            ContributionSpec("gradient-aware-depth", "C1: gradient-aware sensor-depth loss",
                             ("depth-loss",)),
            ContributionSpec("mono-normal-smooth", "C3: monocular normal L1 + TV smoothing",
                             ("normal-loss",)),
            ContributionSpec("scale-flatness", "C4: minimum-scale loss", ("scale-loss",)),
        ]
    return replace(base, name=name, description=description, components=tuple(comps),
                   probes=tuple(probes), contributions=tuple(contribs), citations=CITATIONS)


def method():
    return _build("dn-splatter", "dn",
                  "DN-Splatter native port (official code 97588b4); paper fidelity unverified.")


def method_splatfacto():
    return _build("dn-splatter-splatfacto", "splatfacto",
                  "Paper baseline control: vanilla Splatfacto 1.1.3 (isotropic kNN init, random "
                  "quats, 1/4->1/2->1 resolution schedule) on DN-Splatter's sensor-depth init "
                  "points; not a paper method.")


def method_rgb():
    return _build("dn-splatter-rgb", "rgb",
                  "Priors-off ablation: DN-Splatter renderer/density/oriented init with the "
                  "photometric loss only; NOT the paper's Splatfacto baseline.")
