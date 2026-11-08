"""Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting (Zhang, Hu, Lao,
He, Zhao; ECCV 2024, arXiv 2403.15530). Native SplatStudio port of the official code
https://github.com/zhengzhang01/Pixel-GS (05c9d55); paper fidelity unverified.

Both contributions live in one density component (gsplat DefaultStrategy with the Pixel-GS
densification statistic); renderer and loss are the stock 3DGS/gsplat ones.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

CITATIONS = (
    "https://arxiv.org/abs/2403.15530",
    "https://github.com/zhengzhang01/Pixel-GS",
)

# 3DGS defaults (official arguments/__init__.py) expressed in gsplat DefaultStrategy keys; equal to
# DefaultStrategy defaults, spelled out so the paper values are visible in the spec.
BASE_CONFIG = {
    "grow_grad2d": 0.0002,  # tau_pos (densify_grad_threshold)
    "grow_scale3d": 0.01,  # percent_dense
    "prune_opa": 0.005,
    "prune_scale3d": 0.1,
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "refine_every": 100,
    "reset_every": 3000,
    "refine_scale2d_stop_iter": 0,  # official screen-size prune is a no-op (max_radii2D zeroed)
    "gamma_depth": 0.37,  # paper Sec. 4.1
}


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_pixel_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("pixel-count-kernel", "pixel_count_kernel",
           "CUDA per-Gaussian contributing-pixel counts m_k^i (Eq. 8) equal a brute-force "
           "per-pixel compositing loop and reproduce gsplat's alpha image", True),
    _probe("pixel-aware-accumulation", "pixel_aware_accumulation",
           "After one real render/backward the strategy statistic equals the independent "
           "sum m*f*|g| / sum m (Eq. 11); pixel-only / depth-only / off variants equal their "
           "reduced formulas and the full statistic differs from the 3DGS mean", True),
    _probe("depth-scaling", "depth_scaling",
           "Eq. 10 factor: Gaussians nearer than gamma*radius are attenuated by (z/gamma R)^2, "
           "farther ones keep factor 1; densification flips accordingly"),
)
CONTRIBUTIONS = {
    "pixel": ContributionSpec(
        "pixel-aware-gradient",
        "Sec. 3.2, Eq. 7-8: per-view NDC gradient norms weighted by the number of pixels the "
        "Gaussian is blended into, normalised by the pixel sum (custom CUDA counts)",
        ("pixel-count-kernel", "pixel-aware-accumulation"),
    ),
    "depth": ContributionSpec(
        "scaled-gradient-field",
        "Sec. 3.3, Eq. 9-11: per-view NDC gradient scaled by clip((z/(0.37 radius))^2,0,1)",
        ("depth-scaling", "pixel-aware-accumulation"),
    ),
}


def _make(name, pixel_aware, depth_scaling):
    base = make_method(name)
    density = ComponentSpec(
        "pixel-gs-density", "density", "ss_pixel_gs.density:create_density",
        ("gradient.screen.signed",), (),
        {**BASE_CONFIG, "pixel_aware": pixel_aware, "depth_scaling": depth_scaling},
    )
    contribs = tuple(c for k, c in CONTRIBUTIONS.items()
                     if (k == "pixel" and pixel_aware) or (k == "depth" and depth_scaling))
    label = {(True, True): "", (True, False): " (ablation: pixel-aware gradient only)",
             (False, True): " (ablation: scaled gradient field only)"}[(pixel_aware, depth_scaling)]
    return replace(
        base,
        description=f"Pixel-GS (arXiv 2403.15530) native port{label}; paper fidelity unverified.",
        components=(base.components[0], density, base.components[2]),
        probes=base.probes + PROBES,
        contributions=base.contributions + contribs,
        citations=CITATIONS,
    )


def method():
    return _make("pixel-gs", True, True)


def method_pixel_only():
    return _make("pixel-gs-pixel-only", True, False)


def method_depth_only():
    return _make("pixel-gs-depth-only", False, True)
