"""Experimental runnable configurations, not certified reproductions of papers."""

from dataclasses import replace

from .specs import ComponentSpec, ContributionSpec, MethodSpec, ProbeSpec


def make_method(name, *, absgrad=False, antialiased=False):
    renderer_caps = ["render.rgb", "render.depth.expected_z", "gradient.screen.signed"]
    if absgrad:
        renderer_caps.append("gradient.screen.absolute")
    if antialiased:
        renderer_caps.append("render.antialiased_2d")
    gradient = "gradient.screen.absolute" if absgrad else "gradient.screen.signed"
    components = (
        ComponentSpec(
            "gsplat",
            "renderer",
            "splatstudio.backends.gsplat:create_renderer",
            ("state.gaussian3d", "camera.pinhole"),
            tuple(renderer_caps),
            {"absgrad": absgrad, "antialiased": antialiased},
        ),
        ComponentSpec(
            "adaptive-density",
            "density",
            "splatstudio.backends.gsplat:create_density",
            (gradient,),
            (),
            {"absgrad": absgrad, "grow_grad2d": 0.0008 if absgrad else 0.0002},
        ),
        ComponentSpec(
            "photometric", "loss", "splatstudio.losses:create_photometric", ("render.rgb",), (), {}
        ),
    )
    probes = (
        ProbeSpec(
            "state-transitions",
            "splatstudio.probes:state_transitions",
            "Duplicate, split, prune, optimizer alignment, and checkpoint round trip",
        ),
        ProbeSpec(
            "photometric-gradient",
            "splatstudio.probes:photometric_gradient",
            "Photometric loss has a finite nonzero gradient",
        ),
        ProbeSpec(
            "renderer-gradient",
            "splatstudio.probes:renderer_gradient",
            "Configured renderer executes and supplies declared gradients",
            True,
        ),
    )
    contributions = (
        ContributionSpec(
            "state-lifecycle",
            "Gaussian fields and optimizer state stay aligned",
            ("state-transitions",),
        ),
        ContributionSpec(
            "photometric-objective",
            "Photometric objective reaches rendered RGB",
            ("photometric-gradient",),
        ),
        ContributionSpec(
            "rasterizer-statistics",
            "Renderer supplies density-controller statistics",
            ("renderer-gradient",),
        ),
    )
    return MethodSpec(
        name,
        "Experimental gsplat configuration; paper fidelity unverified.",
        components,
        probes,
        contributions,
        citations=("https://github.com/nerfstudio-project/gsplat",),
    )


def vanilla():
    return make_method("gsplat")


def absolute():
    return make_method("gsplat-absgrad", absgrad=True)


def antialiased():
    # gsplat's 2D compensation is not the full Mip-Splatting method (3D filtering is absent).
    return make_method("gsplat-aa", antialiased=True)


def mcmc():
    base = make_method("gsplat-mcmc")
    density = ComponentSpec(
        "mcmc-density",
        "density",
        "splatstudio.backends.gsplat:create_mcmc",
        (),
        (),
        {"cap_max": 1000000},
    )
    return replace(
        base,
        components=(base.components[0], density, base.components[2]),
        citations=("https://arxiv.org/abs/2404.09591",),
        description="gsplat MCMC density strategy; complete paper reproduction unverified.",
    )


def surfel():
    base = make_method("gsplat-surfel")
    renderer = ComponentSpec(
        "gsplat-2d",
        "renderer",
        "splatstudio.backends.gsplat:create_surfel_renderer",
        ("state.surfel2d", "camera.pinhole"),
        (
            "render.rgb",
            "render.depth.expected_z",
            "render.normal.world",
            "render.normal_from_depth.world",
            "render.distortion.l1",
            "gradient.surfel",
        ),
    )
    density = ComponentSpec(
        "surfel-density",
        "density",
        "splatstudio.backends.gsplat:create_density",
        ("gradient.surfel",),
        (),
        {"key_for_gradient": "gradient_2dgs"},
    )
    geometry = ComponentSpec(
        "surfel-geometry",
        "loss",
        "splatstudio.losses:create_surfel_geometry",
        ("render.normal.world", "render.normal_from_depth.world", "render.distortion.l1"),
        (),
        {
            "normal_weight": 0.05,
            "distortion_weight": 0.01,
            "normal_start": 7000,
            "distortion_start": 3000,
        },
    )
    return replace(
        base,
        representation="surfel2d",
        components=(renderer, density, base.components[2], geometry),
        description="True gsplat 2D rasterization with gsplat's L1 distortion; not the exact original 2DGS objective.",
        citations=("https://surfsplatting.github.io/",),
    )


BUILTINS = {
    "gsplat": "splatstudio.methods:vanilla",
    "gsplat-absgrad": "splatstudio.methods:absolute",
    "gsplat-aa": "splatstudio.methods:antialiased",
    "gsplat-mcmc": "splatstudio.methods:mcmc",
    "gsplat-surfel": "splatstudio.methods:surfel",
}
