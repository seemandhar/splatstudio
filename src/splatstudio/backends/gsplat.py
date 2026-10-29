"""Native gsplat backend. Custom packages may replace either factory completely."""

import torch
from gsplat import DefaultStrategy, MCMCStrategy, rasterization, rasterization_2dgs


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - {"absgrad", "antialiased"}
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        self.absgrad = bool(spec.config.get("absgrad", False))
        self.antialiased = bool(spec.config.get("antialiased", False))
        caps = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}
        if self.absgrad:
            caps.add("gradient.screen.absolute")
        if self.antialiased:
            caps.add("render.antialiased_2d")
        self.capabilities = caps

    def render(self, params, camera, sh_degree=3):
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colors,
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            absgrad=self.absgrad,
            rasterize_mode="antialiased" if self.antialiased else "classic",
        )
        # The dataset convention is RGB over a white background.
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
        }


class Density:
    def __init__(self, spec):
        self.strategy = DefaultStrategy(**spec.config)
        self.requires = {
            "gradient.surfel"
            if self.strategy.key_for_gradient == "gradient_2dgs"
            else "gradient.screen.absolute"
            if self.strategy.absgrad
            else "gradient.screen.signed"
        }
        self.state = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)

    def before_backward(self, params, optimizers, step, outputs):
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        self.strategy.step_post_backward(
            params, optimizers, self.state, step, outputs["info"], packed=False
        )


def create_renderer(spec):
    return Renderer(spec)


def create_density(spec):
    return Density(spec)


class MCMCDensity:
    requires = set()

    def __init__(self, spec):
        self.strategy = MCMCStrategy(**spec.config)
        self.state = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state()

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        lr = optimizers["means"].param_groups[0]["lr"]
        self.strategy.step_post_backward(
            params, optimizers, self.state, step, outputs["info"], lr=lr
        )


class SurfelRenderer:
    capabilities = {
        "render.rgb",
        "render.depth.expected_z",
        "render.normal.world",
        "render.normal_from_depth.world",
        "render.distortion.l1",
        "gradient.surfel",
    }

    def __init__(self, spec):
        if spec.config:
            raise ValueError("SurfelRenderer currently accepts no additional configuration")

    def render(self, params, camera, sh_degree=3):
        render, alpha, normals, surface_normals, distortion, median, info = rasterization_2dgs(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["sh0"], params["shN"]), dim=1),
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            distloss=True,
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "normal": normals[0],
            "surface_normal": surface_normals[0],
            "distortion": distortion[0],
            "median_depth": median[0],
            "info": info,
        }


def create_mcmc(spec):
    return MCMCDensity(spec)


def create_surfel_renderer(spec):
    return SurfelRenderer(spec)
