"""Official 3dgs-mcmc regularisers (train.py):
opacity_reg * mean|sigmoid(o)| + scale_reg * mean|exp(s)| over all Gaussians (and axes)."""

from .scenes import OPACITY_REG, scene_name


def resolve_weights(config):
    wo = config.get("opacity_weight")
    if wo is None:
        wo = OPACITY_REG.get(scene_name(), 0.01)
    return float(wo), float(config.get("scale_weight", 0.01))


def create_regularizers(spec):
    unknown = set(spec.config) - {"opacity_weight", "scale_weight"}
    if unknown:
        raise ValueError(f"Unknown mcmc-regularizers options: {sorted(unknown)}")
    wo, ws = resolve_weights(spec.config)

    def loss(params, outputs, target):
        return wo * params["opacities"].sigmoid().mean() + ws * params["scales"].exp().mean()

    return loss
