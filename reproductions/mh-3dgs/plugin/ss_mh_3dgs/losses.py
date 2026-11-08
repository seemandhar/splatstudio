"""3DGS-MCMC regularisers used by MH-3DGS (Eq. likelihood; train.py):
lambda_opacity * mean(sigmoid(o)) + lambda_scale * mean(exp(s)) over all Gaussians/axes."""


def create_regularizers(spec):
    unknown = set(spec.config) - {"opacity_weight", "scale_weight"}
    if unknown:
        raise ValueError(f"Unknown mcmc-regularizers options: {sorted(unknown)}")
    wo = float(spec.config.get("opacity_weight", 0.01))
    ws = float(spec.config.get("scale_weight", 0.01))

    def loss(params, outputs, target):
        return wo * params["opacities"].sigmoid().mean() + ws * params["scales"].exp().mean()

    return loss
