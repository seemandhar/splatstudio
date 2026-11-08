"""3DGS-MCMC regulariser lambda_o mean|sigmoid(o)| + lambda_s mean|exp(s)| (official 3dgs-mcmc
train.py; same formula as reproductions/confidence-densify). Applied at every step."""

DEFAULTS = {"opacity_reg": 0.01, "scale_reg": 0.01}


def create_reg(spec):
    cfg = {**DEFAULTS, **spec.config}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown mcmc-reg options: {sorted(unknown)}")
    lo, ls = float(cfg["opacity_reg"]), float(cfg["scale_reg"])

    def loss(params, outputs, target):
        return lo * params["opacities"].sigmoid().abs().mean() + ls * params[
            "scales"].exp().abs().mean()

    return loss
