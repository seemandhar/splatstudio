"""3DGS-MCMC regularisers used unchanged by AAA-Gaussians (official train.py:107-108):
opacity_reg * |sigmoid(o)|.mean() + scale_reg * |exp(s)|.mean() (unfiltered scales)."""

DEFAULTS = {"opacity_reg": 0.01, "scale_reg": 0.01}


def create_reg(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown aaa-mcmc-reg options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    w_o, w_s = float(cfg["opacity_reg"]), float(cfg["scale_reg"])

    def loss(params, outputs, target):
        return (w_o * params["opacities"].sigmoid().abs().mean()
                + w_s * params["scales"].exp().abs().mean())

    return loss
