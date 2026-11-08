"""CompGS opacity regularisation (§3): lambda_reg * sum_i sigmoid(o_i), 15000 <= it <= 20000."""

from .schedule import reg_active

DEFAULTS = {"lambda_reg": 1e-7, "start": 15000, "stop": 20000}


def create_opacity_reg(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown compgs-opacity-reg options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    lam = float(cfg["lambda_reg"])

    def loss(params, outputs, target):
        o = params["opacities"]
        if not reg_active(int(outputs["step"]), cfg["start"], cfg["stop"]):
            return o.sum() * 0.0
        return lam * o.sigmoid().sum()

    return loss
