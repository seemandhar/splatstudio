"""Scaffold-GS volume regulariser (official train.py:145):  0.01 * mean(prod(neural Gaussian scales)).
The paper's Eq. 8 text says lambda_vol = 0.001; the released code uses 0.01 (code wins)."""


def create_scaling_reg(spec):
    weight = float(spec.config.get("weight", 0.01))

    def loss(params, outputs, target):
        scaling = outputs["info"]["sgs"]["scaling"]
        if scaling.numel() == 0:
            return params["sgs_scaling"].sum() * 0.0
        return weight * scaling.prod(dim=1).mean()

    return loss
