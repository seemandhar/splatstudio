"""Scaffold-GS volume regulariser (train.py:145):  0.01 * mean(prod(neural Gaussian scales)).

The runtime passes the training image only to losses, while REHGS (HGS Eq. 8) needs
SSIM(I_j, I^_j) inside the density's after_backward. This loss therefore also stores the
(detached) target in `outputs["hgs_target"]`, which the density reads from the same dict.
"""


def create_scaling_reg(spec):
    weight = float(spec.config.get("weight", 0.01))

    def loss(params, outputs, target):
        outputs["hgs_target"] = target.detach()
        scaling = outputs["info"]["hgs"]["scaling"]
        if scaling.numel() == 0:
            return params["hgs_scaling"].sum() * 0.0
        return weight * scaling.prod(dim=1).mean()

    return loss
