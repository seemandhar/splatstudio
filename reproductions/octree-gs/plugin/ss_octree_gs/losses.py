"""Octree-GS loss terms besides the runtime photometric loss (official train.py):

  L = (1 - 0.2) L1 + 0.2 D-SSIM                      (runtime `photometric`, unchanged)
    + 0.01 * mean(prod(scale of neural Gaussians))   (Scaffold-GS volume regulariser, kept by Octree-GS)

Copied from the HAC / ContextGS native ports (identical in the official codes).
"""


def create_scaling_reg(spec):
    weight = float(spec.config.get("weight", 0.01))

    def loss(params, outputs, target):
        scaling = outputs["info"]["ogs"]["scaling"]
        if scaling.numel() == 0:
            return params["ogs_scaling"].sum() * 0.0
        return weight * scaling.prod(dim=1).mean()

    return loss
