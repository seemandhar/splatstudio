"""ConeGS pre-activation opacity penalty (Sec. 4.2, "Opacity-Regularized Pruning").

Paper: L_o^pre = ||o_pre||_1 scaled by lambda_o = 0.0002. Official train.py:
    opacity = gaussians._opacity if use_preactivation_opacity_penalty else abs(get_opacity)
    loss = loss + opacity_penalty * opacity.mean()
i.e. the signed mean of the opacity logits (no absolute value). The code is followed; `mode:
"l1"` gives the paper's literal |logit| mean, `mode: "post"` the MCMC-style sigmoid penalty
(paper ablation (n)).
"""

MODES = ("signed", "l1", "post")


def penalty(opacities, weight, mode="signed"):
    if mode == "signed":
        return weight * opacities.mean()
    if mode == "l1":
        return weight * opacities.abs().mean()
    if mode == "post":
        return weight * opacities.sigmoid().abs().mean()
    raise ValueError(f"mode must be one of {MODES}")


def create_opacity_penalty(spec):
    unknown = set(spec.config) - {"weight", "mode"}
    if unknown:
        raise ValueError(f"Unknown conegs-opacity options: {sorted(unknown)}")
    weight = float(spec.config.get("weight", 0.0002))
    mode = spec.config.get("mode", "signed")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    def loss(params, outputs, target):
        return penalty(params["opacities"], weight, mode)

    return loss
