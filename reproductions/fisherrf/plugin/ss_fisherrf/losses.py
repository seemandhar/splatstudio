"""Official 3DGS objective (L1 + 0.2 D-SSIM) that is inert on skipped active-loop steps."""

from splatstudio.losses import create_photometric as _base


def create_photometric(spec):
    base = _base(spec)

    def loss(params, outputs, target):
        if outputs.get("fisherrf_skip"):
            # zero loss on a non-parameter leaf: every parameter keeps grad=None (no Adam step)
            return outputs["fisherrf_dummy"] * 0.0
        return base(params, outputs, target)

    return loss
