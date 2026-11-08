"""The built-in 3DGS photometric loss (L1 + 0.2 D-SSIM), unchanged, which also hands the
detached target image to the density component (needed for the proxy curvature omega,
App. A.5; the runtime passes targets only to losses)."""

from splatstudio.losses import create_photometric as _base

TARGET_CAPABILITY = "surrogate.target"


def create_photometric(spec):
    base = _base(spec)

    def loss(params, outputs, target):
        outputs["target"] = target.detach()
        return base(params, outputs, target)

    return loss
