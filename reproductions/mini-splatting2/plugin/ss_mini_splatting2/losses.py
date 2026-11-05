"""Photometric loss (L1 + 0.2 D-SSIM, official msv2/train.py) against the renderer's
resolution-matched target: during the 0.5x warm-up the renderer supplies
`outputs["loss_target"]` (the official half-resolution camera image); otherwise the frame target.
"""

from splatstudio.losses import create_photometric as _base


def create_photometric(spec):
    base = _base(spec)

    def loss(params, outputs, target):
        override = outputs.get("loss_target")
        return base(params, outputs, target if override is None else override)

    return loss
