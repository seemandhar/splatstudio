"""gsplat renderer that decodes EAGLES latents while training and renders at the progressive scale.

While a training `DecoderBank` is active and not yet baked, rotation, opacity and SH rest are
`D(STE(q))` of the per-Gaussian latents (paper Eq. 1) and, for gradient-enabled (training)
renders, the camera is downscaled to floor(s H) x floor(s W) with K scaled by s (paper §3.2;
official `render(..., image_shape=gt_image.shape)` after `downsample_image`). The scale s is
returned as `outputs["eagles_scale"]` so the photometric loss downsamples the target with the
same `F.interpolate` factor. Without an active bank (any fresh process, e.g. `ss-eval`) or after
baking, the base Gaussian fields are rendered as-is: they then hold the decoded model.
"""

import math
from dataclasses import replace

import torch
from splatstudio.backends.gsplat import Renderer

from . import decoders


def scaled_camera(camera, s):
    """Camera for an image downsampled by F.interpolate(scale_factor=s): size floor(s*H/W),
    pixel (x+0.5) <-> original (x+0.5)/s, i.e. K[:2] *= s."""
    if s >= 1.0:
        return camera
    cam = dict(camera)
    K = camera["K"].clone()
    K[:2] = K[:2] * s
    cam["K"] = K
    cam["width"] = int(math.floor(float(camera["width"]) * s))
    cam["height"] = int(math.floor(float(camera["height"]) * s))
    return cam


class EaglesRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(replace(spec, config=dict(spec.config)))
        self.bank = None  # explicit bank (probes); otherwise decoders.ACTIVE

    def active_bank(self, params):
        bank = self.bank if self.bank is not None else decoders.ACTIVE["bank"]
        if bank is None or bank.baked or not all(f in params for f in bank.fields):
            return None
        return bank

    def render(self, params, camera, sh_degree=3):
        bank = self.active_bank(params)
        if bank is None:
            return super().render(params, camera, sh_degree)
        decoded = dict(params.items())
        decoded.update(bank.decode_all(params))
        s = float(bank.scale) if torch.is_grad_enabled() else 1.0
        out = super().render(decoded, scaled_camera(camera, s), sh_degree)
        out["eagles_scale"] = s
        return out


def create_renderer(spec):
    return EaglesRenderer(spec)
