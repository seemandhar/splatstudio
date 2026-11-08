"""Depth distillation from the NeRF (paper Sec. 3.3, Eq. 7; Supp. Eq. 13).

    L = L_GS + lambda(i) L_Depth,   L_Depth = E_r |D_NeRF(r) - D_GS(r)|            (Eq. 7)
    lambda(i) = lambda_init * d^(i / S),  lambda_init 0.9, d 0.9, S 100             (Eq. 13)

The expectation is the mean over all pixels of the training view. L_GS (L1 + 0.2 D-SSIM) is the
runtime's unchanged photometric loss.
"""

import torch

DEFAULTS = {"lambda_init": 0.9, "decay": 0.9, "decay_steps": 100, "units": "world"}
# units: "world" = COLMAP units (iteration 0); "nerf" = nerfacc's normalised scene units
# (similarity_from_cameras scale, ~0.23-0.25 on Mip-NeRF 360): post-hoc iteration-1 variant.


def depth_weight(step, lambda_init=0.9, decay=0.9, decay_steps=100):
    return lambda_init * decay ** (step / decay_steps)  # Eq. 13 (continuous exponent)


class DepthDistillation:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown depth-distillation options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["units"] not in ("world", "nerf"):
            raise ValueError("units must be 'world' or 'nerf'")

    def __call__(self, params, outputs, target):
        if "nerf_depth" not in outputs:
            raise KeyError("depth distillation needs outputs['nerf_depth'] (nerfinit sidecar); "
                           "train on a dataset copy built by ss-nerfinit-prep")
        cfg = {k: self.cfg[k] for k in ("lambda_init", "decay", "decay_steps")}
        lam = depth_weight(outputs["step"], **cfg)
        l_depth = (outputs["nerf_depth"] - outputs["depth_accum"]).abs().mean()  # Eq. 7
        if self.cfg["units"] == "nerf":
            if not outputs.get("nerf_scale"):
                raise KeyError("units='nerf' needs the sidecar normalisation scale")
            l_depth = l_depth * outputs["nerf_scale"]
        return torch.as_tensor(lam, dtype=l_depth.dtype, device=l_depth.device) * l_depth


def create_depth_loss(spec):
    return DepthDistillation(spec)
