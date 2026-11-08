"""CRiM-GS regularisers (Sec. 4.3-4.4): lambda_o L_o + lambda_M L_M.

The photometric part (1 - lambda_c) L1 + lambda_c D-SSIM with lambda_c = 0.3 is SplatStudio's
``photometric`` loss with ssim_weight 0.3 on ``outputs["rgb"]`` (= I_out during training).

* L_o = ||R_a^T R_a - I||_2 (Sec. 4.3), the Frobenius norm per trajectory pose, averaged over
  the N poses; active whenever the trajectory is rendered (CoMo: ``ortho_loss`` from
  start_warp). CoMo's code uses mean |R R^T - I| instead (documented deviation).
* L_M = mean(M) (BAGS mask sparsity), active once the pixel weights / mask are on.
"""

import torch

DEFAULTS = {"lambda_o": 1e-4, "lambda_m": 1e-3}


def orthogonality(R_a):
    eye = torch.eye(3, device=R_a.device, dtype=R_a.dtype)
    return torch.linalg.matrix_norm(R_a.transpose(-1, -2) @ R_a - eye).mean()


def create_regularizers(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown crim-gs regulariser options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    lam_o, lam_m = float(cfg["lambda_o"]), float(cfg["lambda_m"])

    def loss(params, outputs, target):
        total = outputs["rgb"].sum() * 0.0  # keeps the term attached before the kernel starts
        if outputs.get("R_a") is not None:
            total = total + lam_o * orthogonality(outputs["R_a"])
        if outputs.get("mask") is not None:
            total = total + lam_m * outputs["mask"].mean()
        return total

    return loss

