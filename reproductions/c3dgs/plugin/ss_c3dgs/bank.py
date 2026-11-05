"""Compressed scene state during quantization-aware fine-tuning (C3) and its process-local slot.

The SplatStudio runtime has no parameter type for global learned tensors, so (like the eagles
plugin's DecoderBank) the c3dgs density owns a `CompressedBank` with the two codebooks, the
per-Gaussian codebook indices and log scale factors, the official FakeQuantize modules and an
Adam over the bank tensors, and registers it in `ACTIVE`; the c3dgs renderer renders from it while
it is active. Opacities and positions stay runtime fields. At the final step the decoded npz is
written into the base fields and the bank is marked `decoded`, after which renderers (and any fresh
process such as `ss-eval`) render the base fields.

Forward attributes follow the official indexed `render()` (`gaussian_renderer/__init__.py`):
means = half(xyz); opacity = FQ(sigmoid(o)); colour = cat(FQ(dc_cb), FQ(rest_cb))[colour index];
scale = exp(FQ(log eta)) * FQ(normalize(relu(scale_cb)))[shape index];
rotation = normalize(FQ(rot_cb))[shape index]. Gradients reaching a codebook entry are the sum
over the Gaussians that index it (index backward), as the paper's "accumulated" update.
"""

import torch
import torch.nn.functional as F

from .quant import half_ste

ACTIVE = {"bank": None}

# official training_setup learning rates (feature_lr, /20, scaling_lr x2, rotation_lr)
DEFAULT_LRS = {"cb_dc": 0.0025, "cb_rest": 0.0025 / 20.0, "cb_scale": 0.005,
               "log_eta": 0.005, "cb_rot": 0.001}


class CompressedBank:
    def __init__(self, cb_dc, cb_rest, color_idx, cb_scale, cb_rot, gauss_idx, log_eta,
                 quantizers, lrs=None):
        self.tensors = {
            "cb_dc": cb_dc.detach().clone().requires_grad_(True),
            "cb_rest": cb_rest.detach().clone().requires_grad_(True),
            "cb_scale": cb_scale.detach().clone().requires_grad_(True),
            "log_eta": log_eta.detach().clone().reshape(-1, 1).requires_grad_(True),
            "cb_rot": cb_rot.detach().clone().requires_grad_(True),
        }
        self.color_idx = color_idx.detach().long()
        self.gauss_idx = gauss_idx.detach().long()
        self.q = quantizers
        self.lrs = {**DEFAULT_LRS, **(lrs or {})}
        # official: one Adam with param groups, lr per group, eps 1e-15, fresh state
        self.optimizer = torch.optim.Adam(
            [{"params": [self.tensors[k]], "lr": self.lrs[k], "name": k} for k in DEFAULT_LRS],
            lr=0.0, eps=1e-15,
        )
        self.decoded = False
        self.steps = 0

    @property
    def n(self):
        return len(self.color_idx)

    def attributes(self, params):
        """Activated, fake-quantised per-Gaussian attributes (one observer update each)."""
        t = self.tensors
        means = half_ste(params["means"])
        opacities = self.q["opacity"](torch.sigmoid(params["opacities"][:, None]))[:, 0]
        colors = torch.cat((self.q["features_dc"](t["cb_dc"]),
                            self.q["features_rest"](t["cb_rest"])), dim=1)[self.color_idx]
        scale_n = self.q["scaling"](F.normalize(F.relu(t["cb_scale"])))
        eta = torch.exp(self.q["scaling_factor"](t["log_eta"]))
        scales = eta * scale_n[self.gauss_idx]
        quats = F.normalize(self.q["rotation"](t["cb_rot"]))[self.gauss_idx]
        return {"means": means, "opacities": opacities, "colors": colors, "scales": scales,
                "quats": quats}

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def summary(self):
        return {
            "color_codebook": int(len(self.tensors["cb_dc"])),
            "shape_codebook": int(len(self.tensors["cb_scale"])),
            "gaussians": int(self.n),
            "bank_steps": int(self.steps),
            "qparams": {k: [float(fq.scale.reshape(-1)[0]), int(fq.zero_point.reshape(-1)[0])]
                        for k, fq in self.q.items()},
        }
