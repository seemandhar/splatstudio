"""FCGS density component: unchanged 3DGS densification + FCGS encode -> decode at the end.

FCGS (Chen et al., ICLR 2025) is an optimisation-free post-training codec: the paper trains a
vanilla 3DGS for 30 K iterations and compresses it with a pretrained network in one pass
(Fig. 3 caption). SplatStudio has no post-training hook, so this component wraps the baseline
gsplat Density (3DGS schedule, opacity reset) and, right after the final optimizer step
(`step == total_steps`), replaces every Gaussian parameter by its FCGS-decoded value (decoded
from real bitstream files). The checkpoint, validation and `ss-eval` therefore measure the
decoded model, and the size report is stored in the density state.
"""

import json
import math
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import codec

DEFAULTS = {
    "total_steps": 30000,  # paper: vanilla 3DGS 30K, then compress
    "lmd": 1e-4,  # "Ours-highrate"; 16e-4 = "Ours-lowrate"
    "encode": True,
    "xyz_coder": "auto",  # official GPCC if tmc3 is installed, else lossless octree substitute
    "weights_dir": None,  # default ~/models/fcgs or $FCGS_WEIGHTS
}


class FCGSDensity:
    def __init__(self, spec):
        config = dict(spec.config)
        base = config.pop("base", {})
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown fcgs-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **config}
        if self.cfg["lmd"] not in codec.LAMBDAS:
            raise ValueError(f"FCGS lambda must be one of {codec.LAMBDAS}")
        if self.cfg["encode"] and not codec.weights_path(self.cfg["lmd"],
                                                         self.cfg["weights_dir"]).exists():
            raise FileNotFoundError(codec.weights_path(self.cfg["lmd"], self.cfg["weights_dir"]))
        self.base = Density(replace(spec, config=dict(base)))
        self.requires = self.base.requires
        self.encoded = None
        self.lr0 = None

    @property
    def state(self):
        return {"strategy": self.base.state, "fcgs": {"encoded": self.encoded}}

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def check_run_length(self, step, lr):
        """Raise if the run length differs from total_steps (the encode would never run).

        The runtime sets the means LR to lr0 * 0.01 ** (step / steps) before after_backward
        (runtime.py), so the step budget is recoverable at step 1.
        """
        if not self.cfg["encode"] or step != 1 or self.lr0 is None:
            return
        if not 0 < lr < self.lr0:
            print("fcgs-density: WARNING cannot infer the run length; the encode runs only at "
                  f"step {self.cfg['total_steps']}", flush=True)
            return
        steps = math.log(0.01) / math.log(lr / self.lr0)
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but fcgs-density total_steps="
                f"{self.cfg['total_steps']}: the final FCGS encode -> decode would never run.")

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        self.base.after_backward(params, optimizers, step, outputs)
        if self.cfg["encode"] and step == self.cfg["total_steps"]:
            start = time.perf_counter()
            report = self.apply_codec(params, optimizers)
            report["seconds"] = round(time.perf_counter() - start, 2)
            print(json.dumps({"fcgs": "encode", "step": step, **report}), flush=True)

    @torch.no_grad()
    def apply_codec(self, params, optimizers):
        decoded, report = codec.encode_decode(
            params, self.cfg["lmd"], weights_dir=self.cfg["weights_dir"],
            xyz_coder=self.cfg["xyz_coder"])

        # Decoding permutes the Gaussians (official sort + seeded shuffle per 1M step). Training
        # is over, so the Adam moments and densification statistics are reset with the swap.
        def param_fn(name, p):
            return torch.nn.Parameter(decoded[name].to(p.dtype).reshape(p.shape),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        state = self.base.state
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.dim() > 0 and len(v) == len(params["means"]):
                state[k] = torch.zeros_like(v)
        self.encoded = report
        return report


def create_density(spec):
    return FCGSDensity(spec)
