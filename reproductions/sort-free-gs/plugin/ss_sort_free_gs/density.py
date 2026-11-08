"""3DGS densification for WSR (paper App. "Densification and pruning for view dependent
opacity": densification unchanged, opacity-threshold pruning removed), plus the plugin-side
handling of the two global learnable scalars (sigma, w_B) and their learning rates.

Global scalars: the native runtime only has per-Gaussian fields and checkpoints `params` only,
so sigma and w_B are stored as per-Gaussian rows that are kept identical. The renderer reads the
value with a straight-through term that gives every row the full scalar gradient g (independent
of N), so identical rows with identical Adam moments take exactly the scalar Adam step. After
every step this controller copies the value and Adam moments of an existing row (one with
non-zero second moment; all such rows are identical) to all rows, so rows added by clone/split
(zero moments) do not drift. Probe `recipe` checks equality with a scalar Adam across N changes.
Every `log_every` steps sigma, w_B, the fraction of Gaussians with u-DC <= 0 (culled) and the
fraction of pixels with S_w + w_B < 1e-3 are printed as a JSON line (review-1 m6).
"""

import json
from dataclasses import replace

import torch
from splatstudio.backends.gsplat import Density

from .renderer import GLOBAL_FIELDS
from .wsr import expon_lr

OPTIONS = {"vi_lr": 0.01, "sigma_lr_init": 0.1, "sigma_lr_final": 0.001,
           "sigma_lr_max_steps": 30000, "wbg_lr": 0.01, "log_every": 1000}


class WSRDensity:
    def __init__(self, spec):
        cfg = dict(spec.config)
        self.opts = {k: float(cfg.pop(k, v)) for k, v in OPTIONS.items()}
        self.base = Density(replace(spec, config=cfg))
        self.requires = self.base.requires
        self.scene_scale = 1.0

    @property
    def state(self):
        return self.base.state

    @property
    def strategy(self):
        return self.base.strategy

    def sigma_lr(self, step):
        o = self.opts
        return expon_lr(step, o["sigma_lr_init"] * self.scene_scale,
                        o["sigma_lr_final"] * self.scene_scale, o["sigma_lr_max_steps"])

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.scene_scale = float(scene_scale)
        optimizers["opacities"].param_groups[0]["lr"] = self.opts["vi_lr"]
        optimizers["wsr_wbg"].param_groups[0]["lr"] = self.opts["wbg_lr"]
        optimizers["wsr_sigma"].param_groups[0]["lr"] = self.sigma_lr(0)

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        n = len(params["means"])
        self.base.after_backward(params, optimizers, step, outputs)
        sync_globals(params, optimizers)
        # lr used by the next optimizer step (3DGS updates the lr at the start of iteration+1)
        optimizers["wsr_sigma"].param_groups[0]["lr"] = self.sigma_lr(step)
        if int(self.opts["log_every"]) > 0 and step % int(self.opts["log_every"]) == 0:
            self.log(params, step, outputs)
        if len(params["means"]) != n:
            for name, p in params.items():  # requires_grad pitfall guard (TODO-runtime.md)
                if not (p.is_leaf and p.requires_grad):
                    raise RuntimeError(f"{name} is no longer a trainable leaf after densify")


    @torch.no_grad()
    def log(self, params, step, outputs):
        from .renderer import global_value

        sigma = global_value(params, "wsr_sigma").item()
        w_bg = global_value(params, "wsr_wbg").item()
        u_dc = 0.1 + 0.28209479177387814 * params["opa_sh0"][:, 0]
        row = {"wsr_step": step, "sigma": sigma, "w_bg": w_bg,
               "frac_u_dc_nonpositive": (u_dc <= 0).float().mean().item(),
               "gaussians": len(params["means"])}
        wsr = outputs.get("info", {}).get("wsr")
        if wsr is not None:
            row["frac_pix_den_lt_1e-3"] = ((wsr["w_sum"] + w_bg) < 1e-3).float().mean().item()
        print(json.dumps(row), flush=True)


@torch.no_grad()
def sync_globals(params, optimizers):
    for name in GLOBAL_FIELDS:
        p = params[name]
        st = optimizers[name].state.get(p, {})
        src = 0
        if "exp_avg_sq" in st:
            src = int(st["exp_avg_sq"].argmax())  # rows from clone/split have zero moments
            for key in ("exp_avg", "exp_avg_sq"):
                st[key].copy_(st[key][src].expand_as(st[key]))
        p.copy_(p[src].expand_as(p))


def create_density(spec):
    return WSRDensity(spec)
