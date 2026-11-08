"""Diagnostic density: 3DGS density + DNGaussian-style separate depth pass (own backward and
optimizer step) every `every` steps, after the photometric step."""

import random

from splatstudio.backends.gsplat import Density
from ss_det_gs import losses, priors
from ss_det_gs.renderer import hard_depth


class SepStep(Density):
    def __init__(self, spec):
        from dataclasses import replace

        super().__init__(replace(spec, config=spec.config["base"]))
        self.cfg = {**losses.DEPTH_DEFAULTS, **spec.config["depth"]}
        self.rng = random.Random(int(self.cfg["seed"]))

    def after_backward(self, params, optimizers, step, outputs):
        super().after_backward(params, optimizers, step, outputs)
        c = self.cfg
        if step % int(c["every"]) != 0:
            return
        cam = outputs["camera"]
        disp, _ = priors.table().lookup(cam["viewmat"], params["means"].device)
        d, t = losses.to_space(hard_depth(params, cam, c["omega"]), disp[None, None], c["space"])
        lo, hi = int(c["patch_min"]), int(c["patch_max"])
        val = losses.hierarchical_depth_loss(d, t, self.rng.randint(lo, hi), self.rng.randint(lo, hi),
                                             float(c["gamma"]), float(c["eta"]), float(c["error_tolerance"]))
        val.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)


def create_density(spec):
    return SepStep(spec)
