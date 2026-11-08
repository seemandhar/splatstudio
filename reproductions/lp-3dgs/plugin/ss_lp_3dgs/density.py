"""3DGS adaptive density control + the LP-3DGS learn-to-prune schedule (paper §3.3, §4.1).

Native order per step s: render -> backward -> optimiser steps -> after_backward(s). The
official loop (train.py) maps onto it as follows (start = 19500, iters = 500, every = 20):

  after_backward(start)            m := mask_init (official set_mask, fresh optimiser)
  start < s < start + iters        AdamW decoupled decay m *= 1 - lr * wd (after the Adam step)
  after_backward(s), s+1 a score step (19520, ..., 19980)
                                   S := max over training views of alpha*T with the *current*
                                   opacity (masked, fresh noise per view, once live); phase train
  after_backward(start + iters-1)  phase frozen (render at 20000 masked, mask not stepped)
  after_backward(start + iters)    remove gs(m) < 0.5 (fresh noise, no score); phase off
"""

import json
from dataclasses import replace

import torch
from gsplat.strategy.ops import remove
from splatstudio.backends.gsplat import Density

from . import kernels
from .core import (
    FIELD,
    PHASE_ATTR,
    SCORE_ATTR,
    effective_opacity,
    in_mask_window,
    is_score_step,
    phase,
    prune_selection,
)

LP_DEFAULTS = {
    "mask_start": 19500,  # arguments prune_iterations
    "mask_iters": 500,  # arguments train_mask_iters
    "score_every": 20,  # train.py (iteration - prune_iterations[0]) % 20
    "mask_init": 1.0,  # mask.py torch.ones
    "weight_decay": 0.01,  # torch.optim.AdamW default (mask.py)
    "tau": 0.5,  # arguments gumbel_temp
}
REQUIRES = {"gradient.screen.signed"}


class LPDensity(Density):
    def __init__(self, spec):
        lp = {k: spec.config[k] for k in LP_DEFAULTS if k in spec.config}
        rest = {k: v for k, v in spec.config.items() if k not in LP_DEFAULTS}
        super().__init__(replace(spec, config=rest))
        cfg = {**LP_DEFAULTS, **lp}
        self.start = int(cfg["mask_start"])
        self.iters = int(cfg["mask_iters"])
        self.every = int(cfg["score_every"])
        self.mask_init = float(cfg["mask_init"])
        self.weight_decay = float(cfg["weight_decay"])
        self.tau = float(cfg["tau"])
        if self.iters < 2 or self.every < 1 or self.tau <= 0:
            raise ValueError("mask_iters >= 2, score_every >= 1 and tau > 0 are required")
        if self.start < self.strategy.refine_stop_iter:
            raise ValueError("the mask window must start after densification stops")
        self.cameras = None
        self.events = []  # schedule log, read by probes and printed

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    # ------------------------------------------------------------------ helpers
    def _event(self, **row):
        self.events.append(row)
        print(json.dumps({"lp3dgs": row}), flush=True)

    @torch.no_grad()
    def compute_score(self, params):
        """Eq. 3, official prune_list: S_i = max over training views of alpha_i T_i."""
        if not self.cameras:
            raise RuntimeError("LP-3DGS needs the training cameras (set_training_cameras)")
        score = torch.zeros(len(params["means"]), device=params["means"].device)
        for camera in self.cameras:
            # official: the render inside prune_list uses get_opacity, i.e. the masked opacity
            # with fresh Gumbel noise per view once the mask is live
            opacity = effective_opacity(params, self.tau)
            kernels.max_contribution_(kernels.project(params, camera, opacity), score)
        return score

    @staticmethod
    def _check_trainable(params):
        for name, p in params.items():
            if not (isinstance(p, torch.nn.Parameter) and p.is_leaf and p.requires_grad):
                raise RuntimeError(f"{name} is no longer a trainable leaf parameter")

    # ------------------------------------------------------------------ hooks
    def after_backward(self, params, optimizers, step, outputs):
        super().after_backward(params, optimizers, step, outputs)
        start, end = self.start, self.start + self.iters
        mask = params[FIELD]
        if step == start:
            with torch.no_grad():
                mask.fill_(self.mask_init)
            optimizers[FIELD].state.pop(mask, None)  # fresh optimiser (normally already empty)
            setattr(params, PHASE_ATTR, "off")
            self._event(step=step, event="mask_init", gaussians=len(mask))
        if in_mask_window(step, start, self.iters) and self.weight_decay > 0:
            lr = optimizers[FIELD].param_groups[0]["lr"]
            with torch.no_grad():
                mask.mul_(1.0 - lr * self.weight_decay)  # AdamW decoupled weight decay
        if is_score_step(step + 1, start, self.iters, self.every):
            setattr(params, SCORE_ATTR, self.compute_score(params))
            setattr(params, PHASE_ATTR, "train")
            self._event(step=step + 1, event="score")
        if step == end - 1 and phase(params) == "train":
            setattr(params, PHASE_ATTR, "frozen")
        if step == end:
            before = len(mask)
            selection = prune_selection(mask, self.tau)
            if bool(selection.all()):
                raise RuntimeError("LP-3DGS mask would remove every Gaussian")
            remove(params, optimizers, self.state, selection)
            setattr(params, PHASE_ATTR, "off")
            if hasattr(params, SCORE_ATTR):
                delattr(params, SCORE_ATTR)
            after = len(params["means"])
            self._check_trainable(params)
            self._event(step=step, event="prune", before=before, after=after,
                        pruning_ratio=1.0 - after / before)


def create_density(spec):
    return LPDensity(spec)
