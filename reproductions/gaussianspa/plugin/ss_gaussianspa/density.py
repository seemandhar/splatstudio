"""GaussianSpa density control: the Mini-Splatting pipeline before 15k (copied, `ms_density.py`)
followed by the "optimizing-sparsifying" phase and the hard prune of the official
`train_opacity.py` / `train_imp_score.py` (https://github.com/noodle-lab/GaussianSpa).

Per training step s (the runtime calls `after_backward` after the optimizer step):
  s < densify_until:  Mini-Splatting densification + depth reinit (unchanged).
  s == restart_step:  importance over all train views, keep int(N(1-r1)) sampled with
                      p ∝ imp + 1, reinit from the centres (z, u reset to 0).
  s == spa_start:     z = prox(a), u = 0                     (OptimizingSpa init + update(False))
  spa_start < s <= spa_stop, s % spa_interval == 0:
                      z = prox(a + u); u += a - z            (Eq. 16-17)
  s == spa_stop:      hard prune to int((1 - r2) N) Gaussians (survivors keep Adam state).
  every step:         means LR and active SH degree of the official schedule for s + 1.
The ADMM penalty itself (Eq. 11) is the `spa-penalty` loss, which reads z/u from the fields.
"""

import torch
from gsplat.strategy.ops import remove

from . import core, spa
from .ms_density import DEFAULTS as MS_DEFAULTS
from .ms_density import MiniSplattingDensity

FIELDS = ("spa_z", "spa_u")

DEFAULTS = {
    **{k: v for k, v in MS_DEFAULTS.items()
       if k not in ("simplify", "simp_iteration2", "sampling_factor", "prune_cdf")},
    # official scripts pass --imp_metric indoor for every scene
    "imp_metric": "indoor",
    # GaussianSpa (arguments/__init__.py OptimizationParams)
    "criterion": "opacity",  # "opacity" (train_opacity.py) | "importance" (train_imp_score.py)
    "prune_ratio1": 0.5,
    "prune_ratio2": 0.8,
    "spa_start": 15200,
    "spa_stop": 25200,
    "spa_interval": 50,
    "log_every_updates": 20,
}


class GaussianSpaDensity(MiniSplattingDensity):
    defaults = DEFAULTS

    def __init__(self, spec):
        super().__init__(spec)
        cfg = self.cfg
        if cfg["criterion"] not in ("opacity", "importance"):
            raise ValueError("criterion must be 'opacity' or 'importance'")
        if not (cfg["densify_until"] <= cfg["restart_step"] < cfg["spa_start"] < cfg["spa_stop"]):
            raise ValueError("Need densify_until <= restart_step < spa_start < spa_stop")
        for key in ("prune_ratio1", "prune_ratio2"):
            if not 0 < cfg[key] < 1:
                raise ValueError(f"{key} must be in (0, 1)")
        self._updates = 0

    def initialize(self, params, optimizers, scene_scale=1.0):
        missing = [f for f in FIELDS if f not in params]
        if missing:
            raise ValueError(f"GaussianSpa needs the ADMM fields {missing}")
        super().initialize(params, optimizers, scene_scale)
        self.state["admm"] = []

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        self._cache_target(outputs, step)
        if step < cfg["densify_until"]:
            self._accumulate(params, outputs["info"])
            n = len(params["means"])
            if (
                step > cfg["densify_from"]
                and step % cfg["densify_every"] == 0
                and step % cfg["reinit_every"] != 0
                and (cfg["num_max"] is None or n < cfg["num_max"])
            ):
                self._densify(params, optimizers, step)
            if step % cfg["reinit_every"] == 0:
                self._depth_reinit(params, optimizers, step)
        if step == cfg["restart_step"]:
            self._simplify_sample(params, optimizers, step)
        if step == cfg["spa_start"]:
            self._sparsify(params, step, update_u=False)
        elif spa.update_active(step, cfg["spa_start"], cfg["spa_stop"], cfg["spa_interval"]):
            self._sparsify(params, step, update_u=True)
        if step == cfg["spa_stop"]:
            self._hard_prune(params, optimizers, step)
        self._schedule(params, optimizers, step + 1)

    # ------------------------------------------------------------------ fields
    def _reinit_extra(self, n, device):
        return {f: torch.zeros(n, device=device) for f in FIELDS}

    # ------------------------------------------------------------------ 15k sampling (C5)
    def _simplify_sample(self, params, optimizers, step):
        """official: prob = (imp+1)/sum, num = int(N (1 - prune_ratio1)), choice without
        replacement, prune, reinitial_pts(centres, SH DC colour), training_setup."""
        if self.cameras is None:
            raise RuntimeError("Simplification needs set_training_cameras()")
        n0 = len(params["means"])
        imp = self._importance(params)
        keep_n = spa.sampling_count(n0, self.cfg["prune_ratio1"])
        idx = core.weighted_sample(spa.sampling_weights(imp), keep_n, replace=False)
        keep = torch.zeros(n0, dtype=torch.bool, device=imp.device)
        keep[idx] = True
        self._reinit(params, optimizers, params["means"].detach()[keep].clone(),
                     params["sh0"].detach()[keep].clone())
        self._log(step, "simplify_sample", n_before=n0, intersected=int((imp > 0).sum()),
                  zero_importance_kept=int((imp[keep] == 0).sum()), n_after=len(params["means"]))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ C2/C3 ADMM
    def _sparsify(self, params, step, update_u):
        a = params["opacities"].detach().sigmoid()
        u = params["spa_u"].detach() if update_u else torch.zeros_like(a)
        imp = self._importance(params) if self.cfg["criterion"] == "importance" else None
        z, u_new = spa.admm_step(a, u, self.cfg["prune_ratio2"], imp)
        params["spa_z"].data.copy_(z)
        if update_u:
            params["spa_u"].data.copy_(u_new)
        else:
            params["spa_u"].data.zero_()
        self._updates += 1
        if not update_u or self._updates % self.cfg["log_every_updates"] == 0:
            row = {"step": int(step), "residual": float(((a - z) ** 2).sum()),
                   "z_nonzero": int((z != 0).sum()), "n": int(a.numel()),
                   "u_abs_mean": float(params["spa_u"].abs().mean()),
                   "a_below_0.005": int((a < 0.005).sum())}
            self.state["admm"].append(row)
            print({"gaussianspa_admm": row}, flush=True)

    # ------------------------------------------------------------------ C4 hard prune
    def _hard_prune(self, params, optimizers, step):
        n0 = len(params["means"])
        ratio = self.cfg["prune_ratio2"]
        if self.cfg["criterion"] == "opacity":
            mask = spa.hard_prune_mask(ratio, opacity=params["opacities"].detach().sigmoid())
        else:
            mask = spa.hard_prune_mask(ratio, importance=self._importance(params))
        remove(params, optimizers, {}, mask)  # official prune_points keeps survivors' Adam state
        self._log(step, "hard_prune", n_before=n0, n_after=len(params["means"]))
        torch.cuda.empty_cache()


def create_density(spec):
    return GaussianSpaDensity(spec)
