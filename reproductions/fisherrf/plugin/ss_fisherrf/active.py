"""Active view-selection training loop of FisherRF (official active_train.py) inside the native
runtime.

``ss-train`` draws a random frame from *all* training views each step. The controller turns
that loop into the official one:

* a training frame that is not in the selected set (or arrives after the effective budget) is
  a *skipped* step: the renderer does not rasterize and the loss is ``0 * dummy`` on a leaf that
  is not a parameter, so every parameter keeps ``grad = None`` and Adam leaves it untouched;
* each selected frame is one official iteration ``it`` (1..total_iterations). When
  ``it`` is in the schedule (``VNSeqMInplace.load_its``) the selector scores the remaining
  training views with the current model, the views are added, and the initial Gaussians and
  fresh optimizers are restored (official ``load_checkpoint(init.ckpt)``, ``base_iter = it-1``);
* xyz LR = lr0 * 0.01^(cur/30000) with cur = it - base_iter (official update_learning_rate);
  SH degree +1 when ``it % sh_up_every == 0`` (absolute it), back to 0 at each restore;
  no optimizer step at it == total_iterations (official ``if iteration < opt.iterations``).

Only the training cameras handed to the density by ``set_training_cameras`` form the candidate
pool; validation/test cameras never reach the controller.
"""

import json
from collections import defaultdict

import torch
from torch import nn

from . import fisher, selection

DEFAULTS = {
    "selector": "fisher",  # "fisher" (H_reg) or "random" (RandSelector)
    "total_views": 20,  # N
    "views_per_round": 1,  # M
    "num_init_views": 4,
    "interval_epochs": 100,
    "total_iterations": 30000,  # scripts/blender_seq1.sh --iterations 30000
    "sh_up_every": 5000,  # active_train.py default
    "sh_up_after": -1,
    "max_sh_degree": 3,
    "reg_lambda": 1e-6,
    "exclude_fields": ["quats"],  # --filter_out_grad rotation
    "position_lr_final_ratio": 0.01,  # 1.6e-6 / 1.6e-4
    "position_lr_max_steps": 30000,
    "seed": 0,
    "background": [1.0, 1.0, 1.0],  # --white_background (training background)
}


class ActiveController:
    def __init__(self, config):
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FisherRF active options: {sorted(unknown)}")
        c = {**DEFAULTS, **config}
        if c["selector"] not in ("fisher", "random"):
            raise ValueError("selector must be 'fisher' or 'random'")
        self.c = c
        self.load_its = selection.schedule(c["total_views"], c["views_per_round"],
                                           c["num_init_views"], c["interval_epochs"])
        if self.load_its and max(self.load_its) >= c["total_iterations"]:
            raise ValueError("selection schedule exceeds total_iterations")
        self.cameras = None
        self.optimizers = None
        self.density = None
        self.initial = None
        self.lr0 = None
        self.it = 0
        self.base_iter = 0
        self.sh_degree = 0
        self.selected = []
        self.history = []
        self.restores = 0

    # -- registration (called by the density component) ---------------------------------
    def register_cameras(self, cameras):
        if len(cameras) < self.c["total_views"]:
            raise ValueError("candidate pool smaller than the view budget")
        self.cameras = cameras
        self.viewmats = torch.stack([cam["viewmat"].float() for cam in cameras])
        centers = torch.linalg.inv(self.viewmats)[:, :3, 3]
        self.selected = selection.initial_views(centers, self.c["num_init_views"])
        self.history.append({"it": 0, "added": list(self.selected), "kind": "init"})

    def register_state(self, params, optimizers, density):
        self.optimizers = optimizers
        self.density = density
        self.initial = {k: v.detach().clone() for k, v in params.items()}
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        self.it = 0
        self.base_iter = 0
        self.sh_degree = 0

    @property
    def active(self):
        return self.cameras is not None and self.initial is not None

    @property
    def done(self):
        return self.it >= self.c["total_iterations"]

    def frame_index(self, camera):
        diff = (self.viewmats - camera["viewmat"].float().to(self.viewmats.device)).abs()
        err, idx = diff.amax(dim=(1, 2)).min(dim=0)
        return int(idx) if float(err) < 1e-5 else None

    # -- per-step decision (called by the renderer before rasterizing) --------------------
    def begin(self, params, camera):
        """Return ``"train"``, ``"skip"``, or ``"eval"`` for this runtime step."""
        if not self.active:
            return "eval"
        idx = self.frame_index(camera)
        if idx is None:
            if not self.done:
                raise RuntimeError(
                    f"FisherRF active schedule incomplete: {self.it}/"
                    f"{self.c['total_iterations']} effective iterations; raise --steps")
            return "eval"
        if self.done or idx not in self.selected:
            return "skip"
        it = self.it + 1
        n_add = self.load_its.get(it, 0)
        if n_add:
            self._acquire(params, n_add, it)
            self._restore(params)
            self.base_iter = it - 1
        if it > self.c["sh_up_after"] and it % self.c["sh_up_every"] == 0:
            self.sh_degree = min(self.sh_degree + 1, self.c["max_sh_degree"])
        self.it = it
        if it >= self.c["total_iterations"]:
            return "skip"  # official: no optimizer step at the last iteration
        t = min(self.cur_iter, self.c["position_lr_max_steps"]) / self.c["position_lr_max_steps"]
        self.optimizers["means"].param_groups[0]["lr"] = (
            self.lr0 * self.c["position_lr_final_ratio"] ** t)
        return "train"

    @property
    def cur_iter(self):
        return self.it - self.base_iter

    # -- acquisition -----------------------------------------------------------------------
    def candidates(self):
        chosen = set(self.selected)
        return [i for i in range(len(self.cameras)) if i not in chosen]

    def _acquire(self, params, n_add, it):
        pool = self.candidates()
        if self.c["selector"] == "random":
            picked = selection.random_select(pool, n_add, self.c["seed"])
            scores = None
        else:
            picked, scores = self.fisher_select(params, pool, n_add)
        self.selected.extend(picked)
        row = {"it": it, "added": picked, "kind": self.c["selector"],
               "n_selected": len(self.selected), "gaussians": len(params["means"])}
        if scores is not None:
            row["best_score"] = scores
        self.history.append(row)
        print(json.dumps({"fisherrf_selection": row}), flush=True)

    @torch.no_grad()
    def fisher_select(self, params, pool, n_add):
        device = params["means"].device
        bg = torch.tensor(self.c["background"], dtype=torch.float32, device=device)
        exclude = tuple(self.c["exclude_fields"])

        def hess(i):
            cam = {k: (v.to(device) if torch.is_tensor(v) else v)
                   for k, v in self.cameras[i].items()}
            return fisher.flat(fisher.diag_hessian(params, cam, self.sh_degree, bg, exclude))

        h_train = None
        for i in self.selected:
            h = hess(i)
            h_train = h if h_train is None else h_train + h
        lam = self.c["reg_lambda"]
        if n_add == 1:
            # H_reg.select_single_view: stream the candidates, keep only the scores
            inv = torch.reciprocal(h_train + lam)
            scores = torch.stack([(hess(i) * inv).sum() for i in pool])
            best = int(torch.sort(scores, descending=True).indices[0])
            return [pool[best]], float(scores[best])
        # batch: greedy Alg. 1 over stored candidate Hessians (official keeps them on CPU)
        h_cand = [hess(i).cpu() for i in pool]
        picked = selection.greedy_batch(h_train.cpu(), h_cand, pool, n_add, lam)
        return picked, None

    def _restore(self, params):
        """load_checkpoint(init.ckpt): initial Gaussians, fresh Adam state, fresh ADC stats."""
        for name, value in self.initial.items():
            new = nn.Parameter(value.clone())
            opt = self.optimizers[name]
            opt.param_groups[0]["params"] = [new]
            opt.state = defaultdict(dict)
            params[name] = new
        self.optimizers["means"].param_groups[0]["lr"] = self.lr0
        self.sh_degree = 0
        self.restores += 1
        self.density.reset(params, self.optimizers)

    def summary(self):
        return {"selected": list(self.selected), "history": json.dumps(self.history),
                "effective_iterations": self.it, "restores": self.restores}
