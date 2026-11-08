"""CoR-GS density control for two radiance fields, co-pruning (C1) and pseudo-camera ownership.

Field 1 (Θ¹) is the runtime's `params` — the field `ss-eval` evaluates; the paper keeps one
field (gs0, `render.py` loads `point_cloud/`) for inference. Field 2 (Θ²) is component-owned
state: a dict of nn.Parameters cloned from Θ¹ at `initialize` (official train.py:72 builds gs1
from the same `scene.init_point_cloud`), with its own Adams (same LRs/eps), its own densification
statistics and the same density rules. The density publishes itself as outputs["corgs"] so the
loss components (`corgs-field2`, `corgs-coreg`) can render Θ².

Densification (official GaussianModel.densify_and_prune, called every 100 its for
500 < it < 10000 with size_threshold=None) — the FSGS fork without proximity unpooling:
1. clone: grad >= tau and max scale <= percent_dense * extent;
2. split (post-clone set): grad >= tau and max scale > percent_dense * extent, OR mean squared
   3-NN distance > dist_thres * extent and max scale > extent (gaussian_model.py:434-439);
   split parents pruned (it > prune_from_iter);
3. prune opacity < 0.005 (it > prune_from_iter).
Opacity reset (train.py:283): min(o, 0.05) at it > 500 with (it - 501) % 3000 == 0.
Co-pruning (train.py:292-309, paper Eq. 2-3): for it > 500 and it % 500 == 0, both
non-matching masks are computed first (nearest Gaussian of the other field farther than
tau = 5), then both fields are pruned. open3d evaluate_registration = KD-tree hybrid search
(radius tau, max_nn 1); here an exact chunked torch.cdist nearest neighbour.

Order per native step: the runtime has already stepped Θ¹'s Adam; after_backward steps Θ²'s Adam
(position LR = the LR Θ¹ used this step), then densifies both, resets both, co-prunes both.
"""

import json

import numpy as np
import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from .pseudo import SCENE_BOUNDS, depth_bounds, official_bounds, random_poses_llff

DEFAULTS = {
    "n_fields": 2,
    "grad_threshold": 0.0005,
    "min_opacity": 0.005,
    "densification_interval": 100,
    "densify_from_iter": 500,
    "densify_until_iter": 10000,
    "prune_from_iter": 500,
    "percent_dense": 0.01,
    "dist_thres": 10.0,
    "opacity_reset_value": 0.05,
    "opacity_reset_interval": 3000,
    "opacity_reset_after": 500,  # start_sample_pseudo (run_llff.sh)
    "coprune": True,
    "coprune_threshold": 5.0,
    "coprune_interval": 500,
    "coprune_after": 500,  # densify_from_iter
    "n_pseudo_poses": 10000,
    "pseudo_seed": 0,
    "pseudo_bounds": "official",
    "scene_bounds": SCENE_BOUNDS,
}


def knn(query_idx, points, k=3, chunk=1024):
    """Exact k-NN of points[query_idx] among all points excluding itself (FSGS distCUDA2).

    Returns (mean squared distance [M], neighbour indices [M, k]).
    """
    m = len(query_idx)
    dist = torch.empty(m, device=points.device)
    idx = torch.empty(m, k, dtype=torch.long, device=points.device)
    for s in range(0, m, chunk):
        q = query_idx[s : s + chunk]
        d2 = torch.cdist(points[q], points).square()
        d2[torch.arange(len(q), device=points.device), q] = float("inf")
        vals, nn = d2.topk(k, dim=1, largest=False)
        dist[s : s + chunk] = vals.mean(dim=1)
        idx[s : s + chunk] = nn
    return dist, idx


@torch.no_grad()
def nearest_distance(source, target, budget=2**27):
    """Euclidean distance from every source point to its nearest target point (exact)."""
    out = torch.empty(len(source), device=source.device, dtype=source.dtype)
    if len(target) == 0:
        return out.fill_(float("inf"))
    chunk = max(1, budget // max(len(target), 1))
    for s in range(0, len(source), chunk):
        out[s : s + chunk] = torch.cdist(source[s : s + chunk], target,
                                         compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values
    return out


@torch.no_grad()
def nonmatching_masks(means_a, means_b, tau):
    """Paper Eq. 3 for both directions: True = no counterpart within tau (to be pruned)."""
    return nearest_distance(means_a, means_b) > tau, nearest_distance(means_b, means_a) > tau


class Field:
    """One radiance field: params, per-field Adams, densification statistics."""

    def __init__(self, params, optimizers):
        self.params = params
        self.optimizers = optimizers
        self.state = {"grad2d": None, "count": None}
        self.info = None  # gsplat info of this step's training-view render

    def reset_stats(self):
        n, dev = len(self.params["means"]), self.params["means"].device
        self.state["grad2d"] = torch.zeros(n, device=dev)
        self.state["count"] = torch.zeros(n, device=dev)


def clone_field(params, optimizers):
    """Θ² = copy of Θ¹ with fresh Adams of identical hyper-parameters."""
    new_params, new_opts = {}, {}
    for name, p in params.items():
        q = torch.nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)
        new_params[name] = q
        if name in optimizers:
            src = optimizers[name]
            g = src.param_groups[0]
            new_opts[name] = torch.optim.Adam([q], lr=g["lr"], betas=g["betas"], eps=g["eps"],
                                              weight_decay=g["weight_decay"])
    return Field(new_params, new_opts)


class CoRGSDensity:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown CoR-GS density options: {sorted(unknown)}")
        self.config = {**DEFAULTS, **spec.config}
        if int(self.config["n_fields"]) not in (1, 2):
            raise ValueError("n_fields must be 1 or 2 (official code supports at most 2)")
        if int(self.config["densify_from_iter"]) < int(self.config["prune_from_iter"]):
            raise ValueError("densify_from_iter must be >= prune_from_iter")
        if bool(self.config["coprune"]) and int(self.config["n_fields"]) != 2:
            raise ValueError("co-pruning needs two fields")
        self.state = {"grad2d": None, "count": None}  # Θ¹ statistics (saved in the checkpoint)
        self.extent = 1.0
        self.field1 = None
        self.field2 = None
        self.pseudo_info = {}
        self._order = []
        self._rng = np.random.default_rng(int(self.config["pseudo_seed"]))
        self._means_lr = None
        self.field2_rendered = 0  # steps on which corgs-field2 rendered Θ²
        self.field2_missed = 0
        self.log = []

    @property
    def fields(self):
        return [f for f in (self.field1, self.field2) if f is not None]

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)
        self._init_points = params["means"].detach().cpu().numpy().astype(np.float64)
        self.field1 = Field(params, optimizers)
        self.field1.state = self.state
        self.field1.reset_stats()
        if int(self.config["n_fields"]) == 2:
            self.field2 = clone_field(params, optimizers)
            self.field2.reset_stats()

    # -- pseudo cameras -----------------------------------------------------------------
    def set_training_cameras(self, cameras):
        viewmats = [c["viewmat"].detach().cpu().numpy().astype(np.float64) for c in cameras]
        mode = self.config["pseudo_bounds"]
        if mode == "official":
            scene, bounds = official_bounds(self.config["scene_bounds"], viewmats)
        elif mode == "points":
            Ks = [c["K"].detach().cpu().numpy().astype(np.float64) for c in cameras]
            sizes = [(int(c["width"]), int(c["height"])) for c in cameras]
            scene, bounds = None, depth_bounds(self._init_points, viewmats, Ks, sizes)
        else:
            raise ValueError("pseudo_bounds must be 'official' or 'points'")
        poses, info = random_poses_llff(viewmats, bounds, int(self.config["n_pseudo_poses"]),
                                        self._rng)
        cam0 = cameras[0]
        w, h = int(cam0["width"]), int(cam0["height"])
        # Official PseudoCamera: FoV of the first training view, centred principal point.
        K = cam0["K"].detach().clone()
        K[0, 2], K[1, 2] = w / 2.0, h / 2.0
        self.pseudo_viewmats = torch.tensor(poses, dtype=torch.float32, device=cam0["K"].device)
        self.pseudo_K, self.pseudo_size = K, (w, h)
        self.pseudo_info = {"bounds": bounds.tolist(), "bounds_source": mode, "scene": scene,
                            **info}

    def next_pseudo_camera(self):
        """Official: pop a random element of a copy of the pose list, refill when empty."""
        if not hasattr(self, "pseudo_viewmats"):
            raise RuntimeError("CoR-GS pseudo views need set_training_cameras (native trainer)")
        if not self._order:
            self._order = list(self._rng.permutation(len(self.pseudo_viewmats)))
        i = int(self._order.pop())
        w, h = self.pseudo_size
        return {"viewmat": self.pseudo_viewmats[i], "K": self.pseudo_K, "width": w, "height": h}

    # -- statistics / densification -------------------------------------------------------
    @staticmethod
    def accumulate(field):
        # add_densification_stats: |dL/d(NDC xy)| on visible Gaussians (gsplat pixel scaling).
        info = field.info
        n = len(field.params["means"])
        if field.state["grad2d"] is None or len(field.state["grad2d"]) != n:
            field.reset_stats()
        if info is None or info["means2d"].grad is None:
            return
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = info["radii"] > 0
        if sel.dim() == 3:
            sel = sel.all(dim=-1)
        ids = torch.where(sel)[1]
        field.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        field.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def densify_and_prune(self, field, step):
        cfg, extent = self.config, self.extent
        params, optimizers, st = field.params, field.optimizers, field.state
        tau = float(cfg["grad_threshold"])
        pd = float(cfg["percent_dense"])
        grads = st["grad2d"] / st["count"]
        grads[grads.isnan()] = 0.0
        is_dup = (grads >= tau) & (params["scales"].exp().max(dim=1).values <= pd * extent)
        n_dup = int(is_dup.sum())
        if n_dup:
            duplicate(params, optimizers, st, is_dup)
        padded = torch.cat([grads, torch.zeros(n_dup, device=grads.device)])
        max_scale = params["scales"].exp().max(dim=1).values
        is_split = (padded >= tau) & (max_scale > pd * extent)
        far = torch.zeros_like(is_split)
        big = torch.where(max_scale > extent)[0]
        if len(big) and len(params["means"]) > 3:
            dist, _ = knn(big, params["means"].detach(), 3)
            far[big[dist > float(cfg["dist_thres"]) * extent]] = True
        is_split |= far
        n_split = int(is_split.sum())
        if n_split:
            split(params, optimizers, st, is_split)
        is_prune = params["opacities"].sigmoid() < float(cfg["min_opacity"])
        n_prune = int(is_prune.sum()) if step > int(cfg["prune_from_iter"]) else 0
        if n_prune:
            remove(params, optimizers, st, is_prune)
        field.reset_stats()
        return {"cloned": n_dup, "split": n_split, "far_split": int(far.sum()),
                "pruned": n_prune, "gaussians": len(params["means"])}

    @torch.no_grad()
    def coprune(self, step):
        """C1: both masks from the current positions, then prune both fields (Eq. 2-3)."""
        f1, f2 = self.field1, self.field2
        tau = float(self.config["coprune_threshold"])
        m1, m2 = nonmatching_masks(f1.params["means"].detach(), f2.params["means"].detach(), tau)
        counts = []
        for field, mask in ((f1, m1), (f2, m2)):
            n = int(mask.sum())
            if n and step > int(self.config["prune_from_iter"]):  # official prune_points gate
                remove(field.params, field.optimizers, field.state, mask)
            else:
                n = 0
            counts.append(n)
        return {"step": step, "coprune": counts,
                "gaussians": [len(f1.params["means"]), len(f2.params["means"])]}

    def coprune_due(self, step):
        cfg = self.config
        return (bool(cfg["coprune"]) and step > int(cfg["coprune_after"])
                and step % int(cfg["coprune_interval"]) == 0)

    def reset_due(self, step):
        after = int(self.config["opacity_reset_after"])
        return step > after and (step - after - 1) % int(self.config["opacity_reset_interval"]) == 0

    # -- runtime hooks ----------------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        if self.field1 is None or self.field1.params is not params:
            raise RuntimeError("CoR-GS density: runtime params changed identity; re-initialize")
        outputs["info"]["means2d"].retain_grad()
        self.field1.info = outputs["info"]
        if self.field2 is not None:
            self.field2.info = None
        self._means_lr = optimizers["means"].param_groups[0]["lr"]  # LR the runtime steps with
        outputs["corgs"] = self

    def step_field2(self):
        f2 = self.field2
        f2.optimizers["means"].param_groups[0]["lr"] = self._means_lr
        for opt in f2.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.config
        if self.field2 is not None:
            if self.field2.info is None:
                # Generic SDK probes drive the hooks without losses; once corgs-field2 has run,
                # a step without the field-2 render is a wiring error.
                if self.field2_rendered:
                    raise RuntimeError("CoR-GS: field 2 was not rendered this step "
                                       "(corgs-field2 loss missing)")
                self.field2_missed += 1
            else:
                self.field2_rendered += 1
                self.step_field2()
        row = {"step": step}
        for k, field in enumerate(self.fields, start=1):
            if step < int(cfg["densify_until_iter"]):
                self.accumulate(field)
                if (step > int(cfg["densify_from_iter"])
                        and step % int(cfg["densification_interval"]) == 0):
                    row[f"field{k}"] = self.densify_and_prune(field, step)
            if self.reset_due(step):
                reset_opa(field.params, field.optimizers, field.state,
                          value=float(cfg["opacity_reset_value"]))
                row[f"reset{k}"] = True
            field.info = None
        if self.field2 is not None and self.coprune_due(step):
            row.update(self.coprune(step))
            print(json.dumps({"corgs": row}), flush=True)  # evidence in the training log
        if len(row) > 1:
            self.log.append(row)
        for k, field in enumerate(self.fields, start=1):
            for name, p in field.params.items():
                if not (p.is_leaf and p.requires_grad):
                    raise RuntimeError(f"CoR-GS density left field{k}.{name} non-trainable")
                opt = field.optimizers.get(name)
                if opt is not None and opt.param_groups[0]["params"][0] is not p:
                    raise RuntimeError(f"CoR-GS density: stale optimizer for field{k}.{name}")


def create_density(spec):
    return CoRGSDensity(spec)
