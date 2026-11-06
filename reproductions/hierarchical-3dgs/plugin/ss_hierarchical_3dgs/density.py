"""Training schedule of the official single-chunk pipeline (scripts/full_train.py):

  steps 1 .. leaves_steps          train_single.py  (leaf / per-chunk optimization, Sec. 5.2)
  end of step leaves_steps         GaussianHierarchyCreator (Sec. 3.1, 4.2) -> state replaced
  steps leaves_steps+1 .. total    train_post.py    (hierarchy optimization, Sec. 4.1)

Run with `--steps leaves_steps` for the unoptimized hierarchy ("Ours (tau)") and with
`--steps leaves_steps + 15000` for the optimized one ("Ours opt (tau)").

Leaf phase (train_single.py, scene/gaussian_model.py; the runtime calls `after_backward` after
the optimizer step, the official loop densifies just before it):
  * opacity init 0.01 (create_from_pcd), per-field lrs of OptimizationParams, Adam eps 1e-15,
    sparse "OurAdam" (optim.SparseAdam), position lr expon 2e-5 -> 2e-7 (x extent) over 30000.
  * s < densify_until: for visible Gaussians (radii > 0)
        max_radii2D = max(max_radii2D, radii);  accum = max(accum, ||dL/dmeans2D[:2]||)   (MAX, Sec. 5.2)
    s > densify_from and s % 300 == 0: densify_and_prune
        clone: accum * max_radii2D * sigma(o)^(1/5) >= 0.015, sigma(o) > 0.15, max s <= pd*extent
        split: same on the pre-clone statistics (clones padded with 0), max s > pd*extent,
               2 samples, scale / (0.8*2);  prune sigma(o) < 0.005;  max_radii2D, accum reset
    s % 3000 == 0: reset_opacity  o <- min(sigma(o), 0.01)
  * big-Gaussian shrink (max scale > 0.02 * extent -> x0.8 every step, train_single.py:180-186) is
    OFF by default: the official single-chunk evaluation command (README.md:550) passes
    --skip_scale_big_gauss. `skip_scale_big_gauss: False` re-enables it (chunk-pipeline variant).
  * extent = 1.1 x the 0.9-quantile of training-camera distances from their mean
    (getNerfppNorm, scene/dataset_readers.py:58), recomputed in set_training_cameras; the
    runtime's scene_scale (1.1 x max) is used only until then. It scales the position lr, the
    percent_dense clone/split boundary and the shrink threshold.
Hierarchy phase (train_post.py): torch Adam over all hierarchy Gaussians with the anchors'
gradients zeroed (optim.FrozenAdam), lrs of full_train.py, position lr schedule restarted.
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from torch import nn

from . import hier
from .optim import FrozenAdam, SparseAdam, StepMask

DEFAULTS = {
    "leaves_steps": 30000,  # OptimizationParams.iterations (train_single)
    "init_opacity": 0.01,  # create_from_pcd
    "position_lr_init": 0.00002, "position_lr_final": 0.0000002,
    "position_lr_max_steps": 30000,
    "feature_lr": 0.0025, "opacity_lr": 0.05, "scaling_lr": 0.005, "rotation_lr": 0.001,
    "percent_dense": 0.0001,
    "densify_from": 500, "densify_until": 15000, "densify_every": 300,
    "grad_threshold": 0.015, "split_min_opacity": 0.15, "min_opacity": 0.005,
    "reset_every": 3000, "reset_value": 0.01,
    # README.md:550 eval protocol passes --skip_scale_big_gauss (train_single.py:180)
    "skip_scale_big_gauss": True, "big_gauss_factor": 0.02, "big_gauss_shrink": 0.8,
    # getNerfppNorm (dataset_readers.py:58): 1.1 x quantile(dist, 0.9); None = runtime scene_scale
    "extent_quantile": 0.9,
    # train_post.py as launched by scripts/full_train.py:143-144
    "post_feature_lr": 0.0005, "post_opacity_lr": 0.01, "post_scaling_lr": 0.001,
    "post_rotation_lr": 0.001,
    "post_steps": 15000,  # train_post.py --iterations (full_train.py:143)
}
REQUIRES = {"gradient.screen.signed"}
BASE = ("means", "scales", "quats", "opacities", "sh0", "shN")


def expon_lr(step, lr_init, lr_final, max_steps):
    """utils/general_utils.get_expon_lr_func with lr_delay_steps = 0."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0) or lr_init == 0:
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def quat_to_rotmat(quats):
    q = quats / quats.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def append(params, optimizers, new):
    """cat_tensors_to_optimizer: append rows with zero Adam moments."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                            requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), dtype=v.dtype, device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def camera_extent(cameras, quantile=0.9):
    """getNerfppNorm of the official code: 1.1 x quantile(||c_i - mean(c)||, q) over the
    training-camera centres (np.quantile, linear interpolation = torch.quantile)."""
    centres = torch.stack([torch.linalg.inv(c["viewmat"].double())[:3, 3] for c in cameras])
    dist = torch.linalg.vector_norm(centres - centres.mean(dim=0), dim=1)
    return 1.1 * float(torch.quantile(dist, quantile))


def clone_select(grads, radii, opacity, max_scale, cfg, extent):
    """densify_and_clone selection (gaussian_model.py:652-656)."""
    sel = grads * radii * opacity.pow(1 / 5.0) >= cfg["grad_threshold"]
    sel &= opacity > cfg["split_min_opacity"]
    return sel & (max_scale <= cfg["percent_dense"] * extent)


def split_select(grads, radii, opacity, max_scale, cfg, extent):
    """densify_and_split selection (gaussian_model.py:625-629)."""
    sel = grads * radii * opacity.pow(1 / 5.0) >= cfg["grad_threshold"]
    sel &= opacity > cfg["split_min_opacity"]
    return sel & (max_scale > cfg["percent_dense"] * extent)


class HierarchicalDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown hierarchical-3dgs density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.cameras = None
        self.generator = None
        self._optimizers = None
        self.extent = None

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        q = self.cfg["extent_quantile"]
        if q is not None and self.cameras:
            self.extent = camera_extent(self.cameras, q)
            if self._optimizers is not None:  # called after initialize by the runtime
                self._set_position_lr(self._optimizers, 1)
                self.state["extent"] = self.extent

    # ------------------------------------------------------------------ optimizers
    def _leaf_optimizers(self, params, optimizers):
        c = self.cfg
        lrs = {"means": c["position_lr_init"] * self.extent, "sh0": c["feature_lr"],
               "shN": c["feature_lr"] / 20.0, "opacities": c["opacity_lr"],
               "scales": c["scaling_lr"], "quats": c["rotation_lr"], hier.FIELD: 0.0}
        self.mask = StepMask(params)
        for name in list(optimizers):
            optimizers[name] = SparseAdam([params[name]], lr=lrs[name], mask=self.mask)

    def _post_optimizers(self, params, optimizers, anchors):
        c = self.cfg
        lrs = {"means": c["position_lr_init"] * self.extent, "sh0": c["post_feature_lr"],
               "shN": c["post_feature_lr"] / 20.0, "opacities": c["post_opacity_lr"],
               "scales": c["post_scaling_lr"], "quats": c["post_rotation_lr"], hier.FIELD: 0.0}
        for name in list(optimizers):
            optimizers[name] = FrozenAdam([params[name]], lr=lrs[name], frozen=anchors)

    def _set_position_lr(self, optimizers, next_step):
        c = self.cfg
        if next_step <= c["leaves_steps"]:
            it = next_step
        else:  # train_post.py restarts update_learning_rate(iteration) from 1
            it = next_step - c["leaves_steps"]
        optimizers["means"].param_groups[0]["lr"] = expon_lr(
            it, c["position_lr_init"] * self.extent, c["position_lr_final"] * self.extent,
            c["position_lr_max_steps"])

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE) - {hier.FIELD}
        if extra or hier.FIELD not in params:
            raise ValueError("hierarchical-3dgs density needs exactly the base fields + h3d")
        if self.extent is None:  # replaced by the 0.9-quantile extent in set_training_cameras
            self.extent = float(scene_scale)
        self.runtime_scene_scale = float(scene_scale)
        self._optimizers = optimizers  # same dict object the runtime steps
        with torch.no_grad():
            logit = math.log(self.cfg["init_opacity"] / (1 - self.cfg["init_opacity"]))
            params["opacities"].fill_(logit)
        self._leaf_optimizers(params, optimizers)
        self._set_position_lr(optimizers, 1)
        n = len(params["means"])
        self._reset_stats(n, params["means"].device)
        self.state = {"events": [], "resets": [], "hierarchy": None, "extent": self.extent,
                      "runtime_scene_scale": self.runtime_scene_scale}

    def before_backward(self, params, optimizers, step, outputs):
        if hasattr(self, "mask"):
            self.mask.begin(step)
        if step <= self.cfg["leaves_steps"] and step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step <= c["leaves_steps"]:
            if step < c["densify_until"]:
                self._accumulate(outputs["info"])
                if step > c["densify_from"] and step % c["densify_every"] == 0:
                    self._densify(params, optimizers, step)
                if c["reset_every"] > 0 and step % c["reset_every"] == 0:
                    self.reset_opacity(params, optimizers, step)
            if not c["skip_scale_big_gauss"]:
                self.shrink_big(params)
            if step == c["leaves_steps"]:
                self.build_hierarchy(params, optimizers)
        elif step == c["leaves_steps"] + c["post_steps"]:
            # Training is complete (train_post.py stops here): drop the Adam moments so the
            # runtime's checkpoint holds the hierarchy only (~3x smaller; no effect on results).
            for opt in optimizers.values():
                opt.state.clear()
        self._set_position_lr(optimizers, step + 1)

    # ------------------------------------------------------------------ leaf phase
    def _reset_stats(self, n, device):
        self.accum = torch.zeros(n, device=device)
        self.max_radii = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        if g2 is None:
            raise RuntimeError("screen-space gradient missing: use the hierarchical-3dgs renderer")
        radii = info["radii"]
        vis = radii > 0
        self.max_radii = torch.where(vis, torch.maximum(self.max_radii, radii.float()),
                                     self.max_radii)
        self.accum = torch.where(vis, torch.maximum(torch.norm(g2[:, :2], dim=-1), self.accum),
                                 self.accum)

    def _densify(self, params, optimizers, step):
        c, ext = self.cfg, self.extent
        n0 = len(params["means"])
        grads = self.accum.clone()
        grads[grads.isnan()] = 0.0
        radii = self.max_radii
        op = params["opacities"].sigmoid()
        ms = params["scales"].exp().max(dim=1).values
        clone = clone_select(grads, radii, op, ms, c, ext)
        n_clone = int(clone.sum())
        append(params, optimizers, {k: params[k][clone] for k in params})
        pad = torch.zeros(n_clone, device=grads.device)
        grads = torch.cat([grads, pad])
        radii = torch.cat([radii, pad])  # densification_postfix: new max_radii2D = 0
        op = params["opacities"].sigmoid()
        ms = params["scales"].exp().max(dim=1).values
        split = split_select(grads, radii, op, ms, c, ext)
        k = int(split.sum())
        stds = params["scales"][split].exp().repeat(2, 1)
        samples = torch.normal(torch.zeros_like(stds), stds, generator=self.generator)
        rots = quat_to_rotmat(params["quats"][split]).repeat(2, 1, 1)
        new = {"means": torch.bmm(rots, samples[..., None])[..., 0]
               + params["means"][split].repeat(2, 1),
               "scales": torch.log(params["scales"][split].exp().repeat(2, 1) / (0.8 * 2)),
               "quats": params["quats"][split].repeat(2, 1),
               "opacities": params["opacities"][split].repeat(2),
               "sh0": params["sh0"][split].repeat(2, 1, 1),
               "shN": params["shN"][split].repeat(2, 1, 1),
               hier.FIELD: params[hier.FIELD][split].repeat(2, 1)}
        append(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat(
            [split, torch.zeros(2 * k, dtype=torch.bool, device=split.device)]))
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        n_prune = int(prune.sum())
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "clone": n_clone, "split": k,
               "pruned": n_prune, "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1500 == 0:
            print({"hierarchical-3dgs": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_opacity(self, params, optimizers, step=0):
        """reset_opacity + replace_tensor_to_optimizer (moments zeroed)."""
        new = torch.logit(torch.clamp(params["opacities"].sigmoid(), max=self.cfg["reset_value"]))

        def param_fn(name, p):
            return nn.Parameter(new.to(p.dtype), requires_grad=p.requires_grad)

        _update_param_with_optimizer(param_fn, lambda key, v: torch.zeros_like(v), params,
                                     optimizers, ["opacities"])
        self.state["resets"].append(int(step))

    def shrink_big(self, params):
        """train_single.py:183-190."""
        s = params["scales"].data
        violators = s.exp().max(dim=1).values > self.extent * self.cfg["big_gauss_factor"]
        s.copy_(torch.where(violators[:, None],
                            torch.log(s.exp() * self.cfg["big_gauss_shrink"]), s))

    # ------------------------------------------------------------------ hierarchy
    def build_hierarchy(self, params, optimizers):
        if not self.cameras:
            raise RuntimeError("hierarchy creation needs set_training_cameras(...)")
        dev = params["means"].device
        n_leaves = len(params["means"])
        h = hier.build(params, self.cameras)
        n = len(h["means"])
        blob = hier.pack(h["nodes"], h["boxes"], h["anchors"], n)
        new = {"means": h["means"], "scales": h["scales"], "quats": h["rots"],
               "opacities": h["alpha"][:, 0],  # falloff, activation abs (create_from_hier)
               "sh0": h["shs"][:, :1].contiguous(), "shN": h["shs"][:, 1:].contiguous(),
               hier.FIELD: blob}
        for name in list(params):
            params[name] = nn.Parameter(new[name].to(dev, torch.float32).contiguous(),
                                        requires_grad=True)
        anchors = torch.zeros(n, dtype=torch.bool, device=dev)
        anchors[h["anchors"].to(dev)] = True
        self._post_optimizers(params, optimizers, anchors)
        self.state["hierarchy"] = {
            "leaves_trained": n_leaves, "leaves_valid": h["valid_leaves"], "gaussians": n,
            "nodes": len(h["nodes"]), "anchors": int(anchors.sum()),
            "max_depth": int(h["nodes"][:, 0].max())}
        print({"hierarchical-3dgs-hierarchy": self.state["hierarchy"]}, flush=True)
        self.accum = self.max_radii = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_density(spec):
    return HierarchicalDensity(spec)
