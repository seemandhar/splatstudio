"""AD-GS density control for both co-trained fields, and ownership of the second field G2.

Densification follows the official `GaussianModel.densify_and_prune` (scene/gaussian_model.py:
425-576): clone (grad >= tau, max scale <= percent_dense * extent), then split (grad >= tau,
max scale > percent_dense * extent, OR the FSGS criterion mean squared 3-NN distance >
dist_thres * extent and max scale > extent), then prune opacity < epsilon (no screen-size
pruning: size_threshold = None). (tau, epsilon) depend on the AD-GS phase (train.py:384-401).
"""

import numpy as np
import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from .pseudo import depth_bounds, random_poses_llff
from .schedule import Schedule, schedule_config

DEFAULTS = {
    "warmup_iters": 1500,
    "cycle_len": 200,
    "high_dens_iters": 100,
    "warmup_grad_threshold": 0.0005,
    "warmup_prune_threshold": 0.005,
    "low_grad_threshold": 0.0005,
    "low_prune_threshold": 0.1,
    "high_grad_threshold": 0.0002,
    "high_prune_threshold": 0.005,
    "densification_interval": 100,
    "densify_from_iter": 500,
    "densify_until_iter": 10000,
    "prune_from_iter": 500,
    "percent_dense": 0.01,
    "dist_thres": 10.0,
    "opacity_reset_value": 0.05,
    "opacity_reset_interval": 3000,
    "opacity_reset_after": 500,  # start_sample_pseudo: resets at it > 500, (it-501) % 3000 == 0
    "n_pseudo_poses": 10000,
    "pseudo_seed": 0,
    "fields": 2,
    # Iteration 1: LLFF near/far of the training views from poses_bounds.npy, as the official
    # generate_random_poses_llff uses them (only bounds.min() / bounds.max() enter). Components
    # cannot read dataset files, so the values (recorded by prepare_llff_fewview.py in
    # prep.json) are matched to the scene by its training-camera centres (OpenGL c2w, world).
    # "points" = iteration-0 behaviour (percentiles of the init cloud).
    "pseudo_bounds": "official",
    "scene_bounds": {
        "horns": {"near": 12.473701480698228, "far": 168.81094102559487,
                  "train_centers": [[-3.77684, -3.147624, -1.43279], [3.351668, 0.198719, 0.574782],
                                    [3.00166, 2.723918, 0.929056]]},
        "fortress": {"near": 13.472437825125102, "far": 128.38342286860328,
                     "train_centers": [[-2.343452, -2.985312, 1.402754],
                                       [-0.596759, 0.169303, -0.406764],
                                       [4.019429, 2.461339, 0.372861]]},
        "room": {"near": 10.729799505777695, "far": 48.929631822500625,
                 "train_centers": [[-3.619974, -3.791045, 0.452161], [0.16049, 2.171683, -0.405748],
                                   [-0.337692, 1.516422, 0.225718]]},
    },
}


def official_bounds(table, viewmats, tol=1e-3):
    """Return (scene, [[near, far]]) whose recorded training-camera centres match these views."""
    centers = np.stack([np.linalg.inv(v)[:3, 3] for v in viewmats])
    for scene, entry in table.items():
        ref = np.asarray(entry["train_centers"], dtype=np.float64)
        if ref.shape == centers.shape and np.abs(ref - centers).max() < tol:
            return scene, np.array([[float(entry["near"]), float(entry["far"])]])
    raise ValueError("pseudo_bounds='official': no scene_bounds entry matches the training "
                     "cameras (add it from prep.json or set pseudo_bounds='points')")


def thresholds(config, phase):
    return float(config[f"{phase}_grad_threshold"]), float(config[f"{phase}_prune_threshold"])


def mean_sq_knn(query, points, k=3, chunk=512):
    """Mean squared distance to the k nearest other points (simple-knn distCUDA2)."""
    out = torch.empty(len(query), device=points.device)
    for s in range(0, len(query), chunk):
        d2 = torch.cdist(query[s : s + chunk], points).square()
        out[s : s + chunk] = d2.topk(k + 1, dim=1, largest=False).values[:, 1:].mean(dim=1)
    return out


class Field:
    """One Gaussian field with per-field Adam optimizers and densification statistics."""

    def __init__(self, params, optimizers):
        self.params, self.optimizers = params, optimizers
        self.state = {"grad2d": None, "count": None}

    @classmethod
    def clone_from(cls, params, optimizers):
        new = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()}
        )
        opts = {
            k: torch.optim.Adam([p], lr=optimizers[k].param_groups[0]["lr"], eps=1e-15)
            for k, p in new.items()
        }
        return cls(new, opts)

    def accumulate(self, info):
        # 3DGS add_densification_stats: |dL/d(NDC xy)| on visible Gaussians (gsplat scaling).
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        grads = means2d.grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(self.params["means"])
        if self.state["grad2d"] is None or len(self.state["grad2d"]) != n:
            self.state["grad2d"] = torch.zeros(n, device=grads.device)
            self.state["count"] = torch.zeros(n, device=grads.device)
        sel = (info["radii"] > 0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold, min_opacity, extent, cfg):
        p, o, st = self.params, self.optimizers, self.state
        n = len(p["means"])
        if st["grad2d"] is None:
            st["grad2d"] = torch.zeros(n, device=p["means"].device)
            st["count"] = torch.zeros(n, device=p["means"].device)
        grads = st["grad2d"] / st["count"]
        grads[grads.isnan()] = 0.0
        pd = float(cfg["percent_dense"])
        # densify_and_clone
        is_dup = (grads >= grad_threshold) & (p["scales"].exp().max(dim=1).values <= pd * extent)
        n_dup = int(is_dup.sum())
        if n_dup:
            duplicate(p, o, st, is_dup)
        # densify_and_split on the post-clone set; clones have zero (padded) gradient
        padded = torch.cat([grads, torch.zeros(n_dup, device=grads.device)])
        max_scale = p["scales"].exp().max(dim=1).values
        is_split = (padded >= grad_threshold) & (max_scale > pd * extent)
        big = max_scale > extent  # FSGS criterion (gaussian_model.py:436-439)
        fsgs = torch.zeros_like(is_split)
        if big.any():
            far = mean_sq_knn(p["means"][big], p["means"]) > float(cfg["dist_thres"]) * extent
            fsgs[torch.where(big)[0][far]] = True
        is_split |= fsgs
        n_split = int(is_split.sum())
        if n_split:
            split(p, o, st, is_split)
        is_prune = p["opacities"].sigmoid() < min_opacity
        n_prune = int(is_prune.sum())
        if n_prune:
            remove(p, o, st, is_prune)
        m = len(p["means"])
        st["grad2d"] = torch.zeros(m, device=p["means"].device)
        st["count"] = torch.zeros(m, device=p["means"].device)
        return {"cloned": n_dup, "split": n_split, "fsgs_split": int(fsgs.sum()),
                "pruned": n_prune, "gaussians": m}

    def reset_opacity(self, value):
        reset_opa(self.params, self.optimizers, self.state, value=value)


class ADGSDensity:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown AD-GS density options: {sorted(unknown)}")
        self.config = {**DEFAULTS, **spec.config}
        self.schedule = Schedule.from_config(schedule_config(self.config))
        if int(self.config["fields"]) not in (1, 2):
            raise ValueError("fields must be 1 or 2")
        if int(self.config["densify_from_iter"]) < int(self.config["prune_from_iter"]):
            # Official prune_points is a no-op for it <= prune_from_iter (split parents kept);
            # with the paper values (500/500) every densification also prunes.
            raise ValueError("densify_from_iter must be >= prune_from_iter")
        self.state = None
        self.partner = None
        self.extent = 1.0
        self.pseudo_cameras = []
        self.pseudo_info = {}
        self._order = []
        self._rng = np.random.default_rng(int(self.config["pseudo_seed"]))
        self.means_lr = None
        self.partner_info = None
        self.log = []

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)
        self.main = Field(params, optimizers)
        self.state = self.main.state
        self._init_points = params["means"].detach().cpu().numpy().astype(np.float64)
        # CoR-GS/AD-GS: G2 is built from the same point cloud with the same initialisation.
        self.partner = (
            Field.clone_from(params, optimizers) if int(self.config["fields"]) == 2 else None
        )

    def set_training_cameras(self, cameras):
        viewmats = [c["viewmat"].detach().cpu().numpy().astype(np.float64) for c in cameras]
        Ks = [c["K"].detach().cpu().numpy().astype(np.float64) for c in cameras]
        sizes = [(int(c["width"]), int(c["height"])) for c in cameras]
        mode = self.config["pseudo_bounds"]
        if mode == "official":
            scene, bounds = official_bounds(self.config["scene_bounds"], viewmats)
        elif mode == "points":
            scene, bounds = None, depth_bounds(self._init_points, viewmats, Ks, sizes)
        else:
            raise ValueError("pseudo_bounds must be 'official' or 'points'")
        poses, info = random_poses_llff(
            viewmats, bounds, int(self.config["n_pseudo_poses"]), self._rng
        )
        cam0 = cameras[0]
        w, h = int(cam0["width"]), int(cam0["height"])
        device = cam0["K"].device
        # Official PseudoCamera: FoV of the first training view, centred principal point.
        K = cam0["K"].detach().clone()
        K[0, 2], K[1, 2] = w / 2.0, h / 2.0
        self.pseudo_viewmats = torch.tensor(poses, dtype=torch.float32, device=device)
        self.pseudo_K, self.pseudo_size = K, (w, h)
        self.pseudo_info = {"bounds": bounds.tolist(), "bounds_source": mode, "scene": scene,
                            **info}

    def next_pseudo_camera(self):
        if not hasattr(self, "pseudo_viewmats"):
            raise RuntimeError("AD-GS pseudo views need set_training_cameras (native trainer)")
        if not self._order:
            self._order = list(self._rng.permutation(len(self.pseudo_viewmats)))
        i = int(self._order.pop())
        w, h = self.pseudo_size
        return {"viewmat": self.pseudo_viewmats[i], "K": self.pseudo_K, "width": w, "height": h}

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()
        self.means_lr = optimizers["means"].param_groups[0]["lr"]
        outputs["adgs"] = self

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.config
        self.main.params, self.main.optimizers = params, optimizers
        fields = [(self.main, outputs["info"])]
        if self.partner is not None:
            # G2 gradients were produced by the co-training loss in this step.
            self.partner.optimizers["means"].param_groups[0]["lr"] = self.means_lr
            for opt in self.partner.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            fields.append((self.partner, self.partner_info))
        if step < int(cfg["densify_until_iter"]):
            for field, info in fields:
                if info is not None:
                    field.accumulate(info)
            if step > int(cfg["densify_from_iter"]) and step % int(cfg["densification_interval"]) == 0:
                phase = self.schedule.phase(step)
                tau, eps = thresholds(cfg, phase)
                for k, (field, _) in enumerate(fields):
                    stats = field.densify_and_prune(tau, eps, self.extent, cfg)
                    self.log.append({"step": step, "field": k, "phase": phase, **stats})
        after = int(cfg["opacity_reset_after"])
        if step > after and (step - after - 1) % int(cfg["opacity_reset_interval"]) == 0:
            for field, _ in fields:
                field.reset_opacity(float(cfg["opacity_reset_value"]))
        self.state = self.main.state
        self.partner_info = None


def create_density(spec):
    return ADGSDensity(spec)
